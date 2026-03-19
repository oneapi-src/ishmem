# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""Optional Torch/XPU interoperability helpers for ``ishmem4py``."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Iterable

from ._common import IshmemStateError, _PointerInfo, _require_initialized
from ._lib import RUNTIME

try:
    import torch as _torch
except Exception:
    _torch = None

try:
    from . import _ishmem4py_torch as _TORCH_EXT
except Exception:
    _TORCH_EXT = None

__all__ = [
    "free_tensor",
    "get_peer_tensor",
    "is_symmetric_tensor",
    "tensor",
    "tensor_base",
]


_DTYPE_TO_SCALAR_TYPE = {
    _torch.uint8 if _torch is not None else object(): 0,
    _torch.int8 if _torch is not None else object(): 1,
    _torch.int16 if _torch is not None else object(): 2,
    _torch.int32 if _torch is not None else object(): 3,
    _torch.int64 if _torch is not None else object(): 4,
    _torch.float16 if _torch is not None else object(): 5,
    _torch.float32 if _torch is not None else object(): 6,
    _torch.float64 if _torch is not None else object(): 7,
    _torch.bool if _torch is not None else object(): 11,
    _torch.bfloat16 if _torch is not None else object(): 15,
}


@dataclass
class _TensorAllocation:
    tensor: "_torch.Tensor"
    ptr: int
    size_bytes: int


_live_tensor_allocations: dict[int, _TensorAllocation] = {}


def _require_torch() -> None:
    if _torch is None:
        raise RuntimeError("Torch/XPU interop requires PyTorch to be installed")
    if _TORCH_EXT is None:
        raise RuntimeError(
            "Torch/XPU interop is unavailable in this ishmem4py build; rebuild ishmem with "
            "-DISHMEM4PY_BUILD_TORCH_INTEROP=ON using a Python executable from a PyTorch XPU environment"
        )


def _normalize_shape(size) -> tuple[int, ...]:
    if isinstance(size, int):
        return (int(size),)
    if isinstance(size, Iterable):
        return tuple(int(dim) for dim in size)
    raise TypeError("size must be an int or an iterable of ints")


def _normalize_device(device) -> "_torch.device":
    if _torch is None:
        raise RuntimeError("PyTorch is not available")
    if device is None:
        return _torch.device("xpu", _torch.xpu.current_device())
    normalized = _torch.device(device)
    if normalized.type != "xpu":
        raise ValueError(f"ishmem4py.torch.tensor only supports XPU devices, got {normalized}")
    if normalized.index is None:
        return _torch.device("xpu", _torch.xpu.current_device())
    return normalized


def _tensor_nbytes(tensor: "_torch.Tensor") -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _register_tensor(base: "_torch.Tensor") -> None:
    base_ptr = int(base.data_ptr())
    _live_tensor_allocations[base_ptr] = _TensorAllocation(
        tensor=base,
        ptr=base_ptr,
        size_bytes=_tensor_nbytes(base),
    )
    base._ishmem_alloc = True  # type: ignore[attr-defined]
    base._ishmem_base = base  # type: ignore[attr-defined]
    base._ishmem_free_owner = True  # type: ignore[attr-defined]


def _allocation_from_base(base: "_torch.Tensor") -> _TensorAllocation:
    allocation = _live_tensor_allocations.get(int(base.data_ptr()))
    if allocation is None:
        raise IshmemStateError("symmetric tensor allocation is not live")
    return allocation


def _find_allocation_for_tensor(tensor: "_torch.Tensor") -> _TensorAllocation | None:
    ptr = int(tensor.data_ptr())
    size_bytes = _tensor_nbytes(tensor)
    for allocation in _live_tensor_allocations.values():
        start = allocation.ptr
        end = allocation.ptr + allocation.size_bytes
        if ptr < start:
            continue
        if ptr > end:
            continue
        if size_bytes > 0 and ptr + size_bytes > end:
            continue
        return allocation
    return None


def _assert_contiguous_tensor(tensor: "_torch.Tensor") -> None:
    if tensor.numel() != 0 and not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")


def _scalar_type_from_dtype(dtype) -> int:
    scalar_type = _DTYPE_TO_SCALAR_TYPE.get(dtype)
    if scalar_type is None:
        raise TypeError(f"unsupported torch dtype for ishmem4py Torch interop: {dtype}")
    return scalar_type


def _remote_tensor_ptr(base: "_torch.Tensor", *, pe: int) -> int:
    address = int(RUNTIME.ishmem4py_ptr(ctypes.c_void_p(int(base.data_ptr())), int(pe)) or 0)
    if address == 0:
        raise IshmemStateError(f"ishmem_ptr returned NULL for PE {pe}")
    return address


def _mark_tensor_alias(tensor: "_torch.Tensor", *, base: "_torch.Tensor", owner: bool) -> None:
    tensor._ishmem_alias = True  # type: ignore[attr-defined]
    tensor._ishmem_alloc = True  # type: ignore[attr-defined]
    tensor._ishmem_base = base  # type: ignore[attr-defined]
    tensor._ishmem_free_owner = owner  # type: ignore[attr-defined]


def _pointer_from_tensor(tensor, *, writable: bool, offset: int = 0) -> _PointerInfo | None:
    if _torch is None or not isinstance(tensor, _torch.Tensor):
        return None
    _assert_contiguous_tensor(tensor)
    size_bytes = _tensor_nbytes(tensor)
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if offset > size_bytes:
        raise ValueError("offset is past the end of the tensor")
    if writable and not tensor.is_leaf and tensor.requires_grad:
        raise ValueError("destination tensor must be writable")
    return _PointerInfo(ptr=int(tensor.data_ptr()) + offset, size=size_bytes - offset, keepalive=tensor)


def _symmetric_tensor_target(obj) -> _PointerInfo | None:
    if _torch is None or not isinstance(obj, _torch.Tensor):
        return None
    _assert_contiguous_tensor(obj)
    allocation = _find_allocation_for_tensor(obj)
    if allocation is None:
        return None
    return _PointerInfo(ptr=int(obj.data_ptr()), size=_tensor_nbytes(obj), keepalive=obj)


def _live_tensor_pointers() -> list[int]:
    return sorted(_live_tensor_allocations)


def tensor(
    size,
    *,
    dtype=None,
    device=None,
    requires_grad: bool = False,
):
    """Allocate a symmetric XPU tensor from the Intel SHMEM heap."""
    _require_initialized()
    _require_torch()
    shape = _normalize_shape(size)
    if dtype is None:
        dtype = _torch.get_default_dtype()
    scalar_type = _scalar_type_from_dtype(dtype)

    normalized_device = _normalize_device(device)
    with _torch.xpu.device(normalized_device):
        result = _TORCH_EXT.alloc_tensor(shape, scalar_type, normalized_device.index)

    if result.numel() > 0:
        _register_tensor(result)
    else:
        result._ishmem_alloc = True  # type: ignore[attr-defined]
        result._ishmem_base = result  # type: ignore[attr-defined]
        result._ishmem_free_owner = True  # type: ignore[attr-defined]
    if requires_grad:
        result.requires_grad_(True)
    return result


def tensor_base(tensor):
    """Return the base symmetric allocation for ``tensor`` when available."""
    if _torch is None or not isinstance(tensor, _torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    base = getattr(tensor, "_ishmem_base", None)
    if base is not None:
        return base
    allocation = _find_allocation_for_tensor(tensor)
    if allocation is None:
        return tensor
    return allocation.tensor


def get_peer_tensor(tensor, pe: int):
    """Return an XPU tensor alias for ``tensor`` on remote PE ``pe``."""
    _require_initialized()
    _require_torch()
    if not isinstance(tensor, _torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    if tensor.device.type != "xpu":
        raise ValueError(f"get_peer_tensor only supports XPU tensors, got {tensor.device}")
    if tensor.device.index is None:
        raise ValueError("get_peer_tensor requires a concrete XPU device index")

    _assert_contiguous_tensor(tensor)
    base = tensor_base(tensor)
    if not is_symmetric_tensor(base):
        raise IshmemStateError("tensor is not backed by ishmem4py symmetric memory")

    byte_offset = int(tensor.data_ptr()) - int(base.data_ptr())
    if byte_offset < 0:
        raise IshmemStateError("tensor points before its tracked symmetric base")
    remote_ptr = _remote_tensor_ptr(base, pe=pe) + byte_offset
    result = _TORCH_EXT.tensor_from_ptr(
        remote_ptr,
        list(tensor.shape),
        _scalar_type_from_dtype(tensor.dtype),
        tensor.device.index,
    )
    _mark_tensor_alias(result, base=base, owner=False)
    result._ishmem_peer_pe = int(pe)  # type: ignore[attr-defined]
    result._ishmem_keepalive = tensor  # type: ignore[attr-defined]
    return result


def is_symmetric_tensor(tensor) -> bool:
    """Return ``True`` when ``tensor`` references live symmetric memory."""
    if _torch is None or not isinstance(tensor, _torch.Tensor):
        return False
    return _find_allocation_for_tensor(tensor) is not None or bool(
        getattr(tensor, "_ishmem_alloc", False)
    )


def free_tensor(tensor) -> None:
    """Free a symmetric tensor previously returned by :func:`tensor`."""
    _require_initialized()
    _require_torch()
    if not isinstance(tensor, _torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    if bool(getattr(tensor, "_ishmem_alias", False)) and not bool(
        getattr(tensor, "_ishmem_free_owner", False)
    ):
        raise IshmemStateError("tensor is a non-owning symmetric alias and cannot be freed")
    base = tensor_base(tensor)
    if not getattr(base, "_ishmem_alloc", False):
        raise IshmemStateError("tensor is not backed by ishmem4py symmetric memory")
    base_ptr = int(base.data_ptr())
    allocation = _live_tensor_allocations.pop(base_ptr, None)
    if allocation is None:
        if base_ptr == 0:
            base._ishmem_alloc = False  # type: ignore[attr-defined]
            return
        raise IshmemStateError("symmetric tensor allocation has already been freed")
    _TORCH_EXT.free_ptr(base_ptr)
    allocation.tensor._ishmem_alloc = False  # type: ignore[attr-defined]
