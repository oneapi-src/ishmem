# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""Shared types and validation helpers for the public ``ishmem4py`` API."""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class IshmemError(RuntimeError):
    """Base exception for ``ishmem4py`` failures."""

    pass


class IshmemStateError(IshmemError):
    """Raised when Intel SHMEM is used in an invalid process state."""

    pass


class Comparison(IntEnum):
    """Comparison operators for wait/test style APIs."""

    EQ = 1
    NE = 2
    GT = 3
    GE = 4
    LT = 5
    LE = 6


class SignalOp(IntEnum):
    """Signal operations retained for API compatibility."""

    SET = 0
    ADD = 1


class InitStatus(IntEnum):
    """Process-local Intel SHMEM initialization state."""

    UNINITIALIZED = 0
    INITIALIZED = 1
    FINALIZED = 2


@dataclass(frozen=True)
class Version:
    """Combined OpenSHMEM, Intel SHMEM, and Python binding version metadata."""

    openshmem_spec_version: str
    ishmem4py_version: str
    libishmem_version: str
    vendor_name: str


@dataclass(frozen=True)
class Team:
    """Python wrapper for an Intel SHMEM team handle."""

    handle: int
    name: Optional[str] = None

    def __int__(self) -> int:
        return self.handle

    @property
    def valid(self) -> bool:
        return self.handle != -1

    def __repr__(self) -> str:
        label = self.name if self.name is not None else f"TEAM_{self.handle}"
        return f"Team(handle={self.handle}, name={label!r})"


@dataclass(frozen=True)
class TeamConfig:
    """Subset of ``ishmem_team_config_t`` exposed by the MVP bindings."""

    num_contexts: int = 0


@dataclass(frozen=True)
class MemoryPointer:
    """A PE-relative address returned by ``ptr`` or ``ishmem_ptr``."""

    address: int
    size: int
    pe: int
    base: "SymmetricMemory"

    def __int__(self) -> int:
        return self.address

    def __bool__(self) -> bool:
        return self.address != 0


@dataclass(frozen=True)
class _DTypeInfo:
    name: str
    code: int
    ctype: type[ctypes._SimpleCData]
    itemsize: int


_DTYPES = {
    "int32": _DTypeInfo("int32", 0, ctypes.c_int32, 4),
    "int64": _DTypeInfo("int64", 1, ctypes.c_int64, 8),
    "uint32": _DTypeInfo("uint32", 2, ctypes.c_uint32, 4),
    "uint64": _DTypeInfo("uint64", 3, ctypes.c_uint64, 8),
    "float32": _DTypeInfo("float32", 4, ctypes.c_float, 4),
    "float64": _DTypeInfo("float64", 5, ctypes.c_double, 8),
}

_DTYPE_ALIASES = {
    "i": "int32",
    "int": "int32",
    "int32": "int32",
    "l": "int64",
    "q": "int64",
    "long": "int64",
    "int64": "int64",
    "i4": "int32",
    "i8": "int64",
    "u": "uint32",
    "I": "uint32",
    "uint": "uint32",
    "uint32": "uint32",
    "Q": "uint64",
    "L": "uint64",
    "ulong": "uint64",
    "uint64": "uint64",
    "u4": "uint32",
    "u8": "uint64",
    "f": "float32",
    "float": "float32",
    "float32": "float32",
    "f4": "float32",
    "d": "float64",
    "double": "float64",
    "float64": "float64",
    "f8": "float64",
}

_CTYPE_ALIASES = {
    ctypes.c_int32: "int32",
    ctypes.c_int64: "int64",
    ctypes.c_uint32: "uint32",
    ctypes.c_uint64: "uint64",
    ctypes.c_float: "float32",
    ctypes.c_double: "float64",
}

_ATOMIC_FETCH_DTYPES = frozenset(_DTYPES)
_ATOMIC_STANDARD_DTYPES = frozenset({"int32", "int64", "uint32", "uint64"})
_ATOMIC_BITWISE_DTYPES = frozenset({"int32", "int64", "uint32", "uint64"})
_REDUCTION_DTYPES = {
    "sum": frozenset(_DTYPES),
    "prod": frozenset(_DTYPES),
    "and": frozenset({"int32", "int64", "uint32", "uint64"}),
    "or": frozenset({"int32", "int64", "uint32", "uint64"}),
    "xor": frozenset({"int32", "int64", "uint32", "uint64"}),
    "min": frozenset(_DTYPES),
    "max": frozenset(_DTYPES),
}
_REDUCTION_CODES = {
    "sum": 0,
    "prod": 1,
    "and": 2,
    "or": 3,
    "xor": 4,
    "min": 5,
    "max": 6,
}


class SymmetricMemory:
    """An opaque handle to a symmetric heap allocation."""

    def __init__(self, ptr: int, size: int):
        self._ptr = int(ptr)
        self._size = int(size)
        self._freed = False

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def size(self) -> int:
        return self._size

    @property
    def freed(self) -> bool:
        return self._freed

    def _mark_freed(self) -> None:
        self._freed = True

    def read(self, size: Optional[int] = None, *, offset: int = 0, pe: Optional[int] = None) -> bytes:
        """Fetch bytes from this symmetric allocation into a new ``bytes`` object."""
        from .rma import get
        from .init_fini import my_pe

        if pe is None:
            pe = my_pe()
        length = _normalize_span(self, size=size, offset=offset)
        dst = bytearray(length)
        get(dst, self, pe=pe, size=length, dest_offset=0, src_offset=offset)
        return bytes(dst)

    def write(self, data, *, offset: int = 0, pe: Optional[int] = None) -> int:
        """Store bytes from a Python buffer into this symmetric allocation."""
        from .rma import put
        from .init_fini import my_pe

        if pe is None:
            pe = my_pe()
        return put(self, data, pe=pe, dest_offset=offset)

    def __int__(self) -> int:
        return self._ptr

    def __repr__(self) -> str:
        state = "freed" if self._freed else "active"
        return f"SymmetricMemory(ptr=0x{self._ptr:x}, size={self._size}, state={state})"


TEAM_INVALID = Team(-1, "TEAM_INVALID")
TEAM_WORLD = Team(0, "TEAM_WORLD")
TEAM_SHARED = Team(1, "TEAM_SHARED")


_initialized = False
_finalized = False
_live_allocations: dict[int, SymmetricMemory] = {}


def _register_allocation(symm: SymmetricMemory) -> None:
    _live_allocations[id(symm)] = symm


def _unregister_allocation(symm: SymmetricMemory) -> None:
    _live_allocations.pop(id(symm), None)


def _require_initialized() -> None:
    if not _initialized:
        raise IshmemStateError("Intel SHMEM is not initialized")


def _set_initialized() -> None:
    global _initialized
    _initialized = True


def _set_finalized() -> None:
    global _initialized, _finalized
    _initialized = False
    _finalized = True


def _get_init_status() -> InitStatus:
    if _initialized:
        return InitStatus.INITIALIZED
    if _finalized:
        return InitStatus.FINALIZED
    return InitStatus.UNINITIALIZED


def _check_can_init() -> None:
    if _initialized:
        raise IshmemStateError("Intel SHMEM is already initialized")
    if _finalized:
        raise IshmemStateError(
            "Intel SHMEM has already been finalized in this process and cannot be reinitialized"
        )


def _check_can_finalize() -> None:
    if not _initialized:
        raise IshmemStateError("Intel SHMEM is not initialized")
    leaked_ptrs = [
        f"0x{symm.ptr:x}" for symm in sorted(_live_allocations.values(), key=lambda item: item.ptr)
    ]
    try:
        from . import torch as _ishmem_torch

        leaked_ptrs.extend(f"0x{ptr:x}" for ptr in _ishmem_torch._live_tensor_pointers())
    except Exception:
        pass
    if leaked_ptrs:
        leaked = ", ".join(leaked_ptrs)
        raise IshmemStateError(
            "cannot finalize while symmetric allocations are still live; free them first: " f"{leaked}"
        )


def _require_active(symm: SymmetricMemory) -> None:
    if not isinstance(symm, SymmetricMemory):
        raise TypeError("expected a SymmetricMemory instance")
    if symm.freed:
        raise IshmemStateError("symmetric allocation has already been freed")


def _normalize_span(obj, *, size: Optional[int], offset: int) -> int:
    if offset < 0:
        raise ValueError("offset must be >= 0")

    available = obj.size - offset
    if available < 0:
        raise ValueError("offset is past the end of the object")

    if size is None:
        return available
    if size < 0:
        raise ValueError("size must be >= 0")
    if size > available:
        raise ValueError("requested size extends past the end of the object")
    return size


def _ensure_contiguous_memoryview(obj, *, writable: bool):
    mv = memoryview(obj)
    if not mv.contiguous:
        raise ValueError("buffer must be contiguous")
    if writable and mv.readonly:
        raise ValueError("destination buffer must be writable")
    return mv


@dataclass(frozen=True)
class _PointerInfo:
    ptr: int
    size: int
    keepalive: object


def _pointer_from_local_buffer(obj, *, writable: bool, offset: int = 0) -> _PointerInfo:
    if isinstance(obj, SymmetricMemory):
        _require_active(obj)
        size = _normalize_span(obj, size=None, offset=offset)
        return _PointerInfo(ptr=obj.ptr + offset, size=size, keepalive=obj)

    try:
        from . import torch as _ishmem_torch

        tensor_info = _ishmem_torch._pointer_from_tensor(obj, writable=writable, offset=offset)
        if tensor_info is not None:
            return tensor_info
    except Exception:
        pass

    mv = _ensure_contiguous_memoryview(obj, writable=writable)

    if offset < 0:
        raise ValueError("offset must be >= 0")
    if offset > mv.nbytes:
        raise ValueError("offset is past the end of the buffer")

    if mv.readonly:
        if writable:
            raise ValueError("destination buffer must be writable")
        copied = (ctypes.c_char * mv.nbytes).from_buffer_copy(mv)
        return _PointerInfo(ptr=ctypes.addressof(copied) + offset, size=mv.nbytes - offset, keepalive=copied)

    raw = (ctypes.c_char * mv.nbytes).from_buffer(mv)
    return _PointerInfo(ptr=ctypes.addressof(raw) + offset, size=mv.nbytes - offset, keepalive=(mv, raw))


def _symmetric_target(obj):
    if isinstance(obj, SymmetricMemory):
        _require_active(obj)
        return obj

    try:
        from . import torch as _ishmem_torch

        target = _ishmem_torch._symmetric_tensor_target(obj)
        if target is not None:
            return target
    except Exception:
        pass

    raise TypeError("expected a SymmetricMemory instance or a live ishmem4py symmetric tensor")


def _check_status(status: int | None, opname: str) -> None:
    if status not in (0, None):
        raise IshmemError(f"{opname} failed with status {status}")


def _normalize_team(team: Team | int | None) -> int:
    if team is None:
        return TEAM_WORLD.handle
    if isinstance(team, Team):
        return team.handle
    return int(team)


def _team_from_handle(handle: int, *, name: Optional[str] = None) -> Team | None:
    if int(handle) == TEAM_INVALID.handle:
        return None
    return Team(int(handle), name)


def _normalize_dtype(dtype) -> _DTypeInfo:
    if isinstance(dtype, _DTypeInfo):
        return dtype
    if dtype in _CTYPE_ALIASES:
        return _DTYPES[_CTYPE_ALIASES[dtype]]

    if isinstance(dtype, str):
        alias = dtype.strip()
        if alias.startswith("="):
            alias = alias[1:]
        normalized = _DTYPE_ALIASES.get(alias.lower(), _DTYPE_ALIASES.get(alias))
        if normalized is not None:
            return _DTYPES[normalized]

    raise TypeError("unsupported dtype; use one of int32, int64, uint32, uint64, float32, float64")


def _require_atomic_dtype(dtype, allowed: frozenset[str], opname: str) -> _DTypeInfo:
    info = _normalize_dtype(dtype)
    if info.name not in allowed:
        supported = ", ".join(sorted(allowed))
        raise TypeError(f"{opname} does not support dtype {info.name}; supported dtypes: {supported}")
    return info


def _require_reduction_dtype(op: str, dtype) -> _DTypeInfo:
    normalized_op = op.strip().lower()
    if normalized_op not in _REDUCTION_CODES:
        supported = ", ".join(sorted(_REDUCTION_CODES))
        raise ValueError(f"unsupported reduction op {op!r}; supported ops: {supported}")

    info = _normalize_dtype(dtype)
    allowed = _REDUCTION_DTYPES[normalized_op]
    if info.name not in allowed:
        supported = ", ".join(sorted(allowed))
        raise TypeError(f"reduction {normalized_op!r} does not support dtype {info.name}; supported dtypes: {supported}")
    return info


def _normalize_reduction_op(op: str) -> int:
    return _REDUCTION_CODES[op.strip().lower()]


def _dtype_element_count(src: SymmetricMemory, dest: SymmetricMemory, dtype, count: Optional[int]) -> tuple[_DTypeInfo, int]:
    info = _normalize_dtype(dtype)
    available = min(src.size, dest.size) // info.itemsize
    if count is None:
        return info, available
    if count < 0:
        raise ValueError("count must be >= 0")
    if count > available:
        raise ValueError("count exceeds the available symmetric storage")
    return info, count


def _scalar_to_bits(value, dtype_info: _DTypeInfo) -> int:
    c_value = dtype_info.ctype(value)
    raw = ctypes.string_at(ctypes.byref(c_value), dtype_info.itemsize)
    return int.from_bytes(raw, sys.byteorder, signed=False)


def _bits_to_scalar(bits: int, dtype_info: _DTypeInfo):
    raw = int(bits).to_bytes(dtype_info.itemsize, sys.byteorder, signed=False)
    value = dtype_info.ctype.from_buffer_copy(raw)
    return value.value
