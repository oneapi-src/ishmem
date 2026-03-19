# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""Host-initiated one-sided RMA operations."""

from __future__ import annotations

import ctypes
from typing import Optional

from ._common import (
    SymmetricMemory,
    _normalize_span,
    _pointer_from_local_buffer,
    _require_initialized,
    _symmetric_target,
)
from ._lib import RUNTIME

__all__ = [
    "fence",
    "get",
    "getmem",
    "put",
    "putmem",
    "quiet",
]


def _queue_handle(queue) -> int:
    if queue is None:
        raise ValueError("queue must not be None")
    if isinstance(queue, int):
        handle = int(queue)
    elif isinstance(queue, ctypes.c_void_p):
        handle = int(queue.value or 0)
    elif hasattr(queue, "sycl_queue"):
        handle = int(queue.sycl_queue)
    elif hasattr(queue, "_as_parameter_"):
        handle = int(getattr(queue._as_parameter_, "value", queue._as_parameter_))
    else:
        raise TypeError("queue must be a torch.xpu.Stream, ctypes.c_void_p, or integer queue pointer")
    if handle == 0:
        raise ValueError("queue pointer must be non-zero")
    return handle


def fence() -> None:
    """Order previously issued put-like operations."""
    _require_initialized()
    RUNTIME.ishmem4py_fence()


def quiet(queue=None) -> None:
    """Wait for completion of previously issued RMA operations."""
    _require_initialized()
    if queue is not None:
        queue_ptr = ctypes.c_void_p(_queue_handle(queue))
        RUNTIME.ishmem4py_queue_sync(queue_ptr)
        RUNTIME.ishmem4py_quiet()
        return
    RUNTIME.ishmem4py_quiet()


def put(
    dest: SymmetricMemory,
    src,
    *,
    pe: int,
    size: Optional[int] = None,
    dest_offset: int = 0,
    src_offset: int = 0,
    queue=None,
) -> int:
    """Copy bytes from a local buffer into symmetric memory on ``pe``."""
    _require_initialized()
    dest = _symmetric_target(dest)
    src_info = _pointer_from_local_buffer(src, writable=False, offset=src_offset)
    dest_size = _normalize_span(dest, size=None, offset=dest_offset)
    nbytes = min(dest_size, src_info.size) if size is None else size

    if nbytes < 0:
        raise ValueError("size must be >= 0")
    if nbytes > dest_size:
        raise ValueError("size extends past the destination symmetric object")
    if nbytes > src_info.size:
        raise ValueError("size extends past the source buffer")

    if queue is None:
        RUNTIME.ishmem4py_putmem(
            ctypes.c_void_p(dest.ptr + dest_offset),
            ctypes.c_void_p(src_info.ptr),
            nbytes,
            pe,
        )
    else:
        RUNTIME.ishmem4py_putmem_on_queue(
            ctypes.c_void_p(dest.ptr + dest_offset),
            ctypes.c_void_p(src_info.ptr),
            nbytes,
            pe,
            ctypes.c_void_p(_queue_handle(queue)),
        )
    return nbytes


def get(
    dest,
    src: SymmetricMemory,
    *,
    pe: int,
    size: Optional[int] = None,
    dest_offset: int = 0,
    src_offset: int = 0,
    queue=None,
) -> int:
    """Copy bytes from symmetric memory on ``pe`` into a local buffer."""
    _require_initialized()
    src = _symmetric_target(src)
    dest_info = _pointer_from_local_buffer(dest, writable=True, offset=dest_offset)
    src_size = _normalize_span(src, size=None, offset=src_offset)
    nbytes = min(dest_info.size, src_size) if size is None else size

    if nbytes < 0:
        raise ValueError("size must be >= 0")
    if nbytes > dest_info.size:
        raise ValueError("size extends past the destination buffer")
    if nbytes > src_size:
        raise ValueError("size extends past the source symmetric object")

    if queue is None:
        RUNTIME.ishmem4py_getmem(
            ctypes.c_void_p(dest_info.ptr),
            ctypes.c_void_p(src.ptr + src_offset),
            nbytes,
            pe,
        )
    else:
        RUNTIME.ishmem4py_getmem_on_queue(
            ctypes.c_void_p(dest_info.ptr),
            ctypes.c_void_p(src.ptr + src_offset),
            nbytes,
            pe,
            ctypes.c_void_p(_queue_handle(queue)),
        )
    return nbytes


def putmem(*args, **kwargs) -> int:
    """Alias for ``put`` for Intel SHMEM naming compatibility."""
    return put(*args, **kwargs)


def getmem(*args, **kwargs) -> int:
    """Alias for ``get`` for Intel SHMEM naming compatibility."""
    return get(*args, **kwargs)
