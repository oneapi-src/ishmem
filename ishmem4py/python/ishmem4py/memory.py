# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetric memory allocation and address helpers."""

from __future__ import annotations

import ctypes

from ._common import (
    IshmemError,
    MemoryPointer,
    SymmetricMemory,
    _register_allocation,
    _require_active,
    _require_initialized,
    _unregister_allocation,
)
from ._lib import RUNTIME

__all__ = [
    "MemoryPointer",
    "SymmetricMemory",
    "buffer",
    "calloc",
    "free",
    "ishmem_ptr",
    "malloc",
    "ptr",
]


def malloc(size: int) -> SymmetricMemory:
    """Allocate ``size`` bytes from the symmetric heap."""
    _require_initialized()
    if size < 0:
        raise ValueError("size must be >= 0")
    ptr_value = int(RUNTIME.ishmem4py_malloc(size) or 0)
    if size > 0 and not ptr_value:
        raise IshmemError(f"ishmem_malloc({size}) returned NULL")
    result = SymmetricMemory(ptr=ptr_value, size=size)
    _register_allocation(result)
    return result


def calloc(count: int, size: int) -> SymmetricMemory:
    """Allocate ``count * size`` zeroed bytes from the symmetric heap."""
    _require_initialized()
    if count < 0 or size < 0:
        raise ValueError("count and size must be >= 0")
    total_size = count * size
    ptr_value = int(RUNTIME.ishmem4py_calloc(count, size) or 0)
    if total_size > 0 and not ptr_value:
        raise IshmemError(f"ishmem_calloc({count}, {size}) returned NULL")
    result = SymmetricMemory(ptr=ptr_value, size=total_size)
    _register_allocation(result)
    return result


def buffer(size: int, *, zero: bool = False) -> SymmetricMemory:
    """Allocate a symmetric byte buffer."""
    if zero:
        return calloc(1, size)
    return malloc(size)


def free(symm: SymmetricMemory) -> None:
    """Free a symmetric allocation previously returned by ``malloc`` or ``calloc``."""
    _require_initialized()
    _require_active(symm)
    RUNTIME.ishmem4py_free(ctypes.c_void_p(symm.ptr))
    _unregister_allocation(symm)
    symm._mark_freed()


def ptr(symm: SymmetricMemory, *, pe: int) -> MemoryPointer | None:
    """Return a directly addressable pointer for ``symm`` on ``pe`` when available."""
    _require_initialized()
    _require_active(symm)
    address = int(RUNTIME.ishmem4py_ptr(ctypes.c_void_p(symm.ptr), pe) or 0)
    if address == 0:
        return None
    return MemoryPointer(address=address, size=symm.size, pe=pe, base=symm)


def ishmem_ptr(symm: SymmetricMemory, *, pe: int) -> MemoryPointer | None:
    """Alias for ``ptr`` for Intel SHMEM naming compatibility."""
    return ptr(symm, pe=pe)
