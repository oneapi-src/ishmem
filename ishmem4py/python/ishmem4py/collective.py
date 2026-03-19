# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""Host-side collective operations over symmetric allocations."""

from __future__ import annotations

import ctypes
from typing import Optional

from ._common import (
    IshmemError,
    SymmetricMemory,
    TEAM_WORLD,
    _check_status,
    _dtype_element_count,
    _normalize_reduction_op,
    _normalize_span,
    _normalize_team,
    _require_active,
    _require_initialized,
    _require_reduction_dtype,
)
from ._lib import RUNTIME
from .teams import team_my_pe, team_n_pes, team_sync

__all__ = [
    "alltoall",
    "barrier",
    "barrier_all",
    "broadcast",
    "collect",
    "fcollect",
    "reduce",
    "reducescatter",
    "sync",
    "sync_all",
]


def _collective_span(
    dest: SymmetricMemory,
    src: SymmetricMemory,
    *,
    size: Optional[int],
    dest_offset: int,
    src_offset: int,
) -> tuple[int, int, int]:
    _require_active(dest)
    _require_active(src)
    dest_size = _normalize_span(dest, size=None, offset=dest_offset)
    src_size = _normalize_span(src, size=None, offset=src_offset)
    nbytes = min(dest_size, src_size) if size is None else size
    if nbytes < 0:
        raise ValueError("size must be >= 0")
    if nbytes > dest_size:
        raise ValueError("size extends past the destination symmetric object")
    if nbytes > src_size:
        raise ValueError("size extends past the source symmetric object")
    return dest_size, src_size, nbytes


def barrier_all() -> None:
    """Block until all PEs in ``TEAM_WORLD`` arrive."""
    _require_initialized()
    RUNTIME.ishmem4py_barrier_all()


def sync_all() -> None:
    """Synchronize all PEs in ``TEAM_WORLD``."""
    _require_initialized()
    RUNTIME.ishmem4py_sync_all()


def sync(team=None) -> None:
    """Synchronize all PEs in ``team``."""
    team_sync(team)


def barrier(team=None) -> None:
    """Barrier on ``team``."""
    if team is None or _normalize_team(team) == TEAM_WORLD.handle:
        barrier_all()
        return
    team_sync(team)


def broadcast(
    dest: SymmetricMemory,
    src: SymmetricMemory,
    *,
    root: int,
    team=None,
    size: Optional[int] = None,
    dest_offset: int = 0,
    src_offset: int = 0,
) -> int:
    """Broadcast ``src`` from ``root`` into ``dest`` across ``team``."""
    _require_initialized()
    _, _, nbytes = _collective_span(dest, src, size=size, dest_offset=dest_offset, src_offset=src_offset)
    status = RUNTIME.ishmem4py_broadcastmem(
        _normalize_team(team),
        ctypes.c_void_p(dest.ptr + dest_offset),
        ctypes.c_void_p(src.ptr + src_offset),
        nbytes,
        root,
    )
    _check_status(status, "ishmem_broadcastmem")
    return nbytes


def collect(
    dest: SymmetricMemory,
    src: SymmetricMemory,
    *,
    team=None,
    size: Optional[int] = None,
    dest_offset: int = 0,
    src_offset: int = 0,
) -> int:
    """Concatenate each PE's contribution from ``src`` into ``dest``."""
    _require_initialized()
    participants = team_n_pes(team)
    dest_size = _normalize_span(dest, size=None, offset=dest_offset)
    src_size = _normalize_span(src, size=None, offset=src_offset)
    per_pe = min(src_size, dest_size // max(1, participants)) if size is None else size
    if per_pe < 0:
        raise ValueError("size must be >= 0")
    if per_pe > src_size:
        raise ValueError("size extends past the source symmetric object")
    if per_pe * max(1, participants) > dest_size:
        raise ValueError("destination symmetric object is too small for collect")
    status = RUNTIME.ishmem4py_collectmem(
        _normalize_team(team),
        ctypes.c_void_p(dest.ptr + dest_offset),
        ctypes.c_void_p(src.ptr + src_offset),
        per_pe,
    )
    _check_status(status, "ishmem_collectmem")
    return per_pe


def fcollect(
    dest: SymmetricMemory,
    src: SymmetricMemory,
    *,
    team=None,
    size: Optional[int] = None,
    dest_offset: int = 0,
    src_offset: int = 0,
) -> int:
    """Fixed-size collect from ``src`` into ``dest``."""
    _require_initialized()
    participants = team_n_pes(team)
    dest_size = _normalize_span(dest, size=None, offset=dest_offset)
    src_size = _normalize_span(src, size=None, offset=src_offset)
    per_pe = min(src_size, dest_size // max(1, participants)) if size is None else size
    if per_pe < 0:
        raise ValueError("size must be >= 0")
    if per_pe > src_size:
        raise ValueError("size extends past the source symmetric object")
    if per_pe * max(1, participants) > dest_size:
        raise ValueError("destination symmetric object is too small for fcollect")
    status = RUNTIME.ishmem4py_fcollectmem(
        _normalize_team(team),
        ctypes.c_void_p(dest.ptr + dest_offset),
        ctypes.c_void_p(src.ptr + src_offset),
        per_pe,
    )
    _check_status(status, "ishmem_fcollectmem")
    return per_pe


def alltoall(
    dest: SymmetricMemory,
    src: SymmetricMemory,
    *,
    team=None,
    size: Optional[int] = None,
    dest_offset: int = 0,
    src_offset: int = 0,
) -> int:
    """Exchange equally sized blocks among all PEs in ``team``."""
    _require_initialized()
    participants = team_n_pes(team)
    if participants <= 0:
        raise IshmemError("alltoall requires a valid team")
    dest_size = _normalize_span(dest, size=None, offset=dest_offset)
    src_size = _normalize_span(src, size=None, offset=src_offset)
    per_pe = min(src_size // participants, dest_size // participants) if size is None else size
    if per_pe < 0:
        raise ValueError("size must be >= 0")
    if per_pe * participants > src_size:
        raise ValueError("source symmetric object is too small for alltoall")
    if per_pe * participants > dest_size:
        raise ValueError("destination symmetric object is too small for alltoall")
    status = RUNTIME.ishmem4py_alltoallmem(
        _normalize_team(team),
        ctypes.c_void_p(dest.ptr + dest_offset),
        ctypes.c_void_p(src.ptr + src_offset),
        per_pe,
    )
    _check_status(status, "ishmem_alltoallmem")
    return per_pe


def reduce(
    op: str,
    dest: SymmetricMemory,
    src: SymmetricMemory,
    *,
    dtype,
    count: Optional[int] = None,
    team=None,
) -> int:
    """Reduce ``src`` into ``dest`` across ``team`` using ``op`` and ``dtype``."""
    _require_initialized()
    _require_active(dest)
    _require_active(src)
    info = _require_reduction_dtype(op, dtype)
    _, count = _dtype_element_count(src, dest, info, count)
    status = RUNTIME.ishmem4py_reduce(
        _normalize_reduction_op(op),
        info.code,
        _normalize_team(team),
        ctypes.c_void_p(dest.ptr),
        ctypes.c_void_p(src.ptr),
        count,
    )
    _check_status(status, f"ishmem_{info.name}_{op}_reduce")
    return count


def reducescatter(
    op: str,
    dest: SymmetricMemory,
    src: SymmetricMemory,
    *,
    dtype,
    count: Optional[int] = None,
    team=None,
) -> int:
    """Reduce ``src`` across ``team`` and return this PE's block in ``dest``.

    Intel SHMEM does not currently expose a host-side reducescatter routine.
    ``ishmem4py`` therefore implements the operation as a small software
    fallback: perform a full ``reduce`` into a temporary symmetric buffer, then
    copy the local team's slice into ``dest``.
    """
    from .init_fini import my_pe
    from .memory import free, malloc
    from .rma import get

    _require_initialized()
    _require_active(dest)
    _require_active(src)
    info = _require_reduction_dtype(op, dtype)
    participants = team_n_pes(team)
    if participants <= 0:
        raise IshmemError("reducescatter requires a valid team")

    dest_available = dest.size // info.itemsize
    src_available = src.size // info.itemsize
    if count is None:
        count = min(dest_available, src_available // participants)
    if count < 0:
        raise ValueError("count must be >= 0")
    if count > dest_available:
        raise ValueError("count exceeds the available destination symmetric storage")

    total_count = count * participants
    if total_count > src_available:
        raise ValueError("count exceeds the available source symmetric storage for reducescatter")

    scratch = malloc(total_count * info.itemsize)
    try:
        reduce(op, scratch, src, dtype=info, count=total_count, team=team)
        get(
            dest,
            scratch,
            pe=my_pe(),
            size=count * info.itemsize,
            src_offset=team_my_pe(team) * count * info.itemsize,
        )
    finally:
        free(scratch)

    return count
