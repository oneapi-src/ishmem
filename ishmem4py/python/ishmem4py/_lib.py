# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ctypes
import os
from pathlib import Path

_RUNTIME_ENVVAR = "ISHMEM4PY_RUNTIME_LIBRARY"


def _candidate_paths():
    env_path = os.environ.get(_RUNTIME_ENVVAR)
    if env_path:
        yield Path(env_path)

    package_dir = Path(__file__).resolve().parent
    yield package_dir / "_ishmem4py_runtime.so"


def _configure_runtime(runtime):
    runtime.ishmem4py_init.argtypes = []
    runtime.ishmem4py_init.restype = None

    runtime.ishmem4py_init_with_device.argtypes = [ctypes.c_int]
    runtime.ishmem4py_init_with_device.restype = None

    runtime.ishmem4py_finalize.argtypes = []
    runtime.ishmem4py_finalize.restype = None

    runtime.ishmem4py_my_pe.argtypes = []
    runtime.ishmem4py_my_pe.restype = ctypes.c_int

    runtime.ishmem4py_n_pes.argtypes = []
    runtime.ishmem4py_n_pes.restype = ctypes.c_int

    runtime.ishmem4py_info_get_version.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    runtime.ishmem4py_info_get_version.restype = None

    runtime.ishmem4py_info_get_name.argtypes = [ctypes.c_void_p]
    runtime.ishmem4py_info_get_name.restype = None

    runtime.ishmem4py_vendor_get_version.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    runtime.ishmem4py_vendor_get_version.restype = None

    runtime.ishmem4py_barrier_all.argtypes = []
    runtime.ishmem4py_barrier_all.restype = None

    runtime.ishmem4py_sync_all.argtypes = []
    runtime.ishmem4py_sync_all.restype = None

    runtime.ishmem4py_fence.argtypes = []
    runtime.ishmem4py_fence.restype = None

    runtime.ishmem4py_quiet.argtypes = []
    runtime.ishmem4py_quiet.restype = None

    runtime.ishmem4py_malloc.argtypes = [ctypes.c_size_t]
    runtime.ishmem4py_malloc.restype = ctypes.c_void_p

    runtime.ishmem4py_calloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    runtime.ishmem4py_calloc.restype = ctypes.c_void_p

    runtime.ishmem4py_free.argtypes = [ctypes.c_void_p]
    runtime.ishmem4py_free.restype = None

    runtime.ishmem4py_putmem.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    runtime.ishmem4py_putmem.restype = None

    runtime.ishmem4py_getmem.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    runtime.ishmem4py_getmem.restype = None

    runtime.ishmem4py_putmem_on_queue.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    runtime.ishmem4py_putmem_on_queue.restype = None

    runtime.ishmem4py_getmem_on_queue.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    runtime.ishmem4py_getmem_on_queue.restype = None

    runtime.ishmem4py_quiet_on_queue.argtypes = [ctypes.c_void_p]
    runtime.ishmem4py_quiet_on_queue.restype = None

    runtime.ishmem4py_queue_sync.argtypes = [ctypes.c_void_p]
    runtime.ishmem4py_queue_sync.restype = None

    runtime.ishmem4py_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    runtime.ishmem4py_ptr.restype = ctypes.c_void_p

    runtime.ishmem4py_team_my_pe.argtypes = [ctypes.c_int]
    runtime.ishmem4py_team_my_pe.restype = ctypes.c_int

    runtime.ishmem4py_team_n_pes.argtypes = [ctypes.c_int]
    runtime.ishmem4py_team_n_pes.restype = ctypes.c_int

    runtime.ishmem4py_team_translate_pe.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    runtime.ishmem4py_team_translate_pe.restype = ctypes.c_int

    runtime.ishmem4py_team_sync.argtypes = [ctypes.c_int]
    runtime.ishmem4py_team_sync.restype = ctypes.c_int

    runtime.ishmem4py_team_split_strided.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_long,
        ctypes.POINTER(ctypes.c_int),
    ]
    runtime.ishmem4py_team_split_strided.restype = ctypes.c_int

    runtime.ishmem4py_team_split_2d.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_long,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_long,
        ctypes.POINTER(ctypes.c_int),
    ]
    runtime.ishmem4py_team_split_2d.restype = ctypes.c_int

    runtime.ishmem4py_team_destroy.argtypes = [ctypes.c_int]
    runtime.ishmem4py_team_destroy.restype = None

    runtime.ishmem4py_broadcastmem.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    runtime.ishmem4py_broadcastmem.restype = ctypes.c_int

    runtime.ishmem4py_collectmem.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    runtime.ishmem4py_collectmem.restype = ctypes.c_int

    runtime.ishmem4py_fcollectmem.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    runtime.ishmem4py_fcollectmem.restype = ctypes.c_int

    runtime.ishmem4py_alltoallmem.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    runtime.ishmem4py_alltoallmem.restype = ctypes.c_int

    runtime.ishmem4py_reduce.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    runtime.ishmem4py_reduce.restype = ctypes.c_int

    runtime.ishmem4py_atomic_fetch.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    runtime.ishmem4py_atomic_fetch.restype = ctypes.c_uint64

    runtime.ishmem4py_atomic_set.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
    ]
    runtime.ishmem4py_atomic_set.restype = None

    runtime.ishmem4py_atomic_swap.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
    ]
    runtime.ishmem4py_atomic_swap.restype = ctypes.c_uint64

    runtime.ishmem4py_atomic_compare_swap.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int,
    ]
    runtime.ishmem4py_atomic_compare_swap.restype = ctypes.c_uint64

    runtime.ishmem4py_atomic_fetch_inc.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    runtime.ishmem4py_atomic_fetch_inc.restype = ctypes.c_uint64

    runtime.ishmem4py_atomic_inc.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    runtime.ishmem4py_atomic_inc.restype = None

    for name in (
        "ishmem4py_atomic_fetch_add",
        "ishmem4py_atomic_fetch_and",
        "ishmem4py_atomic_fetch_or",
        "ishmem4py_atomic_fetch_xor",
    ):
        func = getattr(runtime, name)
        func.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
        func.restype = ctypes.c_uint64

    for name in (
        "ishmem4py_atomic_add",
        "ishmem4py_atomic_and",
        "ishmem4py_atomic_or",
        "ishmem4py_atomic_xor",
    ):
        func = getattr(runtime, name)
        func.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
        func.restype = None

    runtime.ishmem4py_cast_ptr_to_uintptr.argtypes = [ctypes.c_void_p]
    runtime.ishmem4py_cast_ptr_to_uintptr.restype = ctypes.c_uint64

    runtime.ishmem4py_cast_uintptr_to_ptr.argtypes = [ctypes.c_uint64]
    runtime.ishmem4py_cast_uintptr_to_ptr.restype = ctypes.c_void_p

    return runtime


def _load_runtime():
    last_error = None
    for candidate in _candidate_paths():
        if not candidate.exists():
            continue
        try:
            runtime = ctypes.CDLL(str(candidate))
        except OSError as exc:
            last_error = exc
            continue
        return _configure_runtime(runtime)

    if last_error is not None:
        raise RuntimeError(
            "Found ishmem4py runtime library candidate but failed to load it. "
            "Ensure oneAPI runtime libraries are in LD_LIBRARY_PATH and set "
            f"{_RUNTIME_ENVVAR} if needed."
        ) from last_error

    raise RuntimeError(
        "Could not locate _ishmem4py_runtime.so. For a standard install, build ishmem with "
        "-DBUILD_PYTHON_BINDINGS=ON and pip install the build-tree package directory. "
        f"For an editable/dev install, set {_RUNTIME_ENVVAR} to the built runtime library."
    )


class _RuntimeProxy:
    def __init__(self):
        self._runtime = None

    def _get_runtime(self):
        if self._runtime is None:
            self._runtime = _load_runtime()
        return self._runtime

    def __getattr__(self, name):
        return getattr(self._get_runtime(), name)


RUNTIME = _RuntimeProxy()
