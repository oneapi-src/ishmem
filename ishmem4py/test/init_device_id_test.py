# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import subprocess
import sys

import ishmem4py as ishmem
from ishmem4py import init_fini as _init_fini

from utils import expect_equal, pack_int32, unpack_int32

try:
    import torch
except Exception:
    torch = None


def _visible_level_zero_gpu_count() -> int:
    try:
        output = subprocess.check_output(["sycl-ls"], text=True)
    except Exception:
        return 0
    return sum(1 for line in output.splitlines() if "[level_zero:" in line)


def _local_rank() -> int:
    for name in ("MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "PMI_LOCAL_RANK", "SLURM_LOCALID"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return 0


def main() -> int:
    try:
        ishmem.init(device_id="0")  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("init(device_id='0') should raise ValueError")

    try:
        ishmem.init(device_id=-2)
    except ValueError:
        pass
    else:
        raise AssertionError("init(device_id=-2) should raise ValueError")

    try:
        ishmem.init(device_id="cuda:0")
    except ValueError:
        pass
    else:
        raise AssertionError("init(device_id='cuda:0') should raise ValueError")

    visible_gpus = _visible_level_zero_gpu_count()
    local_rank = _local_rank()
    if visible_gpus < 2:
        print("Skipping init_device_id_test.py because fewer than 2 visible Level Zero GPUs were found")
        return 77
    if local_rank >= visible_gpus:
        print(
            f"Skipping init_device_id_test.py because local rank {local_rank} exceeds "
            f"visible Level Zero GPU count {visible_gpus}"
        )
        return 77

    if torch is not None:
        with torch.xpu.device(local_rank):
            expect_equal("xpu string current device normalization", _init_fini._normalize_device_id("xpu"), local_rank)
            expect_equal(
                "torch.device current device normalization",
                _init_fini._normalize_device_id(torch.device("xpu")),
                local_rank,
            )
            expect_equal(
                "torch.device explicit normalization",
                _init_fini._normalize_device_id(torch.device("xpu", local_rank)),
                local_rank,
            )
    expect_equal(
        "xpu explicit string normalization",
        _init_fini._normalize_device_id(f"xpu:{local_rank}"),
        local_rank,
    )

    init_device = torch.device("xpu", local_rank) if torch is not None else f"xpu:{local_rank}"
    ishmem.init(device_id=init_device)
    try:
        my_pe = ishmem.my_pe()
        npes = ishmem.n_pes()
        if npes < 2:
            raise RuntimeError("init_device_id_test.py requires at least 2 PEs")

        next_pe = (my_pe + 1) % npes
        prev_pe = (my_pe + npes - 1) % npes

        src = ishmem.malloc(4)
        dst = ishmem.calloc(1, 4)
        try:
            src.write(pack_int32(my_pe))

            ishmem.barrier_all()
            ishmem.put(dst, src.read(4), pe=next_pe)
            ishmem.barrier_all()

            received = unpack_int32(dst.read(4))
            expect_equal("put ring result with explicit device_id", received, prev_pe)
        finally:
            ishmem.free(dst)
            ishmem.free(src)
    finally:
        ishmem.finalize()

    return 0


if __name__ == "__main__":
    sys.exit(main())
