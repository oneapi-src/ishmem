# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys

import torch

import ishmem4py as ishmem

from utils import expect_equal


def main() -> int:
    ishmem.init()
    try:
        my_pe = ishmem.my_pe()
        npes = ishmem.n_pes()
        if npes < 2:
            raise RuntimeError("torch_ring_tensor_test.py requires at least 2 PEs")

        next_pe = (my_pe + 1) % npes
        prev_pe = (my_pe + npes - 1) % npes

        src = ishmem.tensor((4,), dtype=torch.float32, device="xpu")
        dst = ishmem.tensor((4,), dtype=torch.float32, device="xpu")
        recv = ishmem.tensor((4,), dtype=torch.float32, device="xpu")
        try:
            src.fill_(float(my_pe + 10))
            dst.zero_()
            recv.zero_()
            torch.xpu.synchronize()

            ishmem.barrier_all()

            ishmem.put(dst, src, pe=next_pe)

            ishmem.barrier_all()
            torch.xpu.synchronize()
            expect_equal("symmetric tensor put ring", dst.cpu().tolist(), [float(prev_pe + 10)] * 4)

            ishmem.get(recv, src, pe=next_pe)
            torch.xpu.synchronize()
            expect_equal("symmetric tensor get ring", recv.cpu().tolist(), [float(next_pe + 10)] * 4)
        finally:
            ishmem.free_tensor(recv)
            ishmem.free_tensor(dst)
            ishmem.free_tensor(src)
    finally:
        ishmem.finalize()

    return 0


if __name__ == "__main__":
    sys.exit(main())
