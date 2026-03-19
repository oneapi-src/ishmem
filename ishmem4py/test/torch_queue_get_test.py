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
            raise RuntimeError("torch_queue_get_test.py requires at least 2 PEs")

        next_pe = (my_pe + 1) % npes

        src = ishmem.tensor((4,), dtype=torch.float32, device="xpu")
        recv = ishmem.tensor((4,), dtype=torch.float32, device="xpu")
        try:
            src.fill_(float(my_pe + 10))
            recv.zero_()
            torch.xpu.synchronize()

            ishmem.barrier_all()

            stream = torch.xpu.Stream()
            ishmem.get(recv, src, pe=next_pe, queue=stream)
            ishmem.quiet(queue=stream)
            torch.xpu.synchronize()

            expect_equal("queued symmetric tensor get ring", recv.cpu().tolist(), [float(next_pe + 10)] * 4)
        finally:
            ishmem.free_tensor(recv)
            ishmem.free_tensor(src)
    finally:
        ishmem.finalize()

    return 0


if __name__ == "__main__":
    sys.exit(main())
