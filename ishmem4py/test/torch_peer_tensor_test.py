# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys

import torch

import ishmem4py as ishmem

from utils import expect_equal, expect_true


def main() -> int:
    ishmem.init()
    try:
        my_pe = ishmem.my_pe()
        npes = ishmem.n_pes()
        if npes < 2:
            raise RuntimeError("torch_peer_tensor_test.py requires at least 2 PEs")

        next_pe = (my_pe + 1) % npes
        prev_pe = (my_pe + npes - 1) % npes

        target = ishmem.tensor((4,), dtype=torch.float32, device="xpu")
        try:
            target.fill_(10.0 * my_pe)
            torch.xpu.synchronize()
            ishmem.barrier_all()

            peer = ishmem.get_peer_tensor(target, pe=next_pe)
            expect_true("peer tensor is symmetric", ishmem.is_symmetric_tensor(peer))
            expect_true("peer tensor base returns original allocation", ishmem.tensor_base(peer) is target)

            peer.add_(torch.full_like(peer, float(my_pe + 1)))
            torch.xpu.synchronize()

            try:
                ishmem.free_tensor(peer)
            except ishmem.IshmemStateError:
                pass
            else:
                raise AssertionError("free_tensor(peer) should fail for non-owning aliases")

            ishmem.barrier_all()
            expect_equal(
                "remote peer alias update",
                target.cpu().tolist(),
                [10.0 * my_pe + float(prev_pe + 1)] * 4,
            )
        finally:
            ishmem.free_tensor(target)
    finally:
        ishmem.finalize()

    return 0


if __name__ == "__main__":
    sys.exit(main())
