# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys

import ishmem4py as ishmem

from utils import expect_equal, pack_int32, unpack_int32, unpack_int32_list


def main() -> int:
    ishmem.init()
    try:
        my_pe = ishmem.my_pe()
        npes = ishmem.n_pes()
        if npes < 2:
            raise RuntimeError("collective_test.py requires at least 2 PEs")

        broadcast_src = ishmem.malloc(4)
        broadcast_dst = ishmem.calloc(1, 4)
        collect_src = ishmem.malloc(4)
        collect_dst = ishmem.calloc(npes, 4)
        fcollect_dst = ishmem.calloc(npes, 4)
        alltoall_src = ishmem.malloc(npes * 4)
        alltoall_dst = ishmem.calloc(npes, 4)
        reduce_src = ishmem.malloc(4)
        reduce_sum_dst = ishmem.calloc(1, 4)
        reduce_max_dst = ishmem.calloc(1, 4)
        reducescatter_src = ishmem.malloc(npes * 4)
        reducescatter_dst = ishmem.calloc(1, 4)
        try:
            broadcast_src.write(pack_int32(1000 + my_pe))
            collect_src.write(pack_int32(my_pe))
            alltoall_src.write(pack_int32(*[(my_pe * 100) + peer for peer in range(npes)]))
            reduce_src.write(pack_int32(my_pe + 1))
            reducescatter_src.write(pack_int32(*[(my_pe + 1) * (peer + 1) for peer in range(npes)]))

            ishmem.barrier_all()
            ishmem.sync_all()

            ishmem.broadcast(broadcast_dst, broadcast_src, root=0)
            ishmem.collect(collect_dst, collect_src)
            ishmem.fcollect(fcollect_dst, collect_src)
            ishmem.alltoall(alltoall_dst, alltoall_src)
            ishmem.reduce("sum", reduce_sum_dst, reduce_src, dtype="int32")
            ishmem.reduce("max", reduce_max_dst, reduce_src, dtype="int32")
            ishmem.reducescatter("sum", reducescatter_dst, reducescatter_src, dtype="int32", count=1)

            ishmem.barrier_all()

            expect_equal("broadcast result", unpack_int32(broadcast_dst.read(4)), 1000)
            expect_equal("collect result", unpack_int32_list(collect_dst.read(npes * 4)), list(range(npes)))
            expect_equal("fcollect result", unpack_int32_list(fcollect_dst.read(npes * 4)), list(range(npes)))
            expect_equal(
                "alltoall result",
                unpack_int32_list(alltoall_dst.read(npes * 4)),
                [(peer * 100) + my_pe for peer in range(npes)],
            )
            expect_equal("sum reduce result", unpack_int32(reduce_sum_dst.read(4)), npes * (npes + 1) // 2)
            expect_equal("max reduce result", unpack_int32(reduce_max_dst.read(4)), npes)
            expect_equal(
                "reducescatter result",
                unpack_int32(reducescatter_dst.read(4)),
                (my_pe + 1) * (npes * (npes + 1) // 2),
            )
        finally:
            ishmem.free(reducescatter_dst)
            ishmem.free(reducescatter_src)
            ishmem.free(reduce_max_dst)
            ishmem.free(reduce_sum_dst)
            ishmem.free(reduce_src)
            ishmem.free(alltoall_dst)
            ishmem.free(alltoall_src)
            ishmem.free(fcollect_dst)
            ishmem.free(collect_dst)
            ishmem.free(collect_src)
            ishmem.free(broadcast_dst)
            ishmem.free(broadcast_src)
    finally:
        ishmem.finalize()

    return 0


if __name__ == "__main__":
    sys.exit(main())
