# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys

import ishmem4py as ishmem

from utils import expect_equal, pack_int32, unpack_int32


def main() -> int:
    ishmem.init()
    try:
        my_pe = ishmem.my_pe()
        npes = ishmem.n_pes()
        if npes < 2:
            raise RuntimeError("ring_test.py requires at least 2 PEs")

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
            expect_equal("put ring result", received, prev_pe)

            host_value = bytearray(4)
            ishmem.get(host_value, src, pe=next_pe)
            expect_equal("get ring result", unpack_int32(host_value), next_pe)

            remote_ptr = ishmem.ishmem_ptr(src, pe=next_pe)
            if remote_ptr is not None:
                expect_equal("ptr metadata size", remote_ptr.size, src.size)
                expect_equal("ptr metadata pe", remote_ptr.pe, next_pe)

            ishmem.barrier_all()
        finally:
            ishmem.free(dst)
            ishmem.free(src)
    finally:
        ishmem.finalize()

    return 0


if __name__ == "__main__":
    sys.exit(main())
