# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys

import ishmem4py as ishmem

from utils import expect_equal, expect_true, pack_int32, unpack_int32, unpack_int32_list


def main() -> int:
    version = ishmem.get_version()
    expect_true("version vendor string is populated", bool(version.vendor_name))
    expect_true("version strings are populated", bool(version.libishmem_version))
    expect_equal("pre-init status", ishmem.init_status(), ishmem.InitStatus.UNINITIALIZED)

    ishmem.init()
    try:
        expect_equal("post-init status", ishmem.init_status(), ishmem.InitStatus.INITIALIZED)
        expect_equal("n_pes", ishmem.n_pes(), 1)
        expect_equal("my_pe", ishmem.my_pe(), 0)

        major, minor = ishmem.info_get_version()
        expect_true("OpenSHMEM spec major version", major >= 1)
        expect_true("OpenSHMEM spec minor version", minor >= 0)
        expect_true("library name", ishmem.info_get_name().startswith("Intel"))

        buf = ishmem.malloc(16)
        zeroed = ishmem.calloc(4, 4)
        alias = ishmem.buffer(8, zero=True)
        try:
            buf.write(pack_int32(1, 2, 3, 4))
            expect_equal("local put/get", unpack_int32_list(buf.read(16)), [1, 2, 3, 4])
            expect_equal("calloc zeroed", zeroed.read(16), b"\x00" * 16)
            expect_equal("buffer(zero=True)", alias.read(8), b"\x00" * 8)

            host_value = bytearray(4)
            ishmem.get(host_value, buf, pe=0, size=4, src_offset=8)
            expect_equal("get into host buffer", unpack_int32(host_value), 3)

            ishmem.put(buf, pack_int32(9), pe=0, dest_offset=4)
            expect_equal("put alias", unpack_int32(buf.read(4, offset=4)), 9)

            local_ptr = ishmem.ptr(buf, pe=0)
            expect_true("ptr result type", local_ptr is None or local_ptr.size == buf.size)
            if local_ptr is not None:
                expect_equal("ptr pe", local_ptr.pe, 0)
                expect_true("ptr address", int(local_ptr) != 0)

            ishmem.barrier_all()
        finally:
            ishmem.free(alias)
            ishmem.free(zeroed)
            ishmem.free(buf)
    finally:
        ishmem.finalize()

    expect_equal("post-finalize status", ishmem.init_status(), ishmem.InitStatus.FINALIZED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
