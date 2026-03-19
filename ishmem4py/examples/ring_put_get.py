# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import struct

import ishmem4py as ishmem


def main() -> None:
    ishmem.init()
    try:
        my_pe = ishmem.my_pe()
        npes = ishmem.n_pes()
        next_pe = (my_pe + 1) % npes
        prev_pe = (my_pe + npes - 1) % npes

        src = ishmem.malloc(4)
        dst = ishmem.calloc(1, 4)
        try:
            src.write(struct.pack("=i", my_pe))
            ishmem.barrier_all()

            ishmem.put(dst, src.read(4), pe=next_pe)
            ishmem.barrier_all()

            received = struct.unpack("=i", dst.read(4))[0]

            fetched = bytearray(4)
            ishmem.get(fetched, src, pe=next_pe)
            fetched_value = struct.unpack("=i", fetched)[0]

            print(
                f"PE {my_pe}: dst after put={received} (expected {prev_pe}), "
                f"get from PE {next_pe} returned {fetched_value}"
            )
            ishmem.barrier_all()
        finally:
            ishmem.free(dst)
            ishmem.free(src)
    finally:
        ishmem.finalize()


if __name__ == "__main__":
    main()
