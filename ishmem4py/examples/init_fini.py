# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import ishmem4py as ishmem


def main() -> None:
    ishmem.init()
    try:
        major, minor = ishmem.info_get_version()
        print(f"PE {ishmem.my_pe()} / {ishmem.n_pes()} using Intel SHMEM {major}.{minor}")
    finally:
        ishmem.finalize()


if __name__ == "__main__":
    main()
