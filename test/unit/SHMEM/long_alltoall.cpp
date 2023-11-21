/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <common.h>

constexpr size_t array_size = 1L << 21;

int main(int argc, char **argv)
{
    int exit_code = 0;
    sycl::queue q;
    ishmem_init();
    setbuf(stdout, NULL);

    long *source = (long *) ishmem_malloc(array_size * sizeof(long));
    CHECK_ALLOC(source);
    long *target = (long *) ishmem_malloc(array_size * sizeof(long) * 12);
    CHECK_ALLOC(target);

    ishmem_sync_all();

    /* Perform put operation */
    for (size_t nelems = 1; nelems <= array_size; nelems <<= 1) {
        printf("host call device data nelems %ld\n", nelems);
        for (int i = 0; i < 1024; i += 1) {
            ishmem_long_alltoall(target, source, nelems);
        }
    }
    ishmem_sync_all();

    ishmem_free(source);
    ishmem_free(target);

    ishmem_finalize();

    return exit_code;
}
