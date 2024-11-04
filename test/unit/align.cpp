/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <common.h>

#define ALIGNMENT_TEST_START 16
#define ALIGNMENT_TEST_END   1024

static int test_ishmem_align(size_t alignment);
using std::cerr, std::endl;

static int test_ishmem_align(size_t alignment)
{
    int *buffer = (int *) ishmem_align(alignment, sizeof(int));

    if (buffer == nullptr) return 1;
    if ((uintptr_t) buffer % alignment != 0) {
        ishmem_free(buffer);
        return 1;
    }

    ishmem_free(buffer);
    return 0;
}

int main()
{
    ishmem_init();

    // Test ishmem_align for different values of alignment
    for (size_t alignment = ALIGNMENT_TEST_START; alignment <= ALIGNMENT_TEST_END; alignment *= 2) {
        if (test_ishmem_align(alignment)) {
            cerr << "ishmem_align failed at alignment == " << alignment << endl;
            ishmem_finalize();
            return EXIT_FAILURE;
        }
    }

    ishmem_finalize();
    return EXIT_SUCCESS;
}
