/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#define BW_TEST_HEADER

#define BW_TEST_FUNCTION                                                                           \
    for (int i = 0; i < iterations; i += 1) {                                                      \
        ishmem_long_broadcast((long *) dest, (long *) src, nelems, 0);                             \
    }

#define BW_TEST_FUNCTION_WORK_GROUP                                                                \
    for (int i = 0; i < iterations; i += 1) {                                                      \
        ishmemx_long_broadcast_work_group((long *) dest, (long *) src, nelems, 0, grp);            \
    }
#include "ishmem_tester.h"

STUB_UNIT_TESTS

int main(int argc, char **argv)
{
    class ishmem_tester t(argc, argv);

    size_t bufsize = (t.max_nelems * sizeof(uint64_t)) + 4096;
    t.alloc_memory(bufsize);
    size_t errors = 0;
    t.run_bw_tests(LONG, NOP, t.n_pes, true);
    ishmem_sync_all();
    return (t.finalize_and_report(errors));
}
