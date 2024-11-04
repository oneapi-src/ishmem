/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#define BW_TEST_HEADER int pe = n_pes - 1;
#define BW_TEST_FUNCTION                                                                           \
    for (int i = 0; i < iterations; i += 1) {                                                      \
        ishmem_long_put((long *) dest, (long *) src, nelems, pe);                                  \
    }

#define BW_TEST_FUNCTION_ON_QUEUE                                                                  \
    for (int i = 0; i < iterations; i += 1) {                                                      \
        ishmemx_long_put_on_queue((long *) dest, (long *) src, nelems, pe, q);                     \
    }

#define BW_TEST_FUNCTION_WORK_GROUP                                                                \
    for (int i = 0; i < iterations; i += 1) {                                                      \
        ishmemx_long_put_work_group((long *) dest, (long *) src, nelems, pe, grp);                 \
    }

#include "ishmem_tester.h"

STUB_UNIT_TESTS

int main(int argc, char **argv)
{
    class ishmem_tester t(argc, argv, true);

    size_t bufsize = (t.max_nelems * sizeof(uint64_t)) + 4096;
    t.alloc_memory(bufsize);
    size_t errors = 0;
    if (!t.test_types_set) t.add_test_type(LONG);
    if (!t.test_ops_set) t.add_test_op(NOP);
    t.run_bw_tests(1, false);
    ishmem_sync_all();
    return (t.finalize_and_report(errors));
}
