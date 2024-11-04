/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#define BW_TEST_HEADER
#define BW_TEST_FUNCTION                                                                           \
    for (int i = 0; i < iterations; i += 1) {                                                      \
        ishmem_long_fcollect(ISHMEM_TEAM_WORLD, (long *) dest, (long *) src, nelems);              \
    }

#define BW_TEST_FUNCTION_ON_QUEUE                                                                  \
    for (int i = 0; i < iterations; i += 1) {                                                      \
        ishmemx_long_fcollect_on_queue((long *) dest, (long *) src, nelems, test_return, q);       \
    }

#define BW_TEST_FUNCTION_WORK_GROUP                                                                \
    for (int i = 0; i < iterations; i += 1) {                                                      \
        ishmemx_long_fcollect_work_group(ISHMEM_TEAM_WORLD, (long *) dest, (long *) src, nelems,   \
                                         grp);                                                     \
    }

#include "ishmem_tester.h"

STUB_UNIT_TESTS

int main(int argc, char **argv)
{
    class ishmem_tester t(argc, argv, true);

    size_t bufsize = (t.max_nelems * (size_t) t.n_pes * sizeof(uint64_t)) + 4096;
    t.alloc_memory(bufsize);
    size_t errors = 0;
    long bandwidth_multiplier = t.n_pes * t.n_pes;
    if (!t.test_types_set) t.add_test_type(LONG);
    if (!t.test_ops_set) t.add_test_op(NOP);
    t.run_bw_tests(bandwidth_multiplier, true);
    ishmem_sync_all();
    return (t.finalize_and_report(errors));
}
