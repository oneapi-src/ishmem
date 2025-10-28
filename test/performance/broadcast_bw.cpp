/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#define BW_TEST_HEADER

#define BW_TEST_FUNCTION                                                                           \
    for (size_t i = 0; i < iterations; i += 1) {                                                   \
        ishmem_long_broadcast(ISHMEM_TEAM_WORLD, (long *) dest, (long *) src, nelems, 0);          \
    }

#define BW_TEST_FUNCTION_ON_QUEUE                                                                  \
    for (size_t i = 0; i < iterations; i += 1) {                                                   \
        ishmemx_long_broadcast_on_queue((long *) dest, (long *) src, nelems, 0, test_return, q);   \
    }

#define BW_TEST_FUNCTION_WORK_GROUP                                                                \
    for (size_t i = 0; i < iterations; i += 1) {                                                   \
        ishmemx_long_broadcast_work_group(ISHMEM_TEAM_WORLD, (long *) dest, (long *) src, nelems,  \
                                          0, grp);                                                 \
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
    t.run_bw_tests(t.n_pes, true);
    ishmem_sync_all();
    return (t.finalize_and_report(errors));
}
