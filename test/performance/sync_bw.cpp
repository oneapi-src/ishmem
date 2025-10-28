/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#define BW_TEST_HEADER

#define BW_TEST_FUNCTION                                                                           \
    for (size_t i = 0; i < iterations; i += 1) {                                                   \
        ishmem_team_sync(ISHMEM_TEAM_WORLD);                                                       \
    }

#define BW_TEST_FUNCTION_WORK_GROUP                                                                \
    for (size_t i = 0; i < iterations; i += 1) {                                                   \
        ishmemx_team_sync_work_group(ISHMEM_TEAM_WORLD, grp);                                      \
    }
#include "ishmem_tester.h"

STUB_UNIT_TESTS

int main(int argc, char **argv)
{
    class ishmem_tester t(argc, argv, true);
    t.max_nelems = 1;
    size_t bufsize = (t.max_nelems * sizeof(uint64_t)) + 4096;
    t.alloc_memory(bufsize);
    size_t errors = 0;
    if (!t.test_types_set) t.add_test_type(LONG);
    if (!t.test_ops_set) t.add_test_op(NOP);
    t.run_bw_tests(t.n_pes, true);
    ishmem_sync_all();
    return (t.finalize_and_report(errors));
}
