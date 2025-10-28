/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ishmem_tester.h>
#include "rma_test.h"

#undef TEST_BRANCH_ON_QUEUE

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    *res = ishmem_##typename##_##testname(ISHMEM_TEAM_WORLD, (type *) dest, (type *) src, nelems);

#define TEST_BRANCH_ON_QUEUE(testname, typeenum, typename, type, op, opname)                       \
    ishmemx_##typename##_##testname##_on_queue((type *) dest, (type *) src, nelems, res, q);

GEN_HOST_FNS(sum_inscan, host, NOP, nop)
GEN_ON_QUEUE_FNS(sum_inscan, on_queue, NOP, nop)
GEN_SINGLE_FNS(sum_inscan, single, NOP, nop)

class sum_inscan_tester : public ishmem_tester {
  public:
    sum_inscan_tester(int argc, char *argv[]) : ishmem_tester(argc, argv, true) {}
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
};

/* result written into aligned_source, using host_source as a temp buffer if needed */
size_t sum_inscan_tester::create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                                size_t nelems)
{
    size_t test_size = nelems * typesize(t);
    size_t from_pe = (size_t) my_pe;
    for (size_t idx = 0; idx < ((test_size / sizeof(long)) + 1); idx += 1) {
        aligned_source[idx] = (long) from_pe + (long) idx;
        if (patterndebugflag && (idx < 16)) {
            printf("[%d] source pattern idx %lu val %016lx\n", my_pe, idx, aligned_source[idx]);
        }
    }

    return (test_size);
}

/* check pattern written into host_check, using host_source as a temp buffer */
size_t sum_inscan_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                               size_t nelems)
{
    size_t test_size = nelems * typesize(t);
    size_t to_pe = (size_t) my_pe;
    for (size_t idx = 0; idx < ((test_size / sizeof(long)) + 1); idx += 1) {
        host_check[idx] = ((to_pe + idx) * (to_pe + idx + 1) / 2 - (idx * (idx - 1)) / 2);
        if (patterndebugflag && (idx < 16)) {
            printf("[%d] check pattern idx %lu val %016lx\n", my_pe, idx, host_check[idx]);
        }
    }

    return (test_size);
}

int main(int argc, char **argv)
{
    class sum_inscan_tester t(argc, argv);

    size_t bufsize = (t.max_nelems * t.typesize(SIZE128) * (size_t) t.n_pes) + 4096L;
    t.alloc_memory(bufsize);
    size_t errors = 0;

    GEN_FN_TABLE(sum_inscan, host, NOP, nop)
    GEN_FN_TABLE(sum_inscan, on_queue, NOP, nop)
    GEN_FN_TABLE(sum_inscan, single, NOP, nop)

    if (!t.test_types_set) t.add_test_type_list(scan_types);
    errors += t.run_aligned_tests(NOP);
    errors += t.run_offset_tests(NOP);

    return (t.finalize_and_report(errors));
}
