/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ishmem_tester.h>
#include "rma_test.h"

#undef TEST_BRANCH_ON_QUEUE

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    *res = ishmem_##typename##_##testname(ISHMEM_TEAM_WORLD, (type *) dest, (type *) src, nelems);

#define TEST_BRANCH_ON_QUEUE(testname, typeenum, typename, type, op, opname)                       \
    ishmemx_##typename##_##testname##_on_queue((type *) dest, (type *) src, nelems, res, q);

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    *res = ishmemx_##typename##_##testname##_work_group(ISHMEM_TEAM_WORLD, (type *) dest,          \
                                                        (type *) src, nelems, grp);

GEN_FNS(fcollect, NOP, nop)

#undef TEST_BRANCH_SINGLE
#undef TEST_BRANCH_ON_QUEUE
#undef TEST_BRANCH_WORK_GROUP

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    *res = ishmem_##testname(ISHMEM_TEAM_WORLD, (type *) dest, (type *) src, nelems);

#define TEST_BRANCH_ON_QUEUE(testname, typeenum, typename, type, op, opname)                       \
    ishmemx_##testname##_on_queue((type *) dest, (type *) src, nelems, res, q);

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    *res = ishmemx_##testname##_work_group(ISHMEM_TEAM_WORLD, (type *) dest, (type *) src, nelems, \
                                           grp);

GEN_MEM_FNS(fcollectmem, NOP, nop)

class fcollect_tester : public ishmem_tester {
  public:
    fcollect_tester(int argc, char *argv[]) : ishmem_tester(argc, argv, true) {}
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
};

/* result written into aligned_source, using host_source as a temp buffer if needed */
size_t fcollect_tester::create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                              size_t nelems)
{
    size_t test_size = nelems * typesize(t);
    size_t from_pe = (size_t) my_pe;
    for (size_t idx = 0; idx < ((test_size / sizeof(long)) + 1); idx += 1) {
        aligned_source[idx] =
            (long) ((nelems << 48) + ((0x80L + from_pe) << 40) + (0xffL << 32) + idx);
        if (patterndebugflag && (idx < 16)) {
            printf("[%d] source pattern idx %lu val %016lx\n", my_pe, idx, aligned_source[idx]);
        }
    }

    return (test_size);
}

/* check pattern written into host_check, using host_source as a temp buffer */
size_t fcollect_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                             size_t nelems)
{
    size_t test_size_per_pe = nelems * typesize(t);
    for (size_t from_pe = 0; from_pe < n_pes; from_pe += 1) {
        for (size_t idx = 0; idx < ((test_size_per_pe / sizeof(long)) + 1); idx += 1) {
            host_source[idx] =
                (long) ((nelems << 48) + ((0x80L + from_pe) << 40) + (0xffL << 32) + idx);
        }
        memcpy((void *) (((uintptr_t) host_check) + (from_pe * test_size_per_pe)), host_source,
               test_size_per_pe);
    }
    if (patterndebugflag) {
        for (size_t idx = 0; (idx < 16) && (idx < ((test_size_per_pe / sizeof(long)) + 1)); idx++) {
            printf("[%d] check pattern idx %lu val %016lx\n", my_pe, idx, host_check[idx]);
        }
    }

    return (test_size_per_pe * (size_t) n_pes);
}

int main(int argc, char **argv)
{
    class fcollect_tester t(argc, argv);

    size_t bufsize = (t.max_nelems * t.typesize(SIZE128) * (size_t) t.n_pes) + 4096L;
    t.alloc_memory(bufsize);
    size_t errors = 0;

    GEN_TABLES(fcollect, NOP, nop)
    GEN_MEM_TABLES(fcollectmem, NOP, nop)

    if (!t.test_types_set) t.add_test_type_list(collectives_copy_types);
    errors += t.run_aligned_tests(NOP);
    errors += t.run_offset_tests(NOP);

    return (t.finalize_and_report(errors));
}
