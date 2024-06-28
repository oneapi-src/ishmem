/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ishmem_tester.h>
#include "rma_test.h"

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    *res = ishmem_##typename##_##testname((type *) dest, (type *) src, nelems, 0);

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    *res =                                                                                         \
        ishmemx_##typename##_##testname##_work_group((type *) dest, (type *) src, nelems, 0, grp);

GEN_FNS(broadcast, NOP, nop)

#undef TEST_BRANCH_SINGLE
#undef TEST_BRANCH_WORK_GROUP

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    *res = ishmem_##testname((type *) dest, (type *) src, nelems, 0);

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    *res = ishmemx_##testname##_work_group((type *) dest, (type *) src, nelems, 0, grp);

GEN_MEM_FNS(broadcastmem, NOP, nop)

class broadcast_tester : public ishmem_tester {
  public:
    broadcast_tester(int argc, char *argv[]) : ishmem_tester(argc, argv) {}
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
};

/* result written into aligned_source, using host_source as a temp buffer if needed */
size_t broadcast_tester::create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
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
size_t broadcast_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                              size_t nelems)
{
    size_t from_pe = 0L; /* PE 0's data goes to all the pes */
    size_t test_size = nelems * typesize(t);
    for (size_t idx = 0; idx < ((test_size / sizeof(long)) + 1); idx += 1) {
        host_check[idx] = (long) ((nelems << 48) + ((0x80L + from_pe) << 40) + (0xffL << 32) + idx);
        if (patterndebugflag && (idx < 16)) {
            printf("[%d] check pattern idx %lu val %016lx\n", my_pe, idx, host_check[idx]);
        }
    }

    return (test_size);
}

int main(int argc, char **argv)
{
    class broadcast_tester t(argc, argv);

    size_t bufsize = 2L * ((t.max_nelems * sizeof(uint64_t) * (size_t) t.n_pes) + 4096L);
    t.alloc_memory(bufsize);
    size_t errors = 0;

    GEN_TABLES(broadcast, NOP, nop)
    GEN_MEM_TABLES(broadcastmem, NOP, nop)

    if (!t.test_types_set) t.add_test_type_list(collectives_copy_types);
    errors += t.run_aligned_tests(NOP);
    errors += t.run_offset_tests(NOP);

    return (t.finalize_and_report(errors));
}
