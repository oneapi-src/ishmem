/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem_tester.h"
#include "rma_test.h"

#undef TEST_BRANCH_ON_QUEUE

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    int pe = (ishmem_my_pe() + ishmem_n_pes() - 1) % ishmem_n_pes();                               \
    ishmem_##typename##_##testname((type *) dest, (type *) src, nelems, pe);                       \
    ishmem_quiet();

#define TEST_BRANCH_ON_QUEUE(testname, typeenum, typename, type, op, opname)                       \
    int pe = (ishmem_my_pe() + ishmem_n_pes() - 1) % ishmem_n_pes();                               \
    ishmemx_##typename##_##testname##_on_queue((type *) dest, (type *) src, nelems, pe, q);        \
    ishmemx_quiet_on_queue(q);

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    int pe = (ishmem_my_pe() + ishmem_n_pes() - 1) % ishmem_n_pes();                               \
    ishmemx_##typename##_##testname##_work_group((type *) dest, (type *) src, nelems, pe, grp);    \
    ishmemx_quiet_work_group(grp);

GEN_FNS(get_nbi, NOP, nop)

#undef TEST_BRANCH_SINGLE
#undef TEST_BRANCH_ON_QUEUE
#undef TEST_BRANCH_WORK_GROUP

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    int pe = (ishmem_my_pe() + ishmem_n_pes() - 1) % ishmem_n_pes();                               \
    ishmem_##testname((type *) dest, (type *) src, nelems, pe);                                    \
    ishmem_quiet();

#define TEST_BRANCH_ON_QUEUE(testname, typeenum, typename, type, op, opname)                       \
    int pe = (ishmem_my_pe() + ishmem_n_pes() - 1) % ishmem_n_pes();                               \
    ishmemx_##testname##_on_queue((type *) dest, (type *) src, nelems, pe, q);                     \
    ishmem_quiet();

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    int pe = (ishmem_my_pe() + ishmem_n_pes() - 1) % ishmem_n_pes();                               \
    ishmemx_##testname##_work_group((type *) dest, (type *) src, nelems, pe, grp);                 \
    ishmemx_quiet_work_group(grp);

GEN_MEM_FNS(getmem_nbi, NOP, nop)

#undef TEST_BRANCH_SINGLE
#undef TEST_BRANCH_ON_QUEUE
#undef TEST_BRANCH_WORK_GROUP

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    int pe = (ishmem_my_pe() + ishmem_n_pes() - 1) % ishmem_n_pes();                               \
    ishmem_##testname((type *) dest, (type *) src, nelems, pe);                                    \
    ishmem_quiet();

#define TEST_BRANCH_ON_QUEUE(testname, typeenum, typename, type, op, opname)                       \
    int pe = (ishmem_my_pe() + ishmem_n_pes() - 1) % ishmem_n_pes();                               \
    ishmemx_##testname##_on_queue((type *) dest, (type *) src, nelems, pe, q);                     \
    ishmem_quiet();

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    int pe = (ishmem_my_pe() + ishmem_n_pes() - 1) % ishmem_n_pes();                               \
    ishmemx_##testname##_work_group((type *) dest, (type *) src, nelems, pe, grp);                 \
    ishmemx_quiet_work_group(grp);

GEN_SIZE_FNS(get, _nbi, NOP, nop)

int main(int argc, char **argv)
{
    class ishmem_tester t(argc, argv, true);

    size_t bufsize = (t.max_nelems * t.typesize(SIZE128)) + 4096L;
    t.alloc_memory(bufsize);
    size_t errors = 0;

    GEN_TABLES(get_nbi, NOP, nop)
    GEN_MEM_TABLES(getmem_nbi, NOP, nop)
    GEN_SIZE_TABLES(get, _nbi, NOP, nop)

    if (!t.test_types_set) t.add_test_type_list(rma_copy_types);
    errors += t.run_aligned_tests(NOP);
    errors += t.run_offset_tests(NOP);
    return (t.finalize_and_report(errors));
}
