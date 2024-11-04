/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem_tester.h"
#include "rma_test.h"

#undef TEST_BRANCH_ON_QUEUE

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    int pe = (ishmem_my_pe() + 1) % ishmem_n_pes();                                                \
    ishmemx_##typename##_##testname((type *) dest, (type *) src, ((nelems > 1) ? nelems / 2 : 1),  \
                                    1, 1, ((nelems > 1) ? 2 : 1), pe);                             \
    if (nelems > 2) {                                                                              \
        ishmemx_##typename##_##testname((type *) dest + 1, (type *) src + 2, nelems / 2,           \
                                        ((nelems / 2) - 1), ((nelems / 2) - 1), 2, pe);            \
    }                                                                                              \
    ishmem_quiet();

#define TEST_BRANCH_ON_QUEUE(testname, typeenum, typename, type, op, opname)                       \
    int pe = (ishmem_my_pe() + 1) % ishmem_n_pes();                                                \
    ishmemx_##typename##_##testname##_on_queue((type *) dest, (type *) src,                        \
                                               ((nelems > 1) ? nelems / 2 : 1), 1, 1,              \
                                               ((nelems > 1) ? 2 : 1), pe, q);                     \
    if (nelems > 2) {                                                                              \
        ishmemx_##                                                                                 \
            typename##_##testname##_on_queue((type *) dest + 1, (type *) src + 2, nelems / 2,      \
                                             ((nelems / 2) - 1), ((nelems / 2) - 1), 2, pe, q);    \
    }                                                                                              \
    ishmem_quiet();

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    int pe = (ishmem_my_pe() + 1) % ishmem_n_pes();                                                \
    ishmemx_##typename##_##testname##_work_group((type *) dest, (type *) src,                      \
                                                 ((nelems > 1) ? nelems / 2 : 1), 1, 1,            \
                                                 ((nelems > 1) ? 2 : 1), pe, grp);                 \
    if (nelems > 2) {                                                                              \
        ishmemx_##typename##_##testname##_work_group((type *) dest + 1, (type *) src + 2,          \
                                                     nelems / 2, ((nelems / 2) - 1),               \
                                                     ((nelems / 2) - 1), 2, pe, grp);              \
    }                                                                                              \
    ishmemx_quiet_work_group(grp);

GEN_FNS(ibput, NOP, nop)

#undef TEST_BRANCH_SINGLE
#undef TEST_BRANCH_ON_QUEUE
#undef TEST_BRANCH_WORK_GROUP

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    int pe = (ishmem_my_pe() + 1) % ishmem_n_pes();                                                \
    ishmemx_##testname((type *) dest, (type *) src, ((nelems > 1) ? nelems / 2 : 1), 1, 1,         \
                       ((nelems > 1) ? 2 : 1), pe);                                                \
                                                                                                   \
    if (nelems > 2) {                                                                              \
        ishmemx_##testname((type *) dest + 1, (type *) src + 2, nelems / 2, ((nelems / 2) - 1),    \
                           ((nelems / 2) - 1), 2, pe);                                             \
    }                                                                                              \
    ishmem_quiet();

#define TEST_BRANCH_ON_QUEUE(testname, typeenum, typename, type, op, opname)                       \
    int pe = (ishmem_my_pe() + 1) % ishmem_n_pes();                                                \
    ishmemx_##testname##_on_queue((type *) dest, (type *) src, ((nelems > 1) ? nelems / 2 : 1), 1, \
                                  1, ((nelems > 1) ? 2 : 1), pe, q);                               \
                                                                                                   \
    if (nelems > 2) {                                                                              \
        ishmemx_##testname##_on_queue((type *) dest + 1, (type *) src + 2, nelems / 2,             \
                                      ((nelems / 2) - 1), ((nelems / 2) - 1), 2, pe, q);           \
    }                                                                                              \
    ishmem_quiet();

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    int pe = (ishmem_my_pe() + 1) % ishmem_n_pes();                                                \
    ishmemx_##testname##_work_group((type *) dest, (type *) src, ((nelems > 1) ? nelems / 2 : 1),  \
                                    1, 1, ((nelems > 1) ? 2 : 1), pe, grp);                        \
                                                                                                   \
    if (nelems > 2) {                                                                              \
        ishmemx_##testname##_work_group((type *) dest + 1, (type *) src + 2, nelems / 2,           \
                                        ((nelems / 2) - 1), ((nelems / 2) - 1), 2, pe, grp);       \
    }                                                                                              \
    ishmemx_quiet_work_group(grp);

GEN_SIZE_FNS(ibput, , NOP, nop)

class ibput_tester : public ishmem_tester {
  public:
    ibput_tester(int argc, char *argv[]) : ishmem_tester(argc, argv, true) {}
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
};

#define HOST_CHECK_ASSIGN(type)                                                                    \
    type *check = (type *) host_check;                                                             \
    type val = check[1];                                                                           \
    for (size_t idx = 1; idx < (nelems / 2); idx += 1) {                                           \
        check[idx] = check[idx + 1];                                                               \
    }                                                                                              \
    check[nelems / 2] = val;

size_t ibput_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                          size_t nelems)
{
    long int from_pe = (long int) (my_pe + n_pes - 1) % n_pes;
    long int to_pe = (long int) my_pe;
    size_t test_size = nelems * typesize(t);
    for (size_t idx = 0; idx < ((test_size / sizeof(long)) + 1); idx += 1) {
        host_check[idx] = ((long) nelems << 48) + ((0x80L + from_pe) << 40) +
                          ((0x80L + to_pe) << 32) + (long) idx;
    }

    if (nelems > 2) {
        if (typesize(t) == sizeof(uint8_t)) {
            HOST_CHECK_ASSIGN(uint8_t)
        } else if (typesize(t) == sizeof(uint16_t)) {
            HOST_CHECK_ASSIGN(uint16_t)
        } else if (typesize(t) == sizeof(uint32_t)) {
            HOST_CHECK_ASSIGN(uint32_t)
        } else if (typesize(t) == sizeof(uint64_t)) {
            HOST_CHECK_ASSIGN(uint64_t)
        } else if (typesize(t) == sizeof(__uint128_t)) {
            HOST_CHECK_ASSIGN(__uint128_t)
        } else {
            assert(0);
        }
    }

    if (patterndebugflag) {
        for (size_t idx = 0; idx < 16; idx++) {
            printf("[%d] check pattern idx %lu val %016lx\n", my_pe, idx, host_check[idx]);
        }
    }

    return test_size;
}

int main(int argc, char **argv)
{
    class ibput_tester t(argc, argv);

    size_t bufsize = (t.max_nelems * t.typesize(SIZE128)) + 4096L;
    t.alloc_memory(bufsize);
    size_t errors = 0;

    GEN_TABLES(ibput, NOP, nop)
    GEN_SIZE_TABLES(ibput, , NOP, nop)

    if (!t.test_types_set) t.add_test_type_list(strided_rma_copy_types);
    errors += t.run_aligned_tests(NOP);
    return (t.finalize_and_report(errors));
}
