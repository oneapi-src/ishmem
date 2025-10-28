/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <algorithm>
#include "ishmem_tester.h"
#include "pt2pt_sync_test.h"

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    size_t ret = 0;                                                                                \
    for (size_t cmp = 1; cmp <= NUM_COMPARISON_OPERATORS; cmp++) {                                 \
        type cmp_value = 1;                                                                        \
        switch (cmp) {                                                                             \
            case ISHMEM_CMP_EQ:                                                                    \
                cmp_value = 2;                                                                     \
                break;                                                                             \
            case ISHMEM_CMP_NE:                                                                    \
            case ISHMEM_CMP_GT:                                                                    \
                break;                                                                             \
            case ISHMEM_CMP_GE:                                                                    \
                cmp_value = 2;                                                                     \
                break;                                                                             \
            case ISHMEM_CMP_LT:                                                                    \
                if (sizeof(type) == sizeof(uint64_t)) {                                            \
                    ishmem_uint64_p(&((uint64_t *) src)[nelems / 2], 0, ishmem_my_pe());           \
                } else if (sizeof(type) == sizeof(uint32_t)) {                                     \
                    ishmem_uint32_p(&((uint32_t *) src)[nelems / 2], 0, ishmem_my_pe());           \
                } else if (sizeof(type) == sizeof(uint16_t)) {                                     \
                    ishmem_uint16_p(&((uint16_t *) src)[nelems / 2], 0, ishmem_my_pe());           \
                } else if (sizeof(type) == sizeof(uint8_t)) {                                      \
                    ishmem_uint8_p(&((uint8_t *) src)[nelems / 2], 0, ishmem_my_pe());             \
                }                                                                                  \
                ishmem_quiet();                                                                    \
                break;                                                                             \
            case ISHMEM_CMP_LE:                                                                    \
                cmp_value = 0;                                                                     \
                break;                                                                             \
        }                                                                                          \
        ret = ishmem_##                                                                            \
            typename##_##testname((type *) src, nelems, NULL, static_cast<int>(cmp), cmp_value);   \
                                                                                                   \
        /* Could use memcpy((size_t *) dest + (cmp - 1), &ret, sizeof(size_t)), but this will      \
         * cause seg faults for host_host_device and host_device_device test modes. */             \
        ishmem_size_p((size_t *) dest + (cmp - 1), ret, ishmem_my_pe());                           \
        ishmem_quiet();                                                                            \
    }

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    size_t ret = 0;                                                                                \
    for (size_t cmp = 1; cmp <= NUM_COMPARISON_OPERATORS; cmp++) {                                 \
        type cmp_value = 1;                                                                        \
        switch (cmp) {                                                                             \
            case ISHMEM_CMP_EQ:                                                                    \
                cmp_value = 2;                                                                     \
                break;                                                                             \
            case ISHMEM_CMP_NE:                                                                    \
            case ISHMEM_CMP_GT:                                                                    \
                break;                                                                             \
            case ISHMEM_CMP_GE:                                                                    \
                cmp_value = 2;                                                                     \
                break;                                                                             \
            case ISHMEM_CMP_LT:                                                                    \
                ((type *) src)[nelems / 2] = 0;                                                    \
                break;                                                                             \
            case ISHMEM_CMP_LE:                                                                    \
                cmp_value = 0;                                                                     \
                break;                                                                             \
        }                                                                                          \
        ret = ishmemx_##typename##_##testname##_work_group((type *) src, nelems, NULL,             \
                                                           static_cast<int>(cmp), cmp_value, grp); \
                                                                                                   \
        if (grp.leader()) memcpy((size_t *) dest + (cmp - 1), &ret, sizeof(size_t));               \
    }

GEN_FNS(test_any, NOP, nop)

class test_any_tester : public ishmem_tester {
  public:
    test_any_tester(int argc, char *argv[]) : ishmem_tester(argc, argv) {}
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
};

size_t test_any_tester::create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                              size_t nelems)
{
    size_t test_size = nelems * typesize(t);
    for (size_t i = 0; i < nelems; i++) {
        if (typesize(t) == sizeof(uint64_t)) {
            ((uint64_t *) aligned_source)[i] = (i == nelems / 2) + 1;
        } else if (typesize(t) == sizeof(uint32_t)) {
            ((uint32_t *) aligned_source)[i] = (i == nelems / 2) + 1;
        } else if (typesize(t) == sizeof(uint16_t)) {
            ((uint16_t *) aligned_source)[i] = (i == nelems / 2) + 1;
        } else if (typesize(t) == sizeof(uint8_t)) {
            ((uint8_t *) aligned_source)[i] = (i == nelems / 2) + 1;
        }
    }

    return test_size;
}

size_t test_any_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                             size_t nelems)
{
    size_t check_size = sizeof(size_t) * NUM_COMPARISON_OPERATORS;
    for (size_t i = 0; i < NUM_COMPARISON_OPERATORS; i++) {
        ((size_t *) host_check)[i] = (nelems / 2);
    }

    return check_size;
}

int main(int argc, char **argv)
{
    class test_any_tester t(argc, argv);
    t.max_nelems = 512;

    size_t bufsize =
        std::max(NUM_COMPARISON_OPERATORS * sizeof(uint64_t), t.max_nelems * sizeof(uint64_t));
    t.alloc_memory(bufsize);
    size_t errors = 0;

    GEN_TABLES(test_any, NOP, nop)

    if (!t.test_types_set) t.add_test_type_list(standard_amo_types);
    errors += t.run_aligned_tests(NOP);
    return (t.finalize_and_report(errors));
}
