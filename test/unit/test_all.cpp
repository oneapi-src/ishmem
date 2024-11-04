/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <algorithm>
#include "ishmem_tester.h"
#include "pt2pt_sync_test.h"

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    uint64_t ret = 0;                                                                              \
    for (size_t cmp = 1; cmp <= NUM_COMPARISON_OPERATORS; cmp++) {                                 \
        type cmp_value = 0;                                                                        \
        switch (cmp) {                                                                             \
            case ISHMEM_CMP_EQ:                                                                    \
                cmp_value = 1;                                                                     \
                break;                                                                             \
            case ISHMEM_CMP_NE:                                                                    \
            case ISHMEM_CMP_GT:                                                                    \
            case ISHMEM_CMP_GE:                                                                    \
                break;                                                                             \
            case ISHMEM_CMP_LT:                                                                    \
            case ISHMEM_CMP_LE:                                                                    \
                cmp_value = 2;                                                                     \
                break;                                                                             \
        }                                                                                          \
        ret = static_cast<uint64_t>(ishmem_##typename##_##testname(                                \
            (type *) src, nelems, NULL, static_cast<int>(cmp), cmp_value));                        \
                                                                                                   \
        /* Could use memcpy((uint64_t *) dest + (cmp - 1), &ret, sizeof(uint64_t)), but this will  \
         * cause seg faults for host_host_device and host_device_device test modes. */             \
        ishmem_uint64_p((uint64_t *) dest + (cmp - 1), ret, ishmem_my_pe());                       \
        ishmem_quiet();                                                                            \
    }

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    uint64_t ret = 0;                                                                              \
    for (size_t cmp = 1; cmp <= NUM_COMPARISON_OPERATORS; cmp++) {                                 \
        type cmp_value = 0;                                                                        \
        switch (cmp) {                                                                             \
            case ISHMEM_CMP_EQ:                                                                    \
                cmp_value = 1;                                                                     \
                break;                                                                             \
            case ISHMEM_CMP_NE:                                                                    \
            case ISHMEM_CMP_GT:                                                                    \
            case ISHMEM_CMP_GE:                                                                    \
                break;                                                                             \
            case ISHMEM_CMP_LT:                                                                    \
            case ISHMEM_CMP_LE:                                                                    \
                cmp_value = 2;                                                                     \
                break;                                                                             \
        }                                                                                          \
        ret = static_cast<uint64_t>(ishmemx_##typename##_##testname##_work_group(                  \
            (type *) src, nelems, NULL, static_cast<int>(cmp), cmp_value, grp));                   \
                                                                                                   \
        if (grp.leader()) memcpy((uint64_t *) dest + (cmp - 1), &ret, sizeof(uint64_t));           \
    }

GEN_FNS(test_all, NOP, nop)

class test_all_tester : public ishmem_tester {
  public:
    test_all_tester(int argc, char *argv[]) : ishmem_tester(argc, argv) {}
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
};

size_t test_all_tester::create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                              size_t nelems)
{
    size_t test_size = nelems * typesize(t);
    for (size_t i = 0; i < nelems; i++) {
        if (typesize(t) == sizeof(uint64_t)) {
            ((uint64_t *) aligned_source)[i] = 1;
        } else if (typesize(t) == sizeof(uint32_t)) {
            ((uint32_t *) aligned_source)[i] = 1;
        } else if (typesize(t) == sizeof(uint16_t)) {
            ((uint16_t *) aligned_source)[i] = 1;
        } else if (typesize(t) == sizeof(uint8_t)) {
            ((uint8_t *) aligned_source)[i] = 1;
        }
    }

    return test_size;
}

size_t test_all_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                             size_t nelems)
{
    size_t check_size = sizeof(uint64_t) * NUM_COMPARISON_OPERATORS;
    for (size_t i = 0; i < NUM_COMPARISON_OPERATORS; i++) {
        ((uint64_t *) host_check)[i] = 1;
    }

    return check_size;
}

int main(int argc, char **argv)
{
    class test_all_tester t(argc, argv);

    size_t bufsize =
        std::max(NUM_COMPARISON_OPERATORS * sizeof(uint64_t), t.max_nelems * sizeof(uint64_t));
    t.alloc_memory(bufsize);
    size_t errors = 0;

    GEN_TABLES(test_all, NOP, nop)

    if (!t.test_types_set) t.add_test_type_list(standard_amo_types);
    errors += t.run_aligned_tests(NOP);
    return (t.finalize_and_report(errors));
}
