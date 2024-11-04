/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem_tester.h"
#include "pt2pt_sync_test.h"

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    size_t ret = 0;                                                                                \
    for (size_t cmp = 1; cmp <= NUM_COMPARISON_OPERATORS; cmp++) {                                 \
        type cmp_value = 1;                                                                        \
        switch (cmp) {                                                                             \
            case ISHMEM_CMP_EQ:                                                                    \
                break;                                                                             \
            case ISHMEM_CMP_NE:                                                                    \
            case ISHMEM_CMP_GT:                                                                    \
                cmp_value = 0;                                                                     \
                break;                                                                             \
            case ISHMEM_CMP_GE:                                                                    \
            case ISHMEM_CMP_LT:                                                                    \
                break;                                                                             \
            case ISHMEM_CMP_LE:                                                                    \
                cmp_value = 0;                                                                     \
                break;                                                                             \
        }                                                                                          \
        ret = ishmem_##typename##_##testname(                                                      \
            (type *) src, nelems,                                                                  \
            ((size_t *) dest + (NUM_COMPARISON_OPERATORS + (nelems * (cmp - 1)))), NULL,           \
            static_cast<int>(cmp), cmp_value);                                                     \
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
                break;                                                                             \
            case ISHMEM_CMP_NE:                                                                    \
            case ISHMEM_CMP_GT:                                                                    \
                cmp_value = 0;                                                                     \
                break;                                                                             \
            case ISHMEM_CMP_GE:                                                                    \
            case ISHMEM_CMP_LT:                                                                    \
                break;                                                                             \
            case ISHMEM_CMP_LE:                                                                    \
                cmp_value = 0;                                                                     \
                break;                                                                             \
        }                                                                                          \
        ret = ishmemx_##typename##_##testname##_work_group(                                        \
            (type *) src, nelems,                                                                  \
            ((size_t *) dest) + (NUM_COMPARISON_OPERATORS + (nelems * (cmp - 1))), NULL,           \
            static_cast<int>(cmp), cmp_value, grp);                                                \
                                                                                                   \
        if (grp.leader()) memcpy((size_t *) dest + (cmp - 1), &ret, sizeof(size_t));               \
    }

GEN_FNS(test_some, NOP, nop)

class test_some_tester : public ishmem_tester {
  public:
    test_some_tester(int argc, char *argv[]) : ishmem_tester(argc, argv) {}
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
};

size_t test_some_tester::create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                               size_t nelems)
{
    size_t test_size = nelems * typesize(t);
    for (size_t i = 0; i < nelems; i++) {
        if (typesize(t) == sizeof(uint64_t)) {
            ((uint64_t *) aligned_source)[i] = ((i >= nelems / 2) && (nelems > 1));
        } else if (typesize(t) == sizeof(uint32_t)) {
            ((uint32_t *) aligned_source)[i] = ((i >= nelems / 2) && (nelems > 1));
        } else if (typesize(t) == sizeof(uint16_t)) {
            ((uint16_t *) aligned_source)[i] = ((i >= nelems / 2) && (nelems > 1));
        } else if (typesize(t) == sizeof(uint8_t)) {
            ((uint8_t *) aligned_source)[i] = ((i >= nelems / 2) && (nelems > 1));
        }
    }

    return test_size;
}

size_t test_some_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                              size_t nelems)
{
    int my_pe = ishmem_my_pe();
    size_t check_size =
        sizeof(size_t) * (NUM_COMPARISON_OPERATORS + (NUM_COMPARISON_OPERATORS * nelems));

    for (size_t i = 0; i < NUM_COMPARISON_OPERATORS; i++) {
        ((size_t *) host_check)[i] = (nelems / 2);
    }

    // Special case when nelems == 1
    if (nelems == 1) {
        ((size_t *) host_check)[4] = 1;
        ((size_t *) host_check)[5] = 1;
    }

    for (size_t cmp = 0; cmp < NUM_COMPARISON_OPERATORS; cmp++) {
        size_t starting_idx = NUM_COMPARISON_OPERATORS + (nelems * cmp);
        switch (cmp + 1) {
            case ISHMEM_CMP_EQ:
            case ISHMEM_CMP_NE:
            case ISHMEM_CMP_GT:
            case ISHMEM_CMP_GE:
                for (size_t i = 0; i < (nelems / 2); i++) {
                    ((size_t *) host_check)[starting_idx + i] = (nelems / 2) + i;
                }
                for (size_t i = (nelems / 2); i < nelems; i++) {
                    memset(&((size_t *) host_check)[starting_idx + i], 0x80L + (my_pe), 8);
                }
                break;
            case ISHMEM_CMP_LT:
            case ISHMEM_CMP_LE:
                for (size_t i = 0; i < (nelems / 2); i++) {
                    ((size_t *) host_check)[starting_idx + i] = i;
                }
                for (size_t i = (nelems / 2); i < nelems; i++) {
                    memset(&((size_t *) host_check)[starting_idx + i],
                           (nelems > 1) ? 0x80L + (my_pe) : 0, 8);
                }
                break;
        }
    }

    return check_size;
}

int main(int argc, char **argv)
{
    class test_some_tester t(argc, argv);

    size_t bufsize =
        (NUM_COMPARISON_OPERATORS + (NUM_COMPARISON_OPERATORS * t.max_nelems)) * sizeof(size_t);
    t.alloc_memory(bufsize);
    size_t errors = 0;

    GEN_TABLES(test_some, NOP, nop)

    if (!t.test_types_set) t.add_test_type_list(standard_amo_types);
    errors += t.run_aligned_tests(NOP);
    return (t.finalize_and_report(errors));
}
