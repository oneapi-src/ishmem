/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <algorithm>
#include "ishmem_tester.h"
#include "pt2pt_sync_test.h"

#ifdef TEST_HOST_FN
#undef TEST_HOST_FN
#endif

#ifdef TEST_SINGLE_FN
#undef TEST_SINGLE_FN
#endif

#ifdef TEST_SUBGROUP_FN
#undef TEST_SUBGROUP_FN
#endif

#ifdef TEST_GRP1_FN
#undef TEST_GRP1_FN
#endif

#ifdef TEST_GRP2_FN
#undef TEST_GRP2_FN
#endif

#ifdef TEST_GRP3_FN
#undef TEST_GRP3_FN
#endif

void *_cmp_values = nullptr;

#define ASSIGN_CMP_VALUES(type, _cmp_values, nelems, cmp)                                          \
    switch (cmp) {                                                                                 \
        case ISHMEM_CMP_EQ:                                                                        \
            for (size_t i = 0; i < nelems; i++) {                                                  \
                ((type *) _cmp_values)[i] = (i % 10) + 5;                                          \
            }                                                                                      \
            ((type *) _cmp_values)[nelems / 2] = 2;                                                \
            break;                                                                                 \
        case ISHMEM_CMP_NE:                                                                        \
            for (size_t i = 0; i < nelems; i++) {                                                  \
                ((type *) _cmp_values)[i] = 2;                                                     \
            }                                                                                      \
            ((type *) _cmp_values)[nelems / 2] = 3;                                                \
            break;                                                                                 \
        case ISHMEM_CMP_GT:                                                                        \
            for (size_t i = 0; i < nelems; i++) {                                                  \
                ((type *) _cmp_values)[i] = (i % 10) + 5;                                          \
            }                                                                                      \
            ((type *) _cmp_values)[nelems / 2] = 1;                                                \
            break;                                                                                 \
        case ISHMEM_CMP_GE:                                                                        \
            for (size_t i = 0; i < nelems; i++) {                                                  \
                ((type *) _cmp_values)[i] = (i % 10) + 5;                                          \
            }                                                                                      \
            ((type *) _cmp_values)[nelems / 2] = 2;                                                \
            break;                                                                                 \
        case ISHMEM_CMP_LT:                                                                        \
            for (size_t i = 0; i < nelems; i++) {                                                  \
                ((type *) _cmp_values)[i] = (i % 2);                                               \
            }                                                                                      \
            ((type *) _cmp_values)[nelems / 2] = 3;                                                \
            break;                                                                                 \
        case ISHMEM_CMP_LE:                                                                        \
            for (size_t i = 0; i < nelems; i++) {                                                  \
                ((type *) _cmp_values)[i] = (i % 2);                                               \
            }                                                                                      \
            ((type *) _cmp_values)[nelems / 2] = 2;                                                \
            break;                                                                                 \
        default:                                                                                   \
            break;                                                                                 \
    }                                                                                              \
    const type *cmp_values = (const type *) _cmp_values;

#define TEST_HOST_FN(testname, suffix, typeenum, typename, type, op, opname)                       \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        for (size_t cmp = 1; cmp <= NUM_COMPARISON_OPERATORS; cmp++) {                             \
            ASSIGN_CMP_VALUES(type, _cmp_values, nelems, cmp)                                      \
            ishmem_team_t team __attribute__((unused)) = ISHMEM_TEAM_WORLD;                        \
            ishmem_sync_all();                                                                     \
            TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                     \
            ishmem_sync_all();                                                                     \
        }                                                                                          \
    }

#define TEST_SINGLE_FN(testname, suffix, typeenum, typename, type, op, opname)                     \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        for (size_t cmp = 1; cmp <= NUM_COMPARISON_OPERATORS; cmp++) {                             \
            ASSIGN_CMP_VALUES(type, _cmp_values, nelems, cmp)                                      \
            q.single_task([=]() {                                                                  \
                 ishmem_team_t team __attribute__((unused)) = ISHMEM_TEAM_WORLD;                   \
                 ishmem_sync_all();                                                                \
                 TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                \
                 ishmem_sync_all();                                                                \
             }).wait_and_throw();                                                                  \
        }                                                                                          \
    }

#define TEST_SUBGROUP_FN(testname, suffix, typeenum, typename, type, op, opname)                   \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        for (size_t cmp = 1; cmp <= NUM_COMPARISON_OPERATORS; cmp++) {                             \
            ASSIGN_CMP_VALUES(type, _cmp_values, nelems, cmp)                                      \
            q.parallel_for(sycl::nd_range<1>(sycl::range<1>(x_size), sycl::range<1>(x_size)),      \
                           [=](sycl::nd_item<1> it) {                                              \
                               ishmem_team_t team __attribute__((unused)) = ISHMEM_TEAM_WORLD;     \
                               auto grp = it.get_sub_group();                                      \
                               ishmemx_sync_all_work_group(grp);                                   \
                               TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op,      \
                                                      opname)                                      \
                               ishmemx_sync_all_work_group(grp);                                   \
                           })                                                                      \
                .wait_and_throw();                                                                 \
        }                                                                                          \
    }

#define TEST_GRP1_FN(testname, suffix, typeenum, typename, type, op, opname)                       \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        for (size_t cmp = 1; cmp <= NUM_COMPARISON_OPERATORS; cmp++) {                             \
            ASSIGN_CMP_VALUES(type, _cmp_values, nelems, cmp)                                      \
            q.parallel_for(sycl::nd_range<1>(sycl::range<1>(x_size), sycl::range<1>(x_size)),      \
                           [=](sycl::nd_item<1> it) {                                              \
                               ishmem_team_t team __attribute__((unused)) = ISHMEM_TEAM_WORLD;     \
                               auto grp = it.get_group();                                          \
                               ishmemx_sync_all_work_group(grp);                                   \
                               TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op,      \
                                                      opname)                                      \
                               ishmemx_sync_all_work_group(grp);                                   \
                           })                                                                      \
                .wait_and_throw();                                                                 \
        }                                                                                          \
    }

#define TEST_GRP2_FN(testname, suffix, typeenum, typename, type, op, opname)                       \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        for (size_t cmp = 1; cmp <= NUM_COMPARISON_OPERATORS; cmp++) {                             \
            ASSIGN_CMP_VALUES(type, _cmp_values, nelems, cmp)                                      \
            q.parallel_for(sycl::nd_range<2>(sycl::range<2>(x_size, y_size),                       \
                                             sycl::range<2>(x_size, y_size)),                      \
                           [=](sycl::nd_item<2> it) {                                              \
                               ishmem_team_t team __attribute__((unused)) = ISHMEM_TEAM_WORLD;     \
                               auto grp = it.get_group();                                          \
                               ishmemx_sync_all_work_group(grp);                                   \
                               TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op,      \
                                                      opname)                                      \
                               ishmemx_sync_all_work_group(grp);                                   \
                           })                                                                      \
                .wait_and_throw();                                                                 \
        }                                                                                          \
    }

#define TEST_GRP3_FN(testname, suffix, typeenum, typename, type, op, opname)                       \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        for (size_t cmp = 1; cmp <= NUM_COMPARISON_OPERATORS; cmp++) {                             \
            ASSIGN_CMP_VALUES(type, _cmp_values, nelems, cmp)                                      \
            q.parallel_for(sycl::nd_range<3>(sycl::range<3>(x_size, y_size, z_size),               \
                                             sycl::range<3>(x_size, y_size, z_size)),              \
                           [=](sycl::nd_item<3> it) {                                              \
                               ishmem_team_t team __attribute__((unused)) = ISHMEM_TEAM_WORLD;     \
                               auto grp = it.get_group();                                          \
                               ishmemx_sync_all_work_group(grp);                                   \
                               TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op,      \
                                                      opname);                                     \
                               ishmemx_sync_all_work_group(grp);                                   \
                           })                                                                      \
                .wait_and_throw();                                                                 \
        }                                                                                          \
    }

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    size_t ret = 0;                                                                                \
    ret = ishmem_##                                                                                \
        typename##_##testname((type *) src, nelems, NULL, static_cast<int>(cmp), cmp_values);      \
                                                                                                   \
    /* Could use memcpy((size_t *) dest + (cmp - 1), &ret, sizeof(size_t)), but this will          \
     * cause seg faults for host_host_device and host_device_device test modes. */                 \
    ishmem_size_p((size_t *) dest + (cmp - 1), ret, ishmem_my_pe());                               \
    ishmem_quiet();

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    size_t ret = 0;                                                                                \
    ret = ishmemx_##typename##_##testname##_work_group((type *) src, nelems, NULL,                 \
                                                       static_cast<int>(cmp), cmp_values, grp);    \
                                                                                                   \
    if (grp.leader()) memcpy((size_t *) dest + (cmp - 1), &ret, sizeof(size_t));

GEN_FNS(test_any_vector, NOP, nop)

class test_any_vector_tester : public ishmem_tester {
  public:
    test_any_vector_tester(int argc, char *argv[]) : ishmem_tester(argc, argv) {}
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
};

size_t test_any_vector_tester::create_source_pattern(ishmemi_type_t t, ishmemi_op_t op,
                                                     testmode_t mode, size_t nelems)
{
    size_t test_size = nelems * typesize(t);
    for (size_t i = 0; i < nelems; i++) {
        if (typesize(t) == sizeof(uint64_t)) {
            ((uint64_t *) aligned_source)[i] = 2;
        } else if (typesize(t) == sizeof(uint32_t)) {
            ((uint32_t *) aligned_source)[i] = 2;
        } else if (typesize(t) == sizeof(uint16_t)) {
            ((uint16_t *) aligned_source)[i] = 2;
        } else if (typesize(t) == sizeof(uint8_t)) {
            ((uint8_t *) aligned_source)[i] = 2;
        }
    }

    return test_size;
}

size_t test_any_vector_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op,
                                                    testmode_t mode, size_t nelems)
{
    size_t check_size = sizeof(size_t) * NUM_COMPARISON_OPERATORS;
    for (size_t i = 0; i < NUM_COMPARISON_OPERATORS; i++) {
        ((size_t *) host_check)[i] = (nelems / 2);
    }

    return check_size;
}

int main(int argc, char **argv)
{
    class test_any_vector_tester t(argc, argv);
    t.max_nelems = 512;

    size_t bufsize =
        std::max(NUM_COMPARISON_OPERATORS * sizeof(uint64_t), t.max_nelems * sizeof(uint64_t));
    t.alloc_memory(bufsize);
    _cmp_values = sycl::malloc_host(bufsize, t.q);
    size_t errors = 0;

    GEN_TABLES(test_any_vector, NOP, nop)

    if (!t.test_types_set) t.add_test_type_list(standard_amo_types);
    errors += t.run_aligned_tests(NOP);
    sycl::free(_cmp_values, t.q);
    return (t.finalize_and_report(errors));
}
