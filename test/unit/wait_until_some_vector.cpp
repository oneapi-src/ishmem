/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem_tester.h"
#include "pt2pt_sync_test.h"

#ifdef TEST_HOST_FN
#undef TEST_HOST_FN
#endif

#ifdef TEST_ON_QUEUE_FN
#undef TEST_ON_QUEUE_FN
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

#undef TEST_BRANCH_ON_QUEUE

void *_cmp_values = nullptr;

#define ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                                           \
    do {                                                                                           \
        type val = 0;                                                                              \
        for (size_t i = 0; (i == 0) || (i < nelems / 2); i++) {                                    \
            switch (cmp) {                                                                         \
                case ISHMEM_CMP_EQ:                                                                \
                case ISHMEM_CMP_NE:                                                                \
                    /* Values range from 4-13 */                                                   \
                    val = (i % 10) + 4;                                                            \
                    break;                                                                         \
                case ISHMEM_CMP_GT:                                                                \
                case ISHMEM_CMP_GE:                                                                \
                    /* Values range from 5-14 */                                                   \
                    val = (i % 10) + 5;                                                            \
                    break;                                                                         \
                case ISHMEM_CMP_LT:                                                                \
                    val = (i % 2);                                                                 \
                    break;                                                                         \
                case ISHMEM_CMP_LE:                                                                \
                    /* Values range from 0-2 */                                                    \
                    val = (i % 3);                                                                 \
                    break;                                                                         \
                default:                                                                           \
                    break;                                                                         \
            }                                                                                      \
            ishmem_##typename##_atomic_set(((type *) src) + ((cmp - 1) * nelems) + i, val,         \
                                           (ishmem_my_pe() + 1) % ishmem_n_pes());                 \
        }                                                                                          \
        ishmem_quiet();                                                                            \
    } while (0);

#define ASSIGN_CMP_VALUES(type, _cmp_values, nelems, cmp)                                          \
    switch (cmp) {                                                                                 \
        case ISHMEM_CMP_EQ:                                                                        \
            for (size_t i = 0; i < nelems; i++) {                                                  \
                /* Values range from 4-13 */                                                       \
                ((type *) _cmp_values)[i] = (i % 10) + 4;                                          \
            }                                                                                      \
            break;                                                                                 \
        case ISHMEM_CMP_NE:                                                                        \
            for (size_t i = 0; i < nelems; i++) {                                                  \
                ((type *) _cmp_values)[i] = 3;                                                     \
            }                                                                                      \
            break;                                                                                 \
        case ISHMEM_CMP_GT:                                                                        \
            for (size_t i = 0; i < nelems; i++) {                                                  \
                /* Values range from 4-13 */                                                       \
                ((type *) _cmp_values)[i] = (i % 10) + 4;                                          \
            }                                                                                      \
            break;                                                                                 \
        case ISHMEM_CMP_GE:                                                                        \
            for (size_t i = 0; i < nelems; i++) {                                                  \
                /* Values range from 5-14 */                                                       \
                ((type *) _cmp_values)[i] = (i % 10) + 5;                                          \
            }                                                                                      \
            break;                                                                                 \
        case ISHMEM_CMP_LT:                                                                        \
            for (size_t i = 0; i < nelems; i++) {                                                  \
                /* Values range from 1-2 */                                                        \
                ((type *) _cmp_values)[i] = (i % 2) + 1;                                           \
            }                                                                                      \
            break;                                                                                 \
        case ISHMEM_CMP_LE:                                                                        \
            for (size_t i = 0; i < nelems; i++) {                                                  \
                /* Values range from 0-2 */                                                        \
                ((type *) _cmp_values)[i] = (i % 3);                                               \
            }                                                                                      \
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
            ishmem_sync_all();                                                                     \
            if (ishmem_my_pe() == 0) {                                                             \
                ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                                   \
            }                                                                                      \
            TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                     \
            if (ishmem_my_pe() != 0) {                                                             \
                ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                                   \
            }                                                                                      \
            ishmem_sync_all();                                                                     \
        }                                                                                          \
    }

#define TEST_ON_QUEUE_FN(testname, suffix, typeenum, typename, type, op, opname)                   \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        for (size_t cmp = 1; cmp <= NUM_COMPARISON_OPERATORS; cmp++) {                             \
            ASSIGN_CMP_VALUES(type, _cmp_values, nelems, cmp)                                      \
            ishmem_sync_all();                                                                     \
            if (ishmem_my_pe() == 0) {                                                             \
                ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                                   \
            }                                                                                      \
            TEST_BRANCH_ON_QUEUE(testname, typeenum, typename, type, op, opname)                   \
            if (ishmem_my_pe() != 0) {                                                             \
                ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                                   \
            }                                                                                      \
            ishmem_sync_all();                                                                     \
        }                                                                                          \
    }

#define TEST_SINGLE_FN(testname, suffix, typeenum, typename, type, op, opname)                     \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        int *iter = sycl::malloc_host<int>(1, q);                                                  \
        *iter = -1;                                                                                \
        for (size_t cmp = 1; cmp <= NUM_COMPARISON_OPERATORS; cmp++) {                             \
            ASSIGN_CMP_VALUES(type, _cmp_values, nelems, cmp)                                      \
            q.single_task([=]() {                                                                  \
                 ishmem_sync_all();                                                                \
                 if (ishmem_my_pe() == 0) {                                                        \
                     ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                              \
                 }                                                                                 \
                 TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                \
                 if (ishmem_my_pe() != 0) {                                                        \
                     ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                              \
                 }                                                                                 \
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
                               auto grp = it.get_sub_group();                                      \
                               ishmemx_sync_all_work_group(grp);                                   \
                               if (ishmem_my_pe() == 0 && grp.leader()) {                          \
                                   ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                \
                               }                                                                   \
                               TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op,      \
                                                      opname)                                      \
                               if (ishmem_my_pe() != 0 && grp.leader()) {                          \
                                   ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                \
                               }                                                                   \
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
                               auto grp = it.get_sub_group();                                      \
                               ishmemx_sync_all_work_group(grp);                                   \
                               if (ishmem_my_pe() == 0 && grp.leader()) {                          \
                                   ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                \
                               }                                                                   \
                               TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op,      \
                                                      opname)                                      \
                               if (ishmem_my_pe() != 0 && grp.leader()) {                          \
                                   ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                \
                               }                                                                   \
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
                               auto grp = it.get_group();                                          \
                               ishmemx_sync_all_work_group(grp);                                   \
                               if (ishmem_my_pe() == 0 && grp.leader()) {                          \
                                   ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                \
                               }                                                                   \
                               TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op,      \
                                                      opname)                                      \
                               if (ishmem_my_pe() != 0 && grp.leader()) {                          \
                                   ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                \
                               }                                                                   \
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
                               auto grp = it.get_group();                                          \
                               ishmemx_sync_all_work_group(grp);                                   \
                               if (ishmem_my_pe() == 0 && grp.leader()) {                          \
                                   ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                \
                               }                                                                   \
                               TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op,      \
                                                      opname);                                     \
                               if (ishmem_my_pe() != 0 && grp.leader()) {                          \
                                   ATOMIC_SET_IVARS_WAIT_UNTIL_SOME(typename, type)                \
                               }                                                                   \
                               ishmemx_sync_all_work_group(grp);                                   \
                           })                                                                      \
                .wait_and_throw();                                                                 \
        }                                                                                          \
    }

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    size_t ret = 0;                                                                                \
    size_t exit_val = (nelems > 1) ? nelems / 2 : 1;                                               \
    while (ret != exit_val) {                                                                      \
        ret = ishmem_##typename##_##testname(                                                      \
            ((type *) src) + ((cmp - 1) * nelems), nelems,                                         \
            ((size_t *) dest + (NUM_COMPARISON_OPERATORS + (nelems * (cmp - 1)))), NULL,           \
            static_cast<int>(cmp), cmp_values);                                                    \
    }                                                                                              \
                                                                                                   \
    /* Could use memcpy((size_t *) dest + (i - 1), &ret, sizeof(size_t)), but this will cause      \
     * seg faults for host_host_device and host_device_device test modes. */                       \
    ishmem_size_p((size_t *) dest + (cmp - 1), ret, ishmem_my_pe());                               \
    ishmem_quiet();

#define TEST_BRANCH_ON_QUEUE(testname, typeenum, typename, type, op, opname)                       \
    size_t exit_val = (nelems > 1) ? nelems / 2 : 1;                                               \
    while (*((size_t *) res) != exit_val) {                                                        \
        ishmemx_##typename##_##testname##_on_queue(                                                \
            ((type *) src) + ((cmp - 1) * nelems), nelems,                                         \
            ((size_t *) dest + (NUM_COMPARISON_OPERATORS + (nelems * (cmp - 1)))), NULL,           \
            static_cast<int>(cmp), cmp_values, (size_t *) res, q);                                 \
        q.wait_and_throw();                                                                        \
    }                                                                                              \
                                                                                                   \
    /* Could use memcpy((size_t *) dest + (i - 1), &ret, sizeof(size_t)), but this will cause      \
     * seg faults for host_host_device and host_device_device test modes. */                       \
    ishmem_size_p((size_t *) dest + (cmp - 1), *((size_t *) res), ishmem_my_pe());                 \
    ishmem_quiet();                                                                                \
    *res = 0;

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    size_t ret = 0;                                                                                \
    size_t exit_val = (nelems > 1) ? nelems / 2 : 1;                                               \
    while (ret != exit_val) {                                                                      \
        ret = ishmemx_##typename##_##testname##_work_group(                                        \
            ((type *) src) + ((cmp - 1) * nelems), nelems,                                         \
            ((size_t *) dest) + (NUM_COMPARISON_OPERATORS + (nelems * (cmp - 1))), NULL,           \
            static_cast<int>(cmp), cmp_values, grp);                                               \
    }                                                                                              \
                                                                                                   \
    if (grp.leader()) memcpy((size_t *) dest + (cmp - 1), &ret, sizeof(size_t));

GEN_FNS(wait_until_some_vector, NOP, nop)

class wait_until_some_vector_tester : public ishmem_tester {
  public:
    wait_until_some_vector_tester(int argc, char *argv[]) : ishmem_tester(argc, argv, true) {}
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
};

size_t wait_until_some_vector_tester::create_source_pattern(ishmemi_type_t t, ishmemi_op_t op,
                                                            testmode_t mode, size_t nelems)
{
    size_t test_size = nelems * NUM_COMPARISON_OPERATORS * typesize(t);
    for (size_t i = 0; i < nelems * NUM_COMPARISON_OPERATORS; i++) {
        if (typesize(t) == sizeof(uint64_t)) {
            ((uint64_t *) aligned_source)[i] = 3;
        } else if (typesize(t) == sizeof(uint32_t)) {
            ((uint32_t *) aligned_source)[i] = 3;
        } else if (typesize(t) == sizeof(uint16_t)) {
            ((uint16_t *) aligned_source)[i] = 3;
        } else if (typesize(t) == sizeof(uint8_t)) {
            ((uint8_t *) aligned_source)[i] = 3;
        }
    }

    return test_size;
}

size_t wait_until_some_vector_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op,
                                                           testmode_t mode, size_t nelems)
{
    int my_pe = ishmem_my_pe();
    size_t check_size =
        sizeof(size_t) * (NUM_COMPARISON_OPERATORS + (NUM_COMPARISON_OPERATORS * nelems));

    for (size_t i = 0; i < NUM_COMPARISON_OPERATORS; i++) {
        ((size_t *) host_check)[i] = (nelems > 1) ? (nelems / 2) : 1;
    }

    // Assign the first half of the elements to the value of i, and the second half to 0x80 + my_pe
    for (int cmp = 0; cmp < NUM_COMPARISON_OPERATORS; cmp++) {
        size_t starting_idx = NUM_COMPARISON_OPERATORS + (nelems * static_cast<size_t>(cmp));
        for (size_t i = 0; (i == 0) || (i < nelems); i++) {
            if ((i == 0) || (i < (nelems / 2))) {
                ((size_t *) host_check)[starting_idx + i] = i;
            } else {
                memset(&((size_t *) host_check)[starting_idx + i], 0x80L + my_pe, 8);
            }
        }
    }

    return check_size;
}

int main(int argc, char **argv)
{
    class wait_until_some_vector_tester t(argc, argv);

    // NOTE: Remove this constraint after support added to disable fi_inject_atomic
    t.max_nelems = 2;

    size_t bufsize =
        (NUM_COMPARISON_OPERATORS + (NUM_COMPARISON_OPERATORS * t.max_nelems)) * sizeof(size_t);
    t.alloc_memory(bufsize);
    _cmp_values = sycl::malloc_host(bufsize, t.q);
    size_t errors = 0;

    GEN_TABLES(wait_until_some_vector, NOP, nop)

    if (!t.test_types_set) t.add_test_type_list(standard_amo_types);
    errors += t.run_aligned_tests(NOP);
    sycl::free(_cmp_values, t.q);
    return (t.finalize_and_report(errors));
}
