/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef REDUCE_TEST_H
#define REDUCE_TEST_H

// call with reduce, etc
#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    ishmem_##                                                                                      \
        typename##_##opname##_##testname(ISHMEM_TEAM_WORLD, (type *) dest, (type *) src, nelems)

#define TEST_BRANCH_ON_QUEUE(testname, typeenum, typename, type, op, opname)                       \
    ishmemx_##                                                                                     \
        typename##_##opname##_##testname##_on_queue((type *) dest, (type *) src, nelems, res, q)

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    ishmemx_##typename##_##opname##_##testname##_work_group(team, (type *) dest, (type *) src,     \
                                                            nelems, grp)

#define TEST_HOST_FN(testname, suffix, typeenum, typename, type, op, opname)                       \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        ishmem_team_t team __attribute__((unused)) = ISHMEM_TEAM_WORLD;                            \
        ishmem_sync_all();                                                                         \
        *res = TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname);                 \
        ishmem_sync_all();                                                                         \
    }

#define TEST_ON_QUEUE_FN(testname, suffix, typeenum, typename, type, op, opname)                   \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        ishmem_team_t team __attribute__((unused)) = ISHMEM_TEAM_WORLD;                            \
        ishmem_sync_all();                                                                         \
        TEST_BRANCH_ON_QUEUE(testname, typeenum, typename, type, op, opname);                      \
        q.wait_and_throw();                                                                        \
        ishmem_barrier_all();                                                                      \
    }

#define TEST_SINGLE_FN(testname, suffix, typeenum, typename, type, op, opname)                     \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        q.single_task([=]() {                                                                      \
             ishmem_team_t team __attribute__((unused)) = ISHMEM_TEAM_WORLD;                       \
             ishmem_sync_all();                                                                    \
             *res = TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname);            \
             ishmem_sync_all();                                                                    \
         }).wait_and_throw();                                                                      \
    }

#define TEST_SUBGROUP_FN(testname, suffix, typeenum, typename, type, op, opname)                   \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(x_size), sycl::range<1>(x_size)),          \
                       [=](sycl::nd_item<1> it) {                                                  \
                           ishmem_team_t team __attribute__((unused)) = ISHMEM_TEAM_WORLD;         \
                           auto grp = it.get_sub_group();                                          \
                           ishmemx_sync_all_work_group(grp);                                       \
                           *res = TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op,   \
                                                         opname);                                  \
                           ishmemx_sync_all_work_group(grp);                                       \
                       })                                                                          \
            .wait_and_throw();                                                                     \
    }

#define TEST_GRP1_FN(testname, suffix, typeenum, typename, type, op, opname)                       \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(x_size), sycl::range<1>(x_size)),          \
                       [=](sycl::nd_item<1> it) {                                                  \
                           ishmem_team_t team __attribute__((unused)) = ISHMEM_TEAM_WORLD;         \
                           auto grp = it.get_group();                                              \
                           ishmemx_sync_all_work_group(grp);                                       \
                           *res = TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op,   \
                                                         opname);                                  \
                           ishmemx_sync_all_work_group(grp);                                       \
                       })                                                                          \
            .wait_and_throw();                                                                     \
    }

#define TEST_GRP2_FN(testname, suffix, typeenum, typename, type, op, opname)                       \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        q.parallel_for(                                                                            \
             sycl::nd_range<2>(sycl::range<2>(x_size, y_size), sycl::range<2>(x_size, y_size)),    \
             [=](sycl::nd_item<2> it) {                                                            \
                 ishmem_team_t team __attribute__((unused)) = ISHMEM_TEAM_WORLD;                   \
                 auto grp = it.get_group();                                                        \
                 ishmemx_sync_all_work_group(grp);                                                 \
                 *res = TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname);    \
                 ishmemx_sync_all_work_group(grp);                                                 \
             })                                                                                    \
            .wait_and_throw();                                                                     \
    }

#define TEST_GRP3_FN(testname, suffix, typeenum, typename, type, op, opname)                       \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        q.parallel_for(sycl::nd_range<3>(sycl::range<3>(x_size, y_size, z_size),                   \
                                         sycl::range<3>(x_size, y_size, z_size)),                  \
                       [=](sycl::nd_item<3> it) {                                                  \
                           ishmem_team_t team __attribute__((unused)) = ISHMEM_TEAM_WORLD;         \
                           auto grp = it.get_group();                                              \
                           ishmemx_sync_all_work_group(grp);                                       \
                           *res = TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op,   \
                                                         opname);                                  \
                           ishmemx_sync_all_work_group(grp);                                       \
                       })                                                                          \
            .wait_and_throw();                                                                     \
    }

#define TEST_MULTI_WG_FN(testname, suffix, typeenum, typename, type, op, opname)                   \
    void testname##_##opname##_##typename##_##suffix(                                              \
        sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest, void *src, size_t nelems)  \
    {                                                                                              \
        ishmem_team_t wg_teams[max_wg];                                                            \
        for (size_t wg = 0; wg < max_wg; wg += 1)                                                  \
            wg_teams[wg] = p_wg_teams[wg];                                                         \
        ishmem_sync_all();                                                                         \
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(x_size * num_wg), sycl::range<1>(x_size)), \
                       [=](sycl::nd_item<1> it) {                                                  \
                           auto grp = it.get_group();                                              \
                           auto gid = grp.get_group_linear_id();                                   \
                           ishmem_team_t team __attribute__((unused)) = wg_teams[gid];             \
                           *res = TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op,   \
                                                         opname);                                  \
                       })                                                                          \
            .wait_and_throw();                                                                     \
        ishmem_sync_all();                                                                         \
    }

#define GEN_BITWISE_HOST_FNS(testname, suffix, op, opname)                                         \
    TEST_HOST_FN(testname, suffix, UCHAR, uchar, unsigned char, op, opname)                        \
    TEST_HOST_FN(testname, suffix, USHORT, ushort, unsigned short, op, opname)                     \
    TEST_HOST_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                           \
    TEST_HOST_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                        \
    TEST_HOST_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)           \
    TEST_HOST_FN(testname, suffix, INT8, int8, int8_t, op, opname)                                 \
    TEST_HOST_FN(testname, suffix, INT16, int16, int16_t, op, opname)                              \
    TEST_HOST_FN(testname, suffix, INT32, int32, int32_t, op, opname)                              \
    TEST_HOST_FN(testname, suffix, INT64, int64, int64_t, op, opname)                              \
    TEST_HOST_FN(testname, suffix, UINT8, uint8, uint8_t, op, opname)                              \
    TEST_HOST_FN(testname, suffix, UINT16, uint16, uint16_t, op, opname)                           \
    TEST_HOST_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                           \
    TEST_HOST_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                           \
    TEST_HOST_FN(testname, suffix, SIZE, size, size_t, op, opname)

#define GEN_BITWISE_ON_QUEUE_FNS(testname, suffix, op, opname)                                     \
    TEST_ON_QUEUE_FN(testname, suffix, UCHAR, uchar, unsigned char, op, opname)                    \
    TEST_ON_QUEUE_FN(testname, suffix, USHORT, ushort, unsigned short, op, opname)                 \
    TEST_ON_QUEUE_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                       \
    TEST_ON_QUEUE_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                    \
    TEST_ON_QUEUE_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)       \
    TEST_ON_QUEUE_FN(testname, suffix, INT8, int8, int8_t, op, opname)                             \
    TEST_ON_QUEUE_FN(testname, suffix, INT16, int16, int16_t, op, opname)                          \
    TEST_ON_QUEUE_FN(testname, suffix, INT32, int32, int32_t, op, opname)                          \
    TEST_ON_QUEUE_FN(testname, suffix, INT64, int64, int64_t, op, opname)                          \
    TEST_ON_QUEUE_FN(testname, suffix, UINT8, uint8, uint8_t, op, opname)                          \
    TEST_ON_QUEUE_FN(testname, suffix, UINT16, uint16, uint16_t, op, opname)                       \
    TEST_ON_QUEUE_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                       \
    TEST_ON_QUEUE_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                       \
    TEST_ON_QUEUE_FN(testname, suffix, SIZE, size, size_t, op, opname)

#define GEN_BITWISE_SINGLE_FNS(testname, suffix, op, opname)                                       \
    TEST_SINGLE_FN(testname, suffix, UCHAR, uchar, unsigned char, op, opname)                      \
    TEST_SINGLE_FN(testname, suffix, USHORT, ushort, unsigned short, op, opname)                   \
    TEST_SINGLE_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                         \
    TEST_SINGLE_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                      \
    TEST_SINGLE_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)         \
    TEST_SINGLE_FN(testname, suffix, INT8, int8, int8_t, op, opname)                               \
    TEST_SINGLE_FN(testname, suffix, INT16, int16, int16_t, op, opname)                            \
    TEST_SINGLE_FN(testname, suffix, INT32, int32, int32_t, op, opname)                            \
    TEST_SINGLE_FN(testname, suffix, INT64, int64, int64_t, op, opname)                            \
    TEST_SINGLE_FN(testname, suffix, UINT8, uint8, uint8_t, op, opname)                            \
    TEST_SINGLE_FN(testname, suffix, UINT16, uint16, uint16_t, op, opname)                         \
    TEST_SINGLE_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                         \
    TEST_SINGLE_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                         \
    TEST_SINGLE_FN(testname, suffix, SIZE, size, size_t, op, opname)

#define GEN_BITWISE_SUBGROUP_FNS(testname, suffix, op, opname)                                     \
    TEST_SUBGROUP_FN(testname, suffix, UCHAR, uchar, unsigned char, op, opname)                    \
    TEST_SUBGROUP_FN(testname, suffix, USHORT, ushort, unsigned short, op, opname)                 \
    TEST_SUBGROUP_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                       \
    TEST_SUBGROUP_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                    \
    TEST_SUBGROUP_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)       \
    TEST_SUBGROUP_FN(testname, suffix, INT8, int8, int8_t, op, opname)                             \
    TEST_SUBGROUP_FN(testname, suffix, INT16, int16, int16_t, op, opname)                          \
    TEST_SUBGROUP_FN(testname, suffix, INT32, int32, int32_t, op, opname)                          \
    TEST_SUBGROUP_FN(testname, suffix, INT64, int64, int64_t, op, opname)                          \
    TEST_SUBGROUP_FN(testname, suffix, UINT8, uint8, uint8_t, op, opname)                          \
    TEST_SUBGROUP_FN(testname, suffix, UINT16, uint16, uint16_t, op, opname)                       \
    TEST_SUBGROUP_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                       \
    TEST_SUBGROUP_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                       \
    TEST_SUBGROUP_FN(testname, suffix, SIZE, size, size_t, op, opname)

#define GEN_BITWISE_GRP1_FNS(testname, suffix, op, opname)                                         \
    TEST_GRP1_FN(testname, suffix, UCHAR, uchar, unsigned char, op, opname)                        \
    TEST_GRP1_FN(testname, suffix, USHORT, ushort, unsigned short, op, opname)                     \
    TEST_GRP1_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                           \
    TEST_GRP1_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                        \
    TEST_GRP1_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)           \
    TEST_GRP1_FN(testname, suffix, INT8, int8, int8_t, op, opname)                                 \
    TEST_GRP1_FN(testname, suffix, INT16, int16, int16_t, op, opname)                              \
    TEST_GRP1_FN(testname, suffix, INT32, int32, int32_t, op, opname)                              \
    TEST_GRP1_FN(testname, suffix, INT64, int64, int64_t, op, opname)                              \
    TEST_GRP1_FN(testname, suffix, UINT8, uint8, uint8_t, op, opname)                              \
    TEST_GRP1_FN(testname, suffix, UINT16, uint16, uint16_t, op, opname)                           \
    TEST_GRP1_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                           \
    TEST_GRP1_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                           \
    TEST_GRP1_FN(testname, suffix, SIZE, size, size_t, op, opname)

#define GEN_BITWISE_GRP2_FNS(testname, suffix, op, opname)                                         \
    TEST_GRP2_FN(testname, suffix, UCHAR, uchar, unsigned char, op, opname)                        \
    TEST_GRP2_FN(testname, suffix, USHORT, ushort, unsigned short, op, opname)                     \
    TEST_GRP2_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                           \
    TEST_GRP2_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                        \
    TEST_GRP2_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)           \
    TEST_GRP2_FN(testname, suffix, INT8, int8, int8_t, op, opname)                                 \
    TEST_GRP2_FN(testname, suffix, INT16, int16, int16_t, op, opname)                              \
    TEST_GRP2_FN(testname, suffix, INT32, int32, int32_t, op, opname)                              \
    TEST_GRP2_FN(testname, suffix, INT64, int64, int64_t, op, opname)                              \
    TEST_GRP2_FN(testname, suffix, UINT8, uint8, uint8_t, op, opname)                              \
    TEST_GRP2_FN(testname, suffix, UINT16, uint16, uint16_t, op, opname)                           \
    TEST_GRP2_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                           \
    TEST_GRP2_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                           \
    TEST_GRP2_FN(testname, suffix, SIZE, size, size_t, op, opname)

#define GEN_BITWISE_GRP3_FNS(testname, suffix, op, opname)                                         \
    TEST_GRP3_FN(testname, suffix, UCHAR, uchar, unsigned char, op, opname)                        \
    TEST_GRP3_FN(testname, suffix, USHORT, ushort, unsigned short, op, opname)                     \
    TEST_GRP3_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                           \
    TEST_GRP3_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                        \
    TEST_GRP3_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)           \
    TEST_GRP3_FN(testname, suffix, INT8, int8, int8_t, op, opname)                                 \
    TEST_GRP3_FN(testname, suffix, INT16, int16, int16_t, op, opname)                              \
    TEST_GRP3_FN(testname, suffix, INT32, int32, int32_t, op, opname)                              \
    TEST_GRP3_FN(testname, suffix, INT64, int64, int64_t, op, opname)                              \
    TEST_GRP3_FN(testname, suffix, UINT8, uint8, uint8_t, op, opname)                              \
    TEST_GRP3_FN(testname, suffix, UINT16, uint16, uint16_t, op, opname)                           \
    TEST_GRP3_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                           \
    TEST_GRP3_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                           \
    TEST_GRP3_FN(testname, suffix, SIZE, size, size_t, op, opname)

#define GEN_BITWISE_MULTI_WG_FNS(testname, suffix, op, opname)                                     \
    TEST_MULTI_WG_FN(testname, suffix, UCHAR, uchar, unsigned char, op, opname)                    \
    TEST_MULTI_WG_FN(testname, suffix, USHORT, ushort, unsigned short, op, opname)                 \
    TEST_MULTI_WG_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                       \
    TEST_MULTI_WG_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                    \
    TEST_MULTI_WG_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)       \
    TEST_MULTI_WG_FN(testname, suffix, INT8, int8, int8_t, op, opname)                             \
    TEST_MULTI_WG_FN(testname, suffix, INT16, int16, int16_t, op, opname)                          \
    TEST_MULTI_WG_FN(testname, suffix, INT32, int32, int32_t, op, opname)                          \
    TEST_MULTI_WG_FN(testname, suffix, INT64, int64, int64_t, op, opname)                          \
    TEST_MULTI_WG_FN(testname, suffix, UINT8, uint8, uint8_t, op, opname)                          \
    TEST_MULTI_WG_FN(testname, suffix, UINT16, uint16, uint16_t, op, opname)                       \
    TEST_MULTI_WG_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                       \
    TEST_MULTI_WG_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                       \
    TEST_MULTI_WG_FN(testname, suffix, SIZE, size, size_t, op, opname)

#define GEN_BITWISE_FN_TABLE(testname, suffix, op, opname)                                         \
    t.test_map_fns_##suffix[std::make_pair(UCHAR, op)] = testname##_##opname##_uchar_##suffix;     \
    t.test_map_fns_##suffix[std::make_pair(USHORT, op)] = testname##_##opname##_ushort_##suffix;   \
    t.test_map_fns_##suffix[std::make_pair(UINT, op)] = testname##_##opname##_uint_##suffix;       \
    t.test_map_fns_##suffix[std::make_pair(ULONG, op)] = testname##_##opname##_ulong_##suffix;     \
    t.test_map_fns_##suffix[std::make_pair(ULONGLONG, op)] =                                       \
        testname##_##opname##_ulonglong_##suffix;                                                  \
    t.test_map_fns_##suffix[std::make_pair(INT8, op)] = testname##_##opname##_int8_##suffix;       \
    t.test_map_fns_##suffix[std::make_pair(INT16, op)] = testname##_##opname##_int16_##suffix;     \
    t.test_map_fns_##suffix[std::make_pair(INT32, op)] = testname##_##opname##_int32_##suffix;     \
    t.test_map_fns_##suffix[std::make_pair(INT64, op)] = testname##_##opname##_int64_##suffix;     \
    t.test_map_fns_##suffix[std::make_pair(UINT8, op)] = testname##_##opname##_uint8_##suffix;     \
    t.test_map_fns_##suffix[std::make_pair(UINT16, op)] = testname##_##opname##_uint16_##suffix;   \
    t.test_map_fns_##suffix[std::make_pair(UINT32, op)] = testname##_##opname##_uint32_##suffix;   \
    t.test_map_fns_##suffix[std::make_pair(UINT64, op)] = testname##_##opname##_uint64_##suffix;   \
    t.test_map_fns_##suffix[std::make_pair(SIZE, op)] = testname##_##opname##_size_##suffix;

#define GEN_BITWISE_FNS(testname, op, opname)                                                      \
    GEN_BITWISE_HOST_FNS(testname, host, op, opname)                                               \
    GEN_BITWISE_ON_QUEUE_FNS(testname, on_queue, op, opname)                                       \
    GEN_BITWISE_SINGLE_FNS(testname, single, op, opname)                                           \
    GEN_BITWISE_SUBGROUP_FNS(testname, subgroup, op, opname)                                       \
    GEN_BITWISE_GRP1_FNS(testname, grp1, op, opname)                                               \
    GEN_BITWISE_GRP2_FNS(testname, grp2, op, opname)                                               \
    GEN_BITWISE_GRP3_FNS(testname, grp3, op, opname)                                               \
    GEN_BITWISE_MULTI_WG_FNS(testname, multi_wg, op, opname)

#define GEN_BITWISE_TABLES(testname, op, opname)                                                   \
    GEN_BITWISE_FN_TABLE(testname, host, op, opname)                                               \
    GEN_BITWISE_FN_TABLE(testname, on_queue, op, opname)                                           \
    GEN_BITWISE_FN_TABLE(testname, single, op, opname)                                             \
    GEN_BITWISE_FN_TABLE(testname, subgroup, op, opname)                                           \
    GEN_BITWISE_FN_TABLE(testname, grp1, op, opname)                                               \
    GEN_BITWISE_FN_TABLE(testname, grp2, op, opname)                                               \
    GEN_BITWISE_FN_TABLE(testname, grp3, op, opname)                                               \
    GEN_BITWISE_FN_TABLE(testname, multi_wg, op, opname)

/* These macros are shared by the REDUCE unit tests and the REDUCE performance tests
 */
#define GEN_COMPARISON_HOST_FNS(testname, suffix, op, opname)                                      \
    GEN_BITWISE_HOST_FNS(testname, suffix, op, opname)                                             \
    TEST_HOST_FN(testname, suffix, FLOAT, float, float, op, opname)                                \
    TEST_HOST_FN(testname, suffix, DOUBLE, double, double, op, opname)                             \
    TEST_HOST_FN(testname, suffix, CHAR, char, char, op, opname)                                   \
    TEST_HOST_FN(testname, suffix, SCHAR, schar, signed char, op, opname)                          \
    TEST_HOST_FN(testname, suffix, SHORT, short, short, op, opname)                                \
    TEST_HOST_FN(testname, suffix, INT, int, int, op, opname)                                      \
    TEST_HOST_FN(testname, suffix, LONG, long, long, op, opname)                                   \
    TEST_HOST_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                      \
    TEST_HOST_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#define GEN_COMPARISON_ON_QUEUE_FNS(testname, suffix, op, opname)                                  \
    GEN_BITWISE_ON_QUEUE_FNS(testname, suffix, op, opname)                                         \
    TEST_ON_QUEUE_FN(testname, suffix, FLOAT, float, float, op, opname)                            \
    TEST_ON_QUEUE_FN(testname, suffix, DOUBLE, double, double, op, opname)                         \
    TEST_ON_QUEUE_FN(testname, suffix, CHAR, char, char, op, opname)                               \
    TEST_ON_QUEUE_FN(testname, suffix, SCHAR, schar, signed char, op, opname)                      \
    TEST_ON_QUEUE_FN(testname, suffix, SHORT, short, short, op, opname)                            \
    TEST_ON_QUEUE_FN(testname, suffix, INT, int, int, op, opname)                                  \
    TEST_ON_QUEUE_FN(testname, suffix, LONG, long, long, op, opname)                               \
    TEST_ON_QUEUE_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                  \
    TEST_ON_QUEUE_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#define GEN_COMPARISON_SINGLE_FNS(testname, suffix, op, opname)                                    \
    GEN_BITWISE_SINGLE_FNS(testname, suffix, op, opname)                                           \
    TEST_SINGLE_FN(testname, suffix, FLOAT, float, float, op, opname)                              \
    TEST_SINGLE_FN(testname, suffix, DOUBLE, double, double, op, opname)                           \
    TEST_SINGLE_FN(testname, suffix, CHAR, char, char, op, opname)                                 \
    TEST_SINGLE_FN(testname, suffix, SCHAR, schar, signed char, op, opname)                        \
    TEST_SINGLE_FN(testname, suffix, SHORT, short, short, op, opname)                              \
    TEST_SINGLE_FN(testname, suffix, INT, int, int, op, opname)                                    \
    TEST_SINGLE_FN(testname, suffix, LONG, long, long, op, opname)                                 \
    TEST_SINGLE_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                    \
    TEST_SINGLE_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#define GEN_COMPARISON_SUBGROUP_FNS(testname, suffix, op, opname)                                  \
    GEN_BITWISE_SUBGROUP_FNS(testname, suffix, op, opname)                                         \
    TEST_SUBGROUP_FN(testname, suffix, FLOAT, float, float, op, opname)                            \
    TEST_SUBGROUP_FN(testname, suffix, DOUBLE, double, double, op, opname)                         \
    TEST_SUBGROUP_FN(testname, suffix, CHAR, char, char, op, opname)                               \
    TEST_SUBGROUP_FN(testname, suffix, SCHAR, schar, signed char, op, opname)                      \
    TEST_SUBGROUP_FN(testname, suffix, SHORT, short, short, op, opname)                            \
    TEST_SUBGROUP_FN(testname, suffix, INT, int, int, op, opname)                                  \
    TEST_SUBGROUP_FN(testname, suffix, LONG, long, long, op, opname)                               \
    TEST_SUBGROUP_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                  \
    TEST_SUBGROUP_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#define GEN_COMPARISON_GRP1_FNS(testname, suffix, op, opname)                                      \
    GEN_BITWISE_GRP1_FNS(testname, suffix, op, opname)                                             \
    TEST_GRP1_FN(testname, suffix, FLOAT, float, float, op, opname)                                \
    TEST_GRP1_FN(testname, suffix, DOUBLE, double, double, op, opname)                             \
    TEST_GRP1_FN(testname, suffix, CHAR, char, char, op, opname)                                   \
    TEST_GRP1_FN(testname, suffix, SCHAR, schar, signed char, op, opname)                          \
    TEST_GRP1_FN(testname, suffix, SHORT, short, short, op, opname)                                \
    TEST_GRP1_FN(testname, suffix, INT, int, int, op, opname)                                      \
    TEST_GRP1_FN(testname, suffix, LONG, long, long, op, opname)                                   \
    TEST_GRP1_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                      \
    TEST_GRP1_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#define GEN_COMPARISON_GRP2_FNS(testname, suffix, op, opname)                                      \
    GEN_BITWISE_GRP2_FNS(testname, suffix, op, opname)                                             \
    TEST_GRP2_FN(testname, suffix, FLOAT, float, float, op, opname)                                \
    TEST_GRP2_FN(testname, suffix, DOUBLE, double, double, op, opname)                             \
    TEST_GRP2_FN(testname, suffix, CHAR, char, char, op, opname)                                   \
    TEST_GRP2_FN(testname, suffix, SCHAR, schar, signed char, op, opname)                          \
    TEST_GRP2_FN(testname, suffix, SHORT, short, short, op, opname)                                \
    TEST_GRP2_FN(testname, suffix, INT, int, int, op, opname)                                      \
    TEST_GRP2_FN(testname, suffix, LONG, long, long, op, opname)                                   \
    TEST_GRP2_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                      \
    TEST_GRP2_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#define GEN_COMPARISON_GRP3_FNS(testname, suffix, op, opname)                                      \
    GEN_BITWISE_GRP3_FNS(testname, suffix, op, opname)                                             \
    TEST_GRP3_FN(testname, suffix, FLOAT, float, float, op, opname)                                \
    TEST_GRP3_FN(testname, suffix, DOUBLE, double, double, op, opname)                             \
    TEST_GRP3_FN(testname, suffix, CHAR, char, char, op, opname)                                   \
    TEST_GRP3_FN(testname, suffix, SCHAR, schar, signed char, op, opname)                          \
    TEST_GRP3_FN(testname, suffix, SHORT, short, short, op, opname)                                \
    TEST_GRP3_FN(testname, suffix, INT, int, int, op, opname)                                      \
    TEST_GRP3_FN(testname, suffix, LONG, long, long, op, opname)                                   \
    TEST_GRP3_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                      \
    TEST_GRP3_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#define GEN_COMPARISON_FN_TABLE(testname, suffix, op, opname)                                      \
    GEN_BITWISE_FN_TABLE(testname, suffix, op, opname)                                             \
    t.test_map_fns_##suffix[std::make_pair(FLOAT, op)] = testname##_##opname##_float_##suffix;     \
    t.test_map_fns_##suffix[std::make_pair(DOUBLE, op)] = testname##_##opname##_double_##suffix;   \
    t.test_map_fns_##suffix[std::make_pair(LONGDOUBLE, op)] =                                      \
        testname##_##opname##_double_##suffix;                                                     \
    t.test_map_fns_##suffix[std::make_pair(CHAR, op)] = testname##_##opname##_char_##suffix;       \
    t.test_map_fns_##suffix[std::make_pair(SCHAR, op)] = testname##_##opname##_schar_##suffix;     \
    t.test_map_fns_##suffix[std::make_pair(SHORT, op)] = testname##_##opname##_short_##suffix;     \
    t.test_map_fns_##suffix[std::make_pair(INT, op)] = testname##_##opname##_int_##suffix;         \
    t.test_map_fns_##suffix[std::make_pair(LONG, op)] = testname##_##opname##_long_##suffix;       \
    t.test_map_fns_##suffix[std::make_pair(LONGLONG, op)] =                                        \
        testname##_##opname##_longlong_##suffix;                                                   \
    t.test_map_fns_##suffix[std::make_pair(PTRDIFF, op)] = testname##_##opname##_ptrdiff_##suffix;

#define GEN_COMPARISON_FNS(testname, op, opname)                                                   \
    GEN_COMPARISON_HOST_FNS(testname, host, op, opname)                                            \
    GEN_COMPARISON_ON_QUEUE_FNS(testname, on_queue, op, opname)                                    \
    GEN_COMPARISON_SINGLE_FNS(testname, single, op, opname)                                        \
    GEN_COMPARISON_SUBGROUP_FNS(testname, subgroup, op, opname)                                    \
    GEN_COMPARISON_GRP1_FNS(testname, grp1, op, opname)                                            \
    GEN_COMPARISON_GRP2_FNS(testname, grp2, op, opname)                                            \
    GEN_COMPARISON_GRP3_FNS(testname, grp3, op, opname)

#define GEN_COMPARISON_TABLES(testname, op, opname)                                                \
    GEN_COMPARISON_FN_TABLE(testname, host, op, opname)                                            \
    GEN_COMPARISON_FN_TABLE(testname, on_queue, op, opname)                                        \
    GEN_COMPARISON_FN_TABLE(testname, single, op, opname)                                          \
    GEN_COMPARISON_FN_TABLE(testname, subgroup, op, opname)                                        \
    GEN_COMPARISON_FN_TABLE(testname, grp1, op, opname)                                            \
    GEN_COMPARISON_FN_TABLE(testname, grp2, op, opname)                                            \
    GEN_COMPARISON_FN_TABLE(testname, grp3, op, opname)

/* Arithmetic */

#define GEN_ARITHMETIC_HOST_FNS(testname, suffix, op, opname)                                      \
    GEN_COMPARISON_HOST_FNS(testname, suffix, op, opname)

#define GEN_ARITHMETIC_ON_QUEUE_FNS(testname, suffix, op, opname)                                  \
    GEN_COMPARISON_ON_QUEUE_FNS(testname, suffix, op, opname)

#define GEN_ARITHMETIC_SINGLE_FNS(testname, suffix, op, opname)                                    \
    GEN_COMPARISON_SINGLE_FNS(testname, suffix, op, opname)

#define GEN_ARITHMETIC_SUBGROUP_FNS(testname, suffix, op, opname)                                  \
    GEN_COMPARISON_SUBGROUP_FNS(testname, suffix, op, opname)

#define GEN_ARITHMETIC_GRP1_FNS(testname, suffix, op, opname)                                      \
    GEN_COMPARISON_GRP1_FNS(testname, suffix, op, opname)

#define GEN_ARITHMETIC_GRP2_FNS(testname, suffix, op, opname)                                      \
    GEN_COMPARISON_GRP2_FNS(testname, suffix, op, opname)

#define GEN_ARITHMETIC_GRP3_FNS(testname, suffix, op, opname)                                      \
    GEN_COMPARISON_GRP3_FNS(testname, suffix, op, opname)

#define GEN_ARITHMETIC_FN_TABLE(testname, suffix, op, opname)                                      \
    GEN_COMPARISON_FN_TABLE(testname, suffix, op, opname)

#define GEN_ARITHMETIC_FNS(testname, op, opname)                                                   \
    GEN_ARITHMETIC_HOST_FNS(testname, host, op, opname)                                            \
    GEN_ARITHMETIC_ON_QUEUE_FNS(testname, on_queue, op, opname)                                    \
    GEN_ARITHMETIC_SINGLE_FNS(testname, single, op, opname)                                        \
    GEN_ARITHMETIC_SUBGROUP_FNS(testname, subgroup, op, opname)                                    \
    GEN_ARITHMETIC_GRP1_FNS(testname, grp1, op, opname)                                            \
    GEN_ARITHMETIC_GRP2_FNS(testname, grp2, op, opname)                                            \
    GEN_ARITHMETIC_GRP3_FNS(testname, grp3, op, opname)

#define GEN_ARITHMETIC_TABLES(testname, op, opname)                                                \
    GEN_ARITHMETIC_FN_TABLE(testname, host, op, opname)                                            \
    GEN_ARITHMETIC_FN_TABLE(testname, on_queue, op, opname)                                        \
    GEN_ARITHMETIC_FN_TABLE(testname, single, op, opname)                                          \
    GEN_ARITHMETIC_FN_TABLE(testname, subgroup, op, opname)                                        \
    GEN_ARITHMETIC_FN_TABLE(testname, grp1, op, opname)                                            \
    GEN_ARITHMETIC_FN_TABLE(testname, grp2, op, opname)                                            \
    GEN_ARITHMETIC_FN_TABLE(testname, grp3, op, opname)

#endif /* REDUCE_TEST_H */
