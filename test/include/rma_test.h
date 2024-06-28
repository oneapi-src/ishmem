/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef RMA_TEST2_H
#define RMA_TEST2_H

#define TEST_HOST_FN(testname, suffix, typeenum, typename, type, op, opname)                       \
    void testname##_##opname##_##                                                                  \
        typename##_##suffix(sycl::queue q, int *res, void *dest, void *src, size_t nelems)         \
    {                                                                                              \
        ishmem_sync_all();                                                                         \
        TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
        ishmem_sync_all();                                                                         \
    }

#define TEST_SINGLE_FN(testname, suffix, typeenum, typename, type, op, opname)                     \
    void testname##_##opname##_##                                                                  \
        typename##_##suffix(sycl::queue q, int *res, void *dest, void *src, size_t nelems)         \
    {                                                                                              \
        q.single_task([=]() {                                                                      \
             ishmem_sync_all();                                                                    \
             TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                    \
             ishmem_sync_all();                                                                    \
         }).wait_and_throw();                                                                      \
    }

#define TEST_SUBGROUP_FN(testname, suffix, typeenum, typename, type, op, opname)                   \
    void testname##_##opname##_##                                                                  \
        typename##_##suffix(sycl::queue q, int *res, void *dest, void *src, size_t nelems)         \
    {                                                                                              \
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(x_size), sycl::range<1>(x_size)),          \
                       [=](sycl::nd_item<1> it) {                                                  \
                           auto grp = it.get_sub_group();                                          \
                           ishmemx_sync_all_work_group(grp);                                       \
                           TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)  \
                           ishmemx_sync_all_work_group(grp);                                       \
                       })                                                                          \
            .wait_and_throw();                                                                     \
    }

#define TEST_GRP1_FN(testname, suffix, typeenum, typename, type, op, opname)                       \
    void testname##_##opname##_##                                                                  \
        typename##_##suffix(sycl::queue q, int *res, void *dest, void *src, size_t nelems)         \
    {                                                                                              \
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(x_size), sycl::range<1>(x_size)),          \
                       [=](sycl::nd_item<1> it) {                                                  \
                           auto grp = it.get_sub_group();                                          \
                           ishmemx_sync_all_work_group(grp);                                       \
                           TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)  \
                           ishmemx_sync_all_work_group(grp);                                       \
                       })                                                                          \
            .wait_and_throw();                                                                     \
    }

#define TEST_MULTI_WG_FN(testname, suffix, typeenum, typename, type, op, opname)                   \
    void testname##_##opname##_##                                                                  \
        typename##_##suffix(sycl::queue q, int *res, void *dest, void *src, size_t nelems)         \
    {                                                                                              \
        ishmem_sync_all();                                                                         \
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(x_size * num_wg), sycl::range<1>(x_size)), \
                       [=](sycl::nd_item<1> it) {                                                  \
                           auto grp = it.get_sub_group();                                          \
                           TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)  \
                       })                                                                          \
            .wait_and_throw();                                                                     \
        ishmem_sync_all();                                                                         \
    }

#define TEST_GRP2_FN(testname, suffix, typeenum, typename, type, op, opname)                       \
    void testname##_##opname##_##                                                                  \
        typename##_##suffix(sycl::queue q, int *res, void *dest, void *src, size_t nelems)         \
    {                                                                                              \
        q.parallel_for(                                                                            \
             sycl::nd_range<2>(sycl::range<2>(x_size, y_size), sycl::range<2>(x_size, y_size)),    \
             [=](sycl::nd_item<2> it) {                                                            \
                 auto grp = it.get_group();                                                        \
                 ishmemx_sync_all_work_group(grp);                                                 \
                 TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)            \
                 ishmemx_sync_all_work_group(grp);                                                 \
             })                                                                                    \
            .wait_and_throw();                                                                     \
    }

#define TEST_GRP3_FN(testname, suffix, typeenum, typename, type, op, opname)                       \
    void testname##_##opname##_##                                                                  \
        typename##_##suffix(sycl::queue q, int *res, void *dest, void *src, size_t nelems)         \
    {                                                                                              \
        q.parallel_for(sycl::nd_range<3>(sycl::range<3>(x_size, y_size, z_size),                   \
                                         sycl::range<3>(x_size, y_size, z_size)),                  \
                       [=](sycl::nd_item<3> it) {                                                  \
                           auto grp = it.get_group();                                              \
                           ishmemx_sync_all_work_group(grp);                                       \
                           TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname); \
                           ishmemx_sync_all_work_group(grp);                                       \
                       })                                                                          \
            .wait_and_throw();                                                                     \
    }

#define GEN_HOST_FNS(testname, suffix, op, opname)                                                 \
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
    TEST_HOST_FN(testname, suffix, SIZE, size, size_t, op, opname)                                 \
    TEST_HOST_FN(testname, suffix, FLOAT, float, float, op, opname)                                \
    TEST_HOST_FN(testname, suffix, DOUBLE, double, double, op, opname)                             \
    TEST_HOST_FN(testname, suffix, CHAR, char, char, op, opname)                                   \
    TEST_HOST_FN(testname, suffix, SCHAR, schar, signed char, op, opname)                          \
    TEST_HOST_FN(testname, suffix, SHORT, short, short, op, opname)                                \
    TEST_HOST_FN(testname, suffix, INT, int, int, op, opname)                                      \
    TEST_HOST_FN(testname, suffix, LONG, long, long, op, opname)                                   \
    TEST_HOST_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                      \
    TEST_HOST_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#define GEN_SINGLE_FNS(testname, suffix, op, opname)                                               \
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
    TEST_SINGLE_FN(testname, suffix, SIZE, size, size_t, op, opname)                               \
    TEST_SINGLE_FN(testname, suffix, FLOAT, float, float, op, opname)                              \
    TEST_SINGLE_FN(testname, suffix, DOUBLE, double, double, op, opname)                           \
    TEST_SINGLE_FN(testname, suffix, CHAR, char, char, op, opname)                                 \
    TEST_SINGLE_FN(testname, suffix, SCHAR, schar, signed char, op, opname)                        \
    TEST_SINGLE_FN(testname, suffix, SHORT, short, short, op, opname)                              \
    TEST_SINGLE_FN(testname, suffix, INT, int, int, op, opname)                                    \
    TEST_SINGLE_FN(testname, suffix, LONG, long, long, op, opname)                                 \
    TEST_SINGLE_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                    \
    TEST_SINGLE_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#define GEN_SUBGROUP_FNS(testname, suffix, op, opname)                                             \
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
    TEST_SUBGROUP_FN(testname, suffix, SIZE, size, size_t, op, opname)                             \
    TEST_SUBGROUP_FN(testname, suffix, FLOAT, float, float, op, opname)                            \
    TEST_SUBGROUP_FN(testname, suffix, DOUBLE, double, double, op, opname)                         \
    TEST_SUBGROUP_FN(testname, suffix, CHAR, char, char, op, opname)                               \
    TEST_SUBGROUP_FN(testname, suffix, SCHAR, schar, signed char, op, opname)                      \
    TEST_SUBGROUP_FN(testname, suffix, SHORT, short, short, op, opname)                            \
    TEST_SUBGROUP_FN(testname, suffix, INT, int, int, op, opname)                                  \
    TEST_SUBGROUP_FN(testname, suffix, LONG, long, long, op, opname)                               \
    TEST_SUBGROUP_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                  \
    TEST_SUBGROUP_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#define GEN_GRP1_FNS(testname, suffix, op, opname)                                                 \
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
    TEST_GRP1_FN(testname, suffix, SIZE, size, size_t, op, opname)                                 \
    TEST_GRP1_FN(testname, suffix, FLOAT, float, float, op, opname)                                \
    TEST_GRP1_FN(testname, suffix, DOUBLE, double, double, op, opname)                             \
    TEST_GRP1_FN(testname, suffix, CHAR, char, char, op, opname)                                   \
    TEST_GRP1_FN(testname, suffix, SCHAR, schar, signed char, op, opname)                          \
    TEST_GRP1_FN(testname, suffix, SHORT, short, short, op, opname)                                \
    TEST_GRP1_FN(testname, suffix, INT, int, int, op, opname)                                      \
    TEST_GRP1_FN(testname, suffix, LONG, long, long, op, opname)                                   \
    TEST_GRP1_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                      \
    TEST_GRP1_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#define GEN_MULTI_WG_FNS(testname, suffix, op, opname)                                             \
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
    TEST_MULTI_WG_FN(testname, suffix, SIZE, size, size_t, op, opname)                             \
    TEST_MULTI_WG_FN(testname, suffix, FLOAT, float, float, op, opname)                            \
    TEST_MULTI_WG_FN(testname, suffix, DOUBLE, double, double, op, opname)                         \
    TEST_MULTI_WG_FN(testname, suffix, CHAR, char, char, op, opname)                               \
    TEST_MULTI_WG_FN(testname, suffix, SCHAR, schar, signed char, op, opname)                      \
    TEST_MULTI_WG_FN(testname, suffix, SHORT, short, short, op, opname)                            \
    TEST_MULTI_WG_FN(testname, suffix, INT, int, int, op, opname)                                  \
    TEST_MULTI_WG_FN(testname, suffix, LONG, long, long, op, opname)                               \
    TEST_MULTI_WG_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                  \
    TEST_MULTI_WG_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#define GEN_GRP2_FNS(testname, suffix, op, opname)                                                 \
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
    TEST_GRP2_FN(testname, suffix, SIZE, size, size_t, op, opname)                                 \
    TEST_GRP2_FN(testname, suffix, FLOAT, float, float, op, opname)                                \
    TEST_GRP2_FN(testname, suffix, DOUBLE, double, double, op, opname)                             \
    TEST_GRP2_FN(testname, suffix, CHAR, char, char, op, opname)                                   \
    TEST_GRP2_FN(testname, suffix, SCHAR, schar, signed char, op, opname)                          \
    TEST_GRP2_FN(testname, suffix, SHORT, short, short, op, opname)                                \
    TEST_GRP2_FN(testname, suffix, INT, int, int, op, opname)                                      \
    TEST_GRP2_FN(testname, suffix, LONG, long, long, op, opname)                                   \
    TEST_GRP2_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                      \
    TEST_GRP2_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#define GEN_GRP3_FNS(testname, suffix, op, opname)                                                 \
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
    TEST_GRP3_FN(testname, suffix, SIZE, size, size_t, op, opname)                                 \
    TEST_GRP3_FN(testname, suffix, FLOAT, float, float, op, opname)                                \
    TEST_GRP3_FN(testname, suffix, DOUBLE, double, double, op, opname)                             \
    TEST_GRP3_FN(testname, suffix, CHAR, char, char, op, opname)                                   \
    TEST_GRP3_FN(testname, suffix, SCHAR, schar, signed char, op, opname)                          \
    TEST_GRP3_FN(testname, suffix, SHORT, short, short, op, opname)                                \
    TEST_GRP3_FN(testname, suffix, INT, int, int, op, opname)                                      \
    TEST_GRP3_FN(testname, suffix, LONG, long, long, op, opname)                                   \
    TEST_GRP3_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                      \
    TEST_GRP3_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#define GEN_FN_TABLE(testname, suffix, op, opname)                                                 \
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
    t.test_map_fns_##suffix[std::make_pair(SIZE, op)] = testname##_##opname##_size_##suffix;       \
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

#define GEN_FNS(testname, op, opname)                                                              \
    GEN_HOST_FNS(testname, host, op, opname)                                                       \
    GEN_SINGLE_FNS(testname, single, op, opname)                                                   \
    GEN_SUBGROUP_FNS(testname, subgroup, op, opname)                                               \
    GEN_GRP1_FNS(testname, grp1, op, opname)                                                       \
    GEN_GRP2_FNS(testname, grp2, op, opname)                                                       \
    GEN_GRP3_FNS(testname, grp3, op, opname)

#define GEN_FNS_ALL(testname, op, opname)                                                          \
    GEN_FNS(testname, op, opname)                                                                  \
    GEN_MULTI_WG_FNS(testname, multi_wg, op, opname)

#define GEN_TABLES(testname, op, opname)                                                           \
    GEN_FN_TABLE(testname, host, op, opname)                                                       \
    GEN_FN_TABLE(testname, single, op, opname)                                                     \
    GEN_FN_TABLE(testname, subgroup, op, opname)                                                   \
    GEN_FN_TABLE(testname, grp1, op, opname)                                                       \
    GEN_FN_TABLE(testname, grp2, op, opname)                                                       \
    GEN_FN_TABLE(testname, grp3, op, opname)

#define GEN_TABLES_ALL(testname, op, opname)                                                       \
    GEN_TABLES(testname, op, opname)                                                               \
    GEN_FN_TABLE(testname, multi_wg, op, opname)

#define GEN_MEM_FNS(testname, op, opname)                                                          \
    TEST_HOST_FN(testname, host, MEM, uint8, uint8_t, NOP, nop)                                    \
    TEST_HOST_FN(testname, single, MEM, uint8, uint8_t, NOP, nop)                                  \
    TEST_SUBGROUP_FN(testname, subgroup, MEM, uint8, uint8_t, NOP, nop)                            \
    TEST_GRP1_FN(testname, grp1, MEM, uint8, uint8_t, NOP, nop)                                    \
    TEST_GRP2_FN(testname, grp2, MEM, uint8, uint8_t, NOP, nop)                                    \
    TEST_GRP3_FN(testname, grp3, MEM, uint8, uint8_t, NOP, nop)

#define GEN_MEM_FNS_ALL(testname, op, opname)                                                      \
    GEN_MEM_FNS(testname, op, opname)                                                              \
    TEST_MULTI_WG_FN(testname, multi_wg, MEM, uint8, uint8_t, NOP, nop)

#define GEN_MEM_TABLES(testname, op, opname)                                                       \
    t.test_map_fns_host[std::make_pair(MEM, op)] = testname##_##opname##_uint8_host;               \
    t.test_map_fns_single[std::make_pair(MEM, op)] = testname##_##opname##_uint8_single;           \
    t.test_map_fns_subgroup[std::make_pair(MEM, op)] = testname##_##opname##_uint8_subgroup;       \
    t.test_map_fns_grp1[std::make_pair(MEM, op)] = testname##_##opname##_uint8_grp1;               \
    t.test_map_fns_grp2[std::make_pair(MEM, op)] = testname##_##opname##_uint8_grp2;               \
    t.test_map_fns_grp3[std::make_pair(MEM, op)] = testname##_##opname##_uint8_grp3;

#define GEN_MEM_TABLES_ALL(testname, op, opname)                                                   \
    GEN_MEM_TABLES(testname, op, opname)                                                           \
    t.test_map_fns_multi_wg[std::make_pair(MEM, op)] = testname##_##opname##_uint8_multi_wg;

#define GEN_SIZE_FNS_FOR_SIZE(testname, op, size, bsize)                                           \
    TEST_HOST_FN(testname, host, SIZE##size, void, uint##bsize##_t, NOP, nop)                      \
    TEST_HOST_FN(testname, single, SIZE##size, void, uint##bsize##_t, NOP, nop)                    \
    TEST_SUBGROUP_FN(testname, subgroup, SIZE##size, void, uint##bsize##_t, NOP, nop)              \
    TEST_GRP1_FN(testname, grp1, SIZE##size, void, uint##bsize##_t, NOP, nop)                      \
    TEST_GRP2_FN(testname, grp2, SIZE##size, void, uint##bsize##_t, NOP, nop)                      \
    TEST_GRP3_FN(testname, grp3, SIZE##size, void, uint##bsize##_t, NOP, nop)

#define GEN_SIZE_FNS_FOR_SIZE_ALL(testname, op, size, bsize)                                       \
    GEN_SIZE_FNS_FOR_SIZE(testname, op, size, bsize)                                               \
    TEST_MULTI_WG_FN(testname, multi_wg, SIZE##size, void, uint##bsize___T, NOP, nop)

#define GEN_SIZE_FNS(prefix, suffix, op, opname)                                                   \
    GEN_SIZE_FNS_FOR_SIZE(prefix##8##suffix, op, 8, 8)                                             \
    GEN_SIZE_FNS_FOR_SIZE(prefix##16##suffix, op, 16, 16)                                          \
    GEN_SIZE_FNS_FOR_SIZE(prefix##32##suffix, op, 32, 32)                                          \
    GEN_SIZE_FNS_FOR_SIZE(prefix##64##suffix, op, 64, 64)                                          \
    GEN_SIZE_FNS_FOR_SIZE(prefix##128##suffix, op, 128, 64)

#define GEN_FN_TABLE_SIZE(testname, op, opname, size)                                              \
    t.test_map_fns_host[std::make_pair(SIZE##size, op)] = testname##_##opname##_void_host;         \
    t.test_map_fns_single[std::make_pair(SIZE##size, op)] = testname##_##opname##_void_single;     \
    t.test_map_fns_subgroup[std::make_pair(SIZE##size, op)] = testname##_##opname##_void_subgroup; \
    t.test_map_fns_grp1[std::make_pair(SIZE##size, op)] = testname##_##opname##_void_grp1;         \
    t.test_map_fns_grp2[std::make_pair(SIZE##size, op)] = testname##_##opname##_void_grp2;         \
    t.test_map_fns_grp3[std::make_pair(SIZE##size, op)] = testname##_##opname##_void_grp3;

#define GEN_FN_TABLE_SIZE_ALL(testname, op, opname, size)                                          \
    GEN_FN_TABLE_SIZE(testname, op, opname, size)                                                  \
    t.test_map_fns_multi_wg[std::make_pair(SIZE##size, op)] = testname##_##opname##_void_multi_wg;

#define GEN_SIZE_TABLES(prefix, suffix, op, opname)                                                \
    GEN_FN_TABLE_SIZE(prefix##8##suffix, op, opname, 8)                                            \
    GEN_FN_TABLE_SIZE(prefix##16##suffix, op, opname, 16)                                          \
    GEN_FN_TABLE_SIZE(prefix##32##suffix, op, opname, 32)                                          \
    GEN_FN_TABLE_SIZE(prefix##64##suffix, op, opname, 64)                                          \
    GEN_FN_TABLE_SIZE(prefix##128##suffix, op, opname, 128)

#endif /* RMA_TEST_H */
