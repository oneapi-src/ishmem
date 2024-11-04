/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "rma_test.h"

#define NUM_COMPARISON_OPERATORS 6

#ifdef GEN_HOST_FNS
#undef GEN_HOST_FNS
#endif
#define GEN_HOST_FNS(testname, suffix, op, opname)                                                 \
    TEST_HOST_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                           \
    TEST_HOST_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                        \
    TEST_HOST_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)           \
    TEST_HOST_FN(testname, suffix, INT32, int32, int32_t, op, opname)                              \
    TEST_HOST_FN(testname, suffix, INT64, int64, int64_t, op, opname)                              \
    TEST_HOST_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                           \
    TEST_HOST_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                           \
    TEST_HOST_FN(testname, suffix, SIZE, size, size_t, op, opname)                                 \
    TEST_HOST_FN(testname, suffix, INT, int, int, op, opname)                                      \
    TEST_HOST_FN(testname, suffix, LONG, long, long, op, opname)                                   \
    TEST_HOST_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                      \
    TEST_HOST_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#ifdef GEN_ON_QUEUE_FNS
#undef GEN_ON_QUEUE_FNS
#endif
#define GEN_ON_QUEUE_FNS(testname, suffix, op, opname)                                             \
    TEST_ON_QUEUE_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                       \
    TEST_ON_QUEUE_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                    \
    TEST_ON_QUEUE_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)       \
    TEST_ON_QUEUE_FN(testname, suffix, INT32, int32, int32_t, op, opname)                          \
    TEST_ON_QUEUE_FN(testname, suffix, INT64, int64, int64_t, op, opname)                          \
    TEST_ON_QUEUE_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                       \
    TEST_ON_QUEUE_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                       \
    TEST_ON_QUEUE_FN(testname, suffix, SIZE, size, size_t, op, opname)                             \
    TEST_ON_QUEUE_FN(testname, suffix, INT, int, int, op, opname)                                  \
    TEST_ON_QUEUE_FN(testname, suffix, LONG, long, long, op, opname)                               \
    TEST_ON_QUEUE_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                  \
    TEST_ON_QUEUE_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#ifdef GEN_SINGLE_FNS
#undef GEN_SINGLE_FNS
#endif
#define GEN_SINGLE_FNS(testname, suffix, op, opname)                                               \
    TEST_SINGLE_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                         \
    TEST_SINGLE_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                      \
    TEST_SINGLE_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)         \
    TEST_SINGLE_FN(testname, suffix, INT32, int32, int32_t, op, opname)                            \
    TEST_SINGLE_FN(testname, suffix, INT64, int64, int64_t, op, opname)                            \
    TEST_SINGLE_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                         \
    TEST_SINGLE_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                         \
    TEST_SINGLE_FN(testname, suffix, SIZE, size, size_t, op, opname)                               \
    TEST_SINGLE_FN(testname, suffix, INT, int, int, op, opname)                                    \
    TEST_SINGLE_FN(testname, suffix, LONG, long, long, op, opname)                                 \
    TEST_SINGLE_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                    \
    TEST_SINGLE_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#ifdef GEN_SUBGROUP_FNS
#undef GEN_SUBGROUP_FNS
#endif
#define GEN_SUBGROUP_FNS(testname, suffix, op, opname)                                             \
    TEST_SUBGROUP_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                       \
    TEST_SUBGROUP_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                    \
    TEST_SUBGROUP_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)       \
    TEST_SUBGROUP_FN(testname, suffix, INT32, int32, int32_t, op, opname)                          \
    TEST_SUBGROUP_FN(testname, suffix, INT64, int64, int64_t, op, opname)                          \
    TEST_SUBGROUP_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                       \
    TEST_SUBGROUP_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                       \
    TEST_SUBGROUP_FN(testname, suffix, SIZE, size, size_t, op, opname)                             \
    TEST_SUBGROUP_FN(testname, suffix, INT, int, int, op, opname)                                  \
    TEST_SUBGROUP_FN(testname, suffix, LONG, long, long, op, opname)                               \
    TEST_SUBGROUP_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                  \
    TEST_SUBGROUP_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#ifdef GEN_GRP1_FNS
#undef GEN_GRP1_FNS
#endif
#define GEN_GRP1_FNS(testname, suffix, op, opname)                                                 \
    TEST_GRP1_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                           \
    TEST_GRP1_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                        \
    TEST_GRP1_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)           \
    TEST_GRP1_FN(testname, suffix, INT32, int32, int32_t, op, opname)                              \
    TEST_GRP1_FN(testname, suffix, INT64, int64, int64_t, op, opname)                              \
    TEST_GRP1_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                           \
    TEST_GRP1_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                           \
    TEST_GRP1_FN(testname, suffix, SIZE, size, size_t, op, opname)                                 \
    TEST_GRP1_FN(testname, suffix, INT, int, int, op, opname)                                      \
    TEST_GRP1_FN(testname, suffix, LONG, long, long, op, opname)                                   \
    TEST_GRP1_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                      \
    TEST_GRP1_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#ifdef GEN_GRP2_FNS
#undef GEN_GRP2_FNS
#endif
#define GEN_GRP2_FNS(testname, suffix, op, opname)                                                 \
    TEST_GRP2_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                           \
    TEST_GRP2_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                        \
    TEST_GRP2_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)           \
    TEST_GRP2_FN(testname, suffix, INT32, int32, int32_t, op, opname)                              \
    TEST_GRP2_FN(testname, suffix, INT64, int64, int64_t, op, opname)                              \
    TEST_GRP2_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                           \
    TEST_GRP2_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                           \
    TEST_GRP2_FN(testname, suffix, SIZE, size, size_t, op, opname)                                 \
    TEST_GRP2_FN(testname, suffix, INT, int, int, op, opname)                                      \
    TEST_GRP2_FN(testname, suffix, LONG, long, long, op, opname)                                   \
    TEST_GRP2_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                      \
    TEST_GRP2_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#ifdef GEN_GRP3_FNS
#undef GEN_GRP3_FNS
#endif
#define GEN_GRP3_FNS(testname, suffix, op, opname)                                                 \
    TEST_GRP3_FN(testname, suffix, UINT, uint, unsigned int, op, opname)                           \
    TEST_GRP3_FN(testname, suffix, ULONG, ulong, unsigned long, op, opname)                        \
    TEST_GRP3_FN(testname, suffix, ULONGLONG, ulonglong, unsigned long long, op, opname)           \
    TEST_GRP3_FN(testname, suffix, INT32, int32, int32_t, op, opname)                              \
    TEST_GRP3_FN(testname, suffix, INT64, int64, int64_t, op, opname)                              \
    TEST_GRP3_FN(testname, suffix, UINT32, uint32, uint32_t, op, opname)                           \
    TEST_GRP3_FN(testname, suffix, UINT64, uint64, uint64_t, op, opname)                           \
    TEST_GRP3_FN(testname, suffix, SIZE, size, size_t, op, opname)                                 \
    TEST_GRP3_FN(testname, suffix, INT, int, int, op, opname)                                      \
    TEST_GRP3_FN(testname, suffix, LONG, long, long, op, opname)                                   \
    TEST_GRP3_FN(testname, suffix, LONGLONG, longlong, long long, op, opname)                      \
    TEST_GRP3_FN(testname, suffix, PTRDIFF, ptrdiff, ptrdiff_t, op, opname)

#ifdef GEN_FN_TABLE
#undef GEN_FN_TABLE
#endif
#define GEN_FN_TABLE(testname, suffix, op, opname)                                                 \
    t.test_map_fns_##suffix[std::make_pair(UINT, op)] = testname##_##opname##_uint_##suffix;       \
    t.test_map_fns_##suffix[std::make_pair(ULONG, op)] = testname##_##opname##_ulong_##suffix;     \
    t.test_map_fns_##suffix[std::make_pair(ULONGLONG, op)] =                                       \
        testname##_##opname##_ulonglong_##suffix;                                                  \
    t.test_map_fns_##suffix[std::make_pair(INT32, op)] = testname##_##opname##_int32_##suffix;     \
    t.test_map_fns_##suffix[std::make_pair(INT64, op)] = testname##_##opname##_int64_##suffix;     \
    t.test_map_fns_##suffix[std::make_pair(UINT32, op)] = testname##_##opname##_uint32_##suffix;   \
    t.test_map_fns_##suffix[std::make_pair(UINT64, op)] = testname##_##opname##_uint64_##suffix;   \
    t.test_map_fns_##suffix[std::make_pair(SIZE, op)] = testname##_##opname##_size_##suffix;       \
    t.test_map_fns_##suffix[std::make_pair(INT, op)] = testname##_##opname##_int_##suffix;         \
    t.test_map_fns_##suffix[std::make_pair(LONG, op)] = testname##_##opname##_long_##suffix;       \
    t.test_map_fns_##suffix[std::make_pair(LONGLONG, op)] =                                        \
        testname##_##opname##_longlong_##suffix;                                                   \
    t.test_map_fns_##suffix[std::make_pair(PTRDIFF, op)] = testname##_##opname##_ptrdiff_##suffix;
