/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef AMO_TEST_H
#define AMO_TEST_H

/* These macros are shared by the AMO unit tests and the AMO performance tests
 */

#define ISHMEM_GEN_AMO_STANDARD_FUNCTION(function, returnvar, operator)                            \
    function                                                                                       \
    {                                                                                              \
        returnvar switch (t)                                                                       \
        {                                                                                          \
            case INT:                                                                              \
                ISHMEM_AMO_BRANCH(INT, int, int, operator)                                         \
            case LONG:                                                                             \
                ISHMEM_AMO_BRANCH(LONG, long, long, operator)                                      \
            case LONGLONG:                                                                         \
                ISHMEM_AMO_BRANCH(LONGLONG, longlong, long long, operator)                         \
            case UINT:                                                                             \
                ISHMEM_AMO_BRANCH(UINT, uint, unsigned int, operator)                              \
            case ULONG:                                                                            \
                ISHMEM_AMO_BRANCH(ULONG, ulong, unsigned long, operator)                           \
            case ULONGLONG:                                                                        \
                ISHMEM_AMO_BRANCH(ULONGLONG, ulonglong, unsigned long long, operator)              \
            case INT32:                                                                            \
                ISHMEM_AMO_BRANCH(INT32, int32, int32_t, operator)                                 \
            case INT64:                                                                            \
                ISHMEM_AMO_BRANCH(INT64, int64, int64_t, operator)                                 \
            case UINT32:                                                                           \
                ISHMEM_AMO_BRANCH(UINT32, uint32, uint32_t, operator)                              \
            case UINT64:                                                                           \
                ISHMEM_AMO_BRANCH(UINT64, uint64, uint64_t, operator)                              \
            case SIZE:                                                                             \
                ISHMEM_AMO_BRANCH(SIZE, size, size_t, operator)                                    \
            case PTRDIFF:                                                                          \
                ISHMEM_AMO_BRANCH(PTRDIFF, ptrdiff, ptrdiff_t, operator)                           \
            default:                                                                               \
                break;                                                                             \
        }                                                                                          \
        return 0;                                                                                  \
    }

#define ISHMEM_GEN_AMO_EXTENDED_FUNCTION(function, returnvar, operator)                            \
    function                                                                                       \
    {                                                                                              \
        returnvar switch (t)                                                                       \
        {                                                                                          \
            case FLOAT:                                                                            \
                ISHMEM_AMO_BRANCH(FLOAT, float, float, operator)                                   \
            case DOUBLE:                                                                           \
                ISHMEM_AMO_BRANCH(DOUBLE, double, double, operator)                                \
            case INT:                                                                              \
                ISHMEM_AMO_BRANCH(INT, int, int, operator)                                         \
            case LONG:                                                                             \
                ISHMEM_AMO_BRANCH(LONG, long, long, operator)                                      \
            case LONGLONG:                                                                         \
                ISHMEM_AMO_BRANCH(LONGLONG, longlong, long long, operator)                         \
            case UINT:                                                                             \
                ISHMEM_AMO_BRANCH(UINT, uint, unsigned int, operator)                              \
            case ULONG:                                                                            \
                ISHMEM_AMO_BRANCH(ULONG, ulong, unsigned long, operator)                           \
            case ULONGLONG:                                                                        \
                ISHMEM_AMO_BRANCH(ULONGLONG, ulonglong, unsigned long long, operator)              \
            case INT32:                                                                            \
                ISHMEM_AMO_BRANCH(INT32, int32, int32_t, operator)                                 \
            case INT64:                                                                            \
                ISHMEM_AMO_BRANCH(INT64, int64, int64_t, operator)                                 \
            case UINT32:                                                                           \
                ISHMEM_AMO_BRANCH(UINT32, uint32, uint32_t, operator)                              \
            case UINT64:                                                                           \
                ISHMEM_AMO_BRANCH(UINT64, uint64, uint64_t, operator)                              \
            case SIZE:                                                                             \
                ISHMEM_AMO_BRANCH(SIZE, size, size_t, operator)                                    \
            case PTRDIFF:                                                                          \
                ISHMEM_AMO_BRANCH(PTRDIFF, ptrdiff, ptrdiff_t, operator)                           \
            default:                                                                               \
                break;                                                                             \
        }                                                                                          \
        return 0;                                                                                  \
    }
#define ISHMEM_GEN_AMO_BITWISE_FUNCTION(function, returnvar, operator)                             \
    function                                                                                       \
    {                                                                                              \
        returnvar switch (t)                                                                       \
        {                                                                                          \
            case UINT:                                                                             \
                ISHMEM_AMO_BRANCH(UINT, uint, unsigned int, operator)                              \
            case ULONG:                                                                            \
                ISHMEM_AMO_BRANCH(ULONG, ulong, unsigned long, operator)                           \
            case ULONGLONG:                                                                        \
                ISHMEM_AMO_BRANCH(ULONGLONG, ulonglong, unsigned long long, operator)              \
            case INT32:                                                                            \
                ISHMEM_AMO_BRANCH(INT32, int32, int32_t, operator)                                 \
            case INT64:                                                                            \
                ISHMEM_AMO_BRANCH(INT64, int64, int64_t, operator)                                 \
            case UINT32:                                                                           \
                ISHMEM_AMO_BRANCH(UINT32, uint32, uint32_t, operator)                              \
            case UINT64:                                                                           \
                ISHMEM_AMO_BRANCH(UINT64, uint64, uint64_t, operator)                              \
            default:                                                                               \
                break;                                                                             \
        }                                                                                          \
        return 0;                                                                                  \
    }

#endif /* AMO_TEST_H */
