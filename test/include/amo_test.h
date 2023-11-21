/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef AMO_TEST_H
#define AMO_TEST_H

/* These macros are shared by the AMO unit tests and the AMO performance tests
 */

#define ISHMEM_GEN_AMO_STANDARD_FUNCTION(function, returnvar)                                      \
    function                                                                                       \
    {                                                                                              \
        returnvar switch (t)                                                                       \
        {                                                                                          \
            case INT:                                                                              \
                ISHMEM_AMO_BRANCH(INT, int, int)                                                   \
            case LONG:                                                                             \
                ISHMEM_AMO_BRANCH(LONG, long, long)                                                \
            case LONGLONG:                                                                         \
                ISHMEM_AMO_BRANCH(LONGLONG, longlong, long long)                                   \
            case UINT:                                                                             \
                ISHMEM_AMO_BRANCH(UINT, uint, unsigned int)                                        \
            case ULONG:                                                                            \
                ISHMEM_AMO_BRANCH(ULONG, ulong, unsigned long)                                     \
            case ULONGLONG:                                                                        \
                ISHMEM_AMO_BRANCH(ULONGLONG, ulonglong, unsigned long long)                        \
            case INT32:                                                                            \
                ISHMEM_AMO_BRANCH(INT32, int32, int32_t)                                           \
            case INT64:                                                                            \
                ISHMEM_AMO_BRANCH(INT64, int64, int64_t)                                           \
            case UINT32:                                                                           \
                ISHMEM_AMO_BRANCH(UINT32, uint32, uint32_t)                                        \
            case UINT64:                                                                           \
                ISHMEM_AMO_BRANCH(UINT64, uint64, uint64_t)                                        \
            case SIZE:                                                                             \
                ISHMEM_AMO_BRANCH(SIZE, size, size_t)                                              \
            case PTRDIFF:                                                                          \
                ISHMEM_AMO_BRANCH(PTRDIFF, ptrdiff, ptrdiff_t);                                    \
            default:                                                                               \
                break;                                                                             \
        }                                                                                          \
    }
#define ISHMEM_GEN_AMO_EXTENDED_FUNCTION(function, returnvar)                                      \
    function                                                                                       \
    {                                                                                              \
        returnvar switch (t)                                                                       \
        {                                                                                          \
            case FLOAT:                                                                            \
                ISHMEM_AMO_BRANCH(FLOAT, float, float)                                             \
            case DOUBLE:                                                                           \
                ISHMEM_AMO_BRANCH(DOUBLE, double, double)                                          \
            case INT:                                                                              \
                ISHMEM_AMO_BRANCH(INT, int, int)                                                   \
            case LONG:                                                                             \
                ISHMEM_AMO_BRANCH(LONG, long, long)                                                \
            case LONGLONG:                                                                         \
                ISHMEM_AMO_BRANCH(LONGLONG, longlong, long long)                                   \
            case UINT:                                                                             \
                ISHMEM_AMO_BRANCH(UINT, uint, unsigned int)                                        \
            case ULONG:                                                                            \
                ISHMEM_AMO_BRANCH(ULONG, ulong, unsigned long)                                     \
            case ULONGLONG:                                                                        \
                ISHMEM_AMO_BRANCH(ULONGLONG, ulonglong, unsigned long long)                        \
            case INT32:                                                                            \
                ISHMEM_AMO_BRANCH(INT32, int32, int32_t)                                           \
            case INT64:                                                                            \
                ISHMEM_AMO_BRANCH(INT64, int64, int64_t)                                           \
            case UINT32:                                                                           \
                ISHMEM_AMO_BRANCH(UINT32, uint32, uint32_t)                                        \
            case UINT64:                                                                           \
                ISHMEM_AMO_BRANCH(UINT64, uint64, uint64_t)                                        \
            case SIZE:                                                                             \
                ISHMEM_AMO_BRANCH(SIZE, size, size_t)                                              \
            case PTRDIFF:                                                                          \
                ISHMEM_AMO_BRANCH(PTRDIFF, ptrdiff, ptrdiff_t);                                    \
            default:                                                                               \
                break;                                                                             \
        }                                                                                          \
    }
#define ISHMEM_GEN_AMO_BITWISE_FUNCTION(function, returnvar)                                       \
    function                                                                                       \
    {                                                                                              \
        returnvar switch (t)                                                                       \
        {                                                                                          \
            case UINT:                                                                             \
                ISHMEM_AMO_BRANCH(UINT, uint, unsigned int)                                        \
            case ULONG:                                                                            \
                ISHMEM_AMO_BRANCH(ULONG, ulong, unsigned long)                                     \
            case ULONGLONG:                                                                        \
                ISHMEM_AMO_BRANCH(ULONGLONG, ulonglong, unsigned long long)                        \
            case INT32:                                                                            \
                ISHMEM_AMO_BRANCH(INT32, int32, int32_t)                                           \
            case INT64:                                                                            \
                ISHMEM_AMO_BRANCH(INT64, int64, int64_t)                                           \
            case UINT32:                                                                           \
                ISHMEM_AMO_BRANCH(UINT32, uint32, uint32_t)                                        \
            case UINT64:                                                                           \
                ISHMEM_AMO_BRANCH(UINT64, uint64, uint64_t)                                        \
            default:                                                                               \
                break;                                                                             \
        }                                                                                          \
    }

#endif /* AMO_TEST_H */
