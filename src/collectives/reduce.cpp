/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "collectives/reduce_impl.h"

/* clang-format off */
#define ISHMEMI_API_IMPL_AND_REDUCE(TYPENAME, TYPE)                                                                                                                      \
    int ishmem_##TYPENAME##_and_reduce(TYPE *dest, const TYPE *src, size_t nreduce) { return ishmem_and_reduce(dest, src, nreduce); }                                    \
    sycl::event ishmemx_##TYPENAME##_and_reduce_on_queue(TYPE *dest, const TYPE *src, size_t nreduce, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {  \
        return ishmemx_and_reduce_on_queue(dest, src, nreduce, ret, q, deps);                                                                                            \
    }
#define ISHMEMI_API_IMPL_OR_REDUCE(TYPENAME, TYPE)                                                                                                                       \
    int ishmem_##TYPENAME##_or_reduce(TYPE *dest, const TYPE *src, size_t nreduce) { return ishmem_or_reduce(dest, src, nreduce); }                                      \
    sycl::event ishmemx_##TYPENAME##_or_reduce_on_queue(TYPE *dest, const TYPE *src, size_t nreduce, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {   \
        return ishmemx_or_reduce_on_queue(dest, src, nreduce, ret, q, deps);                                                                                             \
    }
#define ISHMEMI_API_IMPL_XOR_REDUCE(TYPENAME, TYPE)                                                                                                                      \
    int ishmem_##TYPENAME##_xor_reduce(TYPE *dest, const TYPE *src, size_t nreduce) { return ishmem_xor_reduce(dest, src, nreduce); }                                    \
    sycl::event ishmemx_##TYPENAME##_xor_reduce_on_queue(TYPE *dest, const TYPE *src, size_t nreduce, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {  \
        return ishmemx_xor_reduce_on_queue(dest, src, nreduce, ret, q, deps);                                                                                            \
    }
#define ISHMEMI_API_IMPL_MAX_REDUCE(TYPENAME, TYPE)                                                                                                                      \
    int ishmem_##TYPENAME##_max_reduce(TYPE *dest, const TYPE *src, size_t nreduce) { return ishmem_max_reduce(dest, src, nreduce); }                                    \
    sycl::event ishmemx_##TYPENAME##_max_reduce_on_queue(TYPE *dest, const TYPE *src, size_t nreduce, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {  \
        return ishmemx_max_reduce_on_queue(dest, src, nreduce, ret, q, deps);                                                                                            \
    }
#define ISHMEMI_API_IMPL_MIN_REDUCE(TYPENAME, TYPE)                                                                                                                      \
    int ishmem_##TYPENAME##_min_reduce(TYPE *dest, const TYPE *src, size_t nreduce) { return ishmem_min_reduce(dest, src, nreduce); }                                    \
    sycl::event ishmemx_##TYPENAME##_min_reduce_on_queue(TYPE *dest, const TYPE *src, size_t nreduce, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {  \
        return ishmemx_min_reduce_on_queue(dest, src, nreduce, ret, q, deps);                                                                                            \
    }
#define ISHMEMI_API_IMPL_SUM_REDUCE(TYPENAME, TYPE)                                                                                                                      \
    int ishmem_##TYPENAME##_sum_reduce(TYPE *dest, const TYPE *src, size_t nreduce) { return ishmem_sum_reduce(dest, src, nreduce); }                                    \
    sycl::event ishmemx_##TYPENAME##_sum_reduce_on_queue(TYPE *dest, const TYPE *src, size_t nreduce, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {  \
        return ishmemx_sum_reduce_on_queue(dest, src, nreduce, ret, q, deps);                                                                                            \
    }
#define ISHMEMI_API_IMPL_PROD_REDUCE(TYPENAME, TYPE)                                                                                                                     \
    int ishmem_##TYPENAME##_prod_reduce(TYPE *dest, const TYPE *src, size_t nreduce) { return ishmem_prod_reduce(dest, src, nreduce); }                                  \
    sycl::event ishmemx_##TYPENAME##_prod_reduce_on_queue(TYPE *dest, const TYPE *src, size_t nreduce, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) { \
        return ishmemx_prod_reduce_on_queue(dest, src, nreduce, ret, q, deps);                                                                                           \
    }

/* Teams variants */
#define ISHMEMI_API_IMPL_AND_TEAM_REDUCE(TYPENAME, TYPE)                                                                                                                                     \
    int ishmem_##TYPENAME##_and_reduce(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce) { return ishmem_and_reduce(team, dest, src, nreduce); }                              \
    sycl::event ishmemx_##TYPENAME##_and_reduce_on_queue(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {  \
        return ishmemx_and_reduce_on_queue(team, dest, src, nreduce, ret, q, deps);                                                                                                          \
    }
#define ISHMEMI_API_IMPL_OR_TEAM_REDUCE(TYPENAME, TYPE)                                                                                                                                      \
    int ishmem_##TYPENAME##_or_reduce(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce) { return ishmem_or_reduce(team, dest, src, nreduce); }                                \
    sycl::event ishmemx_##TYPENAME##_or_reduce_on_queue(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {   \
        return ishmemx_or_reduce_on_queue(team, dest, src, nreduce, ret, q, deps);                                                                                                           \
    }
#define ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(TYPENAME, TYPE)                                                                                                                                     \
    int ishmem_##TYPENAME##_xor_reduce(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce) { return ishmem_xor_reduce(team, dest, src, nreduce); }                              \
    sycl::event ishmemx_##TYPENAME##_xor_reduce_on_queue(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {  \
        return ishmemx_xor_reduce_on_queue(team, dest, src, nreduce, ret, q, deps);                                                                                                          \
    }
#define ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(TYPENAME, TYPE)                                                                                                                                     \
    int ishmem_##TYPENAME##_max_reduce(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce) { return ishmem_max_reduce(team, dest, src, nreduce); }                              \
    sycl::event ishmemx_##TYPENAME##_max_reduce_on_queue(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {  \
        return ishmemx_max_reduce_on_queue(team, dest, src, nreduce, ret, q, deps);                                                                                                          \
    }
#define ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(TYPENAME, TYPE)                                                                                                                                     \
    int ishmem_##TYPENAME##_min_reduce(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce) { return ishmem_min_reduce(team, dest, src, nreduce); }                              \
    sycl::event ishmemx_##TYPENAME##_min_reduce_on_queue(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {  \
        return ishmemx_min_reduce_on_queue(team, dest, src, nreduce, ret, q, deps);                                                                                                          \
    }
#define ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(TYPENAME, TYPE)                                                                                                                                     \
    int ishmem_##TYPENAME##_sum_reduce(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce) { return ishmem_sum_reduce(team, dest, src, nreduce); }                              \
    sycl::event ishmemx_##TYPENAME##_sum_reduce_on_queue(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {  \
        return ishmemx_sum_reduce_on_queue(team, dest, src, nreduce, ret, q, deps);                                                                                                          \
    }
#define ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(TYPENAME, TYPE)                                                                                                                                    \
    int ishmem_##TYPENAME##_prod_reduce(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce) { return ishmem_prod_reduce(team, dest, src, nreduce); }                            \
    sycl::event ishmemx_##TYPENAME##_prod_reduce_on_queue(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) { \
        return ishmemx_prod_reduce_on_queue(team, dest, src, nreduce, ret, q, deps);                                                                                                         \
    }
/* clang-format on */

/* And Reduce */
ISHMEMI_API_IMPL_AND_REDUCE(uchar, unsigned char)
ISHMEMI_API_IMPL_AND_REDUCE(ushort, unsigned short)
ISHMEMI_API_IMPL_AND_REDUCE(uint, unsigned int)
ISHMEMI_API_IMPL_AND_REDUCE(ulong, unsigned long)
ISHMEMI_API_IMPL_AND_REDUCE(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_AND_REDUCE(int8, int8_t)
ISHMEMI_API_IMPL_AND_REDUCE(int16, int16_t)
ISHMEMI_API_IMPL_AND_REDUCE(int32, int32_t)
ISHMEMI_API_IMPL_AND_REDUCE(int64, int64_t)
ISHMEMI_API_IMPL_AND_REDUCE(uint8, uint8_t)
ISHMEMI_API_IMPL_AND_REDUCE(uint16, uint16_t)
ISHMEMI_API_IMPL_AND_REDUCE(uint32, uint32_t)
ISHMEMI_API_IMPL_AND_REDUCE(uint64, uint64_t)
ISHMEMI_API_IMPL_AND_REDUCE(size, size_t)

/* Or Reduce */
ISHMEMI_API_IMPL_OR_REDUCE(uchar, unsigned char)
ISHMEMI_API_IMPL_OR_REDUCE(ushort, unsigned short)
ISHMEMI_API_IMPL_OR_REDUCE(uint, unsigned int)
ISHMEMI_API_IMPL_OR_REDUCE(ulong, unsigned long)
ISHMEMI_API_IMPL_OR_REDUCE(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_OR_REDUCE(int8, int8_t)
ISHMEMI_API_IMPL_OR_REDUCE(int16, int16_t)
ISHMEMI_API_IMPL_OR_REDUCE(int32, int32_t)
ISHMEMI_API_IMPL_OR_REDUCE(int64, int64_t)
ISHMEMI_API_IMPL_OR_REDUCE(uint8, uint8_t)
ISHMEMI_API_IMPL_OR_REDUCE(uint16, uint16_t)
ISHMEMI_API_IMPL_OR_REDUCE(uint32, uint32_t)
ISHMEMI_API_IMPL_OR_REDUCE(uint64, uint64_t)
ISHMEMI_API_IMPL_OR_REDUCE(size, size_t)

/* Xor Reduce */
ISHMEMI_API_IMPL_XOR_REDUCE(uchar, unsigned char)
ISHMEMI_API_IMPL_XOR_REDUCE(ushort, unsigned short)
ISHMEMI_API_IMPL_XOR_REDUCE(uint, unsigned int)
ISHMEMI_API_IMPL_XOR_REDUCE(ulong, unsigned long)
ISHMEMI_API_IMPL_XOR_REDUCE(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_XOR_REDUCE(int8, int8_t)
ISHMEMI_API_IMPL_XOR_REDUCE(int16, int16_t)
ISHMEMI_API_IMPL_XOR_REDUCE(int32, int32_t)
ISHMEMI_API_IMPL_XOR_REDUCE(int64, int64_t)
ISHMEMI_API_IMPL_XOR_REDUCE(uint8, uint8_t)
ISHMEMI_API_IMPL_XOR_REDUCE(uint16, uint16_t)
ISHMEMI_API_IMPL_XOR_REDUCE(uint32, uint32_t)
ISHMEMI_API_IMPL_XOR_REDUCE(uint64, uint64_t)
ISHMEMI_API_IMPL_XOR_REDUCE(size, size_t)

/* Max Reduce */
ISHMEMI_API_IMPL_MAX_REDUCE(char, char)
ISHMEMI_API_IMPL_MAX_REDUCE(schar, signed char)
ISHMEMI_API_IMPL_MAX_REDUCE(short, short)
ISHMEMI_API_IMPL_MAX_REDUCE(int, int)
ISHMEMI_API_IMPL_MAX_REDUCE(long, long)
ISHMEMI_API_IMPL_MAX_REDUCE(longlong, long long)
ISHMEMI_API_IMPL_MAX_REDUCE(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_MAX_REDUCE(uchar, unsigned char)
ISHMEMI_API_IMPL_MAX_REDUCE(ushort, unsigned short)
ISHMEMI_API_IMPL_MAX_REDUCE(uint, unsigned int)
ISHMEMI_API_IMPL_MAX_REDUCE(ulong, unsigned long)
ISHMEMI_API_IMPL_MAX_REDUCE(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_MAX_REDUCE(int8, int8_t)
ISHMEMI_API_IMPL_MAX_REDUCE(int16, int16_t)
ISHMEMI_API_IMPL_MAX_REDUCE(int32, int32_t)
ISHMEMI_API_IMPL_MAX_REDUCE(int64, int64_t)
ISHMEMI_API_IMPL_MAX_REDUCE(uint8, uint8_t)
ISHMEMI_API_IMPL_MAX_REDUCE(uint16, uint16_t)
ISHMEMI_API_IMPL_MAX_REDUCE(uint32, uint32_t)
ISHMEMI_API_IMPL_MAX_REDUCE(uint64, uint64_t)
ISHMEMI_API_IMPL_MAX_REDUCE(size, size_t)
ISHMEMI_API_IMPL_MAX_REDUCE(float, float)
ISHMEMI_API_IMPL_MAX_REDUCE(double, double)

/* Min Reduce */
ISHMEMI_API_IMPL_MIN_REDUCE(char, char)
ISHMEMI_API_IMPL_MIN_REDUCE(schar, signed char)
ISHMEMI_API_IMPL_MIN_REDUCE(short, short)
ISHMEMI_API_IMPL_MIN_REDUCE(int, int)
ISHMEMI_API_IMPL_MIN_REDUCE(long, long)
ISHMEMI_API_IMPL_MIN_REDUCE(longlong, long long)
ISHMEMI_API_IMPL_MIN_REDUCE(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_MIN_REDUCE(uchar, unsigned char)
ISHMEMI_API_IMPL_MIN_REDUCE(ushort, unsigned short)
ISHMEMI_API_IMPL_MIN_REDUCE(uint, unsigned int)
ISHMEMI_API_IMPL_MIN_REDUCE(ulong, unsigned long)
ISHMEMI_API_IMPL_MIN_REDUCE(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_MIN_REDUCE(int8, int8_t)
ISHMEMI_API_IMPL_MIN_REDUCE(int16, int16_t)
ISHMEMI_API_IMPL_MIN_REDUCE(int32, int32_t)
ISHMEMI_API_IMPL_MIN_REDUCE(int64, int64_t)
ISHMEMI_API_IMPL_MIN_REDUCE(uint8, uint8_t)
ISHMEMI_API_IMPL_MIN_REDUCE(uint16, uint16_t)
ISHMEMI_API_IMPL_MIN_REDUCE(uint32, uint32_t)
ISHMEMI_API_IMPL_MIN_REDUCE(uint64, uint64_t)
ISHMEMI_API_IMPL_MIN_REDUCE(size, size_t)
ISHMEMI_API_IMPL_MIN_REDUCE(float, float)
ISHMEMI_API_IMPL_MIN_REDUCE(double, double)

/* Sum Reduce */
ISHMEMI_API_IMPL_SUM_REDUCE(char, char)
ISHMEMI_API_IMPL_SUM_REDUCE(schar, signed char)
ISHMEMI_API_IMPL_SUM_REDUCE(short, short)
ISHMEMI_API_IMPL_SUM_REDUCE(int, int)
ISHMEMI_API_IMPL_SUM_REDUCE(long, long)
ISHMEMI_API_IMPL_SUM_REDUCE(longlong, long long)
ISHMEMI_API_IMPL_SUM_REDUCE(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_SUM_REDUCE(uchar, unsigned char)
ISHMEMI_API_IMPL_SUM_REDUCE(ushort, unsigned short)
ISHMEMI_API_IMPL_SUM_REDUCE(uint, unsigned int)
ISHMEMI_API_IMPL_SUM_REDUCE(ulong, unsigned long)
ISHMEMI_API_IMPL_SUM_REDUCE(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_SUM_REDUCE(int8, int8_t)
ISHMEMI_API_IMPL_SUM_REDUCE(int16, int16_t)
ISHMEMI_API_IMPL_SUM_REDUCE(int32, int32_t)
ISHMEMI_API_IMPL_SUM_REDUCE(int64, int64_t)
ISHMEMI_API_IMPL_SUM_REDUCE(uint8, uint8_t)
ISHMEMI_API_IMPL_SUM_REDUCE(uint16, uint16_t)
ISHMEMI_API_IMPL_SUM_REDUCE(uint32, uint32_t)
ISHMEMI_API_IMPL_SUM_REDUCE(uint64, uint64_t)
ISHMEMI_API_IMPL_SUM_REDUCE(size, size_t)
ISHMEMI_API_IMPL_SUM_REDUCE(float, float)
ISHMEMI_API_IMPL_SUM_REDUCE(double, double)

/* Prod Reduce */
ISHMEMI_API_IMPL_PROD_REDUCE(char, char)
ISHMEMI_API_IMPL_PROD_REDUCE(schar, signed char)
ISHMEMI_API_IMPL_PROD_REDUCE(short, short)
ISHMEMI_API_IMPL_PROD_REDUCE(int, int)
ISHMEMI_API_IMPL_PROD_REDUCE(long, long)
ISHMEMI_API_IMPL_PROD_REDUCE(longlong, long long)
ISHMEMI_API_IMPL_PROD_REDUCE(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_PROD_REDUCE(uchar, unsigned char)
ISHMEMI_API_IMPL_PROD_REDUCE(ushort, unsigned short)
ISHMEMI_API_IMPL_PROD_REDUCE(uint, unsigned int)
ISHMEMI_API_IMPL_PROD_REDUCE(ulong, unsigned long)
ISHMEMI_API_IMPL_PROD_REDUCE(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_PROD_REDUCE(int8, int8_t)
ISHMEMI_API_IMPL_PROD_REDUCE(int16, int16_t)
ISHMEMI_API_IMPL_PROD_REDUCE(int32, int32_t)
ISHMEMI_API_IMPL_PROD_REDUCE(int64, int64_t)
ISHMEMI_API_IMPL_PROD_REDUCE(uint8, uint8_t)
ISHMEMI_API_IMPL_PROD_REDUCE(uint16, uint16_t)
ISHMEMI_API_IMPL_PROD_REDUCE(uint32, uint32_t)
ISHMEMI_API_IMPL_PROD_REDUCE(uint64, uint64_t)
ISHMEMI_API_IMPL_PROD_REDUCE(size, size_t)
ISHMEMI_API_IMPL_PROD_REDUCE(float, float)
ISHMEMI_API_IMPL_PROD_REDUCE(double, double)

/* Team And Reduce */
ISHMEMI_API_IMPL_AND_TEAM_REDUCE(uchar, unsigned char)
ISHMEMI_API_IMPL_AND_TEAM_REDUCE(ushort, unsigned short)
ISHMEMI_API_IMPL_AND_TEAM_REDUCE(uint, unsigned int)
ISHMEMI_API_IMPL_AND_TEAM_REDUCE(ulong, unsigned long)
ISHMEMI_API_IMPL_AND_TEAM_REDUCE(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_AND_TEAM_REDUCE(int8, int8_t)
ISHMEMI_API_IMPL_AND_TEAM_REDUCE(int16, int16_t)
ISHMEMI_API_IMPL_AND_TEAM_REDUCE(int32, int32_t)
ISHMEMI_API_IMPL_AND_TEAM_REDUCE(int64, int64_t)
ISHMEMI_API_IMPL_AND_TEAM_REDUCE(uint8, uint8_t)
ISHMEMI_API_IMPL_AND_TEAM_REDUCE(uint16, uint16_t)
ISHMEMI_API_IMPL_AND_TEAM_REDUCE(uint32, uint32_t)
ISHMEMI_API_IMPL_AND_TEAM_REDUCE(uint64, uint64_t)
ISHMEMI_API_IMPL_AND_TEAM_REDUCE(size, size_t)

/* Team Or Reduce */
ISHMEMI_API_IMPL_OR_TEAM_REDUCE(uchar, unsigned char)
ISHMEMI_API_IMPL_OR_TEAM_REDUCE(ushort, unsigned short)
ISHMEMI_API_IMPL_OR_TEAM_REDUCE(uint, unsigned int)
ISHMEMI_API_IMPL_OR_TEAM_REDUCE(ulong, unsigned long)
ISHMEMI_API_IMPL_OR_TEAM_REDUCE(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_OR_TEAM_REDUCE(int8, int8_t)
ISHMEMI_API_IMPL_OR_TEAM_REDUCE(int16, int16_t)
ISHMEMI_API_IMPL_OR_TEAM_REDUCE(int32, int32_t)
ISHMEMI_API_IMPL_OR_TEAM_REDUCE(int64, int64_t)
ISHMEMI_API_IMPL_OR_TEAM_REDUCE(uint8, uint8_t)
ISHMEMI_API_IMPL_OR_TEAM_REDUCE(uint16, uint16_t)
ISHMEMI_API_IMPL_OR_TEAM_REDUCE(uint32, uint32_t)
ISHMEMI_API_IMPL_OR_TEAM_REDUCE(uint64, uint64_t)
ISHMEMI_API_IMPL_OR_TEAM_REDUCE(size, size_t)

/* Team Xor Reduce */
ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(uchar, unsigned char)
ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(ushort, unsigned short)
ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(uint, unsigned int)
ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(ulong, unsigned long)
ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(int8, int8_t)
ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(int16, int16_t)
ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(int32, int32_t)
ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(int64, int64_t)
ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(uint8, uint8_t)
ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(uint16, uint16_t)
ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(uint32, uint32_t)
ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(uint64, uint64_t)
ISHMEMI_API_IMPL_XOR_TEAM_REDUCE(size, size_t)

/* Team Max Reduce */
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(char, char)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(schar, signed char)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(short, short)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(int, int)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(long, long)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(longlong, long long)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(uchar, unsigned char)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(ushort, unsigned short)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(uint, unsigned int)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(ulong, unsigned long)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(int8, int8_t)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(int16, int16_t)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(int32, int32_t)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(int64, int64_t)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(uint8, uint8_t)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(uint16, uint16_t)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(uint32, uint32_t)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(uint64, uint64_t)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(size, size_t)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(float, float)
ISHMEMI_API_IMPL_MAX_TEAM_REDUCE(double, double)

/* Team Min Reduce */
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(char, char)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(schar, signed char)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(short, short)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(int, int)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(long, long)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(longlong, long long)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(uchar, unsigned char)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(ushort, unsigned short)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(uint, unsigned int)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(ulong, unsigned long)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(int8, int8_t)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(int16, int16_t)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(int32, int32_t)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(int64, int64_t)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(uint8, uint8_t)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(uint16, uint16_t)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(uint32, uint32_t)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(uint64, uint64_t)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(size, size_t)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(float, float)
ISHMEMI_API_IMPL_MIN_TEAM_REDUCE(double, double)

/* Team Sum Reduce */
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(char, char)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(schar, signed char)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(short, short)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(int, int)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(long, long)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(longlong, long long)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(uchar, unsigned char)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(ushort, unsigned short)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(uint, unsigned int)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(ulong, unsigned long)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(int8, int8_t)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(int16, int16_t)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(int32, int32_t)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(int64, int64_t)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(uint8, uint8_t)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(uint16, uint16_t)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(uint32, uint32_t)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(uint64, uint64_t)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(size, size_t)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(float, float)
ISHMEMI_API_IMPL_SUM_TEAM_REDUCE(double, double)

/* Team Prod Reduce */
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(char, char)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(schar, signed char)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(short, short)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(int, int)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(long, long)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(longlong, long long)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(uchar, unsigned char)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(ushort, unsigned short)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(uint, unsigned int)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(ulong, unsigned long)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(int8, int8_t)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(int16, int16_t)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(int32, int32_t)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(int64, int64_t)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(uint8, uint8_t)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(uint16, uint16_t)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(uint32, uint32_t)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(uint64, uint64_t)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(size, size_t)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(float, float)
ISHMEMI_API_IMPL_PROD_TEAM_REDUCE(double, double)
