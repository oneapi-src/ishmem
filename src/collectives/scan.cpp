/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "collectives/scan_impl.h"

/* clang-format off */
#define ISHMEMI_API_IMPL_INSCAN(TYPENAME, TYPE)                                                                                                                                             \
    int ishmem_##TYPENAME##_sum_inscan(TYPE *dest, const TYPE *src, size_t nelems) { return ishmem_sum_inscan(ISHMEM_TEAM_WORLD, dest, src, nelems); }                                      \
    int ishmem_##TYPENAME##_sum_inscan(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems) { return ishmem_sum_inscan(team, dest, src, nelems); }                               \
    sycl::event ishmemx_##TYPENAME##_sum_inscan_on_queue(TYPE *dest, const TYPE *src, size_t nelems, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {                      \
        return ishmemx_sum_inscan_on_queue(ISHMEM_TEAM_WORLD, dest, src, nelems, ret, q, deps);                                                                                             \
    }                                                                                                                                                                                       \
    sycl::event ishmemx_##TYPENAME##_sum_inscan_on_queue(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {  \
        return ishmemx_sum_inscan_on_queue(team, dest, src, nelems, ret, q, deps);                                                                                                          \
    }
#define ISHMEMI_API_IMPL_EXSCAN(TYPENAME, TYPE)                                                                                                                                             \
    int ishmem_##TYPENAME##_sum_exscan(TYPE *dest, const TYPE *src, size_t nelems) { return ishmem_sum_exscan(ISHMEM_TEAM_WORLD, dest, src, nelems); }                                      \
    int ishmem_##TYPENAME##_sum_exscan(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems) { return ishmem_sum_exscan(team, dest, src, nelems); }                               \
    sycl::event ishmemx_##TYPENAME##_sum_exscan_on_queue(TYPE *dest, const TYPE *src, size_t nelems, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {                      \
        return ishmemx_sum_exscan_on_queue(ISHMEM_TEAM_WORLD, dest, src, nelems, ret, q, deps);                                                                                             \
    }                                                                                                                                                                                       \
    sycl::event ishmemx_##TYPENAME##_sum_exscan_on_queue(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, int *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {  \
        return ishmemx_sum_exscan_on_queue(team, dest, src, nelems, ret, q, deps);                                                                                                          \
    }
/* clang-format on */

ISHMEMI_API_IMPL_INSCAN(float, float)
ISHMEMI_API_IMPL_INSCAN(double, double)
ISHMEMI_API_IMPL_INSCAN(char, char)
ISHMEMI_API_IMPL_INSCAN(schar, signed char)
ISHMEMI_API_IMPL_INSCAN(short, short)
ISHMEMI_API_IMPL_INSCAN(int, int)
ISHMEMI_API_IMPL_INSCAN(long, long)
ISHMEMI_API_IMPL_INSCAN(longlong, long long)
ISHMEMI_API_IMPL_INSCAN(uchar, unsigned char)
ISHMEMI_API_IMPL_INSCAN(ushort, unsigned short)
ISHMEMI_API_IMPL_INSCAN(uint, unsigned int)
ISHMEMI_API_IMPL_INSCAN(ulong, unsigned long)
ISHMEMI_API_IMPL_INSCAN(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_INSCAN(int8, int8_t)
ISHMEMI_API_IMPL_INSCAN(int16, int16_t)
ISHMEMI_API_IMPL_INSCAN(int32, int32_t)
ISHMEMI_API_IMPL_INSCAN(int64, int64_t)
ISHMEMI_API_IMPL_INSCAN(uint8, uint8_t)
ISHMEMI_API_IMPL_INSCAN(uint16, uint16_t)
ISHMEMI_API_IMPL_INSCAN(uint32, uint32_t)
ISHMEMI_API_IMPL_INSCAN(uint64, uint64_t)
ISHMEMI_API_IMPL_INSCAN(size, size_t)
ISHMEMI_API_IMPL_INSCAN(ptrdiff, ptrdiff_t)

ISHMEMI_API_IMPL_EXSCAN(float, float)
ISHMEMI_API_IMPL_EXSCAN(double, double)
ISHMEMI_API_IMPL_EXSCAN(char, char)
ISHMEMI_API_IMPL_EXSCAN(schar, signed char)
ISHMEMI_API_IMPL_EXSCAN(short, short)
ISHMEMI_API_IMPL_EXSCAN(int, int)
ISHMEMI_API_IMPL_EXSCAN(long, long)
ISHMEMI_API_IMPL_EXSCAN(longlong, long long)
ISHMEMI_API_IMPL_EXSCAN(uchar, unsigned char)
ISHMEMI_API_IMPL_EXSCAN(ushort, unsigned short)
ISHMEMI_API_IMPL_EXSCAN(uint, unsigned int)
ISHMEMI_API_IMPL_EXSCAN(ulong, unsigned long)
ISHMEMI_API_IMPL_EXSCAN(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_EXSCAN(int8, int8_t)
ISHMEMI_API_IMPL_EXSCAN(int16, int16_t)
ISHMEMI_API_IMPL_EXSCAN(int32, int32_t)
ISHMEMI_API_IMPL_EXSCAN(int64, int64_t)
ISHMEMI_API_IMPL_EXSCAN(uint8, uint8_t)
ISHMEMI_API_IMPL_EXSCAN(uint16, uint16_t)
ISHMEMI_API_IMPL_EXSCAN(uint32, uint32_t)
ISHMEMI_API_IMPL_EXSCAN(uint64, uint64_t)
ISHMEMI_API_IMPL_EXSCAN(size, size_t)
ISHMEMI_API_IMPL_EXSCAN(ptrdiff, ptrdiff_t)
