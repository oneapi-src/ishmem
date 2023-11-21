/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "collectives/collect_impl.h"

/* Collect */
int ishmem_collectmem(void *dest, const void *src, size_t nelems)
{
    return ishmem_collect((uint8_t *) dest, (uint8_t *) src, nelems);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_COLLECT(TYPENAME, TYPE)  \
    int ishmem_##TYPENAME##_collect(TYPE *dest, const TYPE *src, size_t nelems) { return ishmem_collect(dest, src, nelems); }
/* clang-format on */

ISHMEMI_API_IMPL_COLLECT(float, float)
ISHMEMI_API_IMPL_COLLECT(double, double)
ISHMEMI_API_IMPL_COLLECT(char, char)
ISHMEMI_API_IMPL_COLLECT(schar, signed char)
ISHMEMI_API_IMPL_COLLECT(short, short)
ISHMEMI_API_IMPL_COLLECT(int, int)
ISHMEMI_API_IMPL_COLLECT(long, long)
ISHMEMI_API_IMPL_COLLECT(longlong, long long)
ISHMEMI_API_IMPL_COLLECT(uchar, unsigned char)
ISHMEMI_API_IMPL_COLLECT(ushort, unsigned short)
ISHMEMI_API_IMPL_COLLECT(uint, unsigned int)
ISHMEMI_API_IMPL_COLLECT(ulong, unsigned long)
ISHMEMI_API_IMPL_COLLECT(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_COLLECT(int8, int8_t)
ISHMEMI_API_IMPL_COLLECT(int16, int16_t)
ISHMEMI_API_IMPL_COLLECT(int32, int32_t)
ISHMEMI_API_IMPL_COLLECT(int64, int64_t)
ISHMEMI_API_IMPL_COLLECT(uint8, uint8_t)
ISHMEMI_API_IMPL_COLLECT(uint16, uint16_t)
ISHMEMI_API_IMPL_COLLECT(uint32, uint32_t)
ISHMEMI_API_IMPL_COLLECT(uint64, uint64_t)
ISHMEMI_API_IMPL_COLLECT(size, size_t)
ISHMEMI_API_IMPL_COLLECT(ptrdiff, ptrdiff_t)

/* Fcollect */
int ishmem_fcollectmem(void *dest, const void *src, size_t nelems)
{
    return ishmem_fcollect((uint8_t *) dest, (uint8_t *) src, nelems);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_FCOLLECT(TYPENAME, TYPE) \
    int ishmem_##TYPENAME##_fcollect(TYPE *dest, const TYPE *src, size_t nelems) { return ishmem_fcollect(dest, src, nelems); }
/* clang-format on */

ISHMEMI_API_IMPL_FCOLLECT(float, float)
ISHMEMI_API_IMPL_FCOLLECT(double, double)
ISHMEMI_API_IMPL_FCOLLECT(char, char)
ISHMEMI_API_IMPL_FCOLLECT(schar, signed char)
ISHMEMI_API_IMPL_FCOLLECT(short, short)
ISHMEMI_API_IMPL_FCOLLECT(int, int)
ISHMEMI_API_IMPL_FCOLLECT(long, long)
ISHMEMI_API_IMPL_FCOLLECT(longlong, long long)
ISHMEMI_API_IMPL_FCOLLECT(uchar, unsigned char)
ISHMEMI_API_IMPL_FCOLLECT(ushort, unsigned short)
ISHMEMI_API_IMPL_FCOLLECT(uint, unsigned int)
ISHMEMI_API_IMPL_FCOLLECT(ulong, unsigned long)
ISHMEMI_API_IMPL_FCOLLECT(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_FCOLLECT(int8, int8_t)
ISHMEMI_API_IMPL_FCOLLECT(int16, int16_t)
ISHMEMI_API_IMPL_FCOLLECT(int32, int32_t)
ISHMEMI_API_IMPL_FCOLLECT(int64, int64_t)
ISHMEMI_API_IMPL_FCOLLECT(uint8, uint8_t)
ISHMEMI_API_IMPL_FCOLLECT(uint16, uint16_t)
ISHMEMI_API_IMPL_FCOLLECT(uint32, uint32_t)
ISHMEMI_API_IMPL_FCOLLECT(uint64, uint64_t)
ISHMEMI_API_IMPL_FCOLLECT(size, size_t)
ISHMEMI_API_IMPL_FCOLLECT(ptrdiff, ptrdiff_t)
