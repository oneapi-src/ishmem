/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "collectives/alltoall_impl.h"

/* Alltoall */
int ishmem_alltoallmem(void *dest, const void *src, size_t nelems)
{
    return ishmem_alltoall((uint8_t *) dest, (uint8_t *) src, nelems);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ALLTOALL(TYPENAME, TYPE) \
    int ishmem_##TYPENAME##_alltoall(TYPE *dest, const TYPE *src, size_t nelems) { return ishmem_alltoall(dest, src, nelems); }
/* clang-format on */

ISHMEMI_API_IMPL_ALLTOALL(float, float)
ISHMEMI_API_IMPL_ALLTOALL(double, double)
ISHMEMI_API_IMPL_ALLTOALL(char, char)
ISHMEMI_API_IMPL_ALLTOALL(schar, signed char)
ISHMEMI_API_IMPL_ALLTOALL(short, short)
ISHMEMI_API_IMPL_ALLTOALL(int, int)
ISHMEMI_API_IMPL_ALLTOALL(long, long)
ISHMEMI_API_IMPL_ALLTOALL(longlong, long long)
ISHMEMI_API_IMPL_ALLTOALL(uchar, unsigned char)
ISHMEMI_API_IMPL_ALLTOALL(ushort, unsigned short)
ISHMEMI_API_IMPL_ALLTOALL(uint, unsigned int)
ISHMEMI_API_IMPL_ALLTOALL(ulong, unsigned long)
ISHMEMI_API_IMPL_ALLTOALL(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ALLTOALL(int8, int8_t)
ISHMEMI_API_IMPL_ALLTOALL(int16, int16_t)
ISHMEMI_API_IMPL_ALLTOALL(int32, int32_t)
ISHMEMI_API_IMPL_ALLTOALL(int64, int64_t)
ISHMEMI_API_IMPL_ALLTOALL(uint8, uint8_t)
ISHMEMI_API_IMPL_ALLTOALL(uint16, uint16_t)
ISHMEMI_API_IMPL_ALLTOALL(uint32, uint32_t)
ISHMEMI_API_IMPL_ALLTOALL(uint64, uint64_t)
ISHMEMI_API_IMPL_ALLTOALL(size, size_t)
ISHMEMI_API_IMPL_ALLTOALL(ptrdiff, ptrdiff_t)
