/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "collectives/broadcast_impl.h"

/* Broadcast */
int ishmem_broadcastmem(void *dest, const void *src, size_t nelems, int pe)
{
    return ishmem_broadcast((uint8_t *) dest, (uint8_t *) src, nelems, pe);
}

int ishmem_broadcastmem(ishmem_team_t team, void *dest, const void *src, size_t nelems, int pe)
{
    return ishmem_broadcast(team, (uint8_t *) dest, (uint8_t *) src, nelems, pe);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_BROADCAST(TYPENAME, TYPE)  \
    int ishmem_##TYPENAME##_broadcast(TYPE *dest, const TYPE *src, size_t nelems, int pe) { return ishmem_broadcast(dest, src, nelems, pe); }
#define ISHMEMI_API_IMPL_TEAM_BROADCAST(TYPENAME, TYPE)  \
    int ishmem_##TYPENAME##_broadcast(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, int pe) { return ishmem_broadcast(team, dest, src, nelems, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_BROADCAST(float, float)
ISHMEMI_API_IMPL_BROADCAST(double, double)
ISHMEMI_API_IMPL_BROADCAST(char, char)
ISHMEMI_API_IMPL_BROADCAST(schar, signed char)
ISHMEMI_API_IMPL_BROADCAST(short, short)
ISHMEMI_API_IMPL_BROADCAST(int, int)
ISHMEMI_API_IMPL_BROADCAST(long, long)
ISHMEMI_API_IMPL_BROADCAST(longlong, long long)
ISHMEMI_API_IMPL_BROADCAST(uchar, unsigned char)
ISHMEMI_API_IMPL_BROADCAST(ushort, unsigned short)
ISHMEMI_API_IMPL_BROADCAST(uint, unsigned int)
ISHMEMI_API_IMPL_BROADCAST(ulong, unsigned long)
ISHMEMI_API_IMPL_BROADCAST(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_BROADCAST(int8, int8_t)
ISHMEMI_API_IMPL_BROADCAST(int16, int16_t)
ISHMEMI_API_IMPL_BROADCAST(int32, int32_t)
ISHMEMI_API_IMPL_BROADCAST(int64, int64_t)
ISHMEMI_API_IMPL_BROADCAST(uint8, uint8_t)
ISHMEMI_API_IMPL_BROADCAST(uint16, uint16_t)
ISHMEMI_API_IMPL_BROADCAST(uint32, uint32_t)
ISHMEMI_API_IMPL_BROADCAST(uint64, uint64_t)
ISHMEMI_API_IMPL_BROADCAST(size, size_t)
ISHMEMI_API_IMPL_BROADCAST(ptrdiff, ptrdiff_t)

ISHMEMI_API_IMPL_TEAM_BROADCAST(float, float)
ISHMEMI_API_IMPL_TEAM_BROADCAST(double, double)
ISHMEMI_API_IMPL_TEAM_BROADCAST(char, char)
ISHMEMI_API_IMPL_TEAM_BROADCAST(schar, signed char)
ISHMEMI_API_IMPL_TEAM_BROADCAST(short, short)
ISHMEMI_API_IMPL_TEAM_BROADCAST(int, int)
ISHMEMI_API_IMPL_TEAM_BROADCAST(long, long)
ISHMEMI_API_IMPL_TEAM_BROADCAST(longlong, long long)
ISHMEMI_API_IMPL_TEAM_BROADCAST(uchar, unsigned char)
ISHMEMI_API_IMPL_TEAM_BROADCAST(ushort, unsigned short)
ISHMEMI_API_IMPL_TEAM_BROADCAST(uint, unsigned int)
ISHMEMI_API_IMPL_TEAM_BROADCAST(ulong, unsigned long)
ISHMEMI_API_IMPL_TEAM_BROADCAST(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEAM_BROADCAST(int8, int8_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST(int16, int16_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST(int32, int32_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST(int64, int64_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST(uint8, uint8_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST(uint16, uint16_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST(uint32, uint32_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST(uint64, uint64_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST(size, size_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST(ptrdiff, ptrdiff_t)
