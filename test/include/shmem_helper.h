/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_OPENSHMEM_HELPER_H
#define ISHMEM_OPENSHMEM_HELPER_H

#if defined(ENABLE_OPENSHMEM)
#include <shmem.h>

#define SHMEM_CHECK(call) call
#else
#define SHMEM_CHECK(call)                                                                          \
    fprintf(stderr, "Failure - Attempting to use SHMEM when it's not configured\n");
#endif

void runtime_shmem_init()
{
    SHMEM_CHECK(shmem_init());
}

void runtime_shmem_finalize()
{
    SHMEM_CHECK(shmem_finalize());
}

void *runtime_shmem_calloc(size_t num, size_t size)
{
    SHMEM_CHECK(return shmem_calloc(num, size));

    /* Only returned on failure: */
    return nullptr;
}

void *runtime_shmem_malloc(size_t size)
{
    SHMEM_CHECK(return shmem_malloc(size));

    /* Only returned on failure: */
    return nullptr;
}

void runtime_shmem_free(void *ptr)
{
    SHMEM_CHECK(shmem_free(ptr));
}

void runtime_shmem_sync_all()
{
    SHMEM_CHECK(shmem_sync_all());
}

void runtime_shmem_broadcast(void *dst, void *src, size_t size, int root)
{
    SHMEM_CHECK(shmem_broadcastmem(SHMEM_TEAM_WORLD, dst, src, size, root));
}

void runtime_shmem_uint64_sum_reduce(uint64_t *dst, uint64_t *src, size_t num)
{
    SHMEM_CHECK(shmem_uint64_sum_reduce(SHMEM_TEAM_WORLD, dst, src, num));
}

#endif
