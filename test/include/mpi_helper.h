/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_MPI_HELPER_H
#define ISHMEM_MPI_HELPER_H

#include <stdio.h>
#if defined(ENABLE_MPI)
#include <mpi.h>

#define MPI_CHECK(call)                                                                            \
    do {                                                                                           \
        int mpi_err = call;                                                                        \
        if (mpi_err != MPI_SUCCESS) {                                                              \
            fprintf(stderr, "MPI FAIL: call = '%s' result = '%d'\n", #call, mpi_err);              \
        }                                                                                          \
    } while (0)
#else
#define MPI_CHECK(call)                                                                            \
    fprintf(stderr, "Failure - Attempting to use MPI when it's not configured\n");
#endif

void runtime_mpi_init()
{
    MPI_CHECK(MPI_Init(NULL, NULL));
}

void runtime_mpi_finalize()
{
    MPI_CHECK(MPI_Finalize());
}

void *runtime_mpi_calloc(size_t num, size_t size)
{
    return calloc(num, size);
}

void *runtime_mpi_malloc(size_t size)
{
    return malloc(size);
}

void runtime_mpi_free(void *ptr)
{
    free(ptr);
}

void runtime_mpi_sync_all()
{
    /* TODO: Implement */
}

void runtime_mpi_broadcast(void *dst, void *src, size_t size, int root)
{
    MPI_CHECK(MPI_Bcast(dst, size, MPI_BYTE, root, MPI_COMM_WORLD));
}

void runtime_mpi_uint64_sum_reduce(uint64_t *dst, uint64_t *src, size_t num)
{
    MPI_CHECK(MPI_Allreduce(src, dst, num, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD));
}

#endif
