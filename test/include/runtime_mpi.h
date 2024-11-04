/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_TEST_RUNTIME_MPI_H
#define ISHMEM_TEST_RUNTIME_MPI_H

#include <mpi.h>
#include "runtime.h"

namespace ishmemi_test_mpi_wrappers {
    /* Setup APIs */
    extern void (*Init)(int *, char ***);
    extern void (*Finalize)(void);

    /* Operation APIs */
    extern void (*Barrier)(MPI_Comm);
    extern void (*Bcast)(void *, int, MPI_Datatype, int, MPI_Comm);
    extern void (*Allreduce)(void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm);
}  // namespace ishmemi_test_mpi_wrappers

class ishmemi_test_runtime_mpi : public ishmemi_test_runtime_type {
  public:
    ishmemi_test_runtime_mpi(void);
    ~ishmemi_test_runtime_mpi(void);

    /* Helper APIs */
    ishmemx_runtime_type_t get_type(void) override;

    /* Setup APIs */
    void init(void) override;
    void finalize(void) override;

    /* Memory APIs */
    void *calloc(size_t, size_t) override;
    void *malloc(size_t) override;
    void free(void *) override;

    /* Operation APIs */
    void sync(void) override;
    void broadcast(void *, void *, size_t, int) override;
    void uint64_sum_reduce(uint64_t *, uint64_t *, size_t) override;
    void float_sum_reduce(float *, float *, size_t) override;
};

#endif
