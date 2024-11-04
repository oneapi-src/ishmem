/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_MPI_TYPES_H
#define ISHMEM_RUNTIME_MPI_TYPES_H

/* MPI type wrappers needed for ishmem definitions */
#include "ishmem/config.h"
#include <mpi.h>

namespace ishmemi_runtime_mpi_types {
    typedef int team_t;
    typedef int team_config_t;
}  // namespace ishmemi_runtime_mpi_types

#endif /* ISHMEM_RUNTIME_MPI_TYPES_H */
