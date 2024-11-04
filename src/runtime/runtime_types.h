/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_TYPES_H
#define ISHMEM_RUNTIME_TYPES_H

#include "ishmem/config.h"
/* This file defines common types that are shared across ishmem runtime backends. */
#if defined(ENABLE_OPENSHMEM)
#include "runtime_openshmem_types.h"
#define OPENSHMEM_CLASS(TYPE) ishmemi_runtime_openshmem_types::TYPE shmem
#else
#define OPENSHMEM_CLASS(TYPE)
#endif

#if defined(ENABLE_MPI)
#include "runtime_mpi_types.h"
#define MPI_CLASS(TYPE) ishmemi_runtime_mpi_types::TYPE mpi
#else
#define MPI_CLASS(TYPE)
#endif

#if defined(ENABLE_PMI)
#include "runtime_pmi_types.h"
#define PMI_CLASS(TYPE) ishmemi_runtime_pmi_types::##TYPE pmi
#else
#define PMI_CLASS(TYPE)
#endif

/* Teams definitions */
typedef union {
    OPENSHMEM_CLASS(team_t);
    MPI_CLASS(team_t);
    PMI_CLASS(team_t);
} ishmemi_runtime_team_t;

typedef union {
    OPENSHMEM_CLASS(team_config_t);
    MPI_CLASS(team_config_t);
    PMI_CLASS(team_config_t);
} ishmemi_runtime_team_config_t;

#endif /* ISHMEM_RUNTIME_TYPES_H */
