/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_TYPES_H
#define ISHMEM_RUNTIME_TYPES_H

/* This file defines common types that are shared across ishmem runtime backends. */

#if defined(ENABLE_OPENSHMEM)
#include <shmem.h>
#include <shmemx.h>
#define ISHMEMI_RUNTIME_TEAM_NODE SHMEMX_TEAM_NODE
typedef shmem_team_t ishmemi_runtime_team_t;
typedef shmem_team_config_t ishmemi_runtime_team_config_t;

#else
#define ISHMEMI_RUNTIME_TEAM_NODE ISHMEM_TEAM_INVALID
struct ishmemi_runtime_team_t {
    int dummy = -1;
};
typedef struct ishmemi_runtime_team_t ishmemi_runtime_team_t;
typedef ishmem_team_config_t ishmemi_runtime_team_config_t;
#endif

#endif /* ISHMEM_RUNTIME_TYPES_H */
