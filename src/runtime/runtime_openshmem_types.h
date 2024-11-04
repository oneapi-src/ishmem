/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_OPENSHMEM_TYPES_H
#define ISHMEM_RUNTIME_OPENSHMEM_TYPES_H

/* OpenSHMEM type wrappers needed for ishmem definitions */
#include "ishmem/config.h"
#include <shmem.h>

namespace ishmemi_runtime_openshmem_types {
    typedef shmem_team_t team_t;
    typedef shmem_team_config_t team_config_t;
}  // namespace ishmemi_runtime_openshmem_types

#endif /* ISHMEM_RUNTIME_OPENSHMEM_TYPES_H */
