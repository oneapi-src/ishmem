/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_COLLECTIVES_H
#define ISHMEM_COLLECTIVES_H

#include "runtime_ipc.h"

#define ISHMEM_REDUCE_BUFFER_SIZE  (1L << 16)
#define ISHMEM_SYNC_NUM_PSYNC_ARRS 4

extern size_t *ishmemi_collect_sizes;
extern size_t *ishmemi_my_collect_size;

int ishmemi_collectives_init();
int ishmemi_collectives_fini();

#endif  // ifndef  ISHMEM_COLLECTIVES_H
