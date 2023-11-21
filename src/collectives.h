/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_COLLECTIVES_H
#define ISHMEM_COLLECTIVES_H
#include "internal.h"     // globals
#include "runtime_ipc.h"  // get and put
#include "impl_proxy.h"

#define ISHMEM_REDUCE_BUFFER_SIZE (1L << 16)

#include "collectives/reduce_impl.h"
#include "collectives/broadcast_impl.h"

extern size_t *ishmemi_collect_sizes;
extern size_t *ishmemi_my_collect_size;

int ishmemi_collectives_init();
int ishmemi_collectives_fini();

typedef enum {
    SYNC_ALGORITHM_ATOMIC_EXCHANGE,
    SYNC_ALGORITHM_BITMAP,
    SYNC_ALGORITHM_ATOMIC_ADD,
    SYNC_ALGORITHM_STORE
} ishmemi_sync_algorithm_t;

#endif  // ifndef  ISHMEM_COLLECTIVES_H
