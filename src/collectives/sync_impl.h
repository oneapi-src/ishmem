/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef COLLECTIVES_SYNC_IMPL_H
#define COLLECTIVES_SYNC_IMPL_H

#include "ishmem/err.h"
#include "ishmem/types.h"
#include "proxy_impl.h"
#include "collectives.h"
#include "runtime.h"
#include "teams.h"

typedef enum {
    SYNC_ALGORITHM_ATOMIC_EXCHANGE,
    SYNC_ALGORITHM_BITMAP,
    SYNC_ALGORITHM_ATOMIC_ADD,
    SYNC_ALGORITHM_STORE
} ishmemi_sync_algorithm_t;

constexpr ishmemi_sync_algorithm_t ishmemi_sync_algorithm = SYNC_ALGORITHM_ATOMIC_EXCHANGE;

#define ISHMEMI_SYNC_LOCAL_PES_ATOMIC_EXCHANGE_DEVICE(OPERATION)                                   \
    ishmemi_info_t *info = global_info;                                                            \
    int index = info->OPERATION##_index;                                                           \
    long *my_psync =                                                                               \
        info->OPERATION##_all_psync[info->OPERATION##_index & (ISHMEM_SYNC_NUM_PSYNC_ARRS - 1)];   \
    long n_local_pes = (long) info->n_local_pes;                                                   \
    for (int i = 0; i < n_local_pes; i += 1) {                                                     \
        long *psync = ISHMEMI_ADJUST_PTR(long, (i + 1), my_psync);                                 \
        /* These atomics can be relaxed because we don't care about their ordering */              \
        sycl::atomic_ref<long, sycl::memory_order::relaxed, sycl::memory_scope::system,            \
                         sycl::access::address_space::global_space>                                \
            atomic_psync(*psync);                                                                  \
        atomic_psync += 1L; /* atomic increment info->ipc_buffers[pOffset] */                      \
    }                                                                                              \
    info->OPERATION##_index = index + 1;                                                           \
    /* This atomic has to be seq_cst because we definitely want it to happen in order */           \
    sycl::atomic_ref<long, sycl::memory_order::seq_cst, sycl::memory_scope::system,                \
                     sycl::access::address_space::global_space>                                    \
        atomic_psync(*my_psync);                                                                   \
    long expected;                                                                                 \
    do {                                                                                           \
        expected = n_local_pes;                                                                    \
    } while (!atomic_psync.compare_exchange_strong(expected, 0L, sycl::memory_order::seq_cst,      \
                                                   sycl::memory_order::seq_cst));

#define ISHMEMI_TEAM_SYNC_ATOMIC_EXCHANGE_DEVICE(team)                                             \
    long *my_psync = ishmemi_team_choose_psync(team, TEAM_OP_SYNC);                                \
    int idx = 0;                                                                                   \
    for (int pe = team->start; idx < team->size; pe += team->stride, idx++) {                      \
        long *psync = ISHMEMI_ADJUST_PTR(long, (pe + 1), my_psync);                                \
        /* These atomics can be relaxed because we don't care about their ordering */              \
        sycl::atomic_ref<long, sycl::memory_order::relaxed, sycl::memory_scope::system,            \
                         sycl::access::address_space::global_space>                                \
            atomic_psync(*psync);                                                                  \
        atomic_psync += 1L; /* atomic increment info->ipc_buffers[pOffset] */                      \
    }                                                                                              \
    /* This atomic has to be seq_cst because we definitely want it to happen in order */           \
    sycl::atomic_ref<long, sycl::memory_order::seq_cst, sycl::memory_scope::system,                \
                     sycl::access::address_space::global_space>                                    \
        atomic_psync(*my_psync);                                                                   \
    long expected;                                                                                 \
    do {                                                                                           \
        expected = team->size;                                                                     \
    } while (!atomic_psync.compare_exchange_strong(expected, 0L, sycl::memory_order::seq_cst,      \
                                                   sycl::memory_order::seq_cst));                  \
    ishmemi_team_release_psyncs(team, TEAM_OP_SYNC);

#define ISHMEMI_SYNC_LOCAL_PES_BITMAP_DEVICE(OPERATION)                                            \
    ishmemi_info_t *info = global_info;                                                            \
    int index = info->OPERATION##_index;                                                           \
    long *my_psync =                                                                               \
        info->OPERATION##_all_psync[info->OPERATION##_index & (ISHMEM_SYNC_NUM_PSYNC_ARRS - 1)];   \
    long n_local_pes = (long) info->n_local_pes;                                                   \
    for (int i = 0; i < n_local_pes; i += 1) {                                                     \
        long *psync = ISHMEMI_ADJUST_PTR(long, (i + 1), my_psync);                                 \
        /* These atomics can be relaxed because we don't care about their ordering */              \
        sycl::atomic_ref<long, sycl::memory_order::relaxed, sycl::memory_scope::system,            \
                         sycl::access::address_space::global_space>                                \
            atomic_psync(*psync);                                                                  \
        atomic_psync |= 1L << info->local_rank; /* atomic or */                                    \
    }                                                                                              \
    info->OPERATION##_index = index + 1;                                                           \
    /* This atomic has to be seq_cst because we definitely want it to happen in order */           \
    sycl::atomic_ref<long, sycl::memory_order::seq_cst, sycl::memory_scope::system,                \
                     sycl::access::address_space::global_space>                                    \
        atomic_psync(*my_psync);                                                                   \
    while (atomic_psync != ((1L << n_local_pes) - 1))                                              \
        ;                                                                                          \
    atomic_psync = 0L;

#define ISHMEMI_SYNC_LOCAL_PES_ATOMIC_ADD_DEVICE(OPERATION)                                        \
    ishmemi_info_t *info = global_info;                                                            \
    int index = info->OPERATION##_index;                                                           \
    long *my_psync = info->OPERATION##_all_psync[index & (ISHMEM_SYNC_NUM_PSYNC_ARRS - 1)];        \
    long n_local_pes = info->n_local_pes;                                                          \
    for (int i = 0; i < n_local_pes; i += 1) {                                                     \
        long *psync = ISHMEMI_ADJUST_PTR(long, (i + 1), my_psync);                                 \
        /* These atomics can be relaxed because we don't care about their ordering */              \
        sycl::atomic_ref<long, sycl::memory_order::relaxed, sycl::memory_scope::system,            \
                         sycl::access::address_space::global_space>                                \
            atomic_psync(*psync);                                                                  \
        atomic_psync += 1L; /* atomic increment info->ipc_buffers[pOffset] */                      \
    }                                                                                              \
    info->OPERATION##_index = index + 1;                                                           \
    /* This atomic has to be seq_cst because we definitely want it to happen in order */           \
    sycl::atomic_ref<long, sycl::memory_order::seq_cst, sycl::memory_scope::system,                \
                     sycl::access::address_space::global_space>                                    \
        atomic_psync(*my_psync);                                                                   \
    while (atomic_psync != n_local_pes)                                                            \
        ; /* wait for heap_base[pOffset] to be local_size */                                       \
    atomic_psync = 0;

#define ISHMEMI_SYNC_LOCAL_PES_STORE_DEVICE(OPERATION)                                             \
    ishmemi_info_t *info = global_info;                                                            \
    int index = info->OPERATION##_index;                                                           \
    long *my_psync = info->OPERATION##_all_psync[index & (ISHMEM_SYNC_NUM_PSYNC_ARRS - 1)];        \
    long n_local_pes = info->n_local_pes;                                                          \
    for (int i = 0; i < n_local_pes; i += 1) {                                                     \
        long *psync = ISHMEMI_ADJUST_PTR(long, (i + 1), &my_psync[info->my_pe]);                   \
        *psync = index;                                                                            \
    }                                                                                              \
    info->OPERATION##_index = index + 1;                                                           \
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);                   \
    for (int i = 0; i < n_local_pes; i += 1) {                                                     \
        sycl::atomic_ref<long, sycl::memory_order::seq_cst, sycl::memory_scope::system,            \
                         sycl::access::address_space::global_space>                                \
            atomic_psync(my_psync[i]);                                                             \
        while (atomic_psync != index)                                                              \
            ;                                                                                      \
    }

ISHMEM_DEVICE_ATTRIBUTES int ishmemi_team_sync(ishmemi_team_t *team);

static inline void sync_fallback()
{
    ishmemi_request_t req;
    req.op = SYNC;
    req.type = MEM;

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

static inline void sync_team_fallback(ishmemi_team_t *team)
{
    ishmemi_request_t req;
    ishmemi_ringcompletion_t comp __attribute__((unused));
    req.op = TEAM_SYNC;
    req.type = MEM;
    req.team = team;

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
    atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
#endif
}

inline void ishmemi_sync_atomic_exchange()
{
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (info->n_local_pes == info->n_pes) {
            ISHMEMI_SYNC_LOCAL_PES_ATOMIC_EXCHANGE_DEVICE(sync);
            return;
        }
    }

    /* Otherwise */
    sync_fallback();
}

inline void ishmemi_team_sync_atomic_exchange(ishmemi_team_t *team)
{
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (info->n_local_pes == team->size) {
            ISHMEMI_TEAM_SYNC_ATOMIC_EXCHANGE_DEVICE(team);
            return;
        }
    }

    /* Otherwise */
    sync_team_fallback(team);
}

inline void ishmemi_sync_bitmap()
{
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (info->n_local_pes == info->n_pes) {
            ISHMEMI_SYNC_LOCAL_PES_BITMAP_DEVICE(sync);
            return;
        }
    }

    /* Otherwise */
    sync_fallback();
}

inline void ishmemi_sync_atomic_add()
{
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (info->n_local_pes == info->n_pes) {
            ISHMEMI_SYNC_LOCAL_PES_ATOMIC_ADD_DEVICE(sync);
            return;
        }
    }

    /* Otherwise */
    sync_fallback();
}

inline void ishmemi_sync_store()
{
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (info->n_local_pes == info->n_pes) {
            ISHMEMI_SYNC_LOCAL_PES_STORE_DEVICE(sync);
            return;
        }
    }

    /* Otherwise */
    sync_fallback();
}

#endif  // ifndef COLLECTIVES_SYNC_IMPL_H
