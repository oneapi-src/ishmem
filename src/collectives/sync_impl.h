/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "internal.h"
#include "impl_proxy.h"
#include "collectives.h"
#include "runtime.h"

constexpr ishmemi_sync_algorithm_t ishmemi_sync_algorithm = SYNC_ALGORITHM_ATOMIC_EXCHANGE;

#define ISHMEMI_SYNC_LOCAL_PES_ATOMIC_EXCHANGE_DEVICE(OPERATION)                                   \
    ishmem_info_t *info = global_info;                                                             \
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

#define ISHMEMI_SYNC_LOCAL_PES_BITMAP_DEVICE(OPERATION)                                            \
    ishmem_info_t *info = global_info;                                                             \
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
    ishmem_info_t *info = global_info;                                                             \
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
    ishmem_info_t *info = global_info;                                                             \
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

static inline void sync_fallback()
{
    ishmemi_request_t req = {
        .op = SYNC,
        .type = MEM,
    };

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(&req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

inline void ishmemi_sync_atomic_exchange()
{
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmem_info_t *info = global_info;
        if (info->n_local_pes == info->n_pes) {
            ISHMEMI_SYNC_LOCAL_PES_ATOMIC_EXCHANGE_DEVICE(sync);
            return;
        }
    }

    /* Otherwise */
    sync_fallback();
}

inline void ishmemi_sync_bitmap()
{
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmem_info_t *info = global_info;
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
        ishmem_info_t *info = global_info;
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
        ishmem_info_t *info = global_info;
        if (info->n_local_pes == info->n_pes) {
            ISHMEMI_SYNC_LOCAL_PES_STORE_DEVICE(sync);
            return;
        }
    }

    /* Otherwise */
    sync_fallback();
}
