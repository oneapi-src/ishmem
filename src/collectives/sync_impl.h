/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef COLLECTIVES_SYNC_IMPL_H
#define COLLECTIVES_SYNC_IMPL_H

#include "ishmem/types.h"
#include "proxy_impl.h"
#include "collectives.h"
#include "runtime.h"
#include "teams.h"

static inline void sync_team_fallback(ishmem_team_t team)
{
    ishmemi_request_t req;
    ishmemi_ringcompletion_t comp __attribute__((unused));
    req.op = TEAM_SYNC;
    req.type = NONE;
    req.team = team;

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
    atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
#else
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
#endif
}

ISHMEM_DEVICE_ATTRIBUTES inline void ishmemi_team_sync(ishmem_team_t team)
{
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        ishmemi_team_device_t *team_ptr = &info->team_device_pool[team];
        if (team_ptr->only_intra) {
            sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);

            int index = team_ptr->psync_idx;
            long *my_psync = &team_ptr->psync[team_ptr->psync_idx];
            team_ptr->psync_idx = (index + 1) % N_PSYNCS_PER_TEAM;

            /* This atomic has to be seq_cst because we definitely want it to happen in order */
            sycl::atomic_ref<long, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_psync(*my_psync);

            for (int i = team_ptr->start; i <= team_ptr->last_pe; i += team_ptr->stride) {
                uint8_t local_index = ISHMEMI_LOCAL_PES[i];
                long *remote_psync = ISHMEMI_FAST_ADJUST(long, info, local_index, my_psync);

                /* These atomics can be relaxed because we don't care about their ordering */
                sycl::atomic_ref<long, sycl::memory_order::relaxed, sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>
                    atomic_psync(*remote_psync);
                atomic_psync += 1L;
            }

            while (atomic_psync.load() != team_ptr->size)
                ;
            atomic_psync.store(0);

            return;
        }
    }

    /* Otherwise */
    sync_team_fallback(team);
}

#endif
