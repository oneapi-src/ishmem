/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem.h"
#include "ishmemx.h"
#include "ishmem/types.h"
#include "proxy_impl.h"
#include "collectives.h"
#include "runtime.h"
#include "teams.h"
#include "on_queue.h"

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

inline void ishmemi_team_sync(ishmem_team_t team)
{
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        ishmemi_team_device_t *team_ptr = &info->team_device_pool[team];
        if (team_ptr->only_intra) {
            int index = team_ptr->psync_idx;
            long *my_psync = &team_ptr->psync[index];
            int last_i = team_ptr->last_pe + 1;
            int stride = team_ptr->stride;
            for (int i = team_ptr->start + 1; i <= last_i; i += stride) {
                long *psync = ISHMEMI_FAST_ADJUST(long, info, i, my_psync);
                /* These atomics can be relaxed because we don't care about their ordering */
                sycl::atomic_ref<long, sycl::memory_order::relaxed, sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>
                    atomic_psync(*psync);
                atomic_psync += 1L; /* atomic increment info->ipc_buffers[pOffset] */
            }
            team_ptr->psync_idx = (index + 1) & (ISHMEM_SYNC_NUM_PSYNC_ARRS - 1);
            /* This atomic has to be seq_cst because we definitely want it to happen in order */
            sycl::atomic_ref<long, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_psync(*my_psync);
            long expected;
            int size = team_ptr->size;
            do {
                expected = size;
            } while (!atomic_psync.compare_exchange_strong(
                expected, 0L, sycl::memory_order::seq_cst, sycl::memory_order::seq_cst));
            return;
        }
    }

    /* Otherwise */
    sync_team_fallback(team);
}

void ishmem_sync_all()
{
    ishmemi_team_sync(ISHMEM_TEAM_WORLD);
}

sycl::event ishmemx_sync_all_on_queue(sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return ishmemx_team_sync_on_queue(ISHMEM_TEAM_WORLD, NULL, q, deps);
}

template void ishmemx_sync_all_work_group<sycl::group<1>>(const sycl::group<1> &grp);
template void ishmemx_sync_all_work_group<sycl::group<2>>(const sycl::group<2> &grp);
template void ishmemx_sync_all_work_group<sycl::group<3>>(const sycl::group<3> &grp);
template void ishmemx_sync_all_work_group<sycl::sub_group>(const sycl::sub_group &grp);
template <typename Group>
void ishmemx_sync_all_work_group(const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        sycl::group_barrier(grp);
        if (grp.leader()) ishmemi_team_sync(ISHMEM_TEAM_WORLD);
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_sync_all_work_group not callable from host\n");
    }
}

ISHMEM_DEVICE_ATTRIBUTES int ishmem_team_sync(ishmem_team_t team)
{
    if constexpr (enable_error_checking) {
        if (team <= ISHMEM_TEAM_INVALID || team >= ISHMEMI_N_TEAMS) return -1;
    }
    ishmemi_team_sync(team);
    return 0;
}

sycl::event ishmemx_team_sync_on_queue(ishmem_team_t team, int *ret, sycl::queue &q,
                                       const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    ishmemi_team_host_t *myteam = &ishmemi_cpu_info->team_host_pool[team];
    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        if (myteam->only_intra) {
            cgh.single_task([=]() {
                int tmp_ret = ishmem_team_sync(team);
                if (ret) *ret = tmp_ret;
            });
        } else {
            cgh.host_task([=]() {
                int tmp_ret = ishmem_team_sync(team);
                if (ret) *ret = tmp_ret;
            });
        }
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

template void ishmemx_team_sync_work_group<sycl::group<1>>(ishmem_team_t team,
                                                           const sycl::group<1> &grp);
template void ishmemx_team_sync_work_group<sycl::group<2>>(ishmem_team_t team,
                                                           const sycl::group<2> &grp);
template void ishmemx_team_sync_work_group<sycl::group<3>>(ishmem_team_t team,
                                                           const sycl::group<3> &grp);
template void ishmemx_team_sync_work_group<sycl::sub_group>(ishmem_team_t team,
                                                            const sycl::sub_group &grp);
template <typename Group>
void ishmemx_team_sync_work_group(ishmem_team_t team, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        sycl::group_barrier(grp);
        if (grp.leader()) ishmemi_team_sync(team);
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_team_sync_work_group not callable from host\n");
    }
}
