/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem.h"
#include "ishmemx.h"
#include "ishmem/types.h"
#include "proxy_impl.h"
#include "collectives.h"
#include "sync_impl.h"
#include "runtime.h"
#include "teams.h"
#include "on_queue.h"

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
                ishmemi_team_sync(team);
                if (ret) *ret = 0;
            });
        } else {
            cgh.single_task([=]() {
                ishmemi_team_sync(team);
                if (ret) *ret = 0;
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
