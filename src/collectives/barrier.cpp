/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/err.h"
#include "proxy_impl.h"
#include "collectives.h"
#include "sync_impl.h"
#include "runtime.h"
#include "on_queue.h"

void ishmem_barrier_all()
{
    ishmemi_request_t req;
    req.type = NONE;
    req.op = BARRIER;
#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_info_t *info = global_info;
    if (info->only_intra_node) req.op = QUIET;
    ishmemi_proxy_blocking_request(req);
    if (info->only_intra_node) {
        ishmemi_team_sync(ISHMEM_TEAM_WORLD);
    }
#else
    ishmemi_drain_ring();
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

sycl::event ishmemx_barrier_all_on_queue(sycl::queue &q, const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        cgh.single_task([=]() { ishmem_barrier_all(); });
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

template void ishmemx_barrier_all_work_group<sycl::group<1>>(const sycl::group<1> &grp);
template void ishmemx_barrier_all_work_group<sycl::group<2>>(const sycl::group<2> &grp);
template void ishmemx_barrier_all_work_group<sycl::group<3>>(const sycl::group<3> &grp);
template void ishmemx_barrier_all_work_group<sycl::sub_group>(const sycl::sub_group &grp);
template <typename Group>
void ishmemx_barrier_all_work_group(const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        sycl::group_barrier(grp);
        if (grp.leader()) ishmem_barrier_all();
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_barrier_all_work_group not callable from host\n");
    }
}
