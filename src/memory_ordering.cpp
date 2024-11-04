/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_MEMORY_ORDERING_H
#define ISHMEM_MEMORY_ORDERING_H

#include "proxy_impl.h"
#include "runtime.h"
#include "on_queue.h"

void ishmem_fence()
{
    if constexpr (enable_error_checking) validate_init();
    ishmemi_request_t req;
    req.op = FENCE;
    req.type = NONE;

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
    atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
#else
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
#endif
}

template void ishmemx_fence_work_group<sycl::group<1>>(const sycl::group<1> &grp);
template void ishmemx_fence_work_group<sycl::group<2>>(const sycl::group<2> &grp);
template void ishmemx_fence_work_group<sycl::group<3>>(const sycl::group<3> &grp);
template void ishmemx_fence_work_group<sycl::sub_group>(const sycl::sub_group &grp);
template <typename Group>
void ishmemx_fence_work_group(const Group &grp)
{
    if constexpr (enable_error_checking) validate_init();
    if constexpr (ishmemi_is_device) {
        sycl::group_barrier(grp);
        if (grp.leader()) {
            ishmemi_request_t req;
            req.op = FENCE;
            req.type = NONE;

            ishmemi_proxy_blocking_request(req);
        }
        atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_fence_work_group not callable from host\n");
    }
}

void ishmem_quiet()
{
    if constexpr (enable_error_checking) validate_init();
    ishmemi_request_t req;
    req.op = QUIET;
    req.type = NONE;

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
    atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
#else
    ishmemi_drain_ring();
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
#endif
}

sycl::event ishmemx_quiet_on_queue(sycl::queue &q, const std::vector<sycl::event> &deps)
{
    if constexpr (enable_error_checking) validate_init();
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        cgh.host_task([=]() { ishmem_quiet(); });
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

template void ishmemx_quiet_work_group<sycl::group<1>>(const sycl::group<1> &grp);
template void ishmemx_quiet_work_group<sycl::group<2>>(const sycl::group<2> &grp);
template void ishmemx_quiet_work_group<sycl::group<3>>(const sycl::group<3> &grp);
template void ishmemx_quiet_work_group<sycl::sub_group>(const sycl::sub_group &grp);
template <typename Group>
void ishmemx_quiet_work_group(const Group &grp)
{
    if constexpr (enable_error_checking) validate_init();
    if constexpr (ishmemi_is_device) {
        sycl::group_barrier(grp);
        if (grp.leader()) {
            ishmemi_request_t req;
            req.op = QUIET;
            req.type = NONE;

            ishmemi_proxy_blocking_request(req);
        }
        atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_quiet_work_group not callable from host\n");
    }
}

#endif  // ISHMEM_MEMORY_ORDERING_H
