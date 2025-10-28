/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef COLLECTIVES_SCAN_IMPL_H
#define COLLECTIVES_SCAN_IMPL_H

#include "collectives.h"
#include "runtime.h"
#include "proxy_impl.h"
#include "on_queue.h"

template <typename T, ishmemi_op_t OP>
int scan_impl(ishmem_team_t team, T *dest, const T *src, size_t nelems)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) dest, (void *) src, nelems * sizeof(T));
    }

    ishmemi_request_t req;
    req.src = src;
    req.dst = dest;
    req.nelems = nelems;
    req.op = OP;
    req.type = ishmemi_union_get_base_type<T, OP>();
    req.team = team;

#ifdef __SYCL_DEVICE_ONLY__
    return ishmemi_proxy_blocking_request_status(req);
#else
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    return ishmemi_proxy_get_status(comp.completion.ret);
#endif
}

template <typename T>
int ishmem_sum_inscan(ishmem_team_t team, T *dest, const T *src, size_t nelems)
{
    return scan_impl<T, INSCAN>(team, dest, src, nelems);
}

template <typename T>
int ishmem_sum_exscan(ishmem_team_t team, T *dest, const T *src, size_t nelems)
{
    return scan_impl<T, EXSCAN>(team, dest, src, nelems);
}

template <typename T, ishmemi_op_t OP>
sycl::event scan_on_queue_impl(ishmem_team_t team, T *dest, const T *src, size_t nelems, int *ret,
                               sycl::queue &q, const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        cgh.single_task([=]() {
            int tmp_ret = scan_impl<T, OP>(team, dest, src, nelems);
            if (ret) *ret = tmp_ret;
        });
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

template <typename T>
sycl::event ishmemx_sum_inscan_on_queue(ishmem_team_t team, T *dest, const T *src, size_t nelems,
                                        int *ret, sycl::queue &q,
                                        const std::vector<sycl::event> &deps)
{
    return scan_on_queue_impl<T, INSCAN>(team, dest, src, nelems, ret, q, deps);
}

template <typename T>
sycl::event ishmemx_sum_inscan_on_queue(T *dest, const T *src, size_t nelems, int *ret,
                                        sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return scan_on_queue_impl<T, INSCAN>(ISHMEM_TEAM_WORLD, dest, src, nelems, ret, q, deps);
}

template <typename T>
sycl::event ishmemx_sum_exscan_on_queue(ishmem_team_t team, T *dest, const T *src, size_t nelems,
                                        int *ret, sycl::queue &q,
                                        const std::vector<sycl::event> &deps)
{
    return scan_on_queue_impl<T, EXSCAN>(team, dest, src, nelems, ret, q, deps);
}

template <typename T>
sycl::event ishmemx_sum_exscan_on_queue(T *dest, const T *src, size_t nelems, int *ret,
                                        sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return scan_on_queue_impl<T, EXSCAN>(ISHMEM_TEAM_WORLD, dest, src, nelems, ret, q, deps);
}

#endif
