/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/err.h"
#include "sync_impl.h"
#include "proxy_impl.h"
#include "collectives.h"

void ishmem_barrier_all()
{
    static_assert(ishmemi_sync_algorithm == SYNC_ALGORITHM_ATOMIC_EXCHANGE ||
                  ishmemi_sync_algorithm == SYNC_ALGORITHM_BITMAP ||
                  ishmemi_sync_algorithm == SYNC_ALGORITHM_ATOMIC_ADD ||
                  ishmemi_sync_algorithm == SYNC_ALGORITHM_STORE);

    ishmemi_request_t req;
    req.op = BARRIER;
    req.type = MEM;
#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_info_t *info = global_info;
    if (info->only_intra_node) req.op = QUIET;
    ishmemi_proxy_blocking_request(req);
    if (info->only_intra_node) {
        if constexpr (ishmemi_sync_algorithm == SYNC_ALGORITHM_ATOMIC_EXCHANGE) {
            ISHMEMI_SYNC_LOCAL_PES_ATOMIC_EXCHANGE_DEVICE(barrier);
        } else if constexpr (ishmemi_sync_algorithm == SYNC_ALGORITHM_BITMAP) {
            ISHMEMI_SYNC_LOCAL_PES_BITMAP_DEVICE(barrier);
        } else if constexpr (ishmemi_sync_algorithm == SYNC_ALGORITHM_ATOMIC_ADD) {
            ISHMEMI_SYNC_LOCAL_PES_ATOMIC_ADD_DEVICE(barrier);
        } else if constexpr (ishmemi_sync_algorithm == SYNC_ALGORITHM_STORE) {
            ISHMEMI_SYNC_LOCAL_PES_STORE_DEVICE(barrier);
        }
    }
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
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
