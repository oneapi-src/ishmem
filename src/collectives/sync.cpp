/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "sync_impl.h"

void ishmem_sync_all()
{
    static_assert(ishmemi_sync_algorithm == SYNC_ALGORITHM_ATOMIC_EXCHANGE ||
                  ishmemi_sync_algorithm == SYNC_ALGORITHM_BITMAP ||
                  ishmemi_sync_algorithm == SYNC_ALGORITHM_ATOMIC_ADD ||
                  ishmemi_sync_algorithm == SYNC_ALGORITHM_STORE);
    if constexpr (ishmemi_sync_algorithm == SYNC_ALGORITHM_ATOMIC_EXCHANGE) {
        ishmemi_sync_atomic_exchange();
    } else if constexpr (ishmemi_sync_algorithm == SYNC_ALGORITHM_BITMAP) {
        ishmemi_sync_bitmap();
    } else if constexpr (ishmemi_sync_algorithm == SYNC_ALGORITHM_ATOMIC_ADD) {
        ishmemi_sync_atomic_add();
    } else if constexpr (ishmemi_sync_algorithm == SYNC_ALGORITHM_STORE) {
        ishmemi_sync_store();
    }
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
        if (grp.leader()) ishmem_sync_all();
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_sync_all_work_group not callable from host\n");
    }
}

ISHMEM_DEVICE_ATTRIBUTES void ishmem_exp_sync_atomic_exchange()
{
    ishmemi_sync_atomic_exchange();
}

ISHMEM_DEVICE_ATTRIBUTES void ishmem_exp_sync_bitmap()
{
    ishmemi_sync_bitmap();
}

ISHMEM_DEVICE_ATTRIBUTES void ishmem_exp_sync_atomic_add()
{
    ishmemi_sync_atomic_add();
}

ISHMEM_DEVICE_ATTRIBUTES void ishmem_exp_sync_store()
{
    ishmemi_sync_store();
}
