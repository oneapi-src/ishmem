/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef RMA_IMPL_H
#define RMA_IMPL_H

/* Put */
template <typename T>
ISHMEM_DEVICE_ATTRIBUTES void ishmem_internal_put(T *dest, const T *src, size_t nelems, int pe)
{
    size_t nbytes = nelems * sizeof(T);

    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, (void *) src, nbytes);
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        if ((local_index != 0) && !ISHMEM_RMA_CUTOVER) {
            vec_copy_push(ISHMEMI_ADJUST_PTR(T, local_index, dest), src, nelems);
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req = {
        .op = PUT,
        .type = UINT8,
        .dest_pe = pe,
        .src = src,
        .dst = dest,
        .nelems = nbytes,
    };

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(&req);
#else
    int ret = 1;
    ret = ishmemi_ipc_put(dest, src, nelems, pe);
    if (ret != 0) ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* Put (work-group) */
template <typename T, typename Group>
void ishmemx_internal_put_work_group(T *dest, const T *src, size_t nelems, int pe, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        size_t nbytes = nelems * sizeof(T);
        sycl::group_barrier(grp);
        if constexpr (enable_error_checking) {
            if (grp.leader()) validate_parameters(pe, (void *) dest, (void *) src, nbytes);
        }
        uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
        if ((local_index != 0) && !ISHMEM_RMA_GROUP_CUTOVER) {
            vec_copy_work_group_push(ISHMEMI_ADJUST_PTR(T, local_index, dest), src, nelems, grp);
        } else {
            if (grp.leader()) {
                ishmemi_request_t req = {
                    .op = PUT,
                    .type = UINT8,
                    .dest_pe = pe,
                    .src = src,
                    .dst = dest,
                    .nelems = nbytes,
                };

                ishmemi_proxy_blocking_request(&req);
            }
        }
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_put_work_group not callable from host\n");
    }
}

/* Get */
template <typename T>
ISHMEM_DEVICE_ATTRIBUTES void ishmem_internal_get(T *dest, const T *src, size_t nelems, int pe)
{
    size_t nbytes = nelems * sizeof(T);

    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) src, (void *) dest, nbytes);
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        if ((local_index != 0) && !ISHMEM_RMA_CUTOVER) {
            vec_copy_pull(dest, ISHMEMI_ADJUST_PTR(T, local_index, src), nelems);
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req = {
        .op = GET,
        .type = UINT8,
        .dest_pe = pe,
        .src = src,
        .dst = dest,
        .nelems = nbytes,
    };

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(&req);
#else
    int ret = 1;
    ret = ishmemi_ipc_get(dest, src, nelems, pe);
    if (ret != 0) ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* Get (work-group) */
template <typename T, typename Group>
void ishmemx_internal_get_work_group(T *dest, const T *src, size_t nelems, int pe, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        size_t nbytes = nelems * sizeof(T);
        sycl::group_barrier(grp);
        if constexpr (enable_error_checking) {
            if (grp.leader()) validate_parameters(pe, (void *) src, (void *) dest, nbytes);
        }
        uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
        if ((local_index != 0) && !ISHMEM_RMA_GROUP_CUTOVER) {
            vec_copy_work_group_pull(dest, ISHMEMI_ADJUST_PTR(T, local_index, src), nelems, grp);
        } else {
            if (grp.leader()) {
                ishmemi_request_t req = {
                    .op = GET,
                    .type = UINT8,
                    .dest_pe = pe,
                    .src = src,
                    .dst = dest,
                    .nelems = nbytes,
                };

                ishmemi_proxy_blocking_request(&req);
            }
        }
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_put_work_group not callable from host\n");
    }
}

#endif /* RMA_IMPL_H */
