/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef NBI_IMPL_H
#define NBI_IMPL_H

/* Non-blocking Put */
template <typename T>
ISHMEM_DEVICE_ATTRIBUTES void ishmem_internal_put_nbi(T *dest, const T *src, size_t nelems, int pe)
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
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.src = src;
    req.dst = dest;
    req.nelems = nbytes;
    req.op = PUT_NBI;
    req.type = UINT8;

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_nonblocking_request(req);
#else
    int ret = 1;
    if (local_index != 0) ret = ishmemi_ipc_put_nbi(dest, src, nelems, pe);
    if (ret != 0) ishmemi_runtime->proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* Non-blocking Put (work-group) */
template <typename T, typename Group>
void ishmemx_internal_put_nbi_work_group(T *dest, const T *src, size_t nelems, int pe,
                                         const Group &grp)
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
                ishmemi_request_t req;
                req.dest_pe = pe;
                req.src = src;
                req.dst = dest;
                req.nelems = nbytes;
                req.op = PUT_NBI;
                req.type = UINT8;

                ishmemi_proxy_nonblocking_request(req);
            }
        }
    } else {
        RAISE_ERROR_MSG("ISHMEMX_PUT_NBI_WORK_GROUP routines are not callable from host\n");
    }
}

/* Non-blocking Get */
template <typename T>
ISHMEM_DEVICE_ATTRIBUTES void ishmem_internal_get_nbi(T *dest, const T *src, size_t nelems, int pe)
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
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.src = src;
    req.dst = dest;
    req.nelems = nbytes;
    req.op = GET_NBI;
    req.type = UINT8;

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_nonblocking_request(req);
#else
    int ret = 1;
    if (local_index != 0) ret = ishmemi_ipc_get_nbi(dest, src, nelems, pe);
    if (ret != 0) ishmemi_runtime->proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* Non-blocking Get (work-group) */
template <typename T, typename Group>
void ishmemx_internal_get_nbi_work_group(T *dest, const T *src, size_t nelems, int pe,
                                         const Group &grp)
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
                ishmemi_request_t req;
                req.dest_pe = pe;
                req.src = src;
                req.dst = dest;
                req.nelems = nbytes;
                req.op = GET_NBI;
                req.type = UINT8;

                ishmemi_proxy_nonblocking_request(req);
            }
        }
    } else {
        RAISE_ERROR_MSG("ISHMEMX_GET_NBI_WORK_GROUP routines are not callable from host\n");
    }
}

#endif /* RMA_IMPL_H */
