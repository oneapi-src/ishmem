/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef COLLECTIVES_ALLTOALL_IMPL_H
#define COLLECTIVES_ALLTOALL_IMPL_H

#include "collectives.h"
#include "collectives/sync_impl.h"
#include "runtime.h"

/* Alltoall */
template <typename T>
int ishmem_alltoall(T *dest, const T *src, size_t nelems)
{
    size_t nbytes = nelems * sizeof(T);
    if constexpr (enable_error_checking) {
        int n_pes = ishmem_n_pes();
        validate_parameters((void *) dest, (void *) src, nelems * sizeof(T) * n_pes, nbytes,
                            ishmemi_op_t::ALLTOALL);
    }

    int ret = 0;

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (info->only_intra_node && !ISHMEM_ALLTOALL_CUTOVER) {
            int n_local_pes = info->n_local_pes;
            const T *sptr[MAX_LOCAL_PES]; /* source pointer for each pe */
            T *dptr[MAX_LOCAL_PES];       /* destination pointer for each pe */
            /* compute our address of our section of dest in each PE */
            for (int local_pe = 0; local_pe < n_local_pes;
                 local_pe += 1) {  // index over target local_pe
                dptr[local_pe] = ISHMEMI_ADJUST_PTR(
                    T, (local_pe + 1), (&dest[nelems * static_cast<size_t>(info->local_rank)]));
                sptr[local_pe] = &src[nelems * static_cast<size_t>(local_pe)];
            }
            /* The idea for the inner loop being over local PEs is that the outstanding stores will
             * use different links */
            for (size_t offset = 0; offset < nelems; offset += 1) {
                for (int local_pe = 0; local_pe < n_local_pes; local_pe += 1) {
                    dptr[local_pe][offset] = sptr[local_pe][offset];
                }
            }
            ishmem_sync_all(); /* assure destination buffers complete */
            return ret;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.src = src;
    req.dst = dest;
    req.nelems = nelems * sizeof(T);
    req.op = ALLTOALL;
    req.type = UINT8;

#ifdef __SYCL_DEVICE_ONLY__
    req.team = global_info->team_pool[ISHMEM_TEAM_WORLD];
    ret = ishmemi_proxy_blocking_request_status(req);
#else
    if (ishmemi_only_intra_node && ISHMEMI_HOST_IN_HEAP(dest)) {
        struct put_item items[MAX_LOCAL_PES];
        size_t size = nelems * sizeof(T);
        for (int i = 0; i < ishmemi_n_pes; i += 1) {
            items[i].pe = i;
            items[i].src = pointer_offset(src, static_cast<size_t>(i) * size);
            items[i].size = size;
            items[i].dst = pointer_offset(dest, ((size_t) ishmemi_my_pe) * size);
        }
        int ret = ishmemi_ipc_put_v(ishmemi_n_pes, items);
        ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "ishmemi_ipc_put_v within alltoall failed\n");
    fn_fail:
        ishmem_sync_all(); /* assure destination buffers complete */
        return ret;
    }
    ishmemi_ringcompletion_t comp;
    req.team = ishmemi_mmap_gpu_info->team_pool[ISHMEM_TEAM_WORLD];
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_status(comp.completion.ret);
#endif
    return ret;
}

/* Alltoall on a team */
template <typename T>
int ishmem_alltoall(ishmem_team_t team, T *dest, const T *src, size_t nelems)
{
#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_team_t *myteam = global_info->team_pool[team];
#else
    ishmemi_team_t *myteam = ishmemi_mmap_gpu_info->team_pool[team];
#endif
    size_t nbytes = nelems * sizeof(T);
    if constexpr (enable_error_checking) {
        int n_pes = myteam->size;
        validate_parameters((void *) dest, (void *) src, nelems * sizeof(T) * n_pes, nbytes,
                            ishmemi_op_t::ALLTOALL);
    }

    int ret = 0;

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (info->only_intra_node && !ISHMEM_ALLTOALL_CUTOVER) {
            const T *sptr[MAX_LOCAL_PES]; /* source pointer for each pe */
            T *dptr[MAX_LOCAL_PES];       /* destination pointer for each pe */
            /* compute our address of our section of dest in each PE */
            int idx = 0;
            for (int pe = myteam->start; idx < myteam->size; pe += myteam->stride, idx++) {
                dptr[idx] = ISHMEMI_ADJUST_PTR(T, (pe + 1),
                                               &dest[nelems * static_cast<size_t>(myteam->my_pe)]);
                sptr[idx] = &src[nelems * static_cast<size_t>(idx)];
            }
            /* The idea for the inner loop being over local team PEs is that the outstanding stores
             * will use different links.
             * Because the loop above sets the dptr and sptr buffers contiguously, we can loop over
             * the team members without the stride. */
            for (size_t offset = 0; offset < nelems; offset += 1) {
                for (int pe = 0; pe < myteam->size; pe += 1) {
                    dptr[pe][offset] = sptr[pe][offset];
                }
            }
            ishmemi_team_sync(myteam); /* assure destination buffers complete */
            return ret;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.src = src;
    req.dst = dest;
    req.nelems = nelems * sizeof(T);
    req.op = ALLTOALL;
    req.type = UINT8;
    req.team = myteam;

#ifdef __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_status(req);
#else
    if (ishmemi_only_intra_node && ISHMEMI_HOST_IN_HEAP(dest)) {
        struct put_item items[MAX_LOCAL_PES];
        size_t size = nelems * sizeof(T);
        int idx = 0;
        for (int pe = myteam->start; idx < myteam->size; pe += myteam->stride, idx++) {
            items[idx].pe = pe;
            items[idx].src = pointer_offset(src, static_cast<size_t>(idx) * size);
            items[idx].size = size;
            items[idx].dst = pointer_offset(dest, static_cast<size_t>(myteam->my_pe) * size);
        }
        int ret = ishmemi_ipc_put_v(myteam->size, items);
        ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "ishmemi_ipc_put_v within team alltoall failed\n");
    fn_fail:
        ishmemi_team_sync(myteam); /* assure destination buffers complete */
        return ret;
    }
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_status(comp.completion.ret);
#endif
    return ret;
}

/* Alltoall (work-group) */
template <typename T, typename Group>
int ishmemx_alltoall_work_group(T *dest, const T *src, size_t nelems, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        size_t nbytes = nelems * sizeof(T);
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters((void *) dest, (void *) src, nelems * sizeof(T) * info->n_pes,
                                    nbytes, ishmemi_op_t::ALLTOALL);
        }
        int ret = 0;
        size_t my_nelems_work_item;
        size_t work_item_start_idx;
        ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item, work_item_start_idx);
        sycl::group_barrier(grp); /* assure source buffers complete on all threads */
        if (info->only_intra_node && !ISHMEM_ALLTOALL_GROUP_CUTOVER) {
            int n_local_pes = info->n_local_pes;
            const T *sptr[MAX_LOCAL_PES]; /* source pointer for each pe */
            T *dptr[MAX_LOCAL_PES];       /* destination pointer for each pe*/
            for (size_t local_pe = 0; local_pe < n_local_pes;
                 local_pe += 1) {  // index over target pe
                dptr[local_pe] = ISHMEMI_ADJUST_PTR(
                    T, (local_pe + 1), (&dest[nelems * static_cast<size_t>(info->local_rank)]));
                sptr[local_pe] = &src[nelems * local_pe];
            }
            for (size_t offset = work_item_start_idx;
                 offset < work_item_start_idx + my_nelems_work_item; offset += 1) {
                for (int local_pe = 0; local_pe < n_local_pes; local_pe += 1) {
                    dptr[local_pe][offset] = sptr[local_pe][offset];
                }
            }
            /* assure all threads have finished (group barrier)
             * assure destination buffers complete (sync_all)
             */
            ishmemx_sync_all_work_group(grp);
            /* group broadcast not needed because ret is always 0 here */
            return ret;

        } else {
            if (grp.leader()) {
                ishmemi_request_t req;
                req.src = src;
                req.dst = dest;
                req.nelems = nelems * sizeof(T);
                req.op = ALLTOALL;
                req.type = UINT8;
                req.team = info->team_pool[ISHMEM_TEAM_WORLD];

                ret = ishmemi_proxy_blocking_request_status(req);
            }
        }
        ret = sycl::group_broadcast(grp, ret, 0);
        return ret;
    } else {
        return -1;
    }
}

/* Alltoall (work-group) on a team */
template <typename T, typename Group>
int ishmemx_alltoall_work_group(ishmem_team_t team, T *dest, const T *src, size_t nelems,
                                const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        ishmemi_team_t *myteam = info->team_pool[team];
        size_t nbytes = nelems * sizeof(T);
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters((void *) dest, (void *) src, nelems * sizeof(T) * myteam->size,
                                    nbytes, ishmemi_op_t::ALLTOALL);
        }
        int ret = 0;
        size_t my_nelems_work_item;
        size_t work_item_start_idx;
        ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item, work_item_start_idx);
        sycl::group_barrier(grp); /* assure source buffers complete on all threads */
        if (info->only_intra_node && !ISHMEM_ALLTOALL_GROUP_CUTOVER) {
            const T *sptr[MAX_LOCAL_PES]; /* source pointer for each pe */
            T *dptr[MAX_LOCAL_PES];       /* destination pointer for each pe*/
            int idx = 0;
            for (int pe = myteam->start; idx < myteam->size; pe += myteam->stride, idx++) {
                dptr[idx] = ISHMEMI_ADJUST_PTR(T, (pe + 1),
                                               &dest[nelems * static_cast<size_t>(myteam->my_pe)]);
                sptr[idx] = &src[nelems * static_cast<size_t>(idx)];
            }
            for (size_t offset = work_item_start_idx;
                 offset < work_item_start_idx + my_nelems_work_item; offset += 1) {
                for (int pe = 0; pe < myteam->size; pe += 1) {
                    dptr[pe][offset] = sptr[pe][offset];
                }
            }
            /* assure all threads have finished (group barrier)
             * assure destination buffers complete (sync_all)
             */
            ishmemx_team_sync_work_group(team, grp);
            /* group broadcast not needed because ret is always 0 here */
            return ret;

        } else {
            if (grp.leader()) {
                ishmemi_request_t req;
                req.src = src;
                req.dst = dest;
                req.nelems = nelems * sizeof(T);
                req.op = ALLTOALL;
                req.type = UINT8;
                req.team = myteam;

                ret = ishmemi_proxy_blocking_request_status(req);
            }
        }
        ret = sycl::group_broadcast(grp, ret, 0);
        return ret;
    } else {
        ISHMEM_ERROR_MSG("ISHMEMX_ALLTOALL_WORK_GROUP routines are not callable from host\n");
        return -1;
    }
}

/* clang-format off */
template int ishmemx_alltoallmem_work_group<sycl::group<1>>(void *dest, const void *src, size_t nelems, const sycl::group<1> &grp);
template int ishmemx_alltoallmem_work_group<sycl::group<2>>(void *dest, const void *src, size_t nelems, const sycl::group<2> &grp);
template int ishmemx_alltoallmem_work_group<sycl::group<3>>(void *dest, const void *src, size_t nelems, const sycl::group<3> &grp);
template int ishmemx_alltoallmem_work_group<sycl::sub_group>(void *dest, const void *src, size_t nelems, const sycl::sub_group &grp);

template int ishmemx_alltoallmem_work_group<sycl::group<1>>(ishmem_team_t team, void *dest, const void *src, size_t nelems, const sycl::group<1> &grp);
template int ishmemx_alltoallmem_work_group<sycl::group<2>>(ishmem_team_t team, void *dest, const void *src, size_t nelems, const sycl::group<2> &grp);
template int ishmemx_alltoallmem_work_group<sycl::group<3>>(ishmem_team_t team, void *dest, const void *src, size_t nelems, const sycl::group<3> &grp);
template int ishmemx_alltoallmem_work_group<sycl::sub_group>(ishmem_team_t team, void *dest, const void *src, size_t nelems, const sycl::sub_group &grp);
/* clang-format on */

template <typename Group>
inline int ishmemx_alltoallmem_work_group(void *dest, const void *src, size_t nelems,
                                          const Group &grp)
{
    return ishmemx_alltoall_work_group((uint8_t *) dest, (uint8_t *) src, nelems, grp);
}

template <typename Group>
inline int ishmemx_alltoallmem_work_group(ishmem_team_t team, void *dest, const void *src,
                                          size_t nelems, const Group &grp)
{
    return ishmemx_alltoall_work_group(team, (uint8_t *) dest, (uint8_t *) src, nelems, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_alltoall_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<1> &grp);    \
    template int ishmemx_##TYPENAME##_alltoall_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<2> &grp);    \
    template int ishmemx_##TYPENAME##_alltoall_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<3> &grp);    \
    template int ishmemx_##TYPENAME##_alltoall_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::sub_group &grp);  \
    template <typename Group> int ishmemx_##TYPENAME##_alltoall_work_group(TYPE *dest, const TYPE *src, size_t nelems, const Group &grp) { return ishmemx_alltoall_work_group(dest, src, nelems, grp); }

#define ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_alltoall_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<1> &grp);    \
    template int ishmemx_##TYPENAME##_alltoall_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<2> &grp);    \
    template int ishmemx_##TYPENAME##_alltoall_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<3> &grp);    \
    template int ishmemx_##TYPENAME##_alltoall_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::sub_group &grp);  \
    template <typename Group> int ishmemx_##TYPENAME##_alltoall_work_group(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const Group &grp) { return ishmemx_alltoall_work_group(team, dest, src, nelems, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(double, double)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_ALLTOALL_WORK_GROUP(ptrdiff, ptrdiff_t)

ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(double, double)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_TEAM_ALLTOALL_WORK_GROUP(ptrdiff, ptrdiff_t)

#endif  // COLLECTIVES_ALLTOALL_IMPL.H
