/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef COLLECTIVES_COLLECT_IMPL_H
#define COLLECTIVES_COLLECT_IMPL_H

#include "collectives.h"
#include "collectives/sync_impl.h"
#include "runtime.h"
#include "runtime_ipc.h"

template <typename T>
int ishmem_fcollect(ishmem_team_t team, T *dest, const T *source, size_t nelems);

/* Collect */
template <typename T>
int ishmem_collect(T *dest, const T *src, size_t nelems)
{
    /* TODO: Add validation for cases other than node-local, on-device */
    int ret = 0;
    size_t nbytes = nelems * sizeof(T);

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (info->only_intra_node) {
            int n_local_pes = info->n_local_pes;
            info->collect_mynelems[0] = nelems;  // save our nelems into symmetric space
            ishmem_fcollect(info->collect_nelems, info->collect_mynelems, 1);
            size_t total_nelems = 0;
            for (size_t local_pe = 0; local_pe < info->n_local_pes; local_pe += 1) {
                total_nelems += info->collect_nelems[local_pe];
            }
            size_t total_nbytes = sizeof(T) * total_nelems;
            if (!ISHMEM_COLLECT_CUTOVER) {
                if constexpr (enable_error_checking) {
                    validate_parameters((void *) dest, (void *) src, sizeof(T) * total_nelems,
                                        nbytes, ishmemi_op_t::COLLECT);
                }
                T *ptr[MAX_LOCAL_PES];
                // base is the offset in the dest buffer of where our data will go
                size_t base = 0;
                for (size_t local_pe = 0; local_pe < info->local_rank; local_pe += 1) {
                    base += info->collect_nelems[local_pe];
                }
                /* compute our address of our section of dest in each PE */
                for (int local_pe = 0; local_pe < n_local_pes; local_pe += 1) {
                    ptr[local_pe] = ISHMEMI_ADJUST_PTR(T, (local_pe + 1), &dest[base]);
                }
                /* The idea for the inner loop being over local PEs is that the outstanding stores
                 * will use different links */
                for (size_t offset = 0; offset < nelems; offset += 1) {
                    T data = src[offset];
                    for (int local_pe = 0; local_pe < n_local_pes; local_pe += 1) {
                        ptr[local_pe][offset] = data;
                    }
                }
                /* assure all destination buffers complete */
                ishmem_sync_all();
                return ret;
            }
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.src = src;
    req.dst = dest;
    req.nelems = nbytes;
    req.op = COLLECT;
    req.type = UINT8;

#ifdef __SYCL_DEVICE_ONLY__
    req.team = global_info->team_pool[ISHMEM_TEAM_WORLD];
    ret = ishmemi_proxy_blocking_request_status(req);
#else
    if (ishmemi_only_intra_node && ISHMEMI_HOST_IN_HEAP(dest)) {
        struct put_item items[MAX_LOCAL_PES];
        size_t size = nbytes;
        size_t total_size = 0;
        *ishmemi_my_collect_size = size;
        int ret = ishmem_fcollect(ishmemi_collect_sizes, ishmemi_my_collect_size, 1);
        ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "shmem_size_fcollect within collect failed\n");
        for (int i = 0; i < ishmemi_my_pe; i += 1)
            total_size += ishmemi_collect_sizes[i];
        for (int i = 0; i < ishmemi_n_pes; i += 1) {
            items[i].pe = i;
            items[i].src = src;
            items[i].size = size;
            items[i].dst = pointer_offset(dest, total_size);
        }
        ret = ishmemi_ipc_put_v(ishmemi_n_pes, items);
        ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "ishmemi_ipc_put_v within collect failed\n");
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

/* Collect on a team */
template <typename T>
int ishmem_collect(ishmem_team_t team, T *dest, const T *src, size_t nelems)
{
#if __SYCL_DEVICE_ONLY__
    ishmemi_team_t *myteam = global_info->team_pool[team];
#else
    ishmemi_team_t *myteam = ishmemi_mmap_gpu_info->team_pool[team];
#endif
    int ret = 0;
    size_t nbytes = nelems * sizeof(T);

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (info->only_intra_node) {
            myteam->collect_mynelems[0] = nelems;  // save our nelems into symmetric space
            ishmem_fcollect(team, myteam->collect_nelems, myteam->collect_mynelems, 1);
            size_t total_nelems = 0;
            for (int pe = 0; pe < myteam->size; pe += 1) {
                total_nelems += myteam->collect_nelems[pe];
            }
            size_t total_nbytes = sizeof(T) * total_nelems;
            if (!ISHMEM_COLLECT_CUTOVER) {
                if constexpr (enable_error_checking) {
                    validate_parameters((void *) dest, (void *) src, sizeof(T) * total_nelems,
                                        nbytes, ishmemi_op_t::COLLECT);
                }
                T *ptr[MAX_LOCAL_PES];
                // base is the offset in the dest buffer of where our data will go
                size_t base = 0;
                for (int pe = 0; pe < myteam->size; pe += 1) {
                    base += myteam->collect_nelems[pe];
                }
                /* compute our address of our section of dest in each PE */
                int idx = 0;
                for (int pe = myteam->start; idx < myteam->size; pe += myteam->stride, idx++) {
                    ptr[idx] = ISHMEMI_ADJUST_PTR(T, (pe + 1), &dest[base]);
                }
                /* The idea for the inner loop being over local PEs is that the outstanding stores
                 * will use different links */
                for (size_t offset = 0; offset < nelems; offset += 1) {
                    T data = src[offset];
                    for (int pe = 0; pe < myteam->size; pe += 1) {
                        ptr[pe][offset] = data;
                    }
                }
                /* assure all destination buffers complete */
                ishmemi_team_sync(myteam);
                return ret;
            }
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.src = src;
    req.dst = dest;
    req.nelems = nbytes;
    req.op = COLLECT;
    req.type = UINT8;
    req.team = myteam;

#ifdef __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_status(req);
#else
    if (ishmemi_only_intra_node && ISHMEMI_HOST_IN_HEAP(dest)) {
        struct put_item items[MAX_LOCAL_PES];
        size_t size = nbytes;
        size_t total_size = 0;
        *myteam->collect_my_size = size;
        int ret = ishmem_fcollect(team, myteam->collect_sizes, myteam->collect_my_size, 1);
        ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "shmem_size_fcollect on team collect failed\n");
        for (int i = 0; i < myteam->my_pe; i += 1)
            total_size += myteam->collect_sizes[i];
        for (int i = 0; i < myteam->size; i += 1) {
            items[i].pe = i;
            items[i].src = src;
            items[i].size = size;
            items[i].dst = pointer_offset(dest, total_size);
        }
        ret = ishmemi_ipc_put_v(myteam->size, items);
        ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "ishmemi_ipc_put_v within team collect failed\n");
    fn_fail:
        ishmemi_team_sync(myteam); /* assure all destination buffers complete */
        return ret;
    }
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_status(comp.completion.ret);
#endif
    return ret;
}

/* Collect (work-group) */
template <typename T, typename Group>
int ishmemx_collect_work_group(T *dest, const T *src, size_t nelems, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        int ret = 0;
        ishmemi_info_t *info = global_info;
        size_t nbytes = nelems * sizeof(T);
        size_t my_nelems_work_item;
        size_t work_item_start_idx;
        ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item, work_item_start_idx);
        size_t n_local_pes = (size_t) (info->n_local_pes);
        if (info->only_intra_node) {
            if (grp.leader()) {
                info->collect_mynelems[0] = nelems;  // save our nelems into symmetric space
                ishmem_fcollect(info->collect_nelems, info->collect_mynelems, 1);
                if constexpr (enable_error_checking) {
                    /* this copy of total_nelems is only available to the leader thread */
                    size_t total_nelems = 0;
                    for (int pe = 0; pe < n_local_pes; pe += 1) {
                        total_nelems += info->collect_nelems[pe];
                    }
                    validate_parameters((void *) dest, (void *) src, sizeof(T) * total_nelems,
                                        sizeof(T) * nelems, ishmemi_op_t::COLLECT);
                }
            }
            /* This group barrier assures the fcollect results are available to all threads
             * and makes sure the source buffer is valid on all threads
             */
            sycl::group_barrier(grp);
            /* compute the total data size to make an upcall decision */
            size_t total_nelems = 0;
            for (int pe = 0; pe < n_local_pes; pe += 1) {
                total_nelems += info->collect_nelems[pe];
            }
            /* This copy of total_nbytes is available to all threads */
            size_t total_nbytes = total_nelems * sizeof(T);
            if (!ISHMEM_COLLECT_GROUP_CUTOVER) {
                /* now sum all the lower numbered pe nelems to compute where our data will go
                 */
                size_t base = 0;
                for (int pe = 0; pe < info->local_rank; pe += 1) {
                    base += info->collect_nelems[pe];
                }
                T *ptr[MAX_LOCAL_PES];
                for (int pe = 0; pe < n_local_pes; pe += 1) {
                    ptr[pe] = ISHMEMI_ADJUST_PTR(T, (pe + 1), &dest[base]);
                }
                for (size_t offset = work_item_start_idx;
                     offset < work_item_start_idx + my_nelems_work_item; offset += 1) {
                    T data = src[offset];
                    for (int pe = 0; pe < n_local_pes; pe += 1) {
                        ptr[pe][offset] = data;
                    }
                }
                /* assure all threads have finished copying (group barrier)
                 * assure all destination buffers finished (sync all)
                 */
                ishmemx_sync_all_work_group(grp);
                return 0; /* no need for group_broadcast, since this path never fails */
            }
        }
        if (grp.leader()) {
            ishmemi_request_t req;
            req.src = src;
            req.dst = dest;
            req.nelems = nbytes;
            req.op = COLLECT;
            req.type = UINT8;
            req.team = info->team_pool[ISHMEM_TEAM_WORLD];

            ret = ishmemi_proxy_blocking_request_status(req);
        }
        ret = sycl::group_broadcast(grp, ret, 0);
        return ret;
    } else {
        ISHMEM_ERROR_MSG("ISHMEMX_COLLECT_WORK_GROUP routines are not callable from host\n");
        return -1;
    }
}

/* Collect (work-group) on a team */
template <typename T, typename Group>
int ishmemx_collect_work_group(ishmem_team_t team, T *dest, const T *src, size_t nelems,
                               const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        int ret = 0;
        ishmemi_info_t *info = global_info;
        ishmemi_team_t *myteam = info->team_pool[team];
        size_t nbytes = nelems * sizeof(T);
        size_t my_nelems_work_item;
        size_t work_item_start_idx;
        ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item, work_item_start_idx);
        if (info->only_intra_node) {
            if (grp.leader()) {
                myteam->collect_mynelems[0] = nelems;  // save our nelems into symmetric space
                ishmem_fcollect(team, myteam->collect_nelems, myteam->collect_mynelems, 1);
                if constexpr (enable_error_checking) {
                    /* this copy of total_nelems is only available to the leader thread */
                    size_t total_nelems = 0;
                    for (int pe = 0; pe < myteam->size; pe += 1) {
                        total_nelems += myteam->collect_nelems[pe];
                    }
                    validate_parameters((void *) dest, (void *) src, sizeof(T) * total_nelems,
                                        sizeof(T) * nelems, ishmemi_op_t::COLLECT);
                }
            }
            /* This group barrier assures the fcollect results are available to all threads
             * and makes sure the source buffer is valid on all threads
             */
            sycl::group_barrier(grp);
            /* compute the total data size to make an upcall decision */
            size_t total_nelems = 0;
            for (int pe = 0; pe < myteam->size; pe += 1) {
                total_nelems += myteam->collect_nelems[pe];
            }
            /* This copy of total_nbytes is available to all threads */
            size_t total_nbytes = total_nelems * sizeof(T);
            if (!ISHMEM_COLLECT_GROUP_CUTOVER) {
                /* now sum all the lower numbered pe nelems to compute where our data will go
                 */
                size_t base = 0;
                for (int pe = 0; pe < myteam->size; pe += 1) {
                    base += myteam->collect_nelems[pe];
                }
                T *ptr[MAX_LOCAL_PES];
                int idx = 0;
                for (int pe = myteam->start; idx < myteam->size; pe += myteam->stride, idx++) {
                    ptr[idx] = ISHMEMI_ADJUST_PTR(T, (pe + 1), &dest[base]);
                }
                for (size_t offset = work_item_start_idx;
                     offset < work_item_start_idx + my_nelems_work_item; offset += 1) {
                    T data = src[offset];
                    for (int pe = 0; pe < myteam->size; pe += 1) {
                        ptr[pe][offset] = data;
                    }
                }
                /* assure all threads have finished copying (group barrier)
                 * assure all destination buffers finished (sync all)
                 */
                ishmemx_team_sync_work_group(team, grp);
                return 0; /* no need for group_broadcast, since this path never fails */
            }
        }
        if (grp.leader()) {
            ishmemi_request_t req;
            req.src = src;
            req.dst = dest;
            req.nelems = nbytes;
            req.op = COLLECT;
            req.type = UINT8;
            req.team = myteam;

            ret = ishmemi_proxy_blocking_request_status(req);
        }
        ret = sycl::group_broadcast(grp, ret, 0);
        return ret;
    } else {
        ISHMEM_ERROR_MSG("ISHMEMX_COLLECT_WORK_GROUP routines are not callable from host\n");
        return -1;
    }
}

template int ishmemx_collectmem_work_group<sycl::group<1>>(void *dest, const void *src,
                                                           size_t nelems,
                                                           const sycl::group<1> &grp);
template int ishmemx_collectmem_work_group<sycl::group<2>>(void *dest, const void *src,
                                                           size_t nelems,
                                                           const sycl::group<2> &grp);
template int ishmemx_collectmem_work_group<sycl::group<3>>(void *dest, const void *src,
                                                           size_t nelems,
                                                           const sycl::group<3> &grp);
template int ishmemx_collectmem_work_group<sycl::sub_group>(void *dest, const void *src,
                                                            size_t nelems,
                                                            const sycl::sub_group &grp);
template <typename Group>
inline int ishmemx_collectmem_work_group(void *dest, const void *src, size_t nelems,
                                         const Group &grp)
{
    return ishmemx_collect_work_group((uint8_t *) dest, (uint8_t *) src, nelems, grp);
}

template int ishmemx_collectmem_work_group<sycl::group<1>>(ishmem_team_t team, void *dest,
                                                           const void *src, size_t nelems,
                                                           const sycl::group<1> &grp);
template int ishmemx_collectmem_work_group<sycl::group<2>>(ishmem_team_t team, void *dest,
                                                           const void *src, size_t nelems,
                                                           const sycl::group<2> &grp);
template int ishmemx_collectmem_work_group<sycl::group<3>>(ishmem_team_t team, void *dest,
                                                           const void *src, size_t nelems,
                                                           const sycl::group<3> &grp);
template int ishmemx_collectmem_work_group<sycl::sub_group>(ishmem_team_t team, void *dest,
                                                            const void *src, size_t nelems,
                                                            const sycl::sub_group &grp);
template <typename Group>
inline int ishmemx_collectmem_work_group(ishmem_team_t team, void *dest, const void *src,
                                         size_t nelems, const Group &grp)
{
    return ishmemx_collect_work_group(team, (uint8_t *) dest, (uint8_t *) src, nelems, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(TYPENAME, TYPE)                                                                                        \
    template int ishmemx_##TYPENAME##_collect_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_collect_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_collect_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_collect_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_collect_work_group(TYPE *dest, const TYPE *src, size_t nelems, const Group &grp) { return ishmemx_collect_work_group(dest, src, nelems, grp); }

#define ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(TYPENAME, TYPE)                                                                                        \
    template int ishmemx_##TYPENAME##_collect_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_collect_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_collect_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_collect_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_collect_work_group(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const Group &grp) { return ishmemx_collect_work_group(team, dest, src, nelems, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(double, double)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_COLLECT_WORK_GROUP(ptrdiff, ptrdiff_t)

ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(double, double)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(ptrdiff, ptrdiff_t)

/* Fcollect */
template <typename T>
int ishmem_fcollect(T *dest, const T *src, size_t nelems)
{
    size_t nbytes = nelems * sizeof(T);
    if constexpr (enable_error_checking) {
        int n_pes = ishmem_n_pes();
        validate_parameters((void *) dest, (void *) src, nbytes * n_pes, nbytes,
                            ishmemi_op_t::FCOLLECT);
    }

    int ret = 0;

    /* Node-local, on-device implementaiton */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (info->only_intra_node && !ISHMEM_FCOLLECT_CUTOVER) {
            int n_local_pes = info->n_local_pes;
            size_t base = static_cast<size_t>(info->local_rank) *
                          nelems;  // TODO is my_pe equal to local_rank in this case?
            T *ptr[MAX_LOCAL_PES];
            /* compute our address of our section of dest in each PE */
            for (int pe = 0; pe < n_local_pes; pe += 1) {
                ptr[pe] = ISHMEMI_ADJUST_PTR(T, (pe + 1), (&dest[base]));
            }
            /* The idea for the inner loop being over local PEs is that the outstanding stores will
             * use different links */
            for (size_t offset = 0; offset < nelems; offset += 1) {
                T data = src[offset];
                for (int pe = 0; pe < n_local_pes; pe += 1) {
                    ptr[pe][offset] = data;
                }
            }
            ishmem_sync_all(); /* assure all destination buffers complete */
            return ret;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.src = src;
    req.dst = dest;
    req.nelems = nbytes;
    req.op = FCOLLECT;
    req.type = UINT8;

#ifdef __SYCL_DEVICE_ONLY__
    req.team = global_info->team_pool[ISHMEM_TEAM_WORLD];
    ret = ishmemi_proxy_blocking_request_status(req);
#else
    req.team = ishmemi_mmap_gpu_info->team_pool[ISHMEM_TEAM_WORLD];
    if (ishmemi_only_intra_node && ISHMEMI_HOST_IN_HEAP(dest)) {
        struct put_item items[MAX_LOCAL_PES];
        for (int i = 0; i < ishmemi_n_pes; i += 1) {
            items[i].pe = i;
            items[i].src = src;
            items[i].size = nbytes;
            items[i].dst = pointer_offset(dest, (size_t) ishmemi_my_pe * nbytes);
        }
        int ret = ishmemi_ipc_put_v(ishmemi_n_pes, items);
        ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "ishmemi_ipc_put_v within fcollect failed\n");
    fn_fail:
        ishmem_sync_all(); /* assure all destination buffers complete */
        return ret;
    }
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_status(comp.completion.ret);
#endif
    return ret;
}

/* Fcollect on a team */
template <typename T>
int ishmem_fcollect(ishmem_team_t team, T *dest, const T *src, size_t nelems)
{
#if __SYCL_DEVICE_ONLY__
    ishmemi_team_t *myteam = global_info->team_pool[team];
#else
    ishmemi_team_t *myteam = ishmemi_mmap_gpu_info->team_pool[team];
#endif
    size_t nbytes = nelems * sizeof(T);
    if constexpr (enable_error_checking) {
        int n_pes = ishmem_n_pes();
        validate_parameters((void *) dest, (void *) src, nbytes * n_pes, nbytes,
                            ishmemi_op_t::FCOLLECT);
    }

    int ret = 0;

    /* Node-local, on-device implementaiton */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (info->only_intra_node && !ISHMEM_FCOLLECT_CUTOVER) {
            size_t base = static_cast<size_t>(myteam->my_pe) * nelems;
            T *ptr[MAX_LOCAL_PES];
            /* compute our address of our section of dest in each PE */
            int idx = 0;
            for (int pe = myteam->start; idx < myteam->size; pe += myteam->stride, idx++) {
                ptr[idx] = ISHMEMI_ADJUST_PTR(T, (pe + 1), (&dest[base]));
            }
            /* The idea for the inner loop being over local PEs is that the outstanding stores will
             * use different links */
            for (size_t offset = 0; offset < nelems; offset += 1) {
                T data = src[offset];
                for (int pe = 0; pe < myteam->size; pe += 1) {
                    ptr[pe][offset] = data;
                }
            }
            ishmemi_team_sync(myteam); /* assure all destination buffers complete */
            return ret;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.src = src;
    req.dst = dest;
    req.nelems = nbytes;
    req.op = FCOLLECT;
    req.type = UINT8;
    req.team = myteam;

#ifdef __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_status(req);
#else
    if (ishmemi_only_intra_node && ISHMEMI_HOST_IN_HEAP(dest)) {
        struct put_item items[MAX_LOCAL_PES];
        int idx = 0;
        for (int pe = myteam->start; idx < myteam->size; pe += myteam->stride, idx++) {
            items[idx].pe = pe;
            items[idx].src = src;
            items[idx].size = nbytes;
            items[idx].dst = pointer_offset(dest, static_cast<size_t>(myteam->my_pe) * nbytes);
        }
        int ret = ishmemi_ipc_put_v(myteam->size, items);
        ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "ishmemi_ipc_put_v within team fcollect failed\n");
    fn_fail:
        ishmemi_team_sync(myteam); /* assure all destination buffers complete */
        return ret;
    }
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_status(comp.completion.ret);
#endif
    return ret;
}

/* Fcollect (work-group) */
template <typename T, typename Group>
int ishmemx_fcollect_work_group(T *dest, const T *src, size_t nelems, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        size_t nbytes = nelems * sizeof(T);
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters((void *) dest, (void *) src, nbytes * info->n_pes, nbytes,
                                    ishmemi_op_t::FCOLLECT);
        }
        int ret = 0;
        size_t my_nelems_work_item;
        size_t work_item_start_idx;
        ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item, work_item_start_idx);
        sycl::group_barrier(grp); /* assure source buffers complete */
        if (info->only_intra_node && !ISHMEM_FCOLLECT_GROUP_CUTOVER) {
            int n_local_pes = info->n_local_pes;
            size_t base = static_cast<size_t>(info->local_rank) * nelems;
            T *ptr[MAX_LOCAL_PES];
            for (int pe = 0; pe < n_local_pes; pe += 1) {
                ptr[pe] = ISHMEMI_ADJUST_PTR(T, (pe + 1), &dest[base]);
            }
            for (size_t offset = work_item_start_idx;
                 offset < work_item_start_idx + my_nelems_work_item; offset += 1) {
                T data = src[offset];
                for (int pe = 0; pe < n_local_pes; pe += 1) {
                    ptr[pe][offset] = data;
                }
            }
            /* assure all threads have finished copy  (group barrier)
             * assure all destination buffers complere (sync_all)
             */
            ishmemx_sync_all_work_group(grp);
            return 0;
        } else {
            if (grp.leader()) {
                ishmemi_request_t req;
                req.src = src;
                req.dst = dest;
                req.nelems = nbytes;
                req.op = FCOLLECT;
                req.type = UINT8;
                req.team = info->team_pool[ISHMEM_TEAM_WORLD];

                ret = ishmemi_proxy_blocking_request_status(req);
            }
        }
        ret = sycl::group_broadcast(grp, ret, 0);
        return ret;
    } else {
        ISHMEM_ERROR_MSG("ISHMEMX_FCOLLECT_WORK_GROUP routines are not callable from host\n");
        return -1;
    }
}

/* Fcollect (work-group on a team) */
template <typename T, typename Group>
int ishmemx_fcollect_work_group(ishmem_team_t team, T *dest, const T *src, size_t nelems,
                                const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        ishmemi_team_t *myteam = info->team_pool[team];
        size_t nbytes = nelems * sizeof(T);
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters((void *) dest, (void *) src, nbytes * info->n_pes, nbytes,
                                    ishmemi_op_t::FCOLLECT);
        }
        int ret = 0;
        size_t my_nelems_work_item;
        size_t work_item_start_idx;
        ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item, work_item_start_idx);
        sycl::group_barrier(grp); /* assure source buffers complete */
        if (info->only_intra_node && !ISHMEM_FCOLLECT_GROUP_CUTOVER) {
            size_t base = static_cast<size_t>(myteam->my_pe) * nelems;
            T *ptr[MAX_LOCAL_PES];
            int idx = 0;
            for (int pe = myteam->start; idx < myteam->size; pe += myteam->stride, idx++) {
                ptr[idx] = ISHMEMI_ADJUST_PTR(T, (pe + 1), &dest[base]);
            }
            for (size_t offset = work_item_start_idx;
                 offset < work_item_start_idx + my_nelems_work_item; offset += 1) {
                T data = src[offset];
                for (int pe = 0; pe < myteam->size; pe += 1) {
                    ptr[pe][offset] = data;
                }
            }
            /* assure all threads have finished copy  (group barrier)
             * assure all destination buffers complere (sync_all)
             */
            ishmemx_team_sync_work_group(team, grp);
            return 0;
        } else {
            if (grp.leader()) {
                ishmemi_request_t req;
                req.src = src;
                req.dst = dest;
                req.nelems = nbytes;
                req.op = FCOLLECT;
                req.type = UINT8;
                req.team = myteam;

                ret = ishmemi_proxy_blocking_request_status(req);
            }
        }
        ret = sycl::group_broadcast(grp, ret, 0);
        return ret;
    } else {
        ISHMEM_ERROR_MSG("ISHMEMX_FCOLLECT_WORK_GROUP routines are not callable from host\n");
        return -1;
    }
}

template int ishmemx_fcollectmem_work_group<sycl::group<1>>(void *dest, const void *src,
                                                            size_t nelems,
                                                            const sycl::group<1> &grp);
template int ishmemx_fcollectmem_work_group<sycl::group<2>>(void *dest, const void *src,
                                                            size_t nelems,
                                                            const sycl::group<2> &grp);
template int ishmemx_fcollectmem_work_group<sycl::group<3>>(void *dest, const void *src,
                                                            size_t nelems,
                                                            const sycl::group<3> &grp);
template int ishmemx_fcollectmem_work_group<sycl::sub_group>(void *dest, const void *src,
                                                             size_t nelems,
                                                             const sycl::sub_group &grp);
template <typename Group>
inline int ishmemx_fcollectmem_work_group(void *dest, const void *src, size_t nelems,
                                          const Group &grp)
{
    return ishmemx_fcollect_work_group((uint8_t *) dest, (uint8_t *) src, nelems, grp);
}

template int ishmemx_fcollectmem_work_group<sycl::group<1>>(ishmem_team_t team, void *dest,
                                                            const void *src, size_t nelems,
                                                            const sycl::group<1> &grp);
template int ishmemx_fcollectmem_work_group<sycl::group<2>>(ishmem_team_t team, void *dest,
                                                            const void *src, size_t nelems,
                                                            const sycl::group<2> &grp);
template int ishmemx_fcollectmem_work_group<sycl::group<3>>(ishmem_team_t team, void *dest,
                                                            const void *src, size_t nelems,
                                                            const sycl::group<3> &grp);
template int ishmemx_fcollectmem_work_group<sycl::sub_group>(ishmem_team_t team, void *dest,
                                                             const void *src, size_t nelems,
                                                             const sycl::sub_group &grp);
template <typename Group>
inline int ishmemx_fcollectmem_work_group(ishmem_team_t team, void *dest, const void *src,
                                          size_t nelems, const Group &grp)
{
    return ishmemx_fcollect_work_group(team, (uint8_t *) dest, (uint8_t *) src, nelems, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<1> &grp);    \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<2> &grp);    \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<3> &grp);    \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::sub_group &grp);  \
    template <typename Group> int ishmemx_##TYPENAME##_fcollect_work_group(TYPE *dest, const TYPE *src, size_t nelems, const Group &grp) { return ishmemx_fcollect_work_group(dest, src, nelems, grp); }

#define ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<1> &grp);    \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<2> &grp);    \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<3> &grp);    \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::sub_group &grp);  \
    template <typename Group> int ishmemx_##TYPENAME##_fcollect_work_group(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const Group &grp) { return ishmemx_fcollect_work_group(team, dest, src, nelems, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(double, double)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(ptrdiff, ptrdiff_t)

ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(double, double)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(ptrdiff, ptrdiff_t)

#endif  // COLLECTIVES_COLLECT_IMPL.H
