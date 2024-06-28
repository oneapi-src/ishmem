/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef COLLECTIVES_BROADCAST_IMPL_H
#define COLLECTIVES_BROADCAST_IMPL_H

#include "collectives.h"
#include "collectives/sync_impl.h"
#include "runtime.h"
#include "rma_impl.h"

#define BROADCAST_PUSH 0
#define BROADCAST_PULL 1

/* Broadcast */
template <typename T>
int ishmem_broadcast(T *dest, const T *src, size_t nelems, int PE_root)
{
    size_t nbytes = nelems * sizeof(T);

    if constexpr (enable_error_checking) {
        validate_parameters(PE_root, (void *) dest, (void *) src, nbytes, nbytes);
    }

    int ret = 0;

    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (info->only_intra_node && !ISHMEM_BROADCAST_CUTOVER) {
#if BROADCAST_PUSH
            if (info->my_pe == PE_root) {
                int n_local_pes = info->n_local_pes;
                T *ptr[MAX_LOCAL_PES];
                /* compute our address dest in each PE */
                for (int pe = 0; pe < n_local_pes; pe += 1) {
                    ptr[pe] = ISHMEMI_ADJUST_PTR(T, (pe + 1), dest);
                }
                /* The idea for the inner loop being over local PEs is that the outstanding stores
                 * will use different links */
                for (size_t offset = 0; offset < nelems; offset += 1) {
                    T data = src[offset];
                    for (int pe = 0; pe < n_local_pes; pe += 1) {
                        ptr[pe][offset] = data;
                    }
                }
            }
#else  // BROADCAST_PULL
            ishmem_sync_all(); /* make sure that PE_root's source buffer is ready for use */
            ishmem_internal_get(dest, src, nelems, PE_root);
#endif
            ishmem_sync_all(); /* sync after to let PE_root know we are done */
            return ret;
        }
    }

    ishmemi_request_t req;
    req.root = PE_root;
    req.src = src;
    req.dst = dest;
    req.nelems = nbytes;
    req.op = BCAST;
    req.type = UINT8;

#ifdef __SYCL_DEVICE_ONLY__
    req.team = global_info->team_pool[ISHMEM_TEAM_WORLD];
    ret = ishmemi_proxy_blocking_request_status(req);
#else
#if BROADCAST_PUSH
    if (ishmemi_only_intra_node && ISHMEMI_HOST_IN_HEAP(dest)) {
        if (ishmemi_my_pe == PE_root) {
            struct put_item items[MAX_LOCAL_PES];
            for (int i = 0; i < ishmemi_n_pes; i += 1) {
                items[i].pe = i;
                items[i].src = src;
                items[i].size = nbytes;
                items[i].dst = dest;
            }
            int ret = ishmemi_ipc_put_v(ishmemi_n_pes, items);
            ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "ishmemi_ipc_put_v within broadcast failed\n");
        }
    fn_fail:
        ishmem_sync_all(); /* assure all destination buffers complete */
        return ret;
    }
#else
    if (ishmemi_only_intra_node && ISHMEMI_HOST_IN_HEAP(src)) {
        ishmem_sync_all(); /* assure PE_root source buffer is ready for use */
        ret = ishmemi_ipc_get(dest, src, nelems, PE_root);
        ishmem_sync_all(); /* assure PE_root can reuse source buffer */
        return ret;
    }
#endif
    req.team = ishmemi_mmap_gpu_info->team_pool[ISHMEM_TEAM_WORLD];
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_status(comp.completion.ret);
#endif
    return ret;
}

/* Broadcast on a team: */
template <typename T>
int ishmem_broadcast(ishmem_team_t team, T *dest, const T *src, size_t nelems, int PE_root)
{
#if __SYCL_DEVICE_ONLY__
    ishmemi_team_t *myteam = global_info->team_pool[team];
    ishmemi_team_t *world_team = global_info->team_pool[ISHMEM_TEAM_WORLD];
#else
    ishmemi_team_t *myteam = ishmemi_mmap_gpu_info->team_pool[team];
    ishmemi_team_t *world_team = ishmemi_mmap_gpu_info->team_pool[ISHMEM_TEAM_WORLD];
#endif

    int ret = 0;
    size_t nbytes = nelems * sizeof(T);

    if constexpr (enable_error_checking) {
        validate_parameters(ishmemi_team_translate_pe(myteam, PE_root, world_team), (void *) dest,
                            (void *) src, nbytes, nbytes);
    }

    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (info->only_intra_node && !ISHMEM_BROADCAST_CUTOVER) {
#if BROADCAST_PUSH
            if (myteam->my_pe == PE_root) {
                T *ptr[MAX_LOCAL_PES];
                /* compute our address dest in each PE on the team */
                int idx = 0;
                for (int pe = myteam->start; idx < myteam->size; pe += myteam->stride, idx++) {
                    ptr[idx] = ISHMEMI_ADJUST_PTR(T, (pe + 1), dest);
                }
                /* The idea for the inner loop being over local PEs is that the outstanding stores
                 * will use different links.
                 * Because the loop above sets the dptr and sptr buffers contiguously, we can loop
                 * over the team members without the stride. */
                for (size_t offset = 0; offset < nelems; offset += 1) {
                    T data = src[offset];
                    for (int pe = 0; pe < myteam->size; pe += 1) {
                        ptr[pe][offset] = data;
                    }
                }
            }
            ishmemi_team_sync(myteam); /* assure destination buffers complete */
            return ret;
#else  // BROADCAST_PULL
            ishmemi_team_sync(myteam); /* make sure that PE_root's source buffer is ready for use */
            ishmem_internal_get(dest, src, nelems,
                                ishmemi_team_translate_pe(myteam, PE_root, world_team));
            ishmemi_team_sync(myteam); /* sync after to let PE_root know we are done */
            return ret;
#endif
        }
    }

    ishmemi_request_t req;
    req.root = PE_root;
    req.src = src;
    req.dst = dest;
    req.nelems = nbytes;
    req.op = BCAST;
    req.type = UINT8;
    req.team = myteam;

#ifdef __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_status(req);
#else
#if BROADCAST_PUSH
    if (ishmemi_only_intra_node && ISHMEMI_HOST_IN_HEAP(dest)) {
        if (myteam->my_pe == PE_root) {
            struct put_item items[MAX_LOCAL_PES];
            int idx = 0;
            for (int pe = myteam->start; idx < myteam->size; pe += myteam->stride, idx++) {
                items[idx].pe = pe;
                items[idx].src = src;
                items[idx].size = nbytes;
                items[idx].dst = dest;
            }
            int ret = ishmemi_ipc_put_v(myteam->size, items);
            ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "ishmemi_ipc_put_v within team broadcast failed\n");
        }
    fn_fail:
        ishmemi_team_sync(myteam); /* assure all destination buffers complete */
        return ret;
    }
#else
    if (ishmemi_only_intra_node && ISHMEMI_HOST_IN_HEAP(src)) {
        ishmemi_team_sync(myteam); /* assure PE_root source buffer is ready for use */
        ret = ishmemi_ipc_get(dest, src, nelems,
                              ishmemi_team_translate_pe(myteam, PE_root, world_team));
        ishmemi_team_sync(myteam); /* assure PE_root can reuse source buffer */
        return ret;
    }
#endif
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_status(comp.completion.ret);
#endif
    return ret;
}

/* Broadcast (work-group) */
template <typename T, typename Group>
int ishmemx_broadcast_work_group(T *dest, const T *src, size_t nelems, int PE_root,
                                 const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        size_t nbytes = nelems * sizeof(T);
        if constexpr (enable_error_checking) {
            if (grp.leader()) {
                validate_parameters(PE_root, (void *) dest, (void *) src, nbytes, nbytes);
            }
        }
        ishmemi_info_t *info = global_info;
        int ret = 0;

        if (info->only_intra_node && !ISHMEM_BROADCAST_GROUP_CUTOVER) {
#if BROADCAST_PUSH
            /* make sure all threads have reached here, so source ready for use */
            sycl::group_barrier(grp);
            if (info->my_pe == PE_root) {
                size_t my_nelems_work_item;
                size_t work_item_start_idx;
                ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item,
                                                   work_item_start_idx);
                int n_local_pes = info->n_local_pes;
                T *ptr[MAX_LOCAL_PES];
                for (int pe = 0; pe < n_local_pes; pe += 1) {
                    ptr[pe] = ISHMEMI_ADJUST_PTR(T, (pe + 1), dest);
                }
                for (size_t offset = work_item_start_idx;
                     offset < work_item_start_idx + my_nelems_work_item; offset += 1) {
                    T data = src[offset];
                    for (int pe = 0; pe < n_local_pes; pe += 1) {
                        ptr[pe][offset] = data;
                    }
                }
            }
            /* assure all threads have finished copy  (group barrier)
             * assure all destination buffers complere (sync_all)
             */
            ishmemx_sync_all_work_group(grp);
            return 0;
#else
            /* assure that PE_root's source buffer is ready for use */
            ishmemx_sync_all_work_group(grp);
            ishmemx_internal_get_work_group(dest, src, nelems, PE_root, grp);
            ishmemx_sync_all_work_group(grp); /* inform root PE all threads are done */
#endif
            return ret;
        } else {
            sycl::group_barrier(grp); /* assure source buffer ready for use */
            if (grp.leader()) {
                ishmemi_request_t req;
                req.root = PE_root;
                req.src = src;
                req.dst = dest;
                req.nelems = nbytes;
                req.op = BCAST;
                req.type = UINT8;
                req.team = info->team_pool[ISHMEM_TEAM_WORLD];
                ret = ishmemi_proxy_blocking_request_status(req);
            }
            /* upcall case does not need a sync_all() */
            ret = sycl::group_broadcast(grp, ret, 0);
            return ret;
        }
    } else {
        ISHMEM_ERROR_MSG("ISHMEMX_BROADCAST_WORK_GROUP routines are not callable from host\n");
        return -1;
    }
}

/* Broadcast (work-group) on a team */
template <typename T, typename Group>
int ishmemx_broadcast_work_group(ishmem_team_t team, T *dest, const T *src, size_t nelems,
                                 int PE_root, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        int ret = 0;
        size_t nbytes = nelems * sizeof(T);
        ishmemi_info_t *info = global_info;
        ishmemi_team_t *myteam = info->team_pool[team];
        if constexpr (enable_error_checking) {
            if (grp.leader()) {
                validate_parameters(
                    ishmemi_team_translate_pe(myteam, PE_root, info->team_pool[ISHMEM_TEAM_WORLD]),
                    (void *) dest, (void *) src, nbytes, nbytes);
            }
        }

        if (info->only_intra_node && !ISHMEM_BROADCAST_GROUP_CUTOVER) {
#if BROADCAST_PUSH
            /* make sure all threads have reached here, so source ready for use */
            sycl::group_barrier(grp);
            if (team->my_pe == PE_root) {
                size_t my_nelems_work_item;
                size_t work_item_start_idx;
                ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item,
                                                   work_item_start_idx);
                T *ptr[MAX_LOCAL_PES];
                int idx = 0;
                for (int pe = myteam->start; idx < myteam->size; pe += myteam->stride, idx++) {
                    ptr[idx] = ISHMEMI_ADJUST_PTR(T, (pe + 1), dest);
                }
                for (size_t offset = work_item_start_idx;
                     offset < work_item_start_idx + my_nelems_work_item; offset += 1) {
                    T data = src[offset];
                    for (int pe = 0; pe < myteam->size; pe += 1) {
                        ptr[pe][offset] = data;
                    }
                }
            }
            /* assure all threads have finished copy  (group barrier)
             * assure all destination buffers complere (sync_all)
             */
            ishmemx_team_sync_work_group(team, grp);
            return 0;
#else
            /* assure that PE_root's source buffer is ready for use */
            ishmemx_team_sync_work_group(team, grp);
            ishmemx_internal_get_work_group(
                dest, src, nelems,
                ishmemi_team_translate_pe(myteam, PE_root, info->team_pool[ISHMEM_TEAM_WORLD]),
                grp);
            ishmemx_team_sync_work_group(team, grp); /* inform root PE all threads are done */
#endif
            return ret;
        } else {
            sycl::group_barrier(grp); /* assure source buffer ready for use */
            if (grp.leader()) {
                ishmemi_request_t req;
                req.root = PE_root;
                req.src = src;
                req.dst = dest;
                req.nelems = nbytes;
                req.op = BCAST;
                req.type = UINT8;
                req.team = myteam;
                ret = ishmemi_proxy_blocking_request_status(req);
            }
            /* upcall case does not need a sync_all() */
            ret = sycl::group_broadcast(grp, ret, 0);
            return ret;
        }
    } else {
        ISHMEM_ERROR_MSG("ISHMEMX_BROADCAST_WORK_GROUP routines are not callable from host\n");
        return -1;
    }
}

/* clang-format off */
template int ishmemx_broadcastmem_work_group<sycl::group<1>>(void *dest, const void *src, size_t nelems, int PE_root, const sycl::group<1> &grp);
template int ishmemx_broadcastmem_work_group<sycl::group<2>>(void *dest, const void *src, size_t nelems, int PE_root, const sycl::group<2> &grp);
template int ishmemx_broadcastmem_work_group<sycl::group<3>>(void *dest, const void *src, size_t nelems, int PE_root, const sycl::group<3> &grp);
template int ishmemx_broadcastmem_work_group<sycl::sub_group>(void *dest, const void *src, size_t nelems, int PE_root, const sycl::sub_group &grp);

template int ishmemx_broadcastmem_work_group<sycl::group<1>>(ishmem_team_t team, void *dest, const void *src, size_t nelems, int PE_root, const sycl::group<1> &grp);
template int ishmemx_broadcastmem_work_group<sycl::group<2>>(ishmem_team_t team, void *dest, const void *src, size_t nelems, int PE_root, const sycl::group<2> &grp);
template int ishmemx_broadcastmem_work_group<sycl::group<3>>(ishmem_team_t team, void *dest, const void *src, size_t nelems, int PE_root, const sycl::group<3> &grp);
template int ishmemx_broadcastmem_work_group<sycl::sub_group>(ishmem_team_t team, void *dest, const void *src, size_t nelems, int PE_root, const sycl::sub_group &grp);
/* clang-format on */

template <typename Group>
inline int ishmemx_broadcastmem_work_group(void *dest, const void *src, size_t nelems, int PE_root,
                                           const Group &grp)
{
    return ishmemx_broadcast_work_group((uint8_t *) dest, (uint8_t *) src, nelems, PE_root, grp);
}

template <typename Group>
inline int ishmemx_broadcastmem_work_group(ishmem_team_t team, void *dest, const void *src,
                                           size_t nelems, int PE_root, const Group &grp)
{
    return ishmemx_broadcast_work_group(team, (uint8_t *) dest, (uint8_t *) src, nelems, PE_root,
                                        grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(TYPENAME, TYPE)                                                                                                      \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::group<1> &grp);    \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::group<2> &grp);    \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::group<3> &grp);    \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::sub_group &grp);  \
    template <typename Group> int ishmemx_##TYPENAME##_broadcast_work_group(TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const Group &grp) { return ishmemx_broadcast_work_group(dest, src, nelems, PE_root, grp); }

#define ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(TYPENAME, TYPE)                                                                                                      \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::group<1> &grp);    \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::group<2> &grp);    \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::group<3> &grp);    \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::sub_group &grp);  \
    template <typename Group> int ishmemx_##TYPENAME##_broadcast_work_group(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const Group &grp) { return ishmemx_broadcast_work_group(team, dest, src, nelems, PE_root, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(double, double)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(ptrdiff, ptrdiff_t)

ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(double, double)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(ptrdiff, ptrdiff_t)

#endif  // COLLECTIVES_BROADCAST_IMPL.H
