/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef COLLECTIVES_BROADCAST_IMPL_H
#define COLLECTIVES_BROADCAST_IMPL_H

#include "collectives.h"
#include "sync_impl.h"
#include "runtime.h"
#include "rma_impl.h"
#include "on_queue.h"

#define BROADCAST_PUSH 0
#define BROADCAST_PULL 1

/* Broadcast on a team: */
template <typename T>
int ishmem_broadcast(ishmem_team_t team, T *dest, const T *src, size_t nelems, int PE_root)
{
#if __SYCL_DEVICE_ONLY__
    ishmemi_team_device_t *team_ptr = &global_info->team_device_pool[team];
#else
    ishmemi_team_host_t *team_ptr = &ishmemi_cpu_info->team_host_pool[team];
#endif

    int ret = 0;
    size_t nbytes = nelems * sizeof(T);

    if constexpr (enable_error_checking) {
        validate_parameters(ishmem_team_translate_pe(team, PE_root, ISHMEM_TEAM_WORLD),
                            (void *) dest, (void *) src, nbytes, nbytes);
    }

    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (team_ptr->only_intra && !ISHMEM_BROADCAST_CUTOVER) {
#if BROADCAST_PUSH
            if (team_ptr->my_pe == PE_root) {
                T *ptr[MAX_LOCAL_PES];
                /* compute our address dest in each PE on the team */
                int idx = 0;
                for (int pe = team_ptr->start; idx < team_ptr->size;
                     pe += team_ptr->stride, idx++) {
                    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
                    ptr[idx] = ISHMEMI_ADJUST_PTR(T, local_index, dest);
                }
                /* The idea for the inner loop being over local PEs is that the outstanding stores
                 * will use different links.
                 * Because the loop above sets the dptr and sptr buffers contiguously, we can loop
                 * over the team members without the stride. */
                for (size_t offset = 0; offset < nelems; offset += 1) {
                    T data = src[offset];
                    for (int pe = 0; pe < team_ptr->size; pe += 1) {
                        ptr[pe][offset] = data;
                    }
                }
            }
            ishmemi_team_sync(team); /* assure destination buffers complete */
            return ret;
#else  // BROADCAST_PULL
            ishmemi_team_sync(team); /* make sure that PE_root's source buffer is ready for use */
            ishmem_internal_get(dest, src, nelems,
                                ishmem_team_translate_pe(team, PE_root, ISHMEM_TEAM_WORLD));
            ishmemi_team_sync(team); /* sync after to let PE_root know we are done */
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
    req.team = team;

#ifdef __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_status(req);
#else
#if BROADCAST_PUSH
    if (team_ptr->only_intra && ISHMEMI_HOST_IN_HEAP(dest)) {
        if (team_ptr->my_pe == PE_root) {
            struct put_item items[MAX_LOCAL_PES];
            int idx = 0;
            for (int pe = team_ptr->start; idx < team_ptr->size; pe += team_ptr->stride, idx++) {
                items[idx].pe = pe;
                items[idx].src = src;
                items[idx].size = nbytes;
                items[idx].dst = dest;
            }
            int ret = ishmemi_ipc_put_v(team_ptr->size, items);
            ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "ishmemi_ipc_put_v within team broadcast failed\n");
        }
    fn_fail:
        ishmemi_team_sync(team); /* assure all destination buffers complete */
        return ret;
    }
#else
    if (team_ptr->only_intra && ISHMEMI_HOST_IN_HEAP(src)) {
        ishmemi_team_sync(team); /* assure PE_root source buffer is ready for use */
        ret = ishmem_team_translate_pe(team, PE_root, ISHMEM_TEAM_WORLD);
        ISHMEM_CHECK_GOTO_MSG((ret < 0), fn_fail,
                              "ishmem_team_translate_pe within team broadcast failed\n");
        ret = ishmemi_ipc_get(dest, src, nelems, ret);
    fn_fail:
        ishmemi_team_sync(team); /* assure PE_root can reuse source buffer */
        return ret;
    }
#endif
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_status(comp.completion.ret);
#endif
    return ret;
}

template <typename T>
sycl::event ishmemx_broadcast_on_queue(ishmem_team_t team, T *dest, const T *src, size_t nelems,
                                       int PE_root, int *ret, sycl::queue &q,
                                       const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    size_t nbytes = nelems * sizeof(T);
    ishmemi_team_host_t *myteam = &ishmemi_cpu_info->team_host_pool[team];
    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        if ((nelems != 0) && (myteam->only_intra) && !ISHMEM_BROADCAST_GROUP_CUTOVER) {
            size_t max_work_group_size = iter->second->max_work_group_size;
            size_t range_size = (nelems < max_work_group_size) ? nelems : max_work_group_size;
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(range_size), sycl::range<1>(range_size)),
                [=](sycl::nd_item<1> it) {
                    int tmp_ret = ishmemx_broadcast_work_group(team, dest, src, nelems, PE_root,
                                                               it.get_group());
                    if (ret) *ret = tmp_ret;
                });
        } else {
            cgh.single_task([=]() {
                int tmp_ret = ishmem_broadcast(team, dest, src, nelems, PE_root);
                if (ret) *ret = tmp_ret;
            });
        }
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

template <typename T>
sycl::event ishmemx_broadcast_on_queue(T *dest, const T *src, size_t nelems, int PE_root, int *ret,
                                       sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return ishmemx_broadcast_on_queue(ISHMEM_TEAM_WORLD, dest, src, nelems, PE_root, ret, q, deps);
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
        ishmemi_team_device_t *team_ptr = &info->team_device_pool[team];
        if constexpr (enable_error_checking) {
            if (grp.leader()) {
                validate_parameters(ishmem_team_translate_pe(team, PE_root, ISHMEM_TEAM_WORLD),
                                    (void *) dest, (void *) src, nbytes, nbytes);
            }
        }

        if (team_ptr->only_intra && !ISHMEM_BROADCAST_GROUP_CUTOVER) {
#if BROADCAST_PUSH
            /* make sure all threads have reached here, so source ready for use */
            sycl::group_barrier(grp);
            if (team_ptr->my_pe == PE_root) {
                size_t my_nelems_work_item;
                size_t work_item_start_idx;
                ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item,
                                                   work_item_start_idx);
                T *ptr[MAX_LOCAL_PES];
                int idx = 0;
                for (int pe = team_ptr->start; idx < team_ptr->size;
                     pe += team_ptr->stride, idx++) {
                    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
                    ptr[idx] = ISHMEMI_ADJUST_PTR(T, local_index, dest);
                }
                for (size_t offset = work_item_start_idx;
                     offset < work_item_start_idx + my_nelems_work_item; offset += 1) {
                    T data = src[offset];
                    for (int pe = 0; pe < team_ptr->size; pe += 1) {
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
                dest, src, nelems, ishmem_team_translate_pe(team, PE_root, ISHMEM_TEAM_WORLD), grp);
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
                req.team = team;
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

/* Broadcast */
template <typename T>
int ishmem_broadcast(T *dest, const T *src, size_t nelems, int PE_root)
{
    int ret = ishmem_broadcast(ISHMEM_TEAM_WORLD, dest, src, nelems, PE_root);
    return ret;
}

/* Broadcast (work-group) */
template <typename T, typename Group>
int ishmemx_broadcast_work_group(T *dest, const T *src, size_t nelems, int PE_root,
                                 const Group &grp)
{
    int ret = ishmemx_broadcast_work_group(ISHMEM_TEAM_WORLD, dest, src, nelems, PE_root, grp);
    return (ret);
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
#define ISHMEMI_API_IMPL_BROADCAST_WORK_GROUP(TYPENAME, TYPE)                                                                                                     \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_broadcast_work_group(TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const Group &grp) { return ishmemx_broadcast_work_group(dest, src, nelems, PE_root, grp); }

#define ISHMEMI_API_IMPL_TEAM_BROADCAST_WORK_GROUP(TYPENAME, TYPE)                                                                                                                    \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_broadcast_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, int PE_root, const sycl::sub_group &grp); \
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
