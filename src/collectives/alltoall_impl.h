/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef COLLECTIVES_ALLTOALL_IMPL_H
#define COLLECTIVES_ALLTOALL_IMPL_H

#include "collectives.h"
#include "sync_impl.h"
#include "runtime.h"
#include "on_queue.h"

/* Alltoall */
template <typename T>
int ishmem_alltoall(T *dest, const T *src, size_t nelems)
{
    int ret = ishmem_alltoall(ISHMEM_TEAM_WORLD, dest, src, nelems);
    return ret;
}

/* Alltoall on a team */
template <typename T>
int ishmem_alltoall(ishmem_team_t team, T *dest, const T *src, size_t nelems)
{
#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_team_device_t *team_ptr = &global_info->team_device_pool[team];
#else
    ishmemi_team_host_t *team_ptr = &ishmemi_cpu_info->team_host_pool[team];
#endif
    size_t nbytes = nelems * sizeof(T);
    if constexpr (enable_error_checking) {
        int n_pes = team_ptr->size;
        validate_parameters((void *) dest, (void *) src, nelems * sizeof(T) * n_pes, nbytes,
                            ishmemi_op_t::ALLTOALL);
    }

    int ret = 0;

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        if (team_ptr->only_intra && !ISHMEM_ALLTOALL_CUTOVER) {
            const T *sptr[MAX_LOCAL_PES]; /* source pointer for each pe */
            T *dptr[MAX_LOCAL_PES];       /* destination pointer for each pe */
            /* compute our address of our section of dest in each PE */
            int idx = 0;
            for (int pe = team_ptr->start; idx < team_ptr->size; pe += team_ptr->stride, idx++) {
                uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
                dptr[idx] = ISHMEMI_ADJUST_PTR(
                    T, local_index, &dest[nelems * static_cast<size_t>(team_ptr->my_pe)]);
                sptr[idx] = &src[nelems * static_cast<size_t>(idx)];
            }
            /* The idea for the inner loop being over local team PEs is that the outstanding stores
             * will use different links.
             * Because the loop above sets the dptr and sptr buffers contiguously, we can loop over
             * the team members without the stride. */
            for (size_t offset = 0; offset < nelems; offset += 1) {
                for (int pe = 0; pe < team_ptr->size; pe += 1) {
                    dptr[pe][offset] = sptr[pe][offset];
                }
            }
            ishmemi_team_sync(team); /* assure destination buffers complete */
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
    req.team = team;

#ifdef __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_status(req);
#else
    if (team_ptr->only_intra && ISHMEMI_HOST_IN_HEAP(dest)) {
        struct put_item items[MAX_LOCAL_PES];
        size_t size = nelems * sizeof(T);
        int idx = 0;
        for (int pe = team_ptr->start; idx < team_ptr->size; pe += team_ptr->stride, idx++) {
            items[idx].pe = pe;
            items[idx].src = pointer_offset(src, static_cast<size_t>(idx) * size);
            items[idx].size = size;
            items[idx].dst = pointer_offset(dest, static_cast<size_t>(team_ptr->my_pe) * size);
        }
        int ret = ishmemi_ipc_put_v(team_ptr->size, items);
        ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "ishmemi_ipc_put_v within team alltoall failed\n");
    fn_fail:
        ishmemi_team_sync(team); /* assure destination buffers complete */
        return ret;
    }
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_status(comp.completion.ret);
#endif
    return ret;
}

template <typename T>
sycl::event ishmemx_alltoall_on_queue(ishmem_team_t team, T *dest, const T *src, size_t nelems,
                                      int *ret, sycl::queue &q,
                                      const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    size_t nbytes = nelems * sizeof(T);
    ishmemi_team_host_t *myteam = &ishmemi_cpu_info->team_host_pool[team];
    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        if ((nelems != 0) && (myteam->only_intra) && !ISHMEM_ALLTOALL_GROUP_CUTOVER) {
            size_t max_work_group_size = iter->second->max_work_group_size;
            size_t range_size = (nelems < max_work_group_size) ? nelems : max_work_group_size;
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(range_size), sycl::range<1>(range_size)),
                [=](sycl::nd_item<1> it) {
                    int tmp_ret =
                        ishmemx_alltoall_work_group(team, dest, src, nelems, it.get_group());
                    if (ret) *ret = tmp_ret;
                });
        } else {
            cgh.single_task([=]() {
                int tmp_ret = ishmem_alltoall(team, dest, src, nelems);
                if (ret) *ret = tmp_ret;
            });
        }
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

template <typename T>
sycl::event ishmemx_alltoall_on_queue(T *dest, const T *src, size_t nelems, int *ret,
                                      sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return ishmemx_alltoall_on_queue(ISHMEM_TEAM_WORLD, dest, src, nelems, ret, q, deps);
}

/* Alltoall (work-group) */
template <typename T, typename Group>
int ishmemx_alltoall_work_group(T *dest, const T *src, size_t nelems, const Group &grp)
{
    int ret = ishmemx_alltoall_work_group(ISHMEM_TEAM_WORLD, dest, src, nelems, grp);
    return ret;
}

/* Alltoall (work-group) on a team */
template <typename T, typename Group>
int ishmemx_alltoall_work_group(ishmem_team_t team, T *dest, const T *src, size_t nelems,
                                const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        ishmemi_team_device_t *team_ptr = &info->team_device_pool[team];
        size_t nbytes = nelems * sizeof(T);
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters((void *) dest, (void *) src,
                                    nelems * sizeof(T) * team_ptr->size, nbytes,
                                    ishmemi_op_t::ALLTOALL);
        }
        int ret = 0;
        size_t my_nelems_work_item;
        size_t work_item_start_idx;
        ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item, work_item_start_idx);
        sycl::group_barrier(grp); /* assure source buffers complete on all threads */
        if (team_ptr->only_intra && !ISHMEM_ALLTOALL_GROUP_CUTOVER) {
            const T *sptr[MAX_LOCAL_PES]; /* source pointer for each pe */
            T *dptr[MAX_LOCAL_PES];       /* destination pointer for each pe*/
            int idx = 0;
            for (int pe = team_ptr->start; idx < team_ptr->size; pe += team_ptr->stride, idx++) {
                uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
                dptr[idx] = ISHMEMI_ADJUST_PTR(
                    T, local_index, &dest[nelems * static_cast<size_t>(team_ptr->my_pe)]);
                sptr[idx] = &src[nelems * static_cast<size_t>(idx)];
            }
            for (size_t offset = work_item_start_idx;
                 offset < work_item_start_idx + my_nelems_work_item; offset += 1) {
                for (int pe = 0; pe < team_ptr->size; pe += 1) {
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
                req.team = team;

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
