/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef COLLECTIVES_COLLECT_IMPL_H
#define COLLECTIVES_COLLECT_IMPL_H

#include "collectives.h"
#include "runtime.h"
#include "runtime_ipc.h"
#include "on_queue.h"

template <typename T>
int ishmem_fcollect(ishmem_team_t team, T *dest, const T *source, size_t nelems);

/* Collect */
template <typename T>
int ishmem_collect(T *dest, const T *src, size_t nelems)
{
    int ret = ishmem_collect(ISHMEM_TEAM_WORLD, dest, src, nelems);
    return ret;
}

/* Collect on a team */
template <typename T>
int ishmem_collect(ishmem_team_t team, T *dest, const T *src, size_t nelems)
{
#if __SYCL_DEVICE_ONLY__
    ishmemi_team_device_t *team_ptr = &global_info->team_device_pool[team];
#else
    ishmemi_team_host_t *team_ptr = &ishmemi_cpu_info->team_host_pool[team];
#endif
    int ret = 0;

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;  // duplicate load from above
        if (team_ptr->only_intra) {
            team_ptr->collect_mynelems = nelems;  // save our nelems into symmetric space
            ishmem_team_sync(
                team);  // fcollect requires input buffer be ready everywhere when fcollect starts
            ishmem_fcollect(team, team_ptr->collect_nelems, &team_ptr->collect_mynelems, 1);
            size_t total_nelems = 0;
            // base_nelems is the offset in the dest buffer of where our data will go
            size_t base_nelems;
            for (int teampe = 0; teampe < team_ptr->size; teampe += 1) {
                if (teampe == team_ptr->my_pe)
                    base_nelems = total_nelems;  // partial sum of pes before us
                total_nelems += team_ptr->collect_nelems[teampe];
            }
            size_t total_nbytes __attribute__((unused)) = sizeof(T) * total_nelems;
            if (!ISHMEM_COLLECT_CUTOVER) {
                if constexpr (enable_error_checking) {
                    size_t nbytes = nelems * sizeof(T);
                    validate_parameters((void *) dest, (void *) src, sizeof(T) * total_nelems,
                                        nbytes, ishmemi_op_t::COLLECT);
                }
                T *ptr[MAX_LOCAL_PES];
                /* compute our address of our section of dest in each PE */
                for (int teampe = 0, globalpe = team_ptr->start; teampe < team_ptr->size;
                     globalpe += team_ptr->stride, teampe += 1) {
                    ptr[teampe] = ISHMEMI_ADJUST_PTR(T, (globalpe + 1), &dest[base_nelems]);
                }
                /* The idea for the inner loop being over local PEs is that the outstanding stores
                 * will use different links */
                for (size_t offset = 0; offset < nelems; offset += 1) {
                    T data = src[offset];
                    for (int teampe = 0; teampe < team_ptr->size; teampe += 1) {
                        ptr[teampe][offset] = data;
                    }
                }
                /* assure all destination buffers complete */
                ishmem_team_sync(team);
                return ret;
            }
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.src = src;
    req.dst = dest;
    req.nelems = nelems * sizeof(T);
    req.op = COLLECT;
    req.type = UINT8;
    req.team = team;

#ifdef __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_status(req);
#else
    if (team_ptr->only_intra && ISHMEMI_HOST_IN_HEAP(dest)) {
        struct put_item items[MAX_LOCAL_PES];
        size_t base_nelems = 0;               // nelem index of where our data goes
        team_ptr->collect_mynelems = nelems;  // save our nelems into symmetric space
        ishmem_team_sync(
            team);  // fcollect requires input buffer be ready everywhere when fcollect starts
        ishmem_fcollect(team, team_ptr->collect_nelems, &team_ptr->collect_mynelems, 1);
        ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "shmem_size_fcollect on team collect failed\n");
        for (int teampe = 0; teampe < team_ptr->my_pe; teampe += 1)
            base_nelems += team_ptr->collect_nelems[teampe];
        for (int teampe = 0, globalpe = team_ptr->start; teampe < team_ptr->size;
             teampe += 1, globalpe += team_ptr->stride) {
            items[teampe].pe = globalpe;
            items[teampe].src = src;
            items[teampe].size = nelems * sizeof(T);
            items[teampe].dst = pointer_offset(dest, base_nelems * sizeof(T));
        }
        ret = ishmemi_ipc_put_v(team_ptr->size, items);
        ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "ishmemi_ipc_put_v within team collect failed\n");
    fn_fail:
        ishmem_team_sync(team); /* assure all destination buffers complete */
        return ret;
    }
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_status(comp.completion.ret);
#endif
    return ret;
}

template <typename T>
sycl::event ishmemx_collect_on_queue(ishmem_team_t team, T *dest, const T *src, size_t nelems,
                                     int *ret, sycl::queue &q, const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);

        size_t max_work_group_size = iter->second->max_work_group_size;
        size_t range_size = (nelems < max_work_group_size) ? nelems : max_work_group_size;
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(range_size), sycl::range<1>(range_size)),
                         [=](sycl::nd_item<1> it) {
                             int tmp_ret = ishmemx_collect_work_group(team, dest, src, nelems,
                                                                      it.get_group());
                             if (ret) *ret = tmp_ret;
                         });
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

template <typename T>
sycl::event ishmemx_collect_on_queue(T *dest, const T *src, size_t nelems, int *ret, sycl::queue &q,
                                     const std::vector<sycl::event> &deps)
{
    return ishmemx_collect_on_queue(ISHMEM_TEAM_WORLD, dest, src, nelems, ret, q, deps);
}

/* Collect (work-group) */
template <typename T, typename Group>
int ishmemx_collect_work_group(T *dest, const T *src, size_t nelems, const Group &grp)
{
    int ret = ishmemx_collect_work_group(ISHMEM_TEAM_WORLD, dest, src, nelems, grp);
    return ret;
}

/* Collect (work-group) on a team */
template <typename T, typename Group>
int ishmemx_collect_work_group(ishmem_team_t team, T *dest, const T *src, size_t nelems,
                               const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        int ret = 0;
        ishmemi_info_t *info = global_info;
        ishmemi_team_device_t *team_ptr = &info->team_device_pool[team];
        size_t my_nelems_work_item;
        size_t work_item_start_idx;
        ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item, work_item_start_idx);
        if (team_ptr->only_intra) {
            if (grp.leader()) {
                team_ptr->collect_mynelems = nelems;  // save our nelems into symmetric space
                ishmem_team_sync(team);  // fcollect requires input buffer be ready everywhere when
                                         // fcollect starts
                ishmem_fcollect(team, team_ptr->collect_nelems, &team_ptr->collect_mynelems, 1);
                if constexpr (enable_error_checking) {
                    /* this copy of total_nelems is only available to the leader thread */
                    size_t total_nelems = 0;
                    for (int teampe = 0; teampe < team_ptr->size; teampe += 1) {
                        total_nelems += team_ptr->collect_nelems[teampe];
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
            size_t base_nelems = 0;
            size_t total_nelems = 0;
            for (int teampe = 0; teampe < team_ptr->size; teampe += 1) {
                if (teampe == team_ptr->my_pe) base_nelems = total_nelems;
                total_nelems += team_ptr->collect_nelems[teampe];
            }
            /* This copy of total_nbytes is available to all threads */
            size_t total_nbytes __attribute__((unused)) = total_nelems * sizeof(T);
            if (!ISHMEM_COLLECT_GROUP_CUTOVER) {
                T *ptr[MAX_LOCAL_PES];
                for (int teampe = 0, globalpe = team_ptr->start; teampe < team_ptr->size;
                     globalpe += team_ptr->stride, teampe += 1) {
                    ptr[teampe] = ISHMEMI_ADJUST_PTR(T, (globalpe + 1), &dest[base_nelems]);
                }
                for (size_t offset = work_item_start_idx;
                     offset < work_item_start_idx + my_nelems_work_item; offset += 1) {
                    T data = src[offset];
                    for (int teampe = 0; teampe < team_ptr->size; teampe += 1) {
                        ptr[teampe][offset] = data;
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
            req.nelems = nelems * sizeof(T);
            req.op = COLLECT;
            req.type = UINT8;
            req.team = team;

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

#define ISHMEMI_API_IMPL_TEAM_COLLECT_WORK_GROUP(TYPENAME, TYPE)                                                                                                       \
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
    int ret = ishmem_fcollect(ISHMEM_TEAM_WORLD, dest, src, nelems);
    return ret;
}

/* Fcollect on a team */
template <typename T>
int ishmem_fcollect(ishmem_team_t team, T *dest, const T *src, size_t nelems)
{
#if __SYCL_DEVICE_ONLY__
    ishmemi_team_device_t *team_ptr = &global_info->team_device_pool[team];
#else
    ishmemi_team_host_t *team_ptr = &ishmemi_cpu_info->team_host_pool[team];
#endif
    size_t nbytes = nelems * sizeof(T);
    if constexpr (enable_error_checking) {
        validate_parameters((void *) dest, (void *) src, nbytes * team_ptr->size, nbytes,
                            ishmemi_op_t::FCOLLECT);
    }

    int ret = 0;

    /* Node-local, on-device implementaiton */
    if constexpr (ishmemi_is_device) {
        if (team_ptr->only_intra && !ISHMEM_FCOLLECT_CUTOVER) {
            size_t base_nelems = static_cast<size_t>(team_ptr->my_pe) * nelems;
            T *ptr[MAX_LOCAL_PES];
            /* compute our address of our section of dest in each PE */
            for (int teampe = 0, globalpe = team_ptr->start; teampe < team_ptr->size;
                 globalpe += team_ptr->stride, teampe++) {
                ptr[teampe] = ISHMEMI_ADJUST_PTR(T, (globalpe + 1), (&dest[base_nelems]));
            }
            /* The idea for the inner loop being over local PEs is that the outstanding stores will
             * use different links */
            for (size_t offset = 0; offset < nelems; offset += 1) {
                T data = src[offset];
                for (int teampe = 0; teampe < team_ptr->size; teampe += 1) {
                    ptr[teampe][offset] = data;
                }
            }
            ishmem_team_sync(team); /* assure all destination buffers complete */
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
    req.team = team;

#ifdef __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_status(req);
#else
    if (team_ptr->only_intra && ISHMEMI_HOST_IN_HEAP(dest)) {
        struct put_item items[MAX_LOCAL_PES];
        for (int teampe = 0, globalpe = team_ptr->start; teampe < team_ptr->size;
             globalpe += team_ptr->stride, teampe++) {
            items[teampe].pe = globalpe;
            items[teampe].src = src;
            items[teampe].size = nbytes;
            items[teampe].dst = pointer_offset(dest, static_cast<size_t>(team_ptr->my_pe) * nbytes);
        }
        int ret = ishmemi_ipc_put_v(team_ptr->size, items);
        ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "ishmemi_ipc_put_v within team fcollect failed\n");
    fn_fail:
        ishmem_team_sync(team); /* assure all destination buffers complete */
        return ret;
    }
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_status(comp.completion.ret);
#endif
    return ret;
}

template <typename T>
sycl::event ishmemx_fcollect_on_queue(ishmem_team_t team, T *dest, const T *src, size_t nelems,
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
        if ((nelems != 0) && (myteam->only_intra) && !ISHMEM_FCOLLECT_GROUP_CUTOVER) {
            size_t max_work_group_size = iter->second->max_work_group_size;
            size_t range_size = (nelems < max_work_group_size) ? nelems : max_work_group_size;
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(range_size), sycl::range<1>(range_size)),
                [=](sycl::nd_item<1> it) {
                    int tmp_ret =
                        ishmemx_fcollect_work_group(team, dest, src, nelems, it.get_group());
                    if (ret) *ret = tmp_ret;
                });
        } else {
            cgh.host_task([=]() {
                int tmp_ret = ishmem_fcollect(team, dest, src, nelems);
                if (ret) *ret = tmp_ret;
            });
        }
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

template <typename T>
sycl::event ishmemx_fcollect_on_queue(T *dest, const T *src, size_t nelems, int *ret,
                                      sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return ishmemx_fcollect_on_queue(ISHMEM_TEAM_WORLD, dest, src, nelems, ret, q, deps);
}

/* Fcollect (work-group) */
template <typename T, typename Group>
int ishmemx_fcollect_work_group(T *dest, const T *src, size_t nelems, const Group &grp)
{
    int ret = ishmemx_fcollect_work_group(ISHMEM_TEAM_WORLD, dest, src, nelems, grp);
    return ret;
}

/* Fcollect (work-group on a team) */
template <typename T, typename Group>
int ishmemx_fcollect_work_group(ishmem_team_t team, T *dest, const T *src, size_t nelems,
                                const Group &grp)
{
    size_t nbytes = nelems * sizeof(T);
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        ishmemi_team_device_t *team_ptr = &info->team_device_pool[team];
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters((void *) dest, (void *) src, nbytes * team_ptr->size, nbytes,
                                    ishmemi_op_t::FCOLLECT);
        }
        int ret = 0;
        size_t my_nelems_work_item;
        size_t work_item_start_idx;
        ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item, work_item_start_idx);
        sycl::group_barrier(grp); /* assure source buffers complete */
        if (team_ptr->only_intra && !ISHMEM_FCOLLECT_GROUP_CUTOVER) {
            size_t base = static_cast<size_t>(team_ptr->my_pe) * nelems;
            T *ptr[MAX_LOCAL_PES];
            for (int teampe = 0, globalpe = team_ptr->start; teampe < team_ptr->size;
                 globalpe += team_ptr->stride, teampe++) {
                ptr[teampe] = ISHMEMI_ADJUST_PTR(T, (globalpe + 1), &dest[base]);
            }
            for (size_t offset = work_item_start_idx;
                 offset < work_item_start_idx + my_nelems_work_item; offset += 1) {
                T data = src[offset];
                for (int teampe = 0; teampe < team_ptr->size; teampe += 1) {
                    ptr[teampe][offset] = data;
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
                req.nelems = nelems * sizeof(T);
                req.op = FCOLLECT;
                req.type = UINT8;
                req.team = team;

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
#define ISHMEMI_API_IMPL_FCOLLECT_WORK_GROUP(TYPENAME, TYPE)                                                                                        \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nelems, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_fcollect_work_group(TYPE *dest, const TYPE *src, size_t nelems, const Group &grp) { return ishmemx_fcollect_work_group(dest, src, nelems, grp); }

#define ISHMEMI_API_IMPL_TEAM_FCOLLECT_WORK_GROUP(TYPENAME, TYPE)                                                                                                       \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_fcollect_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nelems, const sycl::sub_group &grp); \
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
