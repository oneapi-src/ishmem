/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef COLLECTIVES_REDUCE_IMPL_H
#define COLLECTIVES_REDUCE_IMPL_H

#include "ishmem/err.h"
#include "ishmem/copy.h"
#include "collectives.h"
#include <type_traits>
#include "memory.h"
#include "runtime.h"
#include "on_queue.h"

#define IN_HEAP(p)                                                                                 \
    ((((uintptr_t) p) >= ((uintptr_t) ishmemi_heap_base)) &&                                       \
     (((uintptr_t) p) < (((uintptr_t) ishmemi_heap_base) + ishmemi_heap_length)))

/* Starting in oneAPI 2025.0, sycl::vec only supports fixed-width integer types */
#define vector_reduce_helper(fn, d, s, n, ...)                                                     \
    if constexpr (std::is_integral_v<T>) {                                                         \
        if constexpr (std::is_signed_v<T>) {                                                       \
            if constexpr (sizeof(T) == sizeof(int8_t))                                             \
                fn<int8_t, OP>(reinterpret_cast<int8_t *>(d), reinterpret_cast<int8_t *>(s),       \
                               n __VA_OPT__(, ) __VA_ARGS__);                                      \
            else if constexpr (sizeof(T) == sizeof(int16_t))                                       \
                fn<int16_t, OP>(reinterpret_cast<int16_t *>(d), reinterpret_cast<int16_t *>(s),    \
                                n __VA_OPT__(, ) __VA_ARGS__);                                     \
            else if constexpr (sizeof(T) == sizeof(int32_t))                                       \
                fn<int32_t, OP>(reinterpret_cast<int32_t *>(d), reinterpret_cast<int32_t *>(s),    \
                                n __VA_OPT__(, ) __VA_ARGS__);                                     \
            else if constexpr (sizeof(T) == sizeof(int64_t))                                       \
                fn<int64_t, OP>(reinterpret_cast<int64_t *>(d), reinterpret_cast<int64_t *>(s),    \
                                n __VA_OPT__(, ) __VA_ARGS__);                                     \
            else                                                                                   \
                fn<int64_t, OP>(reinterpret_cast<int64_t *>(d), reinterpret_cast<int64_t *>(s),    \
                                n * (sizeof(T) / sizeof(int64_t)) __VA_OPT__(, ) __VA_ARGS__);     \
        } else {                                                                                   \
            if constexpr (sizeof(T) == sizeof(uint8_t))                                            \
                fn<uint8_t, OP>(reinterpret_cast<uint8_t *>(d), reinterpret_cast<uint8_t *>(s),    \
                                n __VA_OPT__(, ) __VA_ARGS__);                                     \
            else if constexpr (sizeof(T) == sizeof(uint16_t))                                      \
                fn<uint16_t, OP>(reinterpret_cast<uint16_t *>(d), reinterpret_cast<uint16_t *>(s), \
                                 n __VA_OPT__(, ) __VA_ARGS__);                                    \
            else if constexpr (sizeof(T) == sizeof(uint32_t))                                      \
                fn<uint32_t, OP>(reinterpret_cast<uint32_t *>(d), reinterpret_cast<uint32_t *>(s), \
                                 n __VA_OPT__(, ) __VA_ARGS__);                                    \
            else if constexpr (sizeof(T) == sizeof(uint64_t))                                      \
                fn<uint64_t, OP>(reinterpret_cast<uint64_t *>(d), reinterpret_cast<uint64_t *>(s), \
                                 n __VA_OPT__(, ) __VA_ARGS__);                                    \
            else                                                                                   \
                fn<uint64_t, OP>(reinterpret_cast<uint64_t *>(d), reinterpret_cast<uint64_t *>(s), \
                                 n * (sizeof(T) / sizeof(uint64_t)) __VA_OPT__(, ) __VA_ARGS__);   \
        }                                                                                          \
    } else {                                                                                       \
        fn<T, OP>(d, s, n __VA_OPT__(, ) __VA_ARGS__);                                             \
    }

/* Reduction operators */
template <typename T, int N, ishmemi_op_t OP>
inline void reduce_op(sycl::vec<T, N> &dest, const sycl::vec<T, N> &remote)
{
    if constexpr (std::is_floating_point_v<T>) {
        if constexpr (OP == MAX_REDUCE) dest = sycl::fmax(dest, remote);
        else if constexpr (OP == MIN_REDUCE) dest = sycl::fmin(dest, remote);
        else if constexpr (OP == SUM_REDUCE) dest += remote;
        else if constexpr (OP == PROD_REDUCE) dest *= remote;
        else static_assert(false, "Unknown or unsupported reduction operator");
    } else {
        if constexpr (OP == AND_REDUCE) dest &= remote;
        else if constexpr (OP == OR_REDUCE) dest |= remote;
        else if constexpr (OP == XOR_REDUCE) dest ^= remote;
        else if constexpr (OP == MAX_REDUCE) dest = sycl::max(dest, remote);
        else if constexpr (OP == MIN_REDUCE) dest = sycl::min(dest, remote);
        else if constexpr (OP == SUM_REDUCE) dest += remote;
        else if constexpr (OP == PROD_REDUCE) dest *= remote;
        else static_assert(false, "Unknown or unsupported reduction operator");
    }
}

template <typename T, ishmemi_op_t OP>
inline void reduce_op(T &dest, const T &remote)
{
    if constexpr (std::is_floating_point_v<T>) {
        if constexpr (OP == MAX_REDUCE) dest = sycl::fmax(dest, remote);
        else if constexpr (OP == MIN_REDUCE) dest = sycl::fmin(dest, remote);
        else if constexpr (OP == SUM_REDUCE) dest += remote;
        else if constexpr (OP == PROD_REDUCE) dest *= remote;
        else static_assert(false, "Unknown or unsupported reduction operator");
    } else {
        if constexpr (OP == AND_REDUCE) dest &= remote;
        else if constexpr (OP == OR_REDUCE) dest |= remote;
        else if constexpr (OP == XOR_REDUCE) dest ^= remote;
        else if constexpr (OP == MAX_REDUCE) dest = sycl::max(dest, remote);
        else if constexpr (OP == MIN_REDUCE) dest = sycl::min(dest, remote);
        else if constexpr (OP == SUM_REDUCE) dest += remote;
        else if constexpr (OP == PROD_REDUCE) dest *= remote;
        else static_assert(false, "Unknown or unsupported reduction operator");
    }
}

/* Vector reduce */
template <typename T, ishmemi_op_t OP>
inline void vector_reduce(T *d, const T *s, size_t count)
{
    while ((((uintptr_t) s) & ISHMEMI_ALIGNMASK) && (count > 0)) {
        reduce_op<T, OP>(*d, *s);
        d += 1;
        s += 1;
        count -= 1;
    }
    sycl::multi_ptr<T, sycl::access::address_space::global_space, sycl::access::decorated::yes> dd;
    sycl::multi_ptr<const T, sycl::access::address_space::global_space,
                    sycl::access::decorated::yes>
        ds;
    dd = sycl::address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::yes>(d);
    ds = sycl::address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::yes>(s);
    sycl::vec<T, 16> vs;
    sycl::vec<T, 16> vd;
    while (count >= 16) {
        vs.load(0, ds);
        vd.load(0, dd);
        reduce_op<T, 16, OP>(vd, vs);
        vd.store(0, dd);
        ds += 16;
        dd += 16;
        count -= 16;
    }
    while (count > 0) {
        reduce_op<T, OP>(*dd, *ds);
        dd += 1;
        ds += 1;
        count -= 1;
    }
}

/* Vector reduce (work-group) */
template <typename T, ishmemi_op_t OP, typename Group>
inline void vector_reduce_work_group(T *d, const T *s, size_t count, const Group &grp)
{
    size_t stride = grp.get_local_linear_range();
    long linear_id = static_cast<long>(grp.get_local_linear_id());
    long idx = linear_id;
    T *aligned_s = (T *) sycl::min(((((uintptr_t) s) + ISHMEMI_ALIGNMASK) & (~ISHMEMI_ALIGNMASK)),
                                   (uintptr_t) (s + count));
    while (((uintptr_t) &s[idx]) < ((uintptr_t) (aligned_s))) {
        reduce_op<T, OP>(d[idx], s[idx]);
        idx += stride;
    }
    count -= static_cast<size_t>(aligned_s - s);  // pointer difference is in units of T
    d += (aligned_s - s);
    /* at this point, if count > 0, then d is aligned, s may not be aligned */
    if (count == 0) return;
    idx = linear_id * ishmemi_vec_length;
    size_t vstride = stride * ishmemi_vec_length;

    sycl::multi_ptr<T, sycl::access::address_space::global_space, sycl::access::decorated::yes> dd;
    sycl::multi_ptr<const T, sycl::access::address_space::global_space,
                    sycl::access::decorated::yes>
        ds;
    dd = sycl::address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::yes>(d);
    ds = sycl::address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::yes>(aligned_s);
    sycl::vec<T, 16> vs;
    sycl::vec<T, 16> vd;
    while ((idx + ishmemi_vec_length) <= count) {
        vs.load(0, ds + idx);
        vd.load(0, dd + idx);
        reduce_op<T, 16, OP>(vd, vs);
        vd.store(0, dd + idx);
        idx += vstride;
    }
    idx = linear_id + (static_cast<long>(count) & (~(static_cast<long>(ishmemi_vec_length) - 1)));
    while (idx < count) {
        reduce_op<T, OP>(dd[idx], ds[idx]);
        idx += stride;
    }
}

/* on a team... */
template <typename T, ishmemi_op_t OP>
int ishmemi_generic_op_reduce(ishmem_team_t team, T *dest, const T *src, size_t nreduce)
{
    int ret = 0;
    size_t max_reduce = ISHMEM_REDUCE_BUFFER_SIZE / sizeof(T);
    ishmemi_team_host_t *team_ptr = &ishmemi_cpu_info->team_host_pool[team];

    while (nreduce > 0) {
        size_t this_reduce = nreduce;
        if (this_reduce > max_reduce) this_reduce = max_reduce;
        void *cdst = ishmem_copy((void *) team_ptr->source, (void *) src, this_reduce * sizeof(T));
        if ((uintptr_t) cdst != (uintptr_t) team_ptr->source) {
            ISHMEM_DEBUG_MSG("ishmem_copy in failed\n");
            return (1);
        }

        ishmemi_ringcompletion_t comp;
        ishmemi_request_t req;
        req.src = team_ptr->source;
        req.dst = team_ptr->dest;
        req.nelems = this_reduce;
        req.op = OP;
        req.type = ishmemi_union_get_base_type<T, OP>();
        req.team = team;

        ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
        ret = ishmemi_proxy_get_status(comp.completion.ret);

        if (ret != 0) {
            ISHMEM_DEBUG_MSG("runtime reduction failed\n");
            return ret;
        }
        cdst = ishmem_copy((void *) dest, (void *) team_ptr->dest, this_reduce * sizeof(T));
        if ((uintptr_t) cdst != (uintptr_t) dest) {
            ISHMEM_DEBUG_MSG("ishmem_copy out failed\n");
            return 1;
        }
        dest += this_reduce;
        src += this_reduce;
        nreduce -= this_reduce;
    }
    return (0);
}

/* Sub-reduce - SYCL can't make recursive calls, so this function is used by the top-level reduce */
/* on a team... */
template <typename T, ishmemi_op_t OP>
inline int ishmemi_sub_reduce(ishmem_team_t team, T *dest, const T *source, size_t nreduce)
{
    ishmemi_info_t *info = global_info;
    int ret = 0;
#if __SYCL_DEVICE_ONLY__
    ishmemi_team_device_t *team_ptr = &info->team_device_pool[team];
#else
    ishmemi_team_host_t *team_ptr = &ishmemi_cpu_info->team_host_pool[team];
#endif
    int my_world_pe = ishmem_team_translate_pe(team, team_ptr->my_pe, ISHMEM_TEAM_WORLD);

    ishmem_team_sync(team); /* assure all source buffers are ready for use */

    int idx = 0;
    for (int pe = team_ptr->start; idx < team_ptr->size; pe += team_ptr->stride, idx++) {
        if (pe == my_world_pe) continue;
        T *remote = ISHMEMI_FAST_ADJUST(T, info, info->local_pes[pe], source);

        vector_reduce_helper(vector_reduce, dest, remote, nreduce);
    }
    ishmem_team_sync(team);
    return ret;
}

/* on a team... */
template <typename T, ishmemi_op_t OP>
inline int ishmemi_reduce(ishmem_team_t team, T *dest, const T *source, size_t nreduce)
{
#if __SYCL_DEVICE_ONLY__
    ishmemi_team_device_t *team_ptr = &global_info->team_device_pool[team];
#else
    ishmemi_team_host_t *team_ptr = &ishmemi_cpu_info->team_host_pool[team];
#endif
    if constexpr (enable_error_checking) {
        validate_parameters((void *) dest, (void *) source, nreduce * sizeof(T));
    }

    if constexpr (ishmemi_is_device) {
        /* if this operation involves multiple nodes, just call the proxy */
        if (team_ptr->only_intra) {
            size_t max_nreduce = ISHMEM_REDUCE_BUFFER_SIZE / sizeof(T);
            if (source == dest) {
                while (nreduce > 0) {
                    size_t this_nreduce = (nreduce < max_nreduce) ? nreduce : max_nreduce;
                    vec_copy_push((T *) (team_ptr->buffer), source, this_nreduce);
                    int res =
                        ishmemi_sub_reduce<T, OP>(team, dest, (T *) team_ptr->buffer, this_nreduce);
                    if (res != 0) return res;
                    dest += this_nreduce;
                    source += this_nreduce;
                    nreduce -= this_nreduce;
                }
                return 0;
            }
            vec_copy_push(dest, source, nreduce);
            return ishmemi_sub_reduce<T, OP>(team, dest, source, nreduce);
        }
    }

#ifndef __SYCL_DEVICE_ONLY__
    /* if source and dest are both host memory, then call shmem */
    /* at the moment, device memory must be in the symmetric heap, which is an easier test */
    if (IN_HEAP(dest) || IN_HEAP(source)) {
        return (ishmemi_generic_op_reduce<T, OP>(team, dest, source, nreduce));
    }
#endif

    /* Otherwise */
    ishmemi_request_t req;
    req.src = source;
    req.dst = dest;
    req.nelems = nreduce;
    req.op = OP;
    req.type = ishmemi_union_get_base_type<T, OP>();
    req.team = team;

#ifdef __SYCL_DEVICE_ONLY__
    return ishmemi_proxy_blocking_request_status(req);
#else
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    return ishmemi_proxy_get_status(comp.completion.ret);
#endif
}

/* Reduce - top-level reduce */
/* device code */
template <typename T, ishmemi_op_t OP>
inline int ishmemi_reduce(T *dest, const T *source, size_t nreduce)
{
    int ret = ishmemi_reduce<T, OP>(ISHMEM_TEAM_WORLD, dest, source, nreduce);
    return ret;
}

/* Sub-reduce (work-group) - SYCL can't make recursive calls, so this function is used by the
 * top-level reduce */
/* TODO: figure out how to spread the available threads across all the remote PEs, so as to equalize
 * xe-link loading */
/* on a team... */
template <typename T, ishmemi_op_t OP, typename Group>
inline int ishmemi_sub_reduce_work_group(ishmem_team_t team, T *dest, const T *source,
                                         size_t nreduce, const Group &grp)
{
#if __SYCL_DEVICE_ONLY__
    ishmemi_team_device_t *team_ptr = &global_info->team_device_pool[team];
#else
    ishmemi_team_host_t *team_ptr = &ishmemi_cpu_info->team_host_pool[team];
#endif
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        int ret = 0;
        if (team_ptr->only_intra) {
            /* assure local source buffer ready for use (group_garrier)
             * assure all source buffers ready for use (sync all)
             */
            ishmemx_team_sync_work_group(team, grp);
            int idx = 0;
            int my_world_pe = ishmem_team_translate_pe(team, team_ptr->my_pe, ISHMEM_TEAM_WORLD);
            for (int pe = team_ptr->start; idx < team_ptr->size; pe += team_ptr->stride, idx++) {
                if (pe == my_world_pe) continue;
                T *remote = ISHMEMI_FAST_ADJUST(T, info, info->local_pes[pe], source);

                vector_reduce_helper(vector_reduce_work_group, dest, remote, nreduce, grp);
            }
            /* assure all threads have finished copies (group_barrier)
             * assure destination buffers complete and source buffers may be reused */
            ishmemx_team_sync_work_group(team, grp);
            /* group broadcast not needed because ret is always 0 here */
            return (ret);
        } else {
            if (grp.leader()) {
                ishmemi_request_t req;
                req.src = source;
                req.dst = dest;
                req.nelems = nreduce;
                req.op = OP;
                req.type = ishmemi_union_get_base_type<T, OP>();
                req.team = team;

                ret = ishmemi_proxy_blocking_request_status(req);
                ret = sycl::group_broadcast(grp, ret, 0);
                return (ret);
            }
        }
    } else {
        /* Safe check and return; should never be reached */
        return -1;
    }
}

/* on a team... */
template <typename T, ishmemi_op_t OP, typename Group>
inline int ishmemi_reduce_work_group(ishmem_team_t team, T *dest, const T *source, size_t nreduce,
                                     const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters((void *) dest, (void *) source, nreduce * sizeof(T));
        }
        size_t max_nreduce = ISHMEM_REDUCE_BUFFER_SIZE / sizeof(T);
        if (source == dest) {
            while (nreduce > 0) {
                size_t this_nreduce = (nreduce < max_nreduce) ? nreduce : max_nreduce;
                T *temp_buffer = (T *) info->team_device_pool[team].buffer;

                vec_copy_work_group_push(temp_buffer, source, this_nreduce, grp);
                int res = ishmemi_sub_reduce_work_group<T, OP>(team, dest, temp_buffer,
                                                               this_nreduce, grp);
                if (res != 0) return res;
                dest += this_nreduce;
                source += this_nreduce;
                nreduce -= this_nreduce;
            }
            return 0;
        }
        vec_copy_work_group_push(dest, source, nreduce, grp);
        return ishmemi_sub_reduce_work_group<T, OP>(team, dest, source, nreduce, grp);
    } else {
        ISHMEM_ERROR_MSG("ISHMEMX_REDUCE_WORK_GROUP routines are not callable from host\n");
        return -1;
    }
}

/* Reduce (work-group) - top-level reduce */
/* TODO: figure out how to spread the available threads across all the remote PEs, so as to equalize
 * xe-link loading */
template <typename T, ishmemi_op_t OP, typename Group>
inline int ishmemi_reduce_work_group(T *dest, const T *source, size_t nreduce, const Group &grp)
{
    int ret = ishmemi_reduce_work_group<T, OP>(ISHMEM_TEAM_WORLD, dest, source, nreduce, grp);
    return ret;
}

/* And Reduce */
template <typename T>
int ishmem_and_reduce(T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, AND_REDUCE>(dest, src, nreduce);
}

/* And Reduce on a team */
template <typename T>
int ishmem_and_reduce(ishmem_team_t team, T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, AND_REDUCE>(team, dest, src, nreduce);
}

template <typename T, ishmemi_op_t OP>
sycl::event ishmemi_reduce_on_queue(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                    int *ret, sycl::queue &q, const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    ishmemi_team_host_t *myteam = &ishmemi_cpu_info->team_host_pool[team];
    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        if ((nreduce != 0) && (myteam->only_intra)) {
            size_t max_work_group_size = iter->second->max_work_group_size;
            size_t range_size = (nreduce < max_work_group_size) ? nreduce : max_work_group_size;
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(range_size), sycl::range<1>(range_size)),
                [=](sycl::nd_item<1> it) {
                    int tmp_ret =
                        ishmemi_reduce_work_group<T, OP>(team, dest, src, nreduce, it.get_group());
                    if (ret) *ret = tmp_ret;
                });
        } else {
            cgh.host_task([=]() {
                int tmp_ret = ishmemi_reduce<T, OP>(team, dest, src, nreduce);
                if (ret) *ret = tmp_ret;
            });
        }
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

template <typename T>
sycl::event ishmemx_and_reduce_on_queue(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                        int *ret, sycl::queue &q,
                                        const std::vector<sycl::event> &deps)
{
    return ishmemi_reduce_on_queue<T, AND_REDUCE>(team, dest, src, nreduce, ret, q, deps);
}

template <typename T>
sycl::event ishmemx_and_reduce_on_queue(T *dest, const T *src, size_t nreduce, int *ret,
                                        sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return ishmemi_reduce_on_queue<T, AND_REDUCE>(ISHMEM_TEAM_WORLD, dest, src, nreduce, ret, q,
                                                  deps);
}

template <typename T, typename Group>
int ishmemx_and_reduce_work_group(T *dest, const T *src, size_t nreduce, const Group &grp)
{
    return ishmemi_reduce_work_group<T, AND_REDUCE, Group>(dest, src, nreduce, grp);
}

template <typename T, typename Group>
int ishmemx_and_reduce_work_group(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                  const Group &grp)
{
    return ishmemi_reduce_work_group<T, AND_REDUCE, Group>(team, dest, src, nreduce, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_and_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_and_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_and_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_and_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_and_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_and_reduce_work_group(dest, src, nreduce, grp); }

#define ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                                        \
    template int ishmemx_##TYPENAME##_and_reduce_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_and_reduce_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_and_reduce_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_and_reduce_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_and_reduce_work_group(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_and_reduce_work_group(team, dest, src, nreduce, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(size, size_t)

ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(size, size_t)

/* Or Reduce */
template <typename T>
int ishmem_or_reduce(T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, OR_REDUCE>(dest, src, nreduce);
}

/* Or Reduce on a team */
template <typename T>
int ishmem_or_reduce(ishmem_team_t team, T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, OR_REDUCE>(team, dest, src, nreduce);
}

template <typename T>
sycl::event ishmemx_or_reduce_on_queue(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                       int *ret, sycl::queue &q,
                                       const std::vector<sycl::event> &deps)
{
    return ishmemi_reduce_on_queue<T, OR_REDUCE>(team, dest, src, nreduce, ret, q, deps);
}

template <typename T>
sycl::event ishmemx_or_reduce_on_queue(T *dest, const T *src, size_t nreduce, int *ret,
                                       sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return ishmemi_reduce_on_queue<T, OR_REDUCE>(ISHMEM_TEAM_WORLD, dest, src, nreduce, ret, q,
                                                 deps);
}

template <typename T, typename Group>
int ishmemx_or_reduce_work_group(T *dest, const T *src, size_t nreduce, const Group &grp)
{
    return ishmemi_reduce_work_group<T, OR_REDUCE, Group>(dest, src, nreduce, grp);
}

template <typename T, typename Group>
int ishmemx_or_reduce_work_group(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                 const Group &grp)
{
    return ishmemi_reduce_work_group<T, OR_REDUCE, Group>(team, dest, src, nreduce, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_or_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_or_reduce_work_group(dest, src, nreduce, grp); }

#define ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                                        \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_or_reduce_work_group(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_or_reduce_work_group(team, dest, src, nreduce, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(size, size_t)

ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(size, size_t)

/* Xor Reduce */
template <typename T>
int ishmem_xor_reduce(T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, XOR_REDUCE>(dest, src, nreduce);
}

/* Xor Reduce on a team */
template <typename T>
int ishmem_xor_reduce(ishmem_team_t team, T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, XOR_REDUCE>(team, dest, src, nreduce);
}

template <typename T>
sycl::event ishmemx_xor_reduce_on_queue(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                        int *ret, sycl::queue &q,
                                        const std::vector<sycl::event> &deps)
{
    return ishmemi_reduce_on_queue<T, XOR_REDUCE>(team, dest, src, nreduce, ret, q, deps);
}

template <typename T>
sycl::event ishmemx_xor_reduce_on_queue(T *dest, const T *src, size_t nreduce, int *ret,
                                        sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return ishmemi_reduce_on_queue<T, XOR_REDUCE>(ISHMEM_TEAM_WORLD, dest, src, nreduce, ret, q,
                                                  deps);
}

template <typename T, typename Group>
int ishmemx_xor_reduce_work_group(T *dest, const T *src, size_t nreduce, const Group &grp)
{
    return ishmemi_reduce_work_group<T, XOR_REDUCE, Group>(dest, src, nreduce, grp);
}

template <typename T, typename Group>
int ishmemx_xor_reduce_work_group(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                  const Group &grp)
{
    return ishmemi_reduce_work_group<T, XOR_REDUCE, Group>(team, dest, src, nreduce, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_xor_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_xor_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_xor_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_xor_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_xor_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_xor_reduce_work_group(dest, src, nreduce, grp); }

#define ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                                        \
    template int ishmemx_##TYPENAME##_xor_reduce_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_xor_reduce_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_xor_reduce_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_xor_reduce_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_xor_reduce_work_group(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_xor_reduce_work_group(team, dest, src, nreduce, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(size, size_t)

ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(size, size_t)

/* Max Reduce */
template <typename T>
int ishmem_max_reduce(T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, MAX_REDUCE>(dest, src, nreduce);
}

/* Max Reduce on a team */
template <typename T>
int ishmem_max_reduce(ishmem_team_t team, T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, MAX_REDUCE>(team, dest, src, nreduce);
}

template <typename T>
sycl::event ishmemx_max_reduce_on_queue(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                        int *ret, sycl::queue &q,
                                        const std::vector<sycl::event> &deps)
{
    return ishmemi_reduce_on_queue<T, MAX_REDUCE>(team, dest, src, nreduce, ret, q, deps);
}

template <typename T>
sycl::event ishmemx_max_reduce_on_queue(T *dest, const T *src, size_t nreduce, int *ret,
                                        sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return ishmemi_reduce_on_queue<T, MAX_REDUCE>(ISHMEM_TEAM_WORLD, dest, src, nreduce, ret, q,
                                                  deps);
}

template <typename T, typename Group>
int ishmemx_max_reduce_work_group(T *dest, const T *src, size_t nreduce, const Group &grp)
{
    return ishmemi_reduce_work_group<T, MAX_REDUCE, Group>(dest, src, nreduce, grp);
}

template <typename T, typename Group>
int ishmemx_max_reduce_work_group(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                  const Group &grp)
{
    return ishmemi_reduce_work_group<T, MAX_REDUCE, Group>(team, dest, src, nreduce, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_max_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_max_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_max_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_max_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_max_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_max_reduce_work_group(dest, src, nreduce, grp); }

#define ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                                        \
    template int ishmemx_##TYPENAME##_max_reduce_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_max_reduce_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_max_reduce_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_max_reduce_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_max_reduce_work_group(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_max_reduce_work_group(team, dest, src, nreduce, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(double, double)

ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(double, double)

/* Min Reduce */
template <typename T>
int ishmem_min_reduce(T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, MIN_REDUCE>(dest, src, nreduce);
}

/* Min Reduce on a team */
template <typename T>
int ishmem_min_reduce(ishmem_team_t team, T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, MIN_REDUCE>(team, dest, src, nreduce);
}

template <typename T>
sycl::event ishmemx_min_reduce_on_queue(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                        int *ret, sycl::queue &q,
                                        const std::vector<sycl::event> &deps)
{
    return ishmemi_reduce_on_queue<T, MIN_REDUCE>(team, dest, src, nreduce, ret, q, deps);
}

template <typename T>
sycl::event ishmemx_min_reduce_on_queue(T *dest, const T *src, size_t nreduce, int *ret,
                                        sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return ishmemi_reduce_on_queue<T, MIN_REDUCE>(ISHMEM_TEAM_WORLD, dest, src, nreduce, ret, q,
                                                  deps);
}

template <typename T, typename Group>
int ishmemx_min_reduce_work_group(T *dest, const T *src, size_t nreduce, const Group &grp)
{
    return ishmemi_reduce_work_group<T, MIN_REDUCE, Group>(dest, src, nreduce, grp);
}

template <typename T, typename Group>
int ishmemx_min_reduce_work_group(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                  const Group &grp)
{
    return ishmemi_reduce_work_group<T, MIN_REDUCE, Group>(team, dest, src, nreduce, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_min_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_min_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_min_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_min_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_min_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_min_reduce_work_group(dest, src, nreduce, grp); }

#define ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                                        \
    template int ishmemx_##TYPENAME##_min_reduce_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_min_reduce_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_min_reduce_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_min_reduce_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_min_reduce_work_group(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_min_reduce_work_group(team, dest, src, nreduce, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(double, double)

ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(double, double)

/* Sum Reduce */
template <typename T>
int ishmem_sum_reduce(T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, SUM_REDUCE>(dest, src, nreduce);
}

/* Sum Reduce on a team */
template <typename T>
int ishmem_sum_reduce(ishmem_team_t team, T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, SUM_REDUCE>(team, dest, src, nreduce);
}

template <typename T>
sycl::event ishmemx_sum_reduce_on_queue(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                        int *ret, sycl::queue &q,
                                        const std::vector<sycl::event> &deps)
{
    return ishmemi_reduce_on_queue<T, SUM_REDUCE>(team, dest, src, nreduce, ret, q, deps);
}

template <typename T>
sycl::event ishmemx_sum_reduce_on_queue(T *dest, const T *src, size_t nreduce, int *ret,
                                        sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return ishmemi_reduce_on_queue<T, SUM_REDUCE>(ISHMEM_TEAM_WORLD, dest, src, nreduce, ret, q,
                                                  deps);
}

template <typename T, typename Group>
int ishmemx_sum_reduce_work_group(T *dest, const T *src, size_t nreduce, const Group &grp)
{
    return ishmemi_reduce_work_group<T, SUM_REDUCE, Group>(dest, src, nreduce, grp);
}

template <typename T, typename Group>
int ishmemx_sum_reduce_work_group(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                  const Group &grp)
{
    return ishmemi_reduce_work_group<T, SUM_REDUCE, Group>(team, dest, src, nreduce, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_sum_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_sum_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_sum_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_sum_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_sum_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_sum_reduce_work_group(dest, src, nreduce, grp); }

#define ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                                        \
    template int ishmemx_##TYPENAME##_sum_reduce_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_sum_reduce_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_sum_reduce_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_sum_reduce_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_sum_reduce_work_group(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_sum_reduce_work_group(team, dest, src, nreduce, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(double, double)

ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(double, double)

/* Prod Reduce */
template <typename T>
int ishmem_prod_reduce(T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, PROD_REDUCE>(dest, src, nreduce);
}

/* Prod Reduce on a team */
template <typename T>
int ishmem_prod_reduce(ishmem_team_t team, T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, PROD_REDUCE>(team, dest, src, nreduce);
}

template <typename T>
sycl::event ishmemx_prod_reduce_on_queue(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                         int *ret, sycl::queue &q,
                                         const std::vector<sycl::event> &deps)
{
    return ishmemi_reduce_on_queue<T, PROD_REDUCE>(team, dest, src, nreduce, ret, q, deps);
}

template <typename T>
sycl::event ishmemx_prod_reduce_on_queue(T *dest, const T *src, size_t nreduce, int *ret,
                                         sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return ishmemi_reduce_on_queue<T, PROD_REDUCE>(ISHMEM_TEAM_WORLD, dest, src, nreduce, ret, q,
                                                   deps);
}

template <typename T, typename Group>
int ishmemx_prod_reduce_work_group(T *dest, const T *src, size_t nreduce, const Group &grp)
{
    return ishmemi_reduce_work_group<T, PROD_REDUCE, Group>(dest, src, nreduce, grp);
}

template <typename T, typename Group>
int ishmemx_prod_reduce_work_group(ishmem_team_t team, T *dest, const T *src, size_t nreduce,
                                   const Group &grp)
{
    return ishmemi_reduce_work_group<T, PROD_REDUCE, Group>(team, dest, src, nreduce, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_prod_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_prod_reduce_work_group(dest, src, nreduce, grp); }

#define ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                                        \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_prod_reduce_work_group(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_prod_reduce_work_group(team, dest, src, nreduce, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(double, double)

ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(char, char)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(schar, signed char)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(short, short)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(int, int)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(long, long)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(longlong, long long)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(uchar, unsigned char)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(ushort, unsigned short)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(uint, unsigned int)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(ulong, unsigned long)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(int8, int8_t)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(int16, int16_t)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(int32, int32_t)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(int64, int64_t)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(uint8, uint8_t)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(uint16, uint16_t)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(uint32, uint32_t)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(uint64, uint64_t)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(size, size_t)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(float, float)
ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(double, double)

#endif  // ifndef COLLECTIVES_REDUCE_IMPL_H
