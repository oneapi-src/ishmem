/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef COLLECTIVES_REDUCE_IMPL_H
#define COLLECTIVES_REDUCE_IMPL_H

#include "ishmem/err.h"
#include "ishmem/copy.h"
#include "collectives.h"
#include "collectives/sync_impl.h"
#include <type_traits>
#include "memory.h"
#include "runtime.h"

#define IN_HEAP(p)                                                                                 \
    ((((uintptr_t) p) >= ((uintptr_t) ishmemi_heap_base)) &&                                       \
     (((uintptr_t) p) < (((uintptr_t) ishmemi_heap_base) + ishmemi_heap_length)))

/* Reduction operators */
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
        reduce_op<sycl::vec<T, 16>, OP>(vd, vs);
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
        reduce_op<sycl::vec<T, 16>, OP>(vd, vs);
        vd.store(0, dd + idx);
        idx += vstride;
    }
    idx = linear_id + (static_cast<long>(count) & (~(static_cast<long>(ishmemi_vec_length) - 1)));
    while (idx < count) {
        reduce_op<T, OP>(dd[idx], ds[idx]);
        idx += stride;
    }
}

/* we only need one templated function for all the reductions because the proxy functions take
 * the reduction operator to call the correct backend reduction API */
template <typename T>
int ishmem_generic_op_reduce(T *dest, const T *src, size_t nreduce, ishmemi_op_t op)
{
    int ret = 0;
    size_t max_reduce = ISHMEM_REDUCE_BUFFER_SIZE / sizeof(T);

    while (nreduce > 0) {
        size_t this_reduce = nreduce;
        if (this_reduce > max_reduce) this_reduce = max_reduce;
        void *cdst = ishmem_copy((void *) ishmemi_cpu_info->reduce.source, (void *) src,
                                 this_reduce * sizeof(T));
        if ((uintptr_t) cdst != (uintptr_t) ishmemi_cpu_info->reduce.source) {
            ISHMEM_DEBUG_MSG("Copy-in of source failed in reduce\n");
            return (1);
        }

        ishmemi_ringcompletion_t comp;
        ishmemi_request_t req;
        req.src = ishmemi_cpu_info->reduce.source;
        req.dst = ishmemi_cpu_info->reduce.dest;
        req.nelems = this_reduce;
        req.op = op;
        req.type = ishmemi_proxy_get_base_type<T, true, true>();
        req.team = ishmemi_mmap_gpu_info->team_pool[ISHMEM_TEAM_WORLD];

        ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
        ret = ishmemi_proxy_get_status(comp.completion.ret);

        if (ret != 0) {
            ISHMEM_DEBUG_MSG("Runtime reduction failed\n");
            return ret;
        }
        cdst = ishmem_copy((void *) dest, (void *) ishmemi_cpu_info->reduce.dest,
                           this_reduce * sizeof(T));
        if ((uintptr_t) cdst != (uintptr_t) dest) {
            ISHMEM_DEBUG_MSG("Copy-out of dest failed in reduce\n");
            return 1;
        }
        dest += this_reduce;
        src += this_reduce;
        nreduce -= this_reduce;
    }
    return (0);
}

/* on a team... */
template <typename T>
int ishmemi_generic_op_reduce(ishmemi_team_t *team, T *dest, const T *src, size_t nreduce,
                              ishmemi_op_t op)
{
    int ret = 0;
    size_t max_reduce = ISHMEM_REDUCE_BUFFER_SIZE / sizeof(T);

    while (nreduce > 0) {
        size_t this_reduce = nreduce;
        if (this_reduce > max_reduce) this_reduce = max_reduce;
        void *cdst = ishmem_copy((void *) team->source, (void *) src, this_reduce * sizeof(T));
        if ((uintptr_t) cdst != (uintptr_t) team->source) {
            ISHMEM_DEBUG_MSG("ishmem_copy in failed\n");
            return (1);
        }

        ishmemi_ringcompletion_t comp;
        ishmemi_request_t req;
        req.src = team->source;
        req.dst = team->dest;
        req.nelems = this_reduce;
        req.op = op;
        req.type = ishmemi_proxy_get_base_type<T, true, true>();
        req.team = team;

        ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
        ret = ishmemi_proxy_get_status(comp.completion.ret);

        if (ret != 0) {
            ISHMEM_DEBUG_MSG("runtime reduction failed\n");
            return ret;
        }
        cdst = ishmem_copy((void *) dest, (void *) team->dest, this_reduce * sizeof(T));
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
template <typename T, ishmemi_op_t OP>
inline int ishmemi_sub_reduce(T *dest, const T *source, size_t nreduce)
{
    ishmemi_info_t *info = global_info;
    int ret = 0;
    ishmem_sync_all(); /* assure all source buffers are ready for use */
    for (int pe_idx = 0; pe_idx < info->n_local_pes; pe_idx += 1) {
        if (pe_idx == info->local_rank) continue;
        T *s = ISHMEMI_ADJUST_PTR(T, (pe_idx + 1), source);
        T *d = dest;
        vector_reduce<T, OP>(d, s, nreduce);
    }
    ishmem_sync_all(); /* assure destination buffers ready for use and source can be reused */
    return ret;
}

/* on a team... */
template <typename T, ishmemi_op_t OP>
inline int ishmemi_sub_reduce(ishmemi_team_t *team, T *dest, const T *source, size_t nreduce)
{
    ishmemi_info_t *info = global_info;
    int ret = 0;

    ishmemi_team_sync(team); /* assure all source buffers are ready for use */

    int idx = 0;
    for (int pe = team->start; idx < team->size; pe += team->stride, idx++) {
        if (pe == info->local_rank) continue;
        T *s = ISHMEMI_ADJUST_PTR(T, (pe + 1), source);
        T *d = dest;
        vector_reduce<T, OP>(d, s, nreduce);
    }
    ishmemi_team_sync(team);
    return ret;
}

/* Reduce - top-level reduce */
/* device code */
template <typename T, ishmemi_op_t OP>
inline int ishmemi_reduce(T *dest, const T *source, size_t nreduce)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) dest, (void *) source, nreduce * sizeof(T));
    }

    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        /* if this operation involves multiple nodes, just call the proxy */
        if (info->only_intra_node) {
            size_t max_nreduce = ISHMEM_REDUCE_BUFFER_SIZE / sizeof(T);
            if (source == dest) {
                while (nreduce > 0) {
                    size_t this_nreduce = (nreduce < max_nreduce) ? nreduce : max_nreduce;
                    vec_copy_push((T *) (info->reduce.buffer), source, this_nreduce);
                    int res =
                        ishmemi_sub_reduce<T, OP>(dest, (T *) info->reduce.buffer, this_nreduce);
                    if (res != 0) return res;
                    dest += this_nreduce;
                    source += this_nreduce;
                    nreduce -= this_nreduce;
                }
                return 0;
            }
            vec_copy_push(dest, source, nreduce);
            return ishmemi_sub_reduce<T, OP>(dest, source, nreduce);
        }
    }

#ifndef __SYCL_DEVICE_ONLY__
    /* if source and dest are both host memory, then call shmem */
    /* at the moment, device memory must be in the symmetric heap, which is an easier test */
    if (IN_HEAP(dest) || IN_HEAP(source)) {
        return (ishmem_generic_op_reduce<T>(dest, source, nreduce, OP));
    }
#endif

    /* Otherwise */
    ishmemi_request_t req;
    req.src = source;
    req.dst = dest;
    req.nelems = nreduce;
    req.op = OP;
    req.type = ishmemi_proxy_get_base_type<T, true, true>();

#ifdef __SYCL_DEVICE_ONLY__
    req.team = global_info->team_pool[ISHMEM_TEAM_WORLD];
    return ishmemi_proxy_blocking_request_status(req);
#else
    ishmemi_ringcompletion_t comp;
    req.team = ishmemi_mmap_gpu_info->team_pool[ISHMEM_TEAM_WORLD];
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    return ishmemi_proxy_get_status(comp.completion.ret);
#endif
}

/* on a team... */
template <typename T, ishmemi_op_t OP>
inline int ishmemi_reduce(ishmem_team_t team, T *dest, const T *source, size_t nreduce)
{
#if __SYCL_DEVICE_ONLY__
    ishmemi_team_t *myteam = global_info->team_pool[team];
#else
    ishmemi_team_t *myteam = ishmemi_mmap_gpu_info->team_pool[team];
#endif
    if constexpr (enable_error_checking) {
        validate_parameters((void *) dest, (void *) source, nreduce * sizeof(T));
    }

    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        /* if this operation involves multiple nodes, just call the proxy */
        if (info->only_intra_node) {
            size_t max_nreduce = ISHMEM_REDUCE_BUFFER_SIZE / sizeof(T);
            if (source == dest) {
                while (nreduce > 0) {
                    size_t this_nreduce = (nreduce < max_nreduce) ? nreduce : max_nreduce;
                    vec_copy_push((T *) (myteam->buffer), source, this_nreduce);
                    int res =
                        ishmemi_sub_reduce<T, OP>(myteam, dest, (T *) myteam->buffer, this_nreduce);
                    if (res != 0) return res;
                    dest += this_nreduce;
                    source += this_nreduce;
                    nreduce -= this_nreduce;
                }
                return 0;
            }
            vec_copy_push(dest, source, nreduce);
            return ishmemi_sub_reduce<T, OP>(myteam, dest, source, nreduce);
        }
    }

#ifndef __SYCL_DEVICE_ONLY__
    /* if source and dest are both host memory, then call shmem */
    /* at the moment, device memory must be in the symmetric heap, which is an easier test */
    if (IN_HEAP(dest) || IN_HEAP(source)) {
        return (ishmemi_generic_op_reduce<T>(myteam, dest, source, nreduce, OP));
    }
#endif

    /* Otherwise */
    ishmemi_request_t req;
    req.src = source;
    req.dst = dest;
    req.nelems = nreduce;
    req.op = OP;
    req.type = ishmemi_proxy_get_base_type<T, true, true>();
    req.team = myteam;

#ifdef __SYCL_DEVICE_ONLY__
    return ishmemi_proxy_blocking_request_status(req);
#else
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    return ishmemi_proxy_get_status(comp.completion.ret);
#endif
}

/* Sub-reduce (work-group) - SYCL can't make recursive calls, so this function is used by the
 * top-level reduce */
/* TODO: figure out how to spread the available threads across all the remote PEs, so as to equalize
 * xe-link loading */
template <typename T, ishmemi_op_t OP, typename Group>
inline int ishmemi_sub_reduce_work_group(T *dest, const T *source, size_t nreduce, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        int ret = 0;
        if (info->only_intra_node) {
            /* assure local source buffer ready for use (group_garrier)
             * assure all source buffers ready for use (sync all)
             */
            ishmemx_sync_all_work_group(grp);
            for (int i = 1; i < info->n_local_pes; i += 1) {
                // TODO FP add is not associative, fix ordering
                int pe_idx = info->my_pe + i;
                if (pe_idx >= info->n_local_pes) pe_idx -= info->n_local_pes;
                T *remote = ISHMEMI_ADJUST_PTR(T, (pe_idx + 1), source);
                vector_reduce_work_group<T, OP>(dest, remote, nreduce, grp);
            }
            /* assure all threads have finished copies (group_barrier)
             * assure destination buffers complete and source buffers may be reused */
            ishmemx_sync_all_work_group(grp);
            /* group broadcast not needed because ret is always 0 here */
            return (ret);
        } else {
            if (grp.leader()) {
                ishmemi_request_t req;
                req.src = source;
                req.dst = dest;
                req.nelems = nreduce;
                req.op = OP;
                req.type = ishmemi_proxy_get_base_type<T, true, true>();
                req.team = info->team_pool[ISHMEM_TEAM_WORLD];

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
inline int ishmemi_sub_reduce_work_group(ishmem_team_t team, T *dest, const T *source,
                                         size_t nreduce, const Group &grp)
{
#if __SYCL_DEVICE_ONLY__
    ishmemi_team_t *myteam = global_info->team_pool[team];
#else
    ishmemi_team_t *myteam = ishmemi_mmap_gpu_info->team_pool[team];
#endif
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        int ret = 0;
        if (info->only_intra_node) {
            /* assure local source buffer ready for use (group_garrier)
             * assure all source buffers ready for use (sync all)
             */
            ishmemx_team_sync_work_group(team, grp);
            int idx = 0;
            for (int pe = myteam->start; idx < myteam->size; pe += myteam->stride, idx++) {
                if (pe == info->local_rank) continue;
                T *remote = ISHMEMI_ADJUST_PTR(T, (pe + 1), source);
                vector_reduce_work_group<T, OP>(dest, remote, nreduce, grp);
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
                req.type = ishmemi_proxy_get_base_type<T, true, true>();
                req.team = myteam;

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

/* Reduce (work-group) - top-level reduce */
/* TODO: figure out how to spread the available threads across all the remote PEs, so as to equalize
 * xe-link loading */
template <typename T, ishmemi_op_t OP, typename Group>
inline int ishmemi_reduce_work_group(T *dest, const T *source, size_t nreduce, const Group &grp)
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
                T *temp_buffer = (T *) info->reduce.buffer;

                vec_copy_work_group_push(temp_buffer, source, this_nreduce, grp);
                int res =
                    ishmemi_sub_reduce_work_group<T, OP>(dest, temp_buffer, this_nreduce, grp);
                if (res != 0) return res;
                dest += this_nreduce;
                source += this_nreduce;
                nreduce -= this_nreduce;
            }
            return 0;
        }
        vec_copy_work_group_push(dest, source, nreduce, grp);
        return ishmemi_sub_reduce_work_group<T, OP>(dest, source, nreduce, grp);
    } else {
        ISHMEM_ERROR_MSG("ISHMEMX_REDUCE_WORK_GROUP routines are not callable from host\n");
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
                T *temp_buffer = (T *) info->team_pool[team]->buffer;

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
        return ishmemi_sub_reduce_work_group<T, OP>(dest, source, nreduce, grp);
    } else {
        ISHMEM_ERROR_MSG("ISHMEMX_REDUCE_WORK_GROUP routines are not callable from host\n");
        return -1;
    }
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

#define ISHMEMI_API_IMPL_TEAM_AND_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
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
#define ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                          \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);    \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);    \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);    \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp);  \
    template <typename Group> int ishmemx_##TYPENAME##_or_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_or_reduce_work_group(dest, src, nreduce, grp); }

#define ISHMEMI_API_IMPL_TEAM_OR_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                          \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);    \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);    \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);    \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp);  \
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

#define ISHMEMI_API_IMPL_TEAM_XOR_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
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

#define ISHMEMI_API_IMPL_TEAM_MAX_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
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

#define ISHMEMI_API_IMPL_TEAM_MIN_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
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

#define ISHMEMI_API_IMPL_TEAM_SUM_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
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
#define ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                          \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);    \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);    \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);    \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp);  \
    template <typename Group> int ishmemx_##TYPENAME##_prod_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_prod_reduce_work_group(dest, src, nreduce, grp); }

#define ISHMEMI_API_IMPL_TEAM_PROD_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                          \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<1>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);    \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<2>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);    \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<3>>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);    \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::sub_group>(ishmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp);  \
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
