/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef COLLECTIVES_REDUCE_IMPL_H
#define COLLECTIVES_REDUCE_IMPL_H

#include "internal.h"
#include "collectives.h"
#include <type_traits>
#include "memory.h"  // for ishmem_copy
#include "runtime.h"

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
    while ((((uintptr_t) s) & ALIGNMASK) && (count > 0)) {
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
    T *aligned_s =
        (T *) sycl::min(((((uintptr_t) s) + ALIGNMASK) & (~ALIGNMASK)), (uintptr_t) (s + count));
    while (((uintptr_t) &s[idx]) < ((uintptr_t) (aligned_s))) {
        reduce_op<T, OP>(d[idx], s[idx]);
        idx += stride;
    }
    count -= static_cast<size_t>(aligned_s - s);  // pointer difference is in units of T
    d += (aligned_s - s);
    /* at this point, if count > 0, then d is aligned, s may not be aligned */
    if (count == 0) return;
    idx = linear_id * VL;
    size_t vstride = stride * VL;

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
    while ((idx + VL) <= count) {
        vs.load(0, ds + idx);
        vd.load(0, dd + idx);
        reduce_op<sycl::vec<T, 16>, OP>(vd, vs);
        vd.store(0, dd + idx);
        idx += vstride;
    }
    idx = linear_id + (static_cast<long>(count) & (~(static_cast<long>(VL) - 1)));
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
            ISHMEM_DEBUG_MSG("ishmem_copy in failed\n");
            return (1);
        }

        ishmemi_ringcompletion_t comp;
        ishmemi_request_t req = {
            .op = op,
            .type = ishmemi_proxy_get_base_type<T, true, true>(),
            .src = ishmemi_cpu_info->reduce.source,
            .dst = ishmemi_cpu_info->reduce.dest,
            .nelems = this_reduce,
        };

        ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
        ret = ishmemi_proxy_get_status(comp.completion.ret);

        if (ret != 0) {
            ISHMEM_DEBUG_MSG("runtime reduction failed\n");
            return ret;
        }
        cdst = ishmem_copy((void *) dest, (void *) ishmemi_cpu_info->reduce.dest,
                           this_reduce * sizeof(T));
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
    ishmem_info_t *info = global_info;
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

#define IN_HEAP(p)                                                                                 \
    ((((uintptr_t) p) >= ((uintptr_t) ishmemi_heap_base)) &&                                       \
     (((uintptr_t) p) < (((uintptr_t) ishmemi_heap_base) + ishmemi_heap_length)))

/* Reduce - top-level reduce */
/* device code */
template <typename T, ishmemi_op_t OP>
inline int ishmemi_reduce(T *dest, const T *source, size_t nreduce)
{
    if constexpr (ishmemi_is_device) {
        ishmem_info_t *info = global_info;
        if constexpr (enable_error_checking) {
            validate_parameters((void *) dest, (void *) source, nreduce * sizeof(T));
        }
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
        } else {
            ishmemi_request_t req = {
                .op = OP,
                .type = ishmemi_proxy_get_base_type<T, true, true>(),
                .src = source,
                .dst = dest,
                .nelems = nreduce,
            };

            return ishmemi_proxy_blocking_request_status(&req);
        }
    }
}

/* host code */
template <typename T>
inline int ishmemi_host_reduce(T *dest, const T *source, size_t nreduce, ishmemi_op_t op)
{
    /* if source and dest are both host memory, then call shmem */
    /* at the moment, device memory must be in the symmetric heap, which is an easier test */
    if (IN_HEAP(dest) || IN_HEAP(source)) {
        return (ishmem_generic_op_reduce<T>(dest, source, nreduce, op));
    } else {
        ishmemi_ringcompletion_t comp;
        ishmemi_request_t req = {
            .op = op,
            .type = ishmemi_proxy_get_base_type<T, true, true>(),
            .src = source,
            .dst = dest,
            .nelems = nreduce,
        };

        ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
        return ishmemi_proxy_get_status(comp.completion.ret);
    }
}

/* Sub-reduce (work-group) - SYCL can't make recursive calls, so this function is used by the
 * top-level reduce */
/* TODO: figure out how to spread the available threads across all the remote PEs, so as to equalize
 * xe-link loading */
template <typename T, ishmemi_op_t OP, typename Group>
inline int ishmemi_sub_reduce_work_group(T *dest, const T *source, size_t nreduce, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        ishmem_info_t *info = global_info;
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
                ishmemi_request_t req = {
                    .op = OP,
                    .type = ishmemi_proxy_get_base_type<T, true, true>(),
                    .src = source,
                    .dst = dest,
                    .nelems = nreduce,
                };

                ret = ishmemi_proxy_blocking_request_status(&req);
                ret = sycl::group_broadcast(grp, ret, 0);
                return (ret);
            }
        }
    } else {
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
        ishmem_info_t *info = global_info;
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
        return -1;  // not supported on host
    }
}

/* And Reduce */
template <typename T>
int ishmem_and_reduce(T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, AND_REDUCE>(dest, src, nreduce);
}

template <typename T, typename Group>
int ishmemx_and_reduce_work_group(T *dest, const T *src, size_t nreduce, const Group &grp)
{
    return ishmemi_reduce_work_group<T, AND_REDUCE, Group>(dest, src, nreduce, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_AND_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_and_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_and_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_and_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_and_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_and_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_and_reduce_work_group(dest, src, nreduce, grp); }
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

/* Or Reduce */
template <typename T>
int ishmem_or_reduce(T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, OR_REDUCE>(dest, src, nreduce);
}

template <typename T, typename Group>
int ishmemx_or_reduce_work_group(T *dest, const T *src, size_t nreduce, const Group &grp)
{
    return ishmemi_reduce_work_group<T, OR_REDUCE, Group>(dest, src, nreduce, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_OR_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                          \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);    \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);    \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);    \
    template int ishmemx_##TYPENAME##_or_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp);  \
    template <typename Group> int ishmemx_##TYPENAME##_or_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_or_reduce_work_group(dest, src, nreduce, grp); }
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

/* Xor Reduce */
template <typename T>
int ishmem_xor_reduce(T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, XOR_REDUCE>(dest, src, nreduce);
}

template <typename T, typename Group>
int ishmemx_xor_reduce_work_group(T *dest, const T *src, size_t nreduce, const Group &grp)
{
    return ishmemi_reduce_work_group<T, XOR_REDUCE, Group>(dest, src, nreduce, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_XOR_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_xor_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_xor_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_xor_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_xor_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_xor_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_xor_reduce_work_group(dest, src, nreduce, grp); }
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

/* Max Reduce */
template <typename T>
int ishmem_max_reduce(T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, MAX_REDUCE>(dest, src, nreduce);
}

template <typename T, typename Group>
int ishmemx_max_reduce_work_group(T *dest, const T *src, size_t nreduce, const Group &grp)
{
    return ishmemi_reduce_work_group<T, MAX_REDUCE, Group>(dest, src, nreduce, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_MAX_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_max_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_max_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_max_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_max_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_max_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_max_reduce_work_group(dest, src, nreduce, grp); }
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

/* Min Reduce */
template <typename T>
int ishmem_min_reduce(T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, MIN_REDUCE>(dest, src, nreduce);
}

template <typename T, typename Group>
int ishmemx_min_reduce_work_group(T *dest, const T *src, size_t nreduce, const Group &grp)
{
    return ishmemi_reduce_work_group<T, MIN_REDUCE, Group>(dest, src, nreduce, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_MIN_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_min_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_min_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_min_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_min_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_min_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_min_reduce_work_group(dest, src, nreduce, grp); }
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

/* Sum Reduce */
template <typename T>
int ishmem_sum_reduce(T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, SUM_REDUCE>(dest, src, nreduce);
}

template <typename T, typename Group>
int ishmemx_sum_reduce_work_group(T *dest, const T *src, size_t nreduce, const Group &grp)
{
    return ishmemi_reduce_work_group<T, SUM_REDUCE, Group>(dest, src, nreduce, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_SUM_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                         \
    template int ishmemx_##TYPENAME##_sum_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);   \
    template int ishmemx_##TYPENAME##_sum_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);   \
    template int ishmemx_##TYPENAME##_sum_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);   \
    template int ishmemx_##TYPENAME##_sum_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp); \
    template <typename Group> int ishmemx_##TYPENAME##_sum_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_sum_reduce_work_group(dest, src, nreduce, grp); }
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

/* Prod Reduce */
template <typename T>
int ishmem_prod_reduce(T *dest, const T *src, size_t nreduce)
{
    return ishmemi_reduce<T, PROD_REDUCE>(dest, src, nreduce);
}

template <typename T, typename Group>
int ishmemx_prod_reduce_work_group(T *dest, const T *src, size_t nreduce, const Group &grp)
{
    return ishmemi_reduce_work_group<T, PROD_REDUCE, Group>(dest, src, nreduce, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_PROD_REDUCE_WORK_GROUP(TYPENAME, TYPE)                                                                                          \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<1> &grp);    \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<2> &grp);    \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::group<3> &grp);    \
    template int ishmemx_##TYPENAME##_prod_reduce_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nreduce, const sycl::sub_group &grp);  \
    template <typename Group> int ishmemx_##TYPENAME##_prod_reduce_work_group(TYPE *dest, const TYPE *src, size_t nreduce, const Group &grp) { return ishmemx_prod_reduce_work_group(dest, src, nreduce, grp); }
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

#endif  // ifndef COLLECTIVES_REDUCE_IMPL_H
