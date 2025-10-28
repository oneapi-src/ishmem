/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* internal API and globals */
#ifndef ISHMEM_COPY_H
#define ISHMEM_COPY_H

#include "ishmem/types.h"
#include "ishmem/util.h"

#include <iostream>
#include <cstdlib>

/* Tuning Parameters
 * To tune for a system, run bandwidth tests with CUTOVER_NEVER and again with CUTOVER_ALWAYS. Use
 * the results to choose proper CUTOVER_PRODUCTION values
 *
 * TODO: Allow for runtime-configurable cutovers
 */
#define CUTOVER_PRODUCTION 1
#define CUTOVER_ALWAYS     0
#define CUTOVER_NEVER      0

#if CUTOVER_NEVER

#define ISHMEM_RMA_CUTOVER               (false)
#define ISHMEM_RMA_GROUP_CUTOVER         (false)
#define ISHMEM_STRIDED_RMA_CUTOVER       (false)
#define ISHMEM_STRIDED_RMA_GROUP_CUTOVER (false)
#define ISHMEM_ALLTOALL_CUTOVER          (false)
#define ISHMEM_ALLTOALL_GROUP_CUTOVER    (false)
#define ISHMEM_BROADCAST_CUTOVER         (false)
#define ISHMEM_BROADCAST_GROUP_CUTOVER   (false)
#define ISHMEM_COLLECT_CUTOVER           (false)
#define ISHMEM_COLLECT_GROUP_CUTOVER     (false)
#define ISHMEM_FCOLLECT_CUTOVER          (false)
#define ISHMEM_FCOLLECT_GROUP_CUTOVER    (false)

#elif CUTOVER_ALWAYS

#define ISHMEM_RMA_CUTOVER               (true)
#define ISHMEM_RMA_GROUP_CUTOVER         (true)
#define ISHMEM_STRIDED_RMA_CUTOVER       (true)
#define ISHMEM_STRIDED_RMA_GROUP_CUTOVER (true)
#define ISHMEM_ALLTOALL_CUTOVER          (true)
#define ISHMEM_ALLTOALL_GROUP_CUTOVER    (true)
#define ISHMEM_BROADCAST_CUTOVER         (true)
#define ISHMEM_BROADCAST_GROUP_CUTOVER   (true)
#define ISHMEM_COLLECT_CUTOVER           (true)
#define ISHMEM_COLLECT_GROUP_CUTOVER     (true)
#define ISHMEM_FCOLLECT_CUTOVER          (true)
#define ISHMEM_FCOLLECT_GROUP_CUTOVER    (true)

#else /* CUTOVER_PRODUCTION */

#define ISHMEM_RMA_CUTOVER               (nbytes >= 16384L)
#define ISHMEM_RMA_GROUP_CUTOVER         (nbytes >= 32768L)
#define ISHMEM_STRIDED_RMA_CUTOVER       (nbytes >= 16384L)
#define ISHMEM_STRIDED_RMA_GROUP_CUTOVER (nbytes >= 32768L)
#define ISHMEM_ALLTOALL_CUTOVER          (nbytes >= 128L)
#define ISHMEM_ALLTOALL_GROUP_CUTOVER    (nbytes >= 16384L)
#define ISHMEM_BROADCAST_CUTOVER         ((nbytes * ((size_t) info->n_pes)) >= 8192L)
// preferred BROADCAST_GROUP_CUTOVER is nbytes * threads > 512
#define ISHMEM_BROADCAST_GROUP_CUTOVER   (nbytes >= 65536L)
#define ISHMEM_FCOLLECT_CUTOVER          (nbytes >= 1024L)
#define ISHMEM_FCOLLECT_GROUP_CUTOVER    (nbytes >= 32768L)
#define ISHMEM_COLLECT_CUTOVER           (total_nbytes >= (1024L * ((size_t) info->n_pes)))
#define ISHMEM_COLLECT_GROUP_CUTOVER     (total_nbytes >= (32768L * ((size_t) info->n_pes)))

#endif /* CUTOVER */

/* Copy utility function */
void *ishmem_copy(void *dst, const void *src, size_t size);
void *ishmem_zero(void *dst, size_t size);

// use the routines below rather than a type specific copy loop
constexpr bool ishmemi_use_vec_copy = true;
constexpr long ishmemi_vec_length = 16L;
#define ISHMEMI_ALIGNSIZE (sizeof(T) * ishmemi_vec_length)
#define ISHMEMI_ALIGNMASK (ISHMEMI_ALIGNSIZE - 1)

template <typename Group>
inline void ishmemi_work_item_calculate_offset(size_t nelems, Group grp,
                                               size_t &my_nelems_work_item,
                                               size_t &work_item_start_idx)
{
    size_t num_work_items = grp.get_local_linear_range();
    size_t work_item_id = grp.get_local_linear_id();

    size_t min_nelems_work_item = nelems / num_work_items;
    my_nelems_work_item = min_nelems_work_item;
    size_t carry_over = nelems % num_work_items;

    size_t x = (carry_over < work_item_id) ? carry_over : work_item_id;
    if (work_item_id < carry_over) my_nelems_work_item += 1;

    work_item_start_idx =
        (x * (min_nelems_work_item + 1)) + ((work_item_id - x) * min_nelems_work_item);
}

/* Non-work-group vector copy functions */
template <typename T>
inline void vec_copy_push(T *d, const T *s, size_t count)
{
    if constexpr (ishmemi_is_device) {
        if constexpr (ishmemi_use_vec_copy) {
            while ((((uintptr_t) d) & ISHMEMI_ALIGNMASK) && (count > 0)) {
                *d++ = *s++;
                count -= 1;
            }
            sycl::multi_ptr<T, sycl::access::address_space::global_space,
                            sycl::access::decorated::yes>
                dd;
            sycl::multi_ptr<const T, sycl::access::address_space::global_space,
                            sycl::access::decorated::yes>
                ds;
            dd = sycl::address_space_cast<sycl::access::address_space::global_space,
                                          sycl::access::decorated::yes>(d);
            ds = sycl::address_space_cast<sycl::access::address_space::global_space,
                                          sycl::access::decorated::yes>(s);
            while (count >= ishmemi_vec_length) {
                sycl::vec<T, 16> temp;
                temp.load(0, ds);
                temp.store(0, dd);
                ds += ishmemi_vec_length;
                dd += ishmemi_vec_length;
                count -= ishmemi_vec_length;
            }
            while (count > 0) {
                *dd++ = *ds++;
                count -= 1;
            }
        } else {
            for (size_t i = 0; i < count; i += 1)
                d[i] = s[i];
        }
    } else {
        memcpy(d, s, count * sizeof(T));
    }
}

template <typename T>
inline void vec_copy_pull(T *d, const T *s, size_t count)
{
    if constexpr (ishmemi_is_device) {
        if constexpr (ishmemi_use_vec_copy) {
            while ((((uintptr_t) s) & ISHMEMI_ALIGNMASK) && (count > 0)) {
                *d++ = *s++;
                count -= 1;
            }
            sycl::multi_ptr<T, sycl::access::address_space::global_space,
                            sycl::access::decorated::yes>
                dd;
            sycl::multi_ptr<const T, sycl::access::address_space::global_space,
                            sycl::access::decorated::yes>
                ds;
            dd = sycl::address_space_cast<sycl::access::address_space::global_space,
                                          sycl::access::decorated::yes>(d);
            ds = sycl::address_space_cast<sycl::access::address_space::global_space,
                                          sycl::access::decorated::yes>(s);
            while (count >= ishmemi_vec_length) {
                sycl::vec<T, ishmemi_vec_length> temp;
                temp.load(0, ds);
                temp.store(0, dd);
                ds += ishmemi_vec_length;
                dd += ishmemi_vec_length;
                count -= ishmemi_vec_length;
            }
            while (count > 0) {
                *dd++ = *ds++;
                count -= 1;
            }
        } else {
            for (size_t i = 0; i < count; i += 1)
                d[i] = s[i];
        }
    } else {
        memcpy(d, s, count * sizeof(T));
    }
}

template <typename T>
inline void stride_copy(T *d, const T *s, ptrdiff_t dst, ptrdiff_t sst, size_t count)
{
    if constexpr (ishmemi_is_device) {
        size_t d_idx = 0;
        size_t s_idx = 0;
        for (size_t i = 0; i < count;
             i += 1, d_idx += static_cast<size_t>(dst), s_idx += static_cast<size_t>(sst))
            d[d_idx] = s[s_idx];
    }
}

template <typename T>
inline void stride_bcopy_push(T *d, const T *s, ptrdiff_t dst, ptrdiff_t sst, size_t bsize,
                              size_t nblocks)
{
    if constexpr (ishmemi_is_device) {
        size_t d_idx = 0;
        size_t s_idx = 0;
        for (size_t i = 0; i < nblocks;
             i += 1, d_idx += static_cast<size_t>(dst), s_idx += static_cast<size_t>(sst))
            vec_copy_push(d + d_idx, s + s_idx, bsize);
    }
}

template <typename T>
inline void stride_bcopy_pull(T *d, const T *s, ptrdiff_t dst, ptrdiff_t sst, size_t bsize,
                              size_t nblocks)
{
    if constexpr (ishmemi_is_device) {
        size_t d_idx = 0;
        size_t s_idx = 0;
        for (size_t i = 0; i < nblocks;
             i += 1, d_idx += static_cast<size_t>(dst), s_idx += static_cast<size_t>(sst))
            vec_copy_pull(d + d_idx, s + s_idx, bsize);
    }
}

/* Work-group vector copy functions */
/* These functions are called collectively by all threads in a work group */
typedef enum {
    VEC_COPY_WORK_GROUP_STRIDED,
    VEC_COPY_WORK_GROUP_UNSTRIDED
} ishmemi_vec_copy_work_group_algorithm_t;

constexpr ishmemi_vec_copy_work_group_algorithm_t ishmemi_vec_copy_work_group_algorithm =
    VEC_COPY_WORK_GROUP_UNSTRIDED;

template <typename T, typename Group>
void vec_copy_work_group_unstrided_push(T *d, const T *s, size_t count, Group grp)
{
    if constexpr (ishmemi_is_device) {
        size_t my_nelems_work_item;
        size_t work_item_start_idx;
        ishmemi_work_item_calculate_offset(count, grp, my_nelems_work_item, work_item_start_idx);
        vec_copy_push(d + work_item_start_idx, s + work_item_start_idx, my_nelems_work_item);
    }
}

template <typename T, typename Group>
void vec_copy_work_group_strided_push(T *d, const T *s, size_t count, Group grp)
{
    if constexpr (ishmemi_is_device) {
        size_t stride = grp.get_local_linear_range();
        size_t linear_id = grp.get_local_linear_id();
        size_t idx = linear_id;
        T *aligned_d =
            (T *) sycl::min(((((uintptr_t) d) + ISHMEMI_ALIGNMASK) & (~ISHMEMI_ALIGNMASK)),
                            (uintptr_t) (d + count));
        /* The idea of this loop is that each thread copies every wg_size-th element starting
         * with its own id in the group
         */
        while (((uintptr_t) &d[idx]) < ((uintptr_t) (aligned_d))) {
            d[idx] = s[idx];
            idx += stride;
        }
        count -= (size_t) (aligned_d - d);  // pointer difference is in units of T
        s += (aligned_d - d);
        /* at this point, if count > 0, then d is aligned, s may not be aligned */
        if (count == 0) return;
        idx = (linear_id * ishmemi_vec_length);
        size_t vstride = stride * ishmemi_vec_length;

        sycl::multi_ptr<T, sycl::access::address_space::global_space, sycl::access::decorated::yes>
            dd;
        sycl::multi_ptr<const T, sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>
            ds;
        dd = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::yes>(aligned_d);
        ds = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::yes>(s);
        /* In this loop, each thread handles a len ishmemi_vec_length vector with stride wg_size *
         * ishmemi_vec_length.  */
        while ((idx + ishmemi_vec_length) <= count) {
            sycl::vec<T, ishmemi_vec_length> temp;
            temp.load(0, ds + idx);
            temp.store(0, dd + idx);
            idx += vstride;
        }
        /* at this point, the threads have finished the copy except for the last
         * count & (ishmemi_vec_length-1) items
         * back to item at a time
         */
        /* idx here should be the postfix index + linear_id */
        idx = (linear_id + (count & (~((unsigned long) ishmemi_vec_length - 1))));
        while (idx < count) {
            dd[idx] = ds[idx];
            idx += stride;
        }
    } else {
        memcpy(d, s, count * sizeof(T));
    }
}

template <typename T, typename Group>
void vec_copy_work_group_push(T *d, const T *s, size_t count, Group grp)
{
    if constexpr (ishmemi_vec_copy_work_group_algorithm == VEC_COPY_WORK_GROUP_STRIDED) {
        vec_copy_work_group_strided_push(d, s, count, grp);
    } else if constexpr (ishmemi_vec_copy_work_group_algorithm == VEC_COPY_WORK_GROUP_UNSTRIDED) {
        vec_copy_work_group_unstrided_push(d, s, count, grp);
    } else {
        RAISE_ERROR_MSG("vec_copy_work_group_push: unknown algorithm selected\n");
    }
}

template <typename T, typename Group>
void vec_copy_work_group_unstrided_pull(T *d, const T *s, size_t count, Group grp)
{
    if constexpr (ishmemi_is_device) {
        size_t my_nelems_work_item;
        size_t work_item_start_idx;
        ishmemi_work_item_calculate_offset(count, grp, my_nelems_work_item, work_item_start_idx);
        vec_copy_pull(d + work_item_start_idx, s + work_item_start_idx, my_nelems_work_item);
    }
}

template <typename T, typename Group>
void vec_copy_work_group_strided_pull(T *d, const T *s, size_t count, Group grp)
{
    if constexpr (ishmemi_is_device) {
        size_t stride = grp.get_local_linear_range();
        size_t linear_id = grp.get_local_linear_id();
        size_t idx = linear_id;
        T *aligned_s =
            (T *) sycl::min(((((uintptr_t) s) + ISHMEMI_ALIGNMASK) & (~ISHMEMI_ALIGNMASK)),
                            (uintptr_t) (s + count));
        /* The idea of this loop is that each thread copies every wg_size-th element starting
         * with its own id in the group
         */
        while (((uintptr_t) &s[idx]) < ((uintptr_t) (aligned_s))) {
            d[idx] = s[idx];
            idx += stride;
        }
        count -= (size_t) (aligned_s - s);  // pointer difference is in units of T
        d += (aligned_s - s);
        /* at this point, if count > 0, then d is aligned, s may not be aligned */
        if (count == 0) return;
        idx = (linear_id * ishmemi_vec_length);
        size_t vstride = stride * ishmemi_vec_length;

        sycl::multi_ptr<T, sycl::access::address_space::global_space, sycl::access::decorated::yes>
            dd;
        sycl::multi_ptr<const T, sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>
            ds;
        dd = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::yes>(d);
        ds = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::yes>(aligned_s);
        /* In this loop, each thread handles a len ishmemi_vec_length vector with stride wg_size *
         * ishmemi_vec_length.  */
        while ((idx + ishmemi_vec_length) <= count) {
            sycl::vec<T, ishmemi_vec_length> temp;
            temp.load(0, ds + idx);
            temp.store(0, dd + idx);
            idx += vstride;
        }
        /* at this point, the threads have finished the copy except for the last
         * count & (ishmemi_vec_length-1) items
         * back to item at a time
         */
        idx = (linear_id + (count & (~((unsigned long) ishmemi_vec_length - 1))));
        while (idx < count) {
            dd[idx] = ds[idx];
            idx += stride;
        }
    } else {
        memcpy(d, s, count * sizeof(T));
    }
}

template <typename T, typename Group>
void vec_copy_work_group_pull(T *d, const T *s, size_t count, Group grp)
{
    if constexpr (ishmemi_vec_copy_work_group_algorithm == VEC_COPY_WORK_GROUP_STRIDED) {
        vec_copy_work_group_strided_pull(d, s, count, grp);
    } else if constexpr (ishmemi_vec_copy_work_group_algorithm == VEC_COPY_WORK_GROUP_UNSTRIDED) {
        vec_copy_work_group_unstrided_pull(d, s, count, grp);
    } else {
        RAISE_ERROR_MSG("vec_copy_work_group_pull: unknown algorithm selected\n");
    }
}

template <typename T, typename Group>
void stride_copy_work_group(T *d, const T *s, ptrdiff_t dst, ptrdiff_t sst, size_t count, Group grp)
{
    if constexpr (ishmemi_is_device) {
        size_t my_nelems_work_item;
        size_t work_item_start_idx;
        ishmemi_work_item_calculate_offset(count, grp, my_nelems_work_item, work_item_start_idx);
        size_t d_idx = work_item_start_idx * static_cast<size_t>(dst);
        size_t s_idx = work_item_start_idx * static_cast<size_t>(sst);
        for (size_t i = 0; i < my_nelems_work_item;
             i += 1, d_idx += static_cast<size_t>(dst), s_idx += static_cast<size_t>(sst))
            d[d_idx] = s[s_idx];
    }
}

template <typename T, typename Group>
void stride_bcopy_work_group_push(T *d, const T *s, ptrdiff_t dst, ptrdiff_t sst, size_t bsize,
                                  size_t nblocks, Group grp)
{
    if constexpr (ishmemi_is_device) {
        size_t d_idx = 0;
        size_t s_idx = 0;
        for (size_t i = 0; i < nblocks;
             i += 1, d_idx += static_cast<size_t>(dst), s_idx += static_cast<size_t>(sst))
            vec_copy_work_group_push(d + d_idx, s + s_idx, bsize, grp);
    }
}

template <typename T, typename Group>
void stride_bcopy_work_group_pull(T *d, const T *s, ptrdiff_t dst, ptrdiff_t sst, size_t bsize,
                                  size_t nblocks, Group grp)
{
    if constexpr (ishmemi_is_device) {
        size_t d_idx = 0;
        size_t s_idx = 0;
        for (size_t i = 0; i < nblocks;
             i += 1, d_idx += static_cast<size_t>(dst), s_idx += static_cast<size_t>(sst))
            vec_copy_work_group_pull(d + d_idx, s + s_idx, bsize, grp);
    }
}

#endif /* ISHMEM_INTERNAL_H */
