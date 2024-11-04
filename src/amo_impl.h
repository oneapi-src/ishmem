/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_AMO_IMPL_H
#define ISHMEM_AMO_IMPL_H

#include "ishmem/err.h"
#include "ishmem/types.h"
#include "proxy_impl.h"
#include "runtime.h"
#include "memory.h"

/* Generator for bitwise datatypes */
#define ISHMEMI_API_GENERATE_AMO_BIT_TYPES(ISHMEMI_API)                                            \
    ISHMEMI_API(uint, unsigned int)                                                                \
    ISHMEMI_API(ulong, unsigned long)                                                              \
    ISHMEMI_API(ulonglong, unsigned long long)                                                     \
    ISHMEMI_API(int32, int32_t)                                                                    \
    ISHMEMI_API(int64, int64_t)                                                                    \
    ISHMEMI_API(uint32, uint32_t)                                                                  \
    ISHMEMI_API(uint64, uint64_t)

/* Generator for standard datatypes */
#define ISHMEMI_API_GENERATE_AMO_STD_TYPES(ISHMEMI_API)                                            \
    ISHMEMI_API_GENERATE_AMO_BIT_TYPES(ISHMEMI_API)                                                \
    ISHMEMI_API(int, int)                                                                          \
    ISHMEMI_API(long, long)                                                                        \
    ISHMEMI_API(longlong, long long)                                                               \
    ISHMEMI_API(size, size_t)                                                                      \
    ISHMEMI_API(ptrdiff, ptrdiff_t)

/* Generator for extended datatypes */
#define ISHMEMI_API_GENERATE_AMO_EXT_TYPES(ISHMEMI_API)                                            \
    ISHMEMI_API_GENERATE_AMO_STD_TYPES(ISHMEMI_API)                                                \
    ISHMEMI_API(float, float)                                                                      \
    ISHMEMI_API(double, double)

/* Generalized implementations for all atomic memory operations */
template <typename T, ishmemi_op_t OP>
static inline T amo_impl_return(T *dest, T cond, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    T ret = static_cast<T>(0);
    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);

            if constexpr (OP == AMO_FETCH) {
                ret = atomic_p.load();
            } else if constexpr (OP == AMO_COMPARE_SWAP) {
                ret = cond;
                atomic_p.compare_exchange_strong(ret, val, sycl::memory_order::seq_cst,
                                                 sycl::memory_scope::system);
            } else if constexpr (OP == AMO_SWAP) {
                ret =
                    atomic_p.exchange(val, sycl::memory_order::seq_cst, sycl::memory_scope::system);
            } else if constexpr (OP == AMO_FETCH_ADD) {
                ret = atomic_p.fetch_add(static_cast<T>(val));
            } else if constexpr (OP == AMO_FETCH_INC) {
                ret = atomic_p.fetch_add(static_cast<T>(1));
            } else if constexpr (OP == AMO_FETCH_AND) {
                ret = atomic_p.fetch_and(static_cast<T>(val));
            } else if constexpr (OP == AMO_FETCH_OR) {
                ret = atomic_p.fetch_or(static_cast<T>(val));
            } else if constexpr (OP == AMO_FETCH_XOR) {
                ret = atomic_p.fetch_xor(static_cast<T>(val));
            } else {
                static_assert(false, "Unknown or unsupported type");
            }

            return ret;
        }
    }

    ishmemi_request_t req;
    req.dest_pe = pe;
    if constexpr (OP == AMO_FETCH) {
        req.src = dest;
    } else {
        req.dst = dest;
    }
    req.op = OP;
    req.type = ishmemi_union_get_base_type<T, OP>();

    if constexpr (OP != AMO_FETCH && OP != AMO_FETCH_INC) {
        ishmemi_union_set_field_value<T, OP>(req.value, val);
    }
    if constexpr (OP == AMO_COMPARE_SWAP) {
        ishmemi_union_set_field_value<T, OP>(req.cond, cond);
    }

#if __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_return<T, OP>(req);
#else
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_union_get_field_value<T, OP>(comp.completion.ret);
#endif

    return ret;
}

template <typename T, ishmemi_op_t OP>
static inline void amo_impl(T *dest, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);

            if constexpr (OP == AMO_SET) {
                atomic_p = val;
            } else if constexpr (OP == AMO_ADD) {
                atomic_p += val;
            } else if constexpr (OP == AMO_INC) {
                atomic_p += 1;
            } else if constexpr (OP == AMO_AND) {
                atomic_p &= val;
            } else if constexpr (OP == AMO_OR) {
                atomic_p |= val;
            } else if constexpr (OP == AMO_XOR) {
                atomic_p ^= val;
            } else {
                static_assert(false, "Unknown or unsupported type");
            }

            return;
        }
    }

    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = OP;
    req.type = ishmemi_union_get_base_type<T, OP>();

    ishmemi_union_set_field_value<T, OP>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* Generalized implementation for all non-blocking atomic memory operations */
template <typename T, ishmemi_op_t OP>
static inline void amo_nbi_impl(T *fetch, T *dest, T cond, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);

            if constexpr (OP == AMO_FETCH_NBI) {
                *fetch = atomic_p.load();
            } else if constexpr (OP == AMO_COMPARE_SWAP_NBI) {
                *fetch = cond;
                atomic_p.compare_exchange_strong(*fetch, val, sycl::memory_order::seq_cst,
                                                 sycl::memory_scope::system);
            } else if constexpr (OP == AMO_SWAP_NBI) {
                *fetch =
                    atomic_p.exchange(val, sycl::memory_order::seq_cst, sycl::memory_scope::system);
            } else if constexpr (OP == AMO_FETCH_ADD_NBI) {
                *fetch = atomic_p.fetch_add(static_cast<T>(val));
            } else if constexpr (OP == AMO_FETCH_INC_NBI) {
                *fetch = atomic_p.fetch_add(static_cast<T>(1));
            } else if constexpr (OP == AMO_FETCH_AND_NBI) {
                *fetch = atomic_p.fetch_and(static_cast<T>(val));
            } else if constexpr (OP == AMO_FETCH_OR_NBI) {
                *fetch = atomic_p.fetch_or(static_cast<T>(val));
            } else if constexpr (OP == AMO_FETCH_XOR_NBI) {
                *fetch = atomic_p.fetch_xor(static_cast<T>(val));
            } else {
                static_assert(false, "Unknown or unsupported type");
            }

            return;
        }
    }

    ishmemi_request_t req;
    req.dest_pe = pe;
    /* For amo_fetch_nbi, the dest variable represents src */
    req.dst = dest;
    req.fetch = fetch;
    req.op = OP;
    req.type = ishmemi_union_get_base_type<T, OP>();

    if constexpr (OP != AMO_FETCH_NBI && OP != AMO_FETCH_INC_NBI) {
        ishmemi_union_set_field_value<T, OP>(req.value, val);
    }
    if constexpr (OP == AMO_COMPARE_SWAP_NBI) {
        ishmemi_union_set_field_value<T, OP>(req.cond, cond);
    }

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_nonblocking_request(req);
#else
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

#endif
