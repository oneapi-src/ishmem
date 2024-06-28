/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/err.h"
#include "proxy_impl.h"
#include "runtime.h"
#include "memory.h"

/* Atomic fetch */
template <typename T>
T ishmem_atomic_fetch(T *src, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) src, sizeof(T));
    }

    T ret = static_cast<T>(0);
    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, src);
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            ret = atomic_p.load();
            return ret;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.src = src;
    req.op = AMO_FETCH;
    req.type = ishmemi_proxy_get_base_type<T, true, true>();

#if __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_return<T>(req);
#else
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_field_value<T, true, true>(comp.completion.ret);
#endif
    return ret;
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_FETCH(TYPENAME, TYPE) \
    TYPE ishmem_##TYPENAME##_atomic_fetch(TYPE *src, int pe) { return ishmem_atomic_fetch(src, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_ATOMIC_FETCH(float, float)
ISHMEMI_API_IMPL_ATOMIC_FETCH(double, double)
ISHMEMI_API_IMPL_ATOMIC_FETCH(int, int)
ISHMEMI_API_IMPL_ATOMIC_FETCH(long, long)
ISHMEMI_API_IMPL_ATOMIC_FETCH(longlong, long long)
ISHMEMI_API_IMPL_ATOMIC_FETCH(uint, unsigned int)
ISHMEMI_API_IMPL_ATOMIC_FETCH(ulong, unsigned long)
ISHMEMI_API_IMPL_ATOMIC_FETCH(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ATOMIC_FETCH(int32, int32_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH(int64, int64_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH(uint32, uint32_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH(uint64, uint64_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH(size, size_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH(ptrdiff, ptrdiff_t)

/* Atomic set */
template <typename T>
void ishmem_atomic_set(T *dest, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            atomic_p = val;
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = AMO_SET;
    req.type = ishmemi_proxy_get_base_type<T, true, true>();

    ishmemi_proxy_set_field_value<T, true, true>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_SET(TYPE, TYPENAME) \
    void ishmem_##TYPE##_atomic_set(TYPENAME *dest, TYPENAME val, int pe) { ishmem_atomic_set(dest, val, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_ATOMIC_SET(float, float)
ISHMEMI_API_IMPL_ATOMIC_SET(double, double)
ISHMEMI_API_IMPL_ATOMIC_SET(int, int)
ISHMEMI_API_IMPL_ATOMIC_SET(long, long)
ISHMEMI_API_IMPL_ATOMIC_SET(longlong, long long)
ISHMEMI_API_IMPL_ATOMIC_SET(uint, unsigned int)
ISHMEMI_API_IMPL_ATOMIC_SET(ulong, unsigned long)
ISHMEMI_API_IMPL_ATOMIC_SET(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ATOMIC_SET(int32, int32_t)
ISHMEMI_API_IMPL_ATOMIC_SET(int64, int64_t)
ISHMEMI_API_IMPL_ATOMIC_SET(uint32, uint32_t)
ISHMEMI_API_IMPL_ATOMIC_SET(uint64, uint64_t)
ISHMEMI_API_IMPL_ATOMIC_SET(size, size_t)
ISHMEMI_API_IMPL_ATOMIC_SET(ptrdiff, ptrdiff_t)

/* Atomic compare & swap */
template <typename T>
T ishmem_atomic_compare_swap(T *dest, T cond, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    T ret = cond;
    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            atomic_p.compare_exchange_strong(ret, val, sycl::memory_order::seq_cst,
                                             sycl::memory_scope::system);
            return ret;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = AMO_COMPARE_SWAP;
    req.type = ishmemi_proxy_get_base_type<T, true>();

    ishmemi_proxy_set_field_value<T, true>(req.value, val);
    ishmemi_proxy_set_field_value<T, true>(req.cond, cond);

#if __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_return<T>(req);
#else
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_field_value<T, true>(comp.completion.ret);
#endif
    return ret;
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP(TYPENAME, TYPE)  \
    TYPE ishmem_##TYPENAME##_atomic_compare_swap(TYPE *dest, TYPE cond, TYPE val, int pe) { return ishmem_atomic_compare_swap(dest, cond, val, pe); }
/* clang-format on */

/* Atomic Compare & Swap */
ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP(int, int)
ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP(long, long)
ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP(longlong, long long)
ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP(uint, unsigned int)
ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP(ulong, unsigned long)
ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP(int32, int32_t)
ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP(int64, int64_t)
ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP(uint32, uint32_t)
ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP(uint64, uint64_t)
ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP(size, size_t)
ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP(ptrdiff, ptrdiff_t)

/* Atomic swap */
template <typename T>
T ishmem_atomic_swap(T *dest, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    T ret = static_cast<T>(0);
    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            ret = atomic_p.exchange(val, sycl::memory_order::seq_cst, sycl::memory_scope::system);
            return ret;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = AMO_SWAP;
    req.type = ishmemi_proxy_get_base_type<T, true, true>();

    ishmemi_proxy_set_field_value<T, true, true>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_return<T>(req);
#else
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_field_value<T, true, true>(comp.completion.ret);
#endif
    return ret;
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_SWAP(TYPENAME, TYPE)  \
    TYPE ishmem_##TYPENAME##_atomic_swap(TYPE *dest, TYPE val, int pe) { return ishmem_atomic_swap(dest, val, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_ATOMIC_SWAP(float, float)
ISHMEMI_API_IMPL_ATOMIC_SWAP(double, double)
ISHMEMI_API_IMPL_ATOMIC_SWAP(int, int)
ISHMEMI_API_IMPL_ATOMIC_SWAP(long, long)
ISHMEMI_API_IMPL_ATOMIC_SWAP(longlong, long long)
ISHMEMI_API_IMPL_ATOMIC_SWAP(uint, unsigned int)
ISHMEMI_API_IMPL_ATOMIC_SWAP(ulong, unsigned long)
ISHMEMI_API_IMPL_ATOMIC_SWAP(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ATOMIC_SWAP(int32, int32_t)
ISHMEMI_API_IMPL_ATOMIC_SWAP(int64, int64_t)
ISHMEMI_API_IMPL_ATOMIC_SWAP(uint32, uint32_t)
ISHMEMI_API_IMPL_ATOMIC_SWAP(uint64, uint64_t)
ISHMEMI_API_IMPL_ATOMIC_SWAP(size, size_t)
ISHMEMI_API_IMPL_ATOMIC_SWAP(ptrdiff, ptrdiff_t)

/* Atomic fetch increment */
template <typename T>
T ishmem_atomic_fetch_inc(T *dest, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    T ret = static_cast<T>(0);
    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            ret = atomic_p.fetch_add((T) 1);
            return ret;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = AMO_FETCH_INC;
    req.type = ishmemi_proxy_get_base_type<T, true>();

#if __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_return<T>(req);
#else
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_field_value<T, true>(comp.completion.ret);
#endif
    return ret;
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_INC(TYPENAME, TYPE) \
    TYPE ishmem_##TYPENAME##_atomic_fetch_inc(TYPE *dest, int pe) { return ishmem_atomic_fetch_inc(dest, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_ATOMIC_FETCH_INC(int, int)
ISHMEMI_API_IMPL_ATOMIC_FETCH_INC(long, long)
ISHMEMI_API_IMPL_ATOMIC_FETCH_INC(longlong, long long)
ISHMEMI_API_IMPL_ATOMIC_FETCH_INC(uint, unsigned int)
ISHMEMI_API_IMPL_ATOMIC_FETCH_INC(ulong, unsigned long)
ISHMEMI_API_IMPL_ATOMIC_FETCH_INC(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ATOMIC_FETCH_INC(int32, int32_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_INC(int64, int64_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_INC(uint32, uint32_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_INC(uint64, uint64_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_INC(size, size_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_INC(ptrdiff, ptrdiff_t)

/* Atomic increment */
template <typename T>
void ishmem_atomic_inc(T *dest, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            atomic_p += 1;
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = AMO_INC;
    req.type = ishmemi_proxy_get_base_type<T, true>();

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_INC(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_inc(TYPE *dest, int pe) { ishmem_atomic_inc(dest, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_ATOMIC_INC(int, int)
ISHMEMI_API_IMPL_ATOMIC_INC(long, long)
ISHMEMI_API_IMPL_ATOMIC_INC(longlong, long long)
ISHMEMI_API_IMPL_ATOMIC_INC(uint, unsigned int)
ISHMEMI_API_IMPL_ATOMIC_INC(ulong, unsigned long)
ISHMEMI_API_IMPL_ATOMIC_INC(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ATOMIC_INC(int32, int32_t)
ISHMEMI_API_IMPL_ATOMIC_INC(int64, int64_t)
ISHMEMI_API_IMPL_ATOMIC_INC(uint32, uint32_t)
ISHMEMI_API_IMPL_ATOMIC_INC(uint64, uint64_t)
ISHMEMI_API_IMPL_ATOMIC_INC(size, size_t)
ISHMEMI_API_IMPL_ATOMIC_INC(ptrdiff, ptrdiff_t)

/* Atomic fetch add */
template <typename T>
T ishmem_atomic_fetch_add(T *dest, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    T ret = static_cast<T>(0);
    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            ret = atomic_p.fetch_add(val);
            return ret;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = AMO_FETCH_ADD;
    req.type = ishmemi_proxy_get_base_type<T, true>();

    ishmemi_proxy_set_field_value<T, true>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_return<T>(req);
#else
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_field_value<T, true>(comp.completion.ret);
#endif
    return ret;
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD(TYPENAME, TYPE) \
    TYPE ishmem_##TYPENAME##_atomic_fetch_add(TYPE *dest, TYPE val, int pe) { return ishmem_atomic_fetch_add(dest, val, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD(int, int)
ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD(long, long)
ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD(longlong, long long)
ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD(uint, unsigned int)
ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD(ulong, unsigned long)
ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD(int32, int32_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD(int64, int64_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD(uint32, uint32_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD(uint64, uint64_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD(size, size_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD(ptrdiff, ptrdiff_t)

/* Atomic add */
template <typename T>
void ishmem_atomic_add(T *dest, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            atomic_p += val;
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = AMO_ADD;
    req.type = ishmemi_proxy_get_base_type<T, true>();

    ishmemi_proxy_set_field_value<T, true>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_ADD(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_add(TYPE *dest, TYPE val, int pe) { ishmem_atomic_add(dest, val, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_ATOMIC_ADD(int, int)
ISHMEMI_API_IMPL_ATOMIC_ADD(long, long)
ISHMEMI_API_IMPL_ATOMIC_ADD(longlong, long long)
ISHMEMI_API_IMPL_ATOMIC_ADD(uint, unsigned int)
ISHMEMI_API_IMPL_ATOMIC_ADD(ulong, unsigned long)
ISHMEMI_API_IMPL_ATOMIC_ADD(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ATOMIC_ADD(int32, int32_t)
ISHMEMI_API_IMPL_ATOMIC_ADD(int64, int64_t)
ISHMEMI_API_IMPL_ATOMIC_ADD(uint32, uint32_t)
ISHMEMI_API_IMPL_ATOMIC_ADD(uint64, uint64_t)
ISHMEMI_API_IMPL_ATOMIC_ADD(size, size_t)
ISHMEMI_API_IMPL_ATOMIC_ADD(ptrdiff, ptrdiff_t)

/* Atomic fetch and */
template <typename T>
T ishmem_atomic_fetch_and(T *dest, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    T ret = static_cast<T>(0);
    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            ret = atomic_p.fetch_and(val);
            return ret;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = AMO_FETCH_AND;
    req.type = ishmemi_proxy_get_base_type<T, true>();

    ishmemi_proxy_set_field_value<T, true>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_return<T>(req);
#else
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_field_value<T, true>(comp.completion.ret);
#endif
    return ret;
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_AND(TYPENAME, TYPE) \
    TYPE ishmem_##TYPENAME##_atomic_fetch_and(TYPE *dest, TYPE val, int pe) { return ishmem_atomic_fetch_and(dest, val, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_ATOMIC_FETCH_AND(uint, unsigned int)
ISHMEMI_API_IMPL_ATOMIC_FETCH_AND(ulong, unsigned long)
ISHMEMI_API_IMPL_ATOMIC_FETCH_AND(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ATOMIC_FETCH_AND(int32, int32_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_AND(int64, int64_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_AND(uint32, uint32_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_AND(uint64, uint64_t)

/* Atomic and */
template <typename T>
void ishmem_atomic_and(T *dest, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            atomic_p &= val;
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = AMO_AND;
    req.type = ishmemi_proxy_get_base_type<T, true>();

    ishmemi_proxy_set_field_value<T, true>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_AND(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_and(TYPE *dest, TYPE val, int pe) { ishmem_atomic_and(dest, val, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_ATOMIC_AND(uint, unsigned int)
ISHMEMI_API_IMPL_ATOMIC_AND(ulong, unsigned long)
ISHMEMI_API_IMPL_ATOMIC_AND(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ATOMIC_AND(int32, int32_t)
ISHMEMI_API_IMPL_ATOMIC_AND(int64, int64_t)
ISHMEMI_API_IMPL_ATOMIC_AND(uint32, uint32_t)
ISHMEMI_API_IMPL_ATOMIC_AND(uint64, uint64_t)

/* Atomic fetch or */
template <typename T>
T ishmem_atomic_fetch_or(T *dest, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    T ret = static_cast<T>(0);
    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            ret = atomic_p.fetch_or(val);
            return ret;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = AMO_FETCH_OR;
    req.type = ishmemi_proxy_get_base_type<T, true>();

    ishmemi_proxy_set_field_value<T, true>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_return<T>(req);
#else
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_field_value<T, true>(comp.completion.ret);
#endif
    return ret;
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_OR(TYPENAME, TYPE)  \
    TYPE ishmem_##TYPENAME##_atomic_fetch_or(TYPE *dest, TYPE val, int pe) { return ishmem_atomic_fetch_or(dest, val, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_ATOMIC_FETCH_OR(uint, unsigned int)
ISHMEMI_API_IMPL_ATOMIC_FETCH_OR(ulong, unsigned long)
ISHMEMI_API_IMPL_ATOMIC_FETCH_OR(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ATOMIC_FETCH_OR(int32, int32_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_OR(int64, int64_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_OR(uint32, uint32_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_OR(uint64, uint64_t)

/* Atomic or */
template <typename T>
void ishmem_atomic_or(T *dest, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            atomic_p |= val;
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = AMO_OR;
    req.type = ishmemi_proxy_get_base_type<T, true>();

    ishmemi_proxy_set_field_value<T, true>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_OR(TYPENAME, TYPE)  \
    void ishmem_##TYPENAME##_atomic_or(TYPE *dest, TYPE val, int pe) { ishmem_atomic_or(dest, val, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_ATOMIC_OR(uint, unsigned int)
ISHMEMI_API_IMPL_ATOMIC_OR(ulong, unsigned long)
ISHMEMI_API_IMPL_ATOMIC_OR(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ATOMIC_OR(int32, int32_t)
ISHMEMI_API_IMPL_ATOMIC_OR(int64, int64_t)
ISHMEMI_API_IMPL_ATOMIC_OR(uint32, uint32_t)
ISHMEMI_API_IMPL_ATOMIC_OR(uint64, uint64_t)

/* Atomic fetch xor */
template <typename T>
T ishmem_atomic_fetch_xor(T *dest, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    T ret = static_cast<T>(0);
    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            ret = atomic_p.fetch_xor(val);
            return ret;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = AMO_FETCH_XOR;
    req.type = ishmemi_proxy_get_base_type<T, true>();

    ishmemi_proxy_set_field_value<T, true>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_return<T>(req);
#else
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_field_value<T, true>(comp.completion.ret);
#endif
    return ret;
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_XOR(TYPENAME, TYPE) \
    TYPE ishmem_##TYPENAME##_atomic_fetch_xor(TYPE *dest, TYPE val, int pe) { return ishmem_atomic_fetch_xor(dest, val, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_ATOMIC_FETCH_XOR(uint, unsigned int)
ISHMEMI_API_IMPL_ATOMIC_FETCH_XOR(ulong, unsigned long)
ISHMEMI_API_IMPL_ATOMIC_FETCH_XOR(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ATOMIC_FETCH_XOR(int32, int32_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_XOR(int64, int64_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_XOR(uint32, uint32_t)
ISHMEMI_API_IMPL_ATOMIC_FETCH_XOR(uint64, uint64_t)

/* Atomic xor */
template <typename T>
void ishmem_atomic_xor(T *dest, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            atomic_p ^= val;
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = AMO_XOR;
    req.type = ishmemi_proxy_get_base_type<T, true>();

    ishmemi_proxy_set_field_value<T, true>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_XOR(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_xor(TYPE *dest, TYPE val, int pe) { ishmem_atomic_xor(dest, val, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_ATOMIC_XOR(uint, unsigned int)
ISHMEMI_API_IMPL_ATOMIC_XOR(ulong, unsigned long)
ISHMEMI_API_IMPL_ATOMIC_XOR(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_ATOMIC_XOR(int32, int32_t)
ISHMEMI_API_IMPL_ATOMIC_XOR(int64, int64_t)
ISHMEMI_API_IMPL_ATOMIC_XOR(uint32, uint32_t)
ISHMEMI_API_IMPL_ATOMIC_XOR(uint64, uint64_t)
