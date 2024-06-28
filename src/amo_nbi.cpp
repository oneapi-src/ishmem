/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

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

/* float point types are only for extended AMO types */
template <ishmemi_op_t OP>
struct fp_matters {
    static constexpr bool value = (OP == AMO_FETCH_NBI || OP == AMO_SWAP_NBI);
};

/* Generalized class for all non-blocking atomic memory operations */
template <typename T, ishmemi_op_t OP>
class AMO_NBI {
    T *fetch;
    T *dest;
    T cond;
    T val;
    int pe;

  public:
    AMO_NBI(T *fetch, T *dest, T cond, T val, int pe)
        : fetch(fetch), dest(dest), cond(cond), val(val), pe(pe)
    {
    }

    void impl()
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
                    *fetch = atomic_p.exchange(val, sycl::memory_order::seq_cst,
                                               sycl::memory_scope::system);
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
        req.type = ishmemi_proxy_get_base_type<T, true, fp_matters<OP>::value>();

        if constexpr (OP != AMO_FETCH_NBI && OP != AMO_FETCH_INC_NBI) {
            ishmemi_proxy_set_field_value<T, true, fp_matters<OP>::value>(req.value, val);
        }
        if constexpr (OP == AMO_COMPARE_SWAP_NBI) {
            ishmemi_proxy_set_field_value<T, true>(req.cond, cond);
        }

#if __SYCL_DEVICE_ONLY__
        ishmemi_proxy_nonblocking_request(req);
#else
        ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
    }
};

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_NBI(TYPENAME, TYPE)\
    void ishmem_##TYPENAME##_atomic_fetch_nbi(TYPE *fetch, TYPE *src, int pe) { AMO_NBI<TYPE, AMO_FETCH_NBI>(fetch, src, 0, 0, pe).impl(); }
#define ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP_NBI(TYPENAME, TYPE)\
    void ishmem_##TYPENAME##_atomic_compare_swap_nbi(TYPE *fetch, TYPE *dest, TYPE cond, TYPE val, int pe) {AMO_NBI<TYPE, AMO_COMPARE_SWAP_NBI>(fetch, dest, cond, val, pe).impl();}
#define ISHMEMI_API_IMPL_ATOMIC_SWAP_NBI(TYPENAME, TYPE)\
    void ishmem_##TYPENAME##_atomic_swap_nbi(TYPE *fetch, TYPE *dest, TYPE val, int pe) { AMO_NBI<TYPE, AMO_SWAP_NBI>(fetch, dest, 0, val, pe).impl();}
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_INC_NBI(TYPENAME, TYPE)\
    void ishmem_##TYPENAME##_atomic_fetch_inc_nbi(TYPE *fetch, TYPE *dest, int pe) { AMO_NBI<TYPE, AMO_FETCH_INC_NBI>(fetch, dest, 0, 0, pe).impl();}
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD_NBI(TYPENAME, TYPE)\
    void ishmem_##TYPENAME##_atomic_fetch_add_nbi(TYPE *fetch, TYPE *dest, TYPE val, int pe) { AMO_NBI<TYPE, AMO_FETCH_ADD_NBI>(fetch, dest, 0, val, pe).impl();}
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_AND_NBI(TYPENAME, TYPE)\
    void ishmem_##TYPENAME##_atomic_fetch_and_nbi(TYPE *fetch, TYPE *dest, TYPE val, int pe) { AMO_NBI<TYPE, AMO_FETCH_AND_NBI>(fetch, dest, 0, val, pe).impl();}
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_OR_NBI(TYPENAME, TYPE)\
    void ishmem_##TYPENAME##_atomic_fetch_or_nbi(TYPE *fetch, TYPE *dest, TYPE val, int pe) { AMO_NBI<TYPE, AMO_FETCH_OR_NBI>(fetch, dest, 0, val, pe).impl();}
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_XOR_NBI(TYPENAME, TYPE)\
    void ishmem_##TYPENAME##_atomic_fetch_xor_nbi(TYPE *fetch, TYPE *dest, TYPE val, int pe) { AMO_NBI<TYPE, AMO_FETCH_XOR_NBI>(fetch, dest, 0, val, pe).impl();}
/* clang-format on */

ISHMEMI_API_GENERATE_AMO_EXT_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_NBI)
ISHMEMI_API_GENERATE_AMO_STD_TYPES(ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP_NBI)
ISHMEMI_API_GENERATE_AMO_EXT_TYPES(ISHMEMI_API_IMPL_ATOMIC_SWAP_NBI)
ISHMEMI_API_GENERATE_AMO_STD_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_INC_NBI)
ISHMEMI_API_GENERATE_AMO_STD_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD_NBI)
ISHMEMI_API_GENERATE_AMO_BIT_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_AND_NBI)
ISHMEMI_API_GENERATE_AMO_BIT_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_OR_NBI)
ISHMEMI_API_GENERATE_AMO_BIT_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_XOR_NBI)
