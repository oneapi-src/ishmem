/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "amo_impl.h"

/* Blocking AMOs */
template <typename T>
T ishmem_atomic_fetch(T *src, int pe)
{
    return amo_impl_return<T, AMO_FETCH>(src, 0, 0, pe);
}

template <typename T>
T ishmem_atomic_compare_swap(T *dest, T cond, T val, int pe)
{
    return amo_impl_return<T, AMO_COMPARE_SWAP>(dest, cond, val, pe);
}

template <typename T>
T ishmem_atomic_swap(T *dest, T val, int pe)
{
    return amo_impl_return<T, AMO_SWAP>(dest, 0, val, pe);
}

template <typename T>
T ishmem_atomic_fetch_inc(T *dest, int pe)
{
    return amo_impl_return<T, AMO_FETCH_INC>(dest, 0, 0, pe);
}

template <typename T>
T ishmem_atomic_fetch_add(T *dest, T val, int pe)
{
    return amo_impl_return<T, AMO_FETCH_ADD>(dest, 0, val, pe);
}

template <typename T>
T ishmem_atomic_fetch_and(T *dest, T val, int pe)
{
    return amo_impl_return<T, AMO_FETCH_AND>(dest, 0, val, pe);
}

template <typename T>
T ishmem_atomic_fetch_or(T *dest, T val, int pe)
{
    return amo_impl_return<T, AMO_FETCH_OR>(dest, 0, val, pe);
}

template <typename T>
T ishmem_atomic_fetch_xor(T *dest, T val, int pe)
{
    return amo_impl_return<T, AMO_FETCH_XOR>(dest, 0, val, pe);
}

template <typename T>
void ishmem_atomic_set(T *dest, T val, int pe)
{
    amo_impl<T, AMO_SET>(dest, val, pe);
}

template <typename T>
void ishmem_atomic_inc(T *dest, int pe)
{
    amo_impl<T, AMO_INC>(dest, 0, pe);
}

template <typename T>
void ishmem_atomic_add(T *dest, T val, int pe)
{
    amo_impl<T, AMO_ADD>(dest, val, pe);
}

template <typename T>
void ishmem_atomic_and(T *dest, T val, int pe)
{
    amo_impl<T, AMO_AND>(dest, val, pe);
}

template <typename T>
void ishmem_atomic_or(T *dest, T val, int pe)
{
    amo_impl<T, AMO_OR>(dest, val, pe);
}

template <typename T>
void ishmem_atomic_xor(T *dest, T val, int pe)
{
    amo_impl<T, AMO_XOR>(dest, val, pe);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_FETCH(TYPENAME, TYPE) \
    TYPE ishmem_##TYPENAME##_atomic_fetch(TYPE *src, int pe) { return ishmem_atomic_fetch<TYPE>(src, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP(TYPENAME, TYPE) \
    TYPE ishmem_##TYPENAME##_atomic_compare_swap(TYPE *dest, TYPE cond, TYPE val, int pe) { return ishmem_atomic_compare_swap<TYPE>(dest, cond, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_SWAP(TYPENAME, TYPE) \
    TYPE ishmem_##TYPENAME##_atomic_swap(TYPE *dest, TYPE val, int pe) { return ishmem_atomic_swap<TYPE>(dest, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_INC(TYPENAME, TYPE) \
    TYPE ishmem_##TYPENAME##_atomic_fetch_inc(TYPE *dest, int pe) { return ishmem_atomic_fetch_inc<TYPE>(dest, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD(TYPENAME, TYPE) \
    TYPE ishmem_##TYPENAME##_atomic_fetch_add(TYPE *dest, TYPE val, int pe) { return ishmem_atomic_fetch_add<TYPE>(dest, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_AND(TYPENAME, TYPE) \
    TYPE ishmem_##TYPENAME##_atomic_fetch_and(TYPE *dest, TYPE val, int pe) { return ishmem_atomic_fetch_and<TYPE>(dest, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_OR(TYPENAME, TYPE) \
    TYPE ishmem_##TYPENAME##_atomic_fetch_or(TYPE *dest, TYPE val, int pe) { return ishmem_atomic_fetch_or<TYPE>(dest, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_XOR(TYPENAME, TYPE) \
    TYPE ishmem_##TYPENAME##_atomic_fetch_xor(TYPE *dest, TYPE val, int pe) { return ishmem_atomic_fetch_xor<TYPE>(dest, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_SET(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_set(TYPE *dest, TYPE val, int pe) { ishmem_atomic_set<TYPE>(dest, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_INC(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_inc(TYPE *dest, int pe) { ishmem_atomic_inc<TYPE>(dest, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_ADD(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_add(TYPE *dest, TYPE val, int pe) { ishmem_atomic_add<TYPE>(dest, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_AND(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_and(TYPE *dest, TYPE val, int pe) { ishmem_atomic_and<TYPE>(dest, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_OR(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_or(TYPE *dest, TYPE val, int pe) { ishmem_atomic_or<TYPE>(dest, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_XOR(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_xor(TYPE *dest, TYPE val, int pe) { ishmem_atomic_xor<TYPE>(dest, val, pe); }
/* clang-format on */

ISHMEMI_API_GENERATE_AMO_EXT_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH)
ISHMEMI_API_GENERATE_AMO_STD_TYPES(ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP)
ISHMEMI_API_GENERATE_AMO_EXT_TYPES(ISHMEMI_API_IMPL_ATOMIC_SWAP)
ISHMEMI_API_GENERATE_AMO_STD_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_INC)
ISHMEMI_API_GENERATE_AMO_STD_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD)
ISHMEMI_API_GENERATE_AMO_BIT_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_AND)
ISHMEMI_API_GENERATE_AMO_BIT_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_OR)
ISHMEMI_API_GENERATE_AMO_BIT_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_XOR)
ISHMEMI_API_GENERATE_AMO_EXT_TYPES(ISHMEMI_API_IMPL_ATOMIC_SET)
ISHMEMI_API_GENERATE_AMO_STD_TYPES(ISHMEMI_API_IMPL_ATOMIC_INC)
ISHMEMI_API_GENERATE_AMO_STD_TYPES(ISHMEMI_API_IMPL_ATOMIC_ADD)
ISHMEMI_API_GENERATE_AMO_BIT_TYPES(ISHMEMI_API_IMPL_ATOMIC_AND)
ISHMEMI_API_GENERATE_AMO_BIT_TYPES(ISHMEMI_API_IMPL_ATOMIC_OR)
ISHMEMI_API_GENERATE_AMO_BIT_TYPES(ISHMEMI_API_IMPL_ATOMIC_XOR)

/* Non-Blocking AMOs */
template <typename T>
void ishmem_atomic_fetch_nbi(T *fetch, T *src, int pe)
{
    amo_nbi_impl<T, AMO_FETCH_NBI>(fetch, src, 0, 0, pe);
}

template <typename T>
void ishmem_atomic_compare_swap_nbi(T *fetch, T *dest, T cond, T val, int pe)
{
    amo_nbi_impl<T, AMO_COMPARE_SWAP_NBI>(fetch, dest, cond, val, pe);
}

template <typename T>
void ishmem_atomic_swap_nbi(T *fetch, T *dest, T val, int pe)
{
    amo_nbi_impl<T, AMO_SWAP_NBI>(fetch, dest, 0, val, pe);
}

template <typename T>
void ishmem_atomic_fetch_inc_nbi(T *fetch, T *dest, int pe)
{
    amo_nbi_impl<T, AMO_FETCH_INC_NBI>(fetch, dest, 0, 0, pe);
}

template <typename T>
void ishmem_atomic_fetch_add_nbi(T *fetch, T *dest, T val, int pe)
{
    amo_nbi_impl<T, AMO_FETCH_ADD_NBI>(fetch, dest, 0, val, pe);
}

template <typename T>
void ishmem_atomic_fetch_and_nbi(T *fetch, T *dest, T val, int pe)
{
    amo_nbi_impl<T, AMO_FETCH_AND_NBI>(fetch, dest, 0, val, pe);
}

template <typename T>
void ishmem_atomic_fetch_or_nbi(T *fetch, T *dest, T val, int pe)
{
    amo_nbi_impl<T, AMO_FETCH_OR_NBI>(fetch, dest, 0, val, pe);
}

template <typename T>
void ishmem_atomic_fetch_xor_nbi(T *fetch, T *dest, T val, int pe)
{
    amo_nbi_impl<T, AMO_FETCH_XOR_NBI>(fetch, dest, 0, val, pe);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_NBI(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_fetch_nbi(TYPE *fetch, TYPE *src, int pe) { ishmem_atomic_fetch_nbi<TYPE>(fetch, src, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP_NBI(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_compare_swap_nbi(TYPE *fetch, TYPE *dest, TYPE cond, TYPE val, int pe) { ishmem_atomic_compare_swap_nbi<TYPE>(fetch, dest, cond, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_SWAP_NBI(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_swap_nbi(TYPE *fetch, TYPE *dest, TYPE val, int pe) { ishmem_atomic_swap_nbi<TYPE>(fetch, dest, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_INC_NBI(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_fetch_inc_nbi(TYPE *fetch, TYPE *dest, int pe) { ishmem_atomic_fetch_inc_nbi<TYPE>(fetch, dest, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD_NBI(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_fetch_add_nbi(TYPE *fetch, TYPE *dest, TYPE val, int pe) { ishmem_atomic_fetch_add_nbi<TYPE>(fetch, dest, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_AND_NBI(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_fetch_and_nbi(TYPE *fetch, TYPE *dest, TYPE val, int pe) { ishmem_atomic_fetch_and_nbi<TYPE>(fetch, dest, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_OR_NBI(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_fetch_or_nbi(TYPE *fetch, TYPE *dest, TYPE val, int pe) { ishmem_atomic_fetch_or_nbi<TYPE>(fetch, dest, val, pe); }
#define ISHMEMI_API_IMPL_ATOMIC_FETCH_XOR_NBI(TYPENAME, TYPE) \
    void ishmem_##TYPENAME##_atomic_fetch_xor_nbi(TYPE *fetch, TYPE *dest, TYPE val, int pe) { ishmem_atomic_fetch_xor_nbi<TYPE>(fetch, dest, val, pe); }
/* clang-format on */

ISHMEMI_API_GENERATE_AMO_EXT_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_NBI)
ISHMEMI_API_GENERATE_AMO_STD_TYPES(ISHMEMI_API_IMPL_ATOMIC_COMPARE_SWAP_NBI)
ISHMEMI_API_GENERATE_AMO_EXT_TYPES(ISHMEMI_API_IMPL_ATOMIC_SWAP_NBI)
ISHMEMI_API_GENERATE_AMO_STD_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_INC_NBI)
ISHMEMI_API_GENERATE_AMO_STD_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_ADD_NBI)
ISHMEMI_API_GENERATE_AMO_BIT_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_AND_NBI)
ISHMEMI_API_GENERATE_AMO_BIT_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_OR_NBI)
ISHMEMI_API_GENERATE_AMO_BIT_TYPES(ISHMEMI_API_IMPL_ATOMIC_FETCH_XOR_NBI)
