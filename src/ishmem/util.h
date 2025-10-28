/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portions derived from Sandia OpenSHMEM (https://github.com/Sandia-OpenSHMEM/SOS)
 * For license and copyright information, see the LICENSE and third-party-programs.txt
 * files in the top level directory of this distribution.
 */

/* internal API and globals */
#ifndef ISHMEM_UTIL_H
#define ISHMEM_UTIL_H

#include "ishmem/types.h"

#include <iostream>
#include <cstdlib>

#include "ishmem.h"
#include "ishmemx.h"

#ifdef __SYCL_DEVICE_ONLY__
#define ISHMEMI_LOCAL_PES global_info->local_pes
#define ISHMEMI_N_TEAMS   global_info->n_teams
#else
#define ISHMEMI_LOCAL_PES ishmemi_local_pes
#define ISHMEMI_N_TEAMS   ishmemi_cpu_info->n_teams
#endif

#define MAX_LOCAL_PES 64

extern int ishmemi_my_pe;
extern int ishmemi_n_pes;

typedef struct ishmemi_info_t ishmemi_info_t;

/* TODO should these be combined into ishmem_host_data_t? */
/* Device parameters for the device copy of the data */
extern void *ishmemi_heap_base;
extern size_t ishmemi_heap_length;
extern uintptr_t ishmemi_heap_last;
extern ishmemi_info_t *ishmemi_gpu_info;
/* this is the device global */
ISHMEM_DEVICE_ATTRIBUTES extern sycl::ext::oneapi::experimental::device_global<ishmemi_info_t *>
    global_info;

/* allocated size for info data structure (variable due to n_pes) */
extern size_t ishmemi_info_size;

/* Host parameters for the device data structures */
extern ishmemi_info_t *ishmemi_mmap_gpu_info;
extern void *ishmemi_mmap_heap_base;

/* Host globals to hold the host version of data */
extern uint8_t *ishmemi_local_pes;
extern void *ishmemi_ipc_buffers[MAX_LOCAL_PES + 1];

/* host global for host address of host memory copy of ipc_buffer_delta */
extern ptrdiff_t ishmemi_ipc_buffer_delta[MAX_LOCAL_PES + 1];
extern bool ishmemi_only_intra_node;

/* Used to reduce reliance on macros in function definitions */
#ifdef __SYCL_DEVICE_ONLY__
constexpr bool ishmemi_is_device = true;
#else
constexpr bool ishmemi_is_device = false;
#endif

/* used for garbage collecting nbi cmd lists on synchronize */
/* pre-size this according to ishmem_nbi_count environment, with default 1000 */
/* then auto-cleanup on synchronize or when the number gets to the limit */
template <typename T>
class ishmemi_thread_safe_vector {
  private:
    std::vector<T> vec;

  public:
    std::mutex mtx;

    size_t push_back_thread_safe(T value)
    {
        std::lock_guard<std::mutex> lock(mtx);
        vec.push_back(value);
        return vec.size();
    }

    size_t size()
    {
        return vec.size();
    }

    typename std::vector<T>::iterator begin()
    {
        return vec.begin();
    }

    typename std::vector<T>::iterator end()
    {
        return vec.end();
    }

    typename std::vector<T>::iterator erase(typename std::vector<T>::iterator first,
                                            typename std::vector<T>::iterator last)
    {
        return vec.erase(first, last);
    }

    void reserve(size_t n)
    {
        vec.reserve(n);
    }

    T &operator[](size_t i)
    {
        return vec[i];
    }
};

/* In cleanup, free an object only if not null, then set it to null */
#define ISHMEMI_FREE(freefn, x)                                                                    \
    if ((x) != nullptr) {                                                                          \
        freefn(x);                                                                                 \
        x = nullptr;                                                                               \
    }

template <typename T>
inline int ishmemi_comparison(T val1, T val2, int cmp)
{
    switch (cmp) {
        case (ISHMEM_CMP_EQ):
            return (int) (val1 == val2);
        case (ISHMEM_CMP_NE):
            return (int) (val1 != val2);
        case (ISHMEM_CMP_GT):
            return (int) (val1 > val2);
        case (ISHMEM_CMP_GE):
            return (int) (val1 >= val2);
        case (ISHMEM_CMP_LT):
            return (int) (val1 < val2);
        case (ISHMEM_CMP_LE):
            return (int) (val1 <= val2);
        default:
            // TODO: Add global exit method callable from GPU
            ishmemx_print("invalid 'cmp' value provided.\n", ishmemx_print_msg_type_t::ERROR);
            return -1;
    }
}

static inline void ishmemi_bit_set(unsigned char *ptr, size_t size, size_t index)
{
    /* TODO: add non-persistent assert? */
    assert(size > 0 && (index < size * CHAR_BIT));

    size_t which_byte = index / CHAR_BIT;
    ptr[which_byte] |= (1 << (index % CHAR_BIT));

    return;
}

static inline void ishmemi_bit_clear(unsigned char *ptr, size_t size, size_t index)
{
    assert(size > 0 && (index < size * CHAR_BIT));

    size_t which_byte = index / CHAR_BIT;
    ptr[which_byte] &= ~(1 << (index % CHAR_BIT));

    return;
}

static inline char ishmemi_bit_fetch(unsigned char *ptr, size_t size, size_t index)
{
    assert(size > 0 && (index < size * CHAR_BIT));

    size_t which_byte = index / CHAR_BIT;
    return (ptr[which_byte] >> (index % CHAR_BIT)) & 1;
}

static inline size_t ishmemi_bit_1st_nonzero(const unsigned char *ptr, const size_t size)
{
    /* The following ignores endianess: */
    for (size_t i = 0; i < size; i++) {
        unsigned char bit_val = ptr[i];
        for (size_t j = 0; bit_val && j < CHAR_BIT; j++) {
            if (bit_val & 1) return i * CHAR_BIT + j;
            bit_val >>= 1;
        }
    }

    return static_cast<size_t>(-1);
}

/* Create a bit string of the format AAAAAAAA.BBBBBBBB into str for the byte
 * array passed via ptr. */
static inline void ishmemi_bit_to_string(char *str, size_t str_size, unsigned char *ptr,
                                         size_t ptr_size)
{
    size_t off = 0;

    for (size_t i = 0; i < ptr_size; i++) {
        for (size_t j = 0; j < CHAR_BIT; j++) {
            off += static_cast<size_t>(snprintf(str + off, str_size - off, "%s",
                                                (ptr[i] & (1 << (CHAR_BIT - 1 - j))) ? "1" : "0"));
            if (off >= str_size) return;
        }
        if (i < ptr_size - 1) {
            off += static_cast<size_t>(snprintf(str + off, str_size - off, "."));
            if (off >= str_size) return;
        }
    }
}

template <typename TYPE>
ISHMEM_DEVICE_ATTRIBUTES constexpr ishmemi_type_t ishmemi_get_type()
{
    if constexpr (std::is_same_v<TYPE, float>) return FLOAT;
    else if constexpr (std::is_same_v<TYPE, double>) return DOUBLE;
    else if constexpr (std::is_same_v<TYPE, long double>) return LONGDOUBLE;
    else if constexpr (std::is_same_v<TYPE, char>) return CHAR;
    else if constexpr (std::is_same_v<TYPE, signed char>) return SCHAR;
    else if constexpr (std::is_same_v<TYPE, short>) return SHORT;
    else if constexpr (std::is_same_v<TYPE, int>) return INT;
    else if constexpr (std::is_same_v<TYPE, long>) return LONG;
    else if constexpr (std::is_same_v<TYPE, long long>) return LONGLONG;
    else if constexpr (std::is_same_v<TYPE, unsigned char>) return UCHAR;
    else if constexpr (std::is_same_v<TYPE, unsigned short>) return USHORT;
    else if constexpr (std::is_same_v<TYPE, unsigned int>) return UINT;
    else if constexpr (std::is_same_v<TYPE, unsigned long>) return ULONG;
    else if constexpr (std::is_same_v<TYPE, unsigned long long>) return ULONGLONG;
    else if constexpr (std::is_same_v<TYPE, int8_t>) return INT8;
    else if constexpr (std::is_same_v<TYPE, int16_t>) return INT16;
    else if constexpr (std::is_same_v<TYPE, int32_t>) return INT32;
    else if constexpr (std::is_same_v<TYPE, int64_t>) return INT64;
    else if constexpr (std::is_same_v<TYPE, uint8_t>) return UINT8;
    else if constexpr (std::is_same_v<TYPE, uint16_t>) return UINT16;
    else if constexpr (std::is_same_v<TYPE, uint32_t>) return UINT32;
    else if constexpr (std::is_same_v<TYPE, uint64_t>) return UINT64;
    else if constexpr (std::is_same_v<TYPE, size_t>) return SIZE;
    else if constexpr (std::is_same_v<TYPE, ptrdiff_t>) return PTRDIFF;
    else if constexpr (std::is_same_v<TYPE, void>) return MEM;
    else return ISHMEMI_TYPE_END;
}

template <ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr bool ishmemi_op_is_bitwise_reduction()
{
    if constexpr (OP == AND_REDUCE) return true;
    else if constexpr (OP == OR_REDUCE) return true;
    else if constexpr (OP == XOR_REDUCE) return true;
    else return false;
}

template <ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr bool ishmemi_op_is_value_reduction()
{
    if constexpr (OP == MAX_REDUCE) return true;
    else if constexpr (OP == MIN_REDUCE) return true;
    else if constexpr (OP == SUM_REDUCE) return true;
    else if constexpr (OP == PROD_REDUCE) return true;
    else return false;
}

template <ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr bool ishmemi_op_is_reduction()
{
    if constexpr (ishmemi_op_is_bitwise_reduction<OP>()) return true;
    else if constexpr (ishmemi_op_is_value_reduction<OP>()) return true;
    else return false;
}

template <ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr bool ishmemi_op_is_scan()
{
    if constexpr (OP == INSCAN) return true;
    else if constexpr (OP == EXSCAN) return true;
    else return false;
}

template <ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr bool ishmemi_op_is_standard_amo()
{
    if constexpr (OP == AMO_COMPARE_SWAP) return true;
    else if constexpr (OP == AMO_FETCH_INC) return true;
    else if constexpr (OP == AMO_INC) return true;
    else if constexpr (OP == AMO_FETCH_ADD) return true;
    else if constexpr (OP == AMO_ADD) return true;
    else if constexpr (OP == AMO_COMPARE_SWAP_NBI) return true;
    else if constexpr (OP == AMO_FETCH_INC_NBI) return true;
    else if constexpr (OP == AMO_FETCH_ADD_NBI) return true;
    else return false;
}

template <ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr bool ishmemi_op_is_extended_amo()
{
    if constexpr (OP == AMO_FETCH) return true;
    else if constexpr (OP == AMO_SET) return true;
    else if constexpr (OP == AMO_SWAP) return true;
    else if constexpr (OP == AMO_FETCH_NBI) return true;
    else if constexpr (OP == AMO_SWAP_NBI) return true;
    else return false;
}

template <ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr bool ishmemi_op_is_bitwise_amo()
{
    if constexpr (OP == AMO_FETCH_AND) return true;
    else if constexpr (OP == AMO_AND) return true;
    else if constexpr (OP == AMO_FETCH_OR) return true;
    else if constexpr (OP == AMO_OR) return true;
    else if constexpr (OP == AMO_FETCH_XOR) return true;
    else if constexpr (OP == AMO_XOR) return true;
    else if constexpr (OP == AMO_FETCH_AND_NBI) return true;
    else if constexpr (OP == AMO_FETCH_OR_NBI) return true;
    else if constexpr (OP == AMO_FETCH_XOR_NBI) return true;
    else return false;
}

template <ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr bool ishmemi_op_is_amo()
{
    if constexpr (ishmemi_op_is_standard_amo<OP>()) return true;
    else if constexpr (ishmemi_op_is_bitwise_amo<OP>()) return true;
    else if constexpr (ishmemi_op_is_extended_amo<OP>()) return true;
    else return false;
}

template <ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr bool ishmemi_op_is_sync()
{
    if constexpr (OP == WAIT) return true;
    else if constexpr (OP == WAIT_ALL) return true;
    else if constexpr (OP == WAIT_ALL_VECTOR) return true;
    else if constexpr (OP == WAIT_ANY) return true;
    else if constexpr (OP == WAIT_ANY_VECTOR) return true;
    else if constexpr (OP == WAIT_SOME) return true;
    else if constexpr (OP == WAIT_SOME_VECTOR) return true;
    else if constexpr (OP == TEST) return true;
    else if constexpr (OP == TEST_ALL) return true;
    else if constexpr (OP == TEST_ALL_VECTOR) return true;
    else if constexpr (OP == TEST_ANY) return true;
    else if constexpr (OP == TEST_ANY_VECTOR) return true;
    else if constexpr (OP == TEST_SOME) return true;
    else if constexpr (OP == TEST_SOME_VECTOR) return true;
    else if constexpr (OP == SIGNAL_WAIT_UNTIL) return true;
    else return false;
}

template <ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr bool ishmemi_op_floating_point_matters()
{
    if constexpr (ishmemi_op_is_value_reduction<OP>()) return true;
    else if constexpr (ishmemi_op_is_scan<OP>()) return true;
    else if constexpr (ishmemi_op_is_extended_amo<OP>()) return true;
    else if constexpr (OP == P) return true;
    else if constexpr (OP == G) return true;
    else return false;
}

template <ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr bool ishmemi_op_sign_matters()
{
    if constexpr (ishmemi_op_is_reduction<OP>()) return true;
    else if constexpr (ishmemi_op_is_scan<OP>()) return true;
    else if constexpr (ishmemi_op_is_amo<OP>()) return true;
    else if constexpr (OP == SIGNAL_WAIT_UNTIL) return false;
    else if constexpr (ishmemi_op_is_sync<OP>()) return true;
    else return false;
}

template <typename T, ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr void ishmemi_union_set_field_value(ishmemi_union_type &field,
                                                                      const T val)
{
    /* Floating-point types */
    if constexpr (ishmemi_op_floating_point_matters<OP>() && std::is_floating_point_v<T>) {
        if constexpr (std::is_same_v<T, float>) {
            field.f = static_cast<float>(val);
            return;
        } else if constexpr (std::is_same_v<T, double>) {
            field.ld = static_cast<double>(val);
            return;
        } else static_assert(false, "Unknown or unsupported type");
    }

    /* Signed types */
    if constexpr (ishmemi_op_sign_matters<OP>() && std::is_signed_v<T>) {
        if constexpr (sizeof(T) == sizeof(int8_t)) {
            field.i8 = static_cast<int8_t>(val);
            return;
        } else if constexpr (sizeof(T) == sizeof(int16_t)) {
            field.i16 = static_cast<int16_t>(val);
            return;
        } else if constexpr (sizeof(T) == sizeof(int32_t)) {
            field.i32 = static_cast<int32_t>(val);
            return;
        } else if constexpr (sizeof(T) == sizeof(int64_t)) {
            field.i64 = static_cast<int64_t>(val);
            return;
        } else if constexpr (sizeof(T) == sizeof(long long)) {
            field.ll = static_cast<long long>(val);
            return;
        } else static_assert(false, "Unknown or unsupported type");
    }

    /* Unsigned types */
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        field.ui8 = static_cast<uint8_t>(val);
        return;
    } else if constexpr (sizeof(T) == sizeof(uint16_t)) {
        field.ui16 = static_cast<uint16_t>(val);
        return;
    } else if constexpr (sizeof(T) == sizeof(uint32_t)) {
        field.ui32 = static_cast<uint32_t>(val);
        return;
    } else if constexpr (sizeof(T) == sizeof(uint64_t)) {
        field.ui64 = static_cast<uint64_t>(val);
        return;
    } else if constexpr (sizeof(T) == sizeof(unsigned long long)) {
        field.ull = static_cast<unsigned long long>(val);
        return;
    } else static_assert(false, "Unknown or unsupported type");
}

template <typename T, ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr T ishmemi_union_get_field_value(const ishmemi_union_type &field)
{
    /* Floating-point types */
    if constexpr (ishmemi_op_floating_point_matters<OP>() && std::is_floating_point_v<T>) {
        if constexpr (std::is_same_v<T, float>) return static_cast<T>(field.f);
        else if constexpr (std::is_same_v<T, double>) return static_cast<T>(field.ld);
        else static_assert(false, "Unknown or unsupported type");
    }

    /* Signed types */
    if constexpr (ishmemi_op_sign_matters<OP>() && std::is_signed_v<T>) {
        if constexpr (sizeof(T) == sizeof(int8_t)) return static_cast<T>(field.i8);
        else if constexpr (sizeof(T) == sizeof(int16_t)) return static_cast<T>(field.i16);
        else if constexpr (sizeof(T) == sizeof(int32_t)) return static_cast<T>(field.i32);
        else if constexpr (sizeof(T) == sizeof(int64_t)) return static_cast<T>(field.i64);
        else if constexpr (sizeof(T) == sizeof(long long)) return static_cast<T>(field.ll);
        else static_assert(false, "Unknown or unsupported type");
    }

    /* Unsigned types */
    if constexpr (sizeof(T) == sizeof(uint8_t)) return static_cast<T>(field.ui8);
    else if constexpr (sizeof(T) == sizeof(uint16_t)) return static_cast<T>(field.ui16);
    else if constexpr (sizeof(T) == sizeof(uint32_t)) return static_cast<T>(field.ui32);
    else if constexpr (sizeof(T) == sizeof(uint64_t)) return static_cast<T>(field.ui64);
    else if constexpr (sizeof(T) == sizeof(unsigned long long)) return static_cast<T>(field.ull);
    else static_assert(false, "Unknown or unsupported type");
}

template <typename T, ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr ishmemi_type_t ishmemi_union_get_base_type()
{
    /* Floating-point types */
    if constexpr (ishmemi_op_floating_point_matters<OP>() && std::is_floating_point_v<T>) {
        if constexpr (std::is_same_v<T, float>) return FLOAT;
        else if constexpr (std::is_same_v<T, double>) return DOUBLE;
        else static_assert(false, "Unknown or unsupported type");
    }

    /* Signed types */
    if constexpr (ishmemi_op_sign_matters<OP>() && std::is_signed_v<T>) {
        if constexpr (sizeof(T) == sizeof(int8_t)) return INT8;
        else if constexpr (sizeof(T) == sizeof(int16_t)) return INT16;
        else if constexpr (sizeof(T) == sizeof(int32_t)) return INT32;
        else if constexpr (sizeof(T) == sizeof(int64_t)) return INT64;
        else if constexpr (sizeof(T) == sizeof(long long)) return LONGLONG;
        else static_assert(false, "Unknown or unsupported type");
    }

    /* Unsigned types */
    if constexpr (sizeof(T) == sizeof(uint8_t)) return UINT8;
    else if constexpr (sizeof(T) == sizeof(uint16_t)) return UINT16;
    else if constexpr (sizeof(T) == sizeof(uint32_t)) return UINT32;
    else if constexpr (sizeof(T) == sizeof(uint64_t)) return UINT64;
    else if constexpr (sizeof(T) == sizeof(unsigned long long)) return ULONGLONG;
    else static_assert(false, "Unknown or unsupported type");
}

template <ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES constexpr bool ishmemi_op_uses_team()
{
    if constexpr (ishmemi_op_is_reduction<OP>()) return true;
    else if constexpr (ishmemi_op_is_scan<OP>()) return true;
    else if constexpr (OP == ALLTOALL) return true;
    else if constexpr (OP == BCAST) return true;
    else if constexpr (OP == COLLECT) return true;
    else if constexpr (OP == FCOLLECT) return true;
    else if constexpr (OP == TEAM_MY_PE) return true;
    else if constexpr (OP == TEAM_N_PES) return true;
    else if constexpr (OP == TEAM_SYNC) return true;
    else return false;
}

#endif /* ISHMEM_UTIL_H */
