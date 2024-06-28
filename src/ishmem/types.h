/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* ishmem common types and definitions */
#ifndef ISHMEM_TYPES_H
#define ISHMEM_TYPES_H

#include <CL/sycl.hpp>

/* Note: this should mirror ISHMEM_DEVICE_ATTRIBUTES definition in ishmem.h */
#define ISHMEM_DEVICE_ATTRIBUTES SYCL_EXTERNAL

typedef enum : uint16_t {
    PUT = 0,
    IPUT,
    IBPUT,
    P,
    GET,
    IGET,
    IBGET,
    G,
    PUT_NBI,
    GET_NBI,
    AMO_FETCH,
    AMO_SET,
    AMO_COMPARE_SWAP,
    AMO_SWAP,
    AMO_FETCH_INC,
    AMO_INC,
    AMO_FETCH_ADD,
    AMO_ADD,
    AMO_FETCH_AND,
    AMO_AND,
    AMO_FETCH_OR,
    AMO_OR,
    AMO_FETCH_XOR,
    AMO_XOR,
    AMO_FETCH_NBI,
    AMO_COMPARE_SWAP_NBI,
    AMO_SWAP_NBI,
    AMO_FETCH_INC_NBI,
    AMO_FETCH_ADD_NBI,
    AMO_FETCH_AND_NBI,
    AMO_FETCH_OR_NBI,
    AMO_FETCH_XOR_NBI,
    PUT_SIGNAL,
    PUT_SIGNAL_NBI,
    SIGNAL_FETCH,
    SIGNAL_ADD,
    SIGNAL_SET,
    BARRIER,
    SYNC,
    ALLTOALL,
    BCAST,
    COLLECT,
    FCOLLECT,
    AND_REDUCE,
    OR_REDUCE,
    XOR_REDUCE,
    MAX_REDUCE,
    MIN_REDUCE,
    SUM_REDUCE,
    PROD_REDUCE,
    WAIT,
    WAIT_ALL,
    WAIT_ANY,
    WAIT_SOME,
    TEST,
    TEST_ALL,
    TEST_ANY,
    TEST_SOME,
    SIGNAL_WAIT_UNTIL,
    FENCE,
    QUIET,
    TEAM_MY_PE,
    TEAM_N_PES,
    TEAM_SYNC,
    KILL,
    NOP,
    NOP_NO_R,
    TIMESTAMP,
    PRINT,
    DEBUG_TEST,
    ISHMEMI_OP_END
} ishmemi_op_t;

constexpr ishmemi_op_t ISHMEMI_OP_put = PUT;
constexpr ishmemi_op_t ISHMEMI_OP_iput = IPUT;
constexpr ishmemi_op_t ISHMEMI_OP_p = P;
constexpr ishmemi_op_t ISHMEMI_OP_get = GET;
constexpr ishmemi_op_t ISHMEMI_OP_iget = IGET;
constexpr ishmemi_op_t ISHMEMI_OP_g = G;
constexpr ishmemi_op_t ISHMEMI_OP_put_nbi = PUT_NBI;
constexpr ishmemi_op_t ISHMEMI_OP_get_nbi = GET_NBI;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_fetch = AMO_FETCH;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_set = AMO_SET;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_compare_swap = AMO_COMPARE_SWAP;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_swap = AMO_SWAP;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_fetch_inc = AMO_FETCH_INC;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_inc = AMO_INC;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_fetch_add = AMO_FETCH_ADD;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_add = AMO_ADD;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_fetch_and = AMO_FETCH_AND;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_and = AMO_AND;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_fetch_or = AMO_FETCH_OR;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_or = AMO_OR;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_fetch_xor = AMO_FETCH_XOR;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_xor = AMO_XOR;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_fetch_nbi = AMO_FETCH_NBI;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_compare_swap_nbi = AMO_COMPARE_SWAP_NBI;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_swap_nbi = AMO_SWAP_NBI;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_fetch_inc_nbi = AMO_FETCH_INC_NBI;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_fetch_add_nbi = AMO_FETCH_ADD_NBI;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_fetch_and_nbi = AMO_FETCH_AND_NBI;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_fetch_or_nbi = AMO_FETCH_OR_NBI;
constexpr ishmemi_op_t ISHMEMI_OP_atomic_fetch_xor_nbi = AMO_FETCH_XOR_NBI;
constexpr ishmemi_op_t ISHMEMI_OP_put_signal = PUT_SIGNAL;
constexpr ishmemi_op_t ISHMEMI_OP_put_signal_nbi = PUT_SIGNAL_NBI;
constexpr ishmemi_op_t ISHMEMI_OP_signal_fetch = SIGNAL_FETCH;
constexpr ishmemi_op_t ISHMEMI_OP_signal_add = SIGNAL_ADD;
constexpr ishmemi_op_t ISHMEMI_OP_signal_set = SIGNAL_SET;
constexpr ishmemi_op_t ISHMEMI_OP_barrier_all = BARRIER;
constexpr ishmemi_op_t ISHMEMI_OP_sync_all = SYNC;
constexpr ishmemi_op_t ISHMEMI_OP_alltoall = ALLTOALL;
constexpr ishmemi_op_t ISHMEMI_OP_broadcast = BCAST;
constexpr ishmemi_op_t ISHMEMI_OP_collect = COLLECT;
constexpr ishmemi_op_t ISHMEMI_OP_fcollect = FCOLLECT;
constexpr ishmemi_op_t ISHMEMI_OP_and_reduce = AND_REDUCE;
constexpr ishmemi_op_t ISHMEMI_OP_or_reduce = OR_REDUCE;
constexpr ishmemi_op_t ISHMEMI_OP_xor_reduce = XOR_REDUCE;
constexpr ishmemi_op_t ISHMEMI_OP_max_reduce = MAX_REDUCE;
constexpr ishmemi_op_t ISHMEMI_OP_min_reduce = MIN_REDUCE;
constexpr ishmemi_op_t ISHMEMI_OP_sum_reduce = SUM_REDUCE;
constexpr ishmemi_op_t ISHMEMI_OP_prod_reduce = PROD_REDUCE;
constexpr ishmemi_op_t ISHMEMI_OP_wait_until = WAIT;
constexpr ishmemi_op_t ISHMEMI_OP_wait_until_all = WAIT_ALL;
constexpr ishmemi_op_t ISHMEMI_OP_wait_until_any = WAIT_ANY;
constexpr ishmemi_op_t ISHMEMI_OP_wait_until_some = WAIT_SOME;
constexpr ishmemi_op_t ISHMEMI_OP_test = TEST;
constexpr ishmemi_op_t ISHMEMI_OP_test_all = TEST_ALL;
constexpr ishmemi_op_t ISHMEMI_OP_test_any = TEST_ANY;
constexpr ishmemi_op_t ISHMEMI_OP_test_some = TEST_SOME;
constexpr ishmemi_op_t ISHMEMI_OP_fence = FENCE;
constexpr ishmemi_op_t ISHMEMI_OP_quiet = QUIET;
constexpr ishmemi_op_t ISHMEMI_OP_kill = KILL;
constexpr ishmemi_op_t ISHMEMI_OP_nop = NOP;
constexpr ishmemi_op_t ISHMEMI_OP_nop_no_r = NOP_NO_R;
constexpr ishmemi_op_t ISHMEMI_OP_timestamp = TIMESTAMP;
constexpr ishmemi_op_t ISHMEMI_OP_print = PRINT;
constexpr ishmemi_op_t ISHMEMI_OP_debug_test = DEBUG_TEST;

typedef enum : uint16_t {
    MEM = 0,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    ULONGLONG,
    INT8,
    INT16,
    INT32,
    INT64,
    LONGLONG,
    FLOAT,
    DOUBLE,
    LONGDOUBLE,
    /* DO NOT MODIFY THE ORDERING ABOVE - It's necessary for runtime function pointer definitions */
    CHAR,
    SCHAR,
    SHORT,
    INT,
    LONG,
    UCHAR,
    USHORT,
    UINT,
    ULONG,
    SIZE,
    PTRDIFF,
    SIZE8,
    SIZE16,
    SIZE32,
    SIZE64,
    SIZE128,
    ISHMEMI_TYPE_END
} ishmemi_type_t;

constexpr ishmemi_type_t ISHMEMI_TYPE_float = FLOAT;
constexpr ishmemi_type_t ISHMEMI_TYPE_double = DOUBLE;
constexpr ishmemi_type_t ISHMEMI_TYPE_longdouble = LONGDOUBLE;
constexpr ishmemi_type_t ISHMEMI_TYPE_char = CHAR;
constexpr ishmemi_type_t ISHMEMI_TYPE_schar = SCHAR;
constexpr ishmemi_type_t ISHMEMI_TYPE_short = SHORT;
constexpr ishmemi_type_t ISHMEMI_TYPE_int = INT;
constexpr ishmemi_type_t ISHMEMI_TYPE_long = LONG;
constexpr ishmemi_type_t ISHMEMI_TYPE_longlong = LONGLONG;
constexpr ishmemi_type_t ISHMEMI_TYPE_uchar = UCHAR;
constexpr ishmemi_type_t ISHMEMI_TYPE_ushort = USHORT;
constexpr ishmemi_type_t ISHMEMI_TYPE_uint = UINT;
constexpr ishmemi_type_t ISHMEMI_TYPE_ulong = ULONG;
constexpr ishmemi_type_t ISHMEMI_TYPE_ulonglong = ULONGLONG;
constexpr ishmemi_type_t ISHMEMI_TYPE_int8 = INT8;
constexpr ishmemi_type_t ISHMEMI_TYPE_int16 = INT16;
constexpr ishmemi_type_t ISHMEMI_TYPE_int32 = INT32;
constexpr ishmemi_type_t ISHMEMI_TYPE_int64 = INT64;
constexpr ishmemi_type_t ISHMEMI_TYPE_uint8 = UINT8;
constexpr ishmemi_type_t ISHMEMI_TYPE_uint16 = UINT16;
constexpr ishmemi_type_t ISHMEMI_TYPE_uint32 = UINT32;
constexpr ishmemi_type_t ISHMEMI_TYPE_uint64 = UINT64;
constexpr ishmemi_type_t ISHMEMI_TYPE_size = SIZE;
constexpr ishmemi_type_t ISHMEMI_TYPE_ptrdiff = PTRDIFF;
constexpr ishmemi_type_t ISHMEMI_TYPE_void = MEM;

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

extern const char *ishmemi_op_str[ISHMEMI_OP_END + 1];
extern const char *ishmemi_type_str[ISHMEMI_TYPE_END + 1];

/* support for printf style upcalls and parameter validation */
#define MAX_PROXY_MSG_SIZE 128
#define NUM_MESSAGES       32

struct ishmemi_message_t {
    char message[MAX_PROXY_MSG_SIZE];
    char file[MAX_PROXY_MSG_SIZE]; /*  __FILE__ */
    char func[MAX_PROXY_MSG_SIZE]; /*  __func__ */
    long int line;                 /* __LINE__ */
};

union ishmemi_union_type {
    float f;
    double ld;
    // long double lld;
    char c;
    signed char sc;
    short sh;
    int i;
    long l;
    long long ll;
    unsigned char uc;
    unsigned short us;
    unsigned int ui;
    unsigned long ul;
    unsigned long long ull;
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    uint8_t ui8;
    uint16_t ui16;
    uint32_t ui32;
    uint64_t ui64;
    size_t szt;
    ptrdiff_t pd;
};

typedef sycl::vec<ulong, 8> ulong8;

#endif /* ISHMEM_TYPES_H */
