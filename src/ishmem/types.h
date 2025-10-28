/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* ishmem common types and definitions */
#ifndef ISHMEM_TYPES_H
#define ISHMEM_TYPES_H

#if __INTEL_CLANG_COMPILER >= 20250000
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

/* Note: this should mirror ISHMEM_DEVICE_ATTRIBUTES definition in ishmem.h */
#define ISHMEM_DEVICE_ATTRIBUTES SYCL_EXTERNAL

// Needed to make GEN_SIZE_FNS_FOR_SIZE macro compatible for SIZE128 operations
typedef __uint128_t uint128_t;

typedef enum : uint16_t {
    UNDEFINED = 0,
    PUT,
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
    INSCAN,
    EXSCAN,
    WAIT,
    WAIT_ALL,
    WAIT_ALL_VECTOR,
    WAIT_ANY,
    WAIT_ANY_VECTOR,
    WAIT_SOME,
    WAIT_SOME_VECTOR,
    TEST,
    TEST_ALL,
    TEST_ALL_VECTOR,
    TEST_ANY,
    TEST_ANY_VECTOR,
    TEST_SOME,
    TEST_SOME_VECTOR,
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

constexpr ishmemi_op_t ISHMEMI_OP_undefined = UNDEFINED;
constexpr ishmemi_op_t ISHMEMI_OP_put = PUT;
constexpr ishmemi_op_t ISHMEMI_OP_iput = IPUT;
constexpr ishmemi_op_t ISHMEMI_OP_ibput = IBPUT;
constexpr ishmemi_op_t ISHMEMI_OP_p = P;
constexpr ishmemi_op_t ISHMEMI_OP_get = GET;
constexpr ishmemi_op_t ISHMEMI_OP_iget = IGET;
constexpr ishmemi_op_t ISHMEMI_OP_ibget = IBGET;
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
constexpr ishmemi_op_t ISHMEMI_OP_in_scan = INSCAN;
constexpr ishmemi_op_t ISHMEMI_OP_ex_scan = EXSCAN;
constexpr ishmemi_op_t ISHMEMI_OP_wait_until = WAIT;
constexpr ishmemi_op_t ISHMEMI_OP_wait_until_all = WAIT_ALL;
constexpr ishmemi_op_t ISHMEMI_OP_wait_until_all_vector = WAIT_ALL_VECTOR;
constexpr ishmemi_op_t ISHMEMI_OP_wait_until_any = WAIT_ANY;
constexpr ishmemi_op_t ISHMEMI_OP_wait_until_any_vector = WAIT_ANY_VECTOR;
constexpr ishmemi_op_t ISHMEMI_OP_wait_until_some = WAIT_SOME;
constexpr ishmemi_op_t ISHMEMI_OP_wait_until_some_vector = WAIT_SOME_VECTOR;
constexpr ishmemi_op_t ISHMEMI_OP_test = TEST;
constexpr ishmemi_op_t ISHMEMI_OP_test_all = TEST_ALL;
constexpr ishmemi_op_t ISHMEMI_OP_test_all_vector = TEST_ALL_VECTOR;
constexpr ishmemi_op_t ISHMEMI_OP_test_any = TEST_ANY;
constexpr ishmemi_op_t ISHMEMI_OP_test_any_vector = TEST_ANY_VECTOR;
constexpr ishmemi_op_t ISHMEMI_OP_test_some = TEST_SOME;
constexpr ishmemi_op_t ISHMEMI_OP_test_some_vector = TEST_SOME_VECTOR;
constexpr ishmemi_op_t ISHMEMI_OP_signal_wait_until = SIGNAL_WAIT_UNTIL;
constexpr ishmemi_op_t ISHMEMI_OP_fence = FENCE;
constexpr ishmemi_op_t ISHMEMI_OP_quiet = QUIET;
constexpr ishmemi_op_t ISHMEMI_OP_team_my_pe = TEAM_MY_PE;
constexpr ishmemi_op_t ISHMEMI_OP_team_n_pes = TEAM_N_PES;
constexpr ishmemi_op_t ISHMEMI_OP_team_sync = TEAM_SYNC;
constexpr ishmemi_op_t ISHMEMI_OP_kill = KILL;
constexpr ishmemi_op_t ISHMEMI_OP_nop = NOP;
constexpr ishmemi_op_t ISHMEMI_OP_nop_no_r = NOP_NO_R;
constexpr ishmemi_op_t ISHMEMI_OP_timestamp = TIMESTAMP;
constexpr ishmemi_op_t ISHMEMI_OP_print = PRINT;
constexpr ishmemi_op_t ISHMEMI_OP_debug_test = DEBUG_TEST;

typedef enum : uint16_t {
    NONE = 0,
    MEM,
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

constexpr ishmemi_type_t ISHMEMI_TYPE_none = NONE;
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

/* Macros for ensuring linker finds template specializations */
#define ISHMEM_INSTANTIATE_TYPE_float(TYPE)  ISHMEM_INSTANTIATE_TYPE(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_double(TYPE) ISHMEM_INSTANTIATE_TYPE(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_char(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_uchar(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_schar(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_short(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_ushort(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_int(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_uint(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_long(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_ulong(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_longlong(TYPE)  ISHMEM_INSTANTIATE_TYPE(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_ulonglong(TYPE) ISHMEM_INSTANTIATE_TYPE(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_int8(TYPE)      ISHMEM_INSTANTIATE_TYPE(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_int16(TYPE)     ISHMEM_INSTANTIATE_TYPE(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_int32(TYPE)     ISHMEM_INSTANTIATE_TYPE(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_int64(TYPE)     ISHMEM_INSTANTIATE_TYPE(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_uint8(TYPE)     ISHMEM_INSTANTIATE_TYPE(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_uint16(TYPE)    ISHMEM_INSTANTIATE_TYPE(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_uint32(TYPE)    ISHMEM_INSTANTIATE_TYPE(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_uint64(TYPE)    ISHMEM_INSTANTIATE_TYPE(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_size(TYPE)
#define ISHMEM_INSTANTIATE_TYPE_ptrdiff(TYPE)

#endif /* ISHMEM_TYPES_H */
