/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* internal API and globals */
#ifndef ISHMEM_INTERNAL_H
#define ISHMEM_INTERNAL_H

#include "env_utils.h"

#include <iostream>
#include <cstdlib>
#include <CL/sycl.hpp>

/* Note: this should mirror ISHMEM_DEVICE_ATTRIBUTES definition in ishmem.h */
#define ISHMEM_DEVICE_ATTRIBUTES SYCL_EXTERNAL

#define ISHMEMI_TYPE_float      FLOAT
#define ISHMEMI_TYPE_double     DOUBLE
#define ISHMEMI_TYPE_longdouble LONGDOUBLE
#define ISHMEMI_TYPE_char       CHAR
#define ISHMEMI_TYPE_schar      SCHAR
#define ISHMEMI_TYPE_short      SHORT
#define ISHMEMI_TYPE_int        INT
#define ISHMEMI_TYPE_long       LONG
#define ISHMEMI_TYPE_longlong   LONGLONG
#define ISHMEMI_TYPE_uchar      UCHAR
#define ISHMEMI_TYPE_ushort     USHORT
#define ISHMEMI_TYPE_uint       UINT
#define ISHMEMI_TYPE_ulong      ULONG
#define ISHMEMI_TYPE_ulonglong  ULONGLONG
#define ISHMEMI_TYPE_int8       INT8
#define ISHMEMI_TYPE_int16      INT16
#define ISHMEMI_TYPE_int32      INT32
#define ISHMEMI_TYPE_int64      INT64
#define ISHMEMI_TYPE_uint8      UINT8
#define ISHMEMI_TYPE_uint16     UINT16
#define ISHMEMI_TYPE_uint32     UINT32
#define ISHMEMI_TYPE_uint64     UINT64
#define ISHMEMI_TYPE_size       SIZE
#define ISHMEMI_TYPE_ptrdiff    PTRDIFF
#define ISHMEMI_TYPE_void       MEM

#define ISHMEMI_OP_nop                       NOP
#define ISHMEMI_OP_nop_no_r                  NOP_NO_R
#define ISHMEMI_OP_debug_test                DEBUG_TEST
#define ISHMEMI_OP_put                       PUT
#define ISHMEMI_OP_put_work_group            PUT_WORK_GROUP
#define ISHMEMI_OP_iput                      IPUT
#define ISHMEMI_OP_iput_work_group           IPUT_WORK_GROUP
#define ISHMEMI_OP_p                         P
#define ISHMEMI_OP_get                       GET
#define ISHMEMI_OP_get_work_group            GET_WORK_GROUP
#define ISHMEMI_OP_iget                      IGET
#define ISHMEMI_OP_iget_work_group           IGET_WORK_GROUP
#define ISHMEMI_OP_g                         G
#define ISHMEMI_OP_put_nbi                   PUT_NBI
#define ISHMEMI_OP_put_nbi_work_group        PUT_NBI_WORK_GROUP
#define ISHMEMI_OP_get_nbi                   GET_NBI
#define ISHMEMI_OP_get_nbi_work_group        GET_NBI_WORK_GROUP
#define ISHMEMI_OP_atomic_fetch              AMO_FETCH
#define ISHMEMI_OP_atomic_set                AMO_SET
#define ISHMEMI_OP_atomic_compare_swap       AMO_COMPARE_SWAP
#define ISHMEMI_OP_atomic_swap               AMO_SWAP
#define ISHMEMI_OP_atomic_fetch_inc          AMO_FETCH_INC
#define ISHMEMI_OP_atomic_inc                AMO_INC
#define ISHMEMI_OP_atomic_fetch_add          AMO_FETCH_ADD
#define ISHMEMI_OP_atomic_add                AMO_ADD
#define ISHMEMI_OP_atomic_fetch_and          AMO_FETCH_AND
#define ISHMEMI_OP_atomic_and                AMO_AND
#define ISHMEMI_OP_atomic_fetch_or           AMO_FETCH_OR
#define ISHMEMI_OP_atomic_or                 AMO_OR
#define ISHMEMI_OP_atomic_fetch_xor          AMO_FETCH_XOR
#define ISHMEMI_OP_atomic_xor                AMO_XOR
#define ISHMEMI_OP_put_signal                PUT_SIGNAL
#define ISHMEMI_OP_put_signal_work_group     PUT_SIGNAL_WORK_GROUP
#define ISHMEMI_OP_put_signal_nbi            PUT_SIGNAL_NBI
#define ISHMEMI_OP_put_signal_nbi_work_group PUT_SIGNAL_NBI_WORK_GROUP
#define ISHMEMI_OP_signal_fetch              SIGNAL_FETCH
#define ISHMEMI_OP_test                      TEST
#define ISHMEMI_OP_wait_until                WAIT
#define ISHMEMI_OP_fence                     FENCE
#define ISHMEMI_OP_quiet                     QUIET
#define ISHMEMI_OP_barrier_all               BARRIER
#define ISHMEMI_OP_barrier_all_work_group    BARRIER_WORK_GROUP
#define ISHMEMI_OP_sync_all                  SYNC
#define ISHMEMI_OP_sync_all_work_group       SYNC_WORK_GROUP
#define ISHMEMI_OP_alltoall                  ALLTOALL
#define ISHMEMI_OP_alltoall_work_group       ALLTOALL_WORK_GROUP
#define ISHMEMI_OP_broadcast                 BCAST
#define ISHMEMI_OP_broadcast_work_group      BCAST_WORK_GROUP
#define ISHMEMI_OP_collect                   COLLECT
#define ISHMEMI_OP_collect_work_group        COLLECT_WORK_GROUP
#define ISHMEMI_OP_fcollect                  FCOLLECT
#define ISHMEMI_OP_fcollect_work_group       FCOLLECT_WORK_GROUP
#define ISHMEMI_OP_and_reduce                AND_REDUCE
#define ISHMEMI_OP_and_reduce_work_group     AND_REDUCE_WORK_GROUP
#define ISHMEMI_OP_or_reduce                 OR_REDUCE
#define ISHMEMI_OP_or_reduce_work_group      OR_REDUCE_WORK_GROUP
#define ISHMEMI_OP_xor_reduce                XOR_REDUCE
#define ISHMEMI_OP_xor_reduce_work_group     XOR_REDUCE_WORK_GROUP
#define ISHMEMI_OP_max_reduce                MAX_REDUCE
#define ISHMEMI_OP_max_reduce_work_group     MAX_REDUCE_WORK_GROUP
#define ISHMEMI_OP_min_reduce                MIN_REDUCE
#define ISHMEMI_OP_min_reduce_work_group     MIN_REDUCE_WORK_GROUP
#define ISHMEMI_OP_sum_reduce                SUM_REDUCE
#define ISHMEMI_OP_sum_reduce_work_group     SUM_REDUCE_WORK_GROUP
#define ISHMEMI_OP_prod_reduce               PROD_REDUCE
#define ISHMEMI_OP_prod_reduce_work_group    PROD_REDUCE_WORK_GROUP
#define ISHMEMI_OP_kill                      KILL
#define ISHMEMI_OP_timestamp                 TIMESTAMP
#define ISHMEMI_OP_print                     PRINT

#ifdef __SYCL_DEVICE_ONLY__
#define ISHMEMI_LOCAL_PES global_info->local_pes
#else
#define ISHMEMI_LOCAL_PES ishmemi_local_pes
#endif

#define MAX_LOCAL_PES 64

extern int ishmemi_my_pe;
extern int ishmemi_n_pes;

typedef struct ishmem_info_t ishmem_info_t;

/* TODO should these be combined into ishmem_host_data_t? */
/* Device parameters for the device copy of the data */
extern void *ishmemi_heap_base;
extern size_t ishmemi_heap_length;
extern uintptr_t ishmemi_heap_last;
extern ishmem_info_t *ishmemi_gpu_info;
/* this is the device global */
ISHMEM_DEVICE_ATTRIBUTES extern sycl::ext::oneapi::experimental::device_global<ishmem_info_t *>
    global_info;

/* allocated size for info data structure (variable due to n_pes) */
extern size_t ishmemi_info_size;

/* Host parameters for the device data structures */
extern ishmem_info_t *ishmemi_mmap_gpu_info;
extern void *ishmemi_mmap_heap_base;

/* Host globals to hold the host version of data */
extern uint8_t *ishmemi_local_pes;
extern void *ishmemi_ipc_buffers[MAX_LOCAL_PES + 1];

/* host global for host address of host memory copy of ipc_buffer_delta */
extern ptrdiff_t ishmemi_ipc_buffer_delta[MAX_LOCAL_PES + 1];
extern bool ishmemi_only_intra_node;

typedef enum : uint16_t {
    PUT = 0,
    PUT_WORK_GROUP,
    IPUT,
    IPUT_WORK_GROUP,
    P,
    GET,
    GET_WORK_GROUP,
    IGET,
    IGET_WORK_GROUP,
    G,
    PUT_NBI,
    PUT_NBI_WORK_GROUP,
    GET_NBI,
    GET_NBI_WORK_GROUP,
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
    PUT_SIGNAL,
    PUT_SIGNAL_WORK_GROUP,
    PUT_SIGNAL_NBI,
    PUT_SIGNAL_NBI_WORK_GROUP,
    SIGNAL_FETCH,
    BARRIER,
    BARRIER_WORK_GROUP,
    SYNC,
    SYNC_WORK_GROUP,
    ALLTOALL,
    ALLTOALL_WORK_GROUP,
    BCAST,
    BCAST_WORK_GROUP,
    COLLECT,
    COLLECT_WORK_GROUP,
    FCOLLECT,
    FCOLLECT_WORK_GROUP,
    AND_REDUCE,
    AND_REDUCE_WORK_GROUP,
    OR_REDUCE,
    OR_REDUCE_WORK_GROUP,
    XOR_REDUCE,
    XOR_REDUCE_WORK_GROUP,
    MAX_REDUCE,
    MAX_REDUCE_WORK_GROUP,
    MIN_REDUCE,
    MIN_REDUCE_WORK_GROUP,
    SUM_REDUCE,
    SUM_REDUCE_WORK_GROUP,
    PROD_REDUCE,
    PROD_REDUCE_WORK_GROUP,
    TEST,
    WAIT,
    FENCE,
    QUIET,
    KILL,
    NOP,
    NOP_NO_R,
    TIMESTAMP,
    PRINT,
    DEBUG_TEST,
    ISHMEMI_OP_END
} ishmemi_op_t;

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
    ISHMEMI_TYPE_END
} ishmemi_type_t;

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

extern const char *ishmemi_op_str[];
extern const char *ishmemi_type_str[];

/* support for printf style upcalls and parameter validation */
#define MAX_PROXY_MSG_SIZE 128
#define NUM_MESSAGES       32
struct ishmemi_message_t {
    char message[MAX_PROXY_MSG_SIZE];
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

/* This block of code enables the Intel® Graphics Compiler
 * intrinsic instruction for an uncached store
 *
 * if USE_BUILTIN==1, then the intrinsic is used
 * if USE_BUILTIN==0, then the intrinsic is not used
 */

/* This enables use of the graphics compiler intrinsic for Send */
/* TODO figure out whether this is needed */
#define USE_BUILTIN 0

enum LSC_STCC {
    LSC_STCC_DEFAULT = 0,
    LSC_STCC_L1UC_L3UC = 1,  // Override to L1 uncached and L3 uncached
    LSC_STCC_L1UC_L3WB = 2,  // Override to L1 uncached and L3 written back
    LSC_STCC_L1WT_L3UC = 3,  // Override to L1 written through and L3 uncached
    LSC_STCC_L1WT_L3WB = 4,  // Override to L1 written through and L3 written back
    LSC_STCC_L1S_L3UC = 5,   // Override to L1 streaming and L3 uncached
    LSC_STCC_L1S_L3WB = 6,   // Override to L1 streaming and L3 written back
    LSC_STCC_L1WB_L3WB = 7,  // Override to L1 written through and L3 written back
};

#ifdef __SYCL_DEVICE_ONLY__
#define __global

SYCL_EXTERNAL extern "C" void __builtin_IB_lsc_store_global_ulong8(
    __global ulong8 *base, int immElemOff, ulong8 val, enum LSC_STCC cacheOpt);  // D64V8

#endif  // end __SYCL_DEVICE_ONLY__

static inline void ucs_ulong8(ulong8 *base, ulong8 val)
{
#ifdef __SYCL_DEVICE_ONLY__
#if USE_BUILTIN == 1
    __builtin_IB_lsc_store_global_ulong8(base, 0, val, LSC_STCC_L1UC_L3UC);
#else   // end USE_BUILTIN start !USE_BUILTIN
    base[0][1] = val[1];
    base[0][2] = val[2];
    base[0][3] = val[3];
    base[0][4] = val[4];
    base[0][5] = val[5];
    base[0][6] = val[6];
    base[0][7] = val[7];
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    base[0][0] = val[0];
#endif  // end !USE_BUILTIN
#else   // end __SYCL_DEVICE_ONLY__ start !__SYCL_DEVICE_ONLY__
    *base = val;
#endif  // end !__SYCL_DEVICE_ONLY__
}
/* end of compiler intrinsic block */

/* TODO move this to configuration */
/* The idea is LG(MAX_PES) + 2   2^30 PEs seems like enough for now */
/* This is copied from SOS. It is copied from shmem, the logic is that it allows
 * for 2^30 PEs with binary trees or recursive doubling. In fact I only use one cell for intra-node
 * but the same type is used for host side. */

#define ISHMEM_REDUCE_SYNC_SIZE    32
#define ISHMEM_SYNC_NUM_PSYNC_ARRS 4

/* Device side data structure used for collectives */
struct ishmemi_team_t {
    int my_pe;  // my pe number within the team
    int PE_Start;
    int PE_Stride;
    int PE_Size;
    int n_local_pes;  // number of local PEs in the team
    int pow2_proc;    // largest power of 2 <= PE_Size
    int log2_proc;    // lg of pow2_proc
    long psync[ISHMEM_REDUCE_SYNC_SIZE];
    void *buffer;  // reduction buffer for in-place intra-node
};

/* Host side data structurre used for collectives */
struct ishmemi_host_team_t {
    void *source;  // bounce buffer for internode
    void *dest;    // bounce buffer for internode
};

/* Begin: Error/warning and debug related macros: */
#define ISHMEMI_ERROR_MPI  1
#define ISHMEMI_ERROR_ZE   2
#define ISHMEMI_ERROR_SOCK 3

#define RAISE_PE_PREFIX     "[%04d]        "
#define ISHMEMI_DIAG_STRLEN 1024

#define ZE_ERR_NAME_EXPANSION(name)                                                                \
    case name:                                                                                     \
        err_name = #name;                                                                          \
        break;

/* Level Zero API doesn't provide an err-to-str function, so make our own */
#define ZE_ERR_GET_NAME(err, err_name)                                                             \
    switch (err) {                                                                                 \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_DEVICE_LOST);                                        \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_UNINITIALIZED);                                      \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_UNSUPPORTED_VERSION);                                \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);                                \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_UNSUPPORTED_SIZE);                                   \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_INVALID_ARGUMENT);                                   \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_INVALID_ENUMERATION);                                \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_INVALID_NULL_HANDLE);                                \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_INVALID_NULL_POINTER);                               \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_INVALID_SIZE);                                       \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT);                     \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY);                                 \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_OVERLAPPING_REGIONS);                                \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE);                               \
        default:                                                                                   \
            err_name = "Unknown";                                                                  \
            break;                                                                                 \
    }

#define ISHMEM_COMMON_MSG(typestring, ...)                                                         \
    do {                                                                                           \
        char str[ISHMEMI_DIAG_STRLEN];                                                             \
        int off;                                                                                   \
        off = snprintf(str, sizeof(str), "[%04d] %s:  %s:%d: %s\n", ishmemi_my_pe, typestring,     \
                       __FILE__, __LINE__, __func__);                                              \
        off += snprintf(str + off, sizeof(str) - static_cast<size_t>(off), RAISE_PE_PREFIX,        \
                        ishmemi_my_pe);                                                            \
        off += snprintf(str + off, sizeof(str) - static_cast<size_t>(off), __VA_ARGS__);           \
        fprintf(stderr, "%s", str);                                                                \
    } while (0)

#define ISHMEM_WARN_MSG(...) ISHMEM_COMMON_MSG("WARN", __VA_ARGS__)

#define ISHMEM_DEBUG_MSG(...)                                                                      \
    do {                                                                                           \
        if (ishmemi_params.DEBUG) {                                                                \
            ISHMEM_COMMON_MSG("DEBUG", __VA_ARGS__);                                               \
        }                                                                                          \
    } while (0)

#ifdef __SYCL_DEVICE_ONLY__
#define ISHMEM_ERROR_MSG(...)
#else
#define ISHMEM_ERROR_MSG(...) ISHMEM_COMMON_MSG("ERROR", __VA_ARGS__)
#endif

// name changed for consistency.  MSG means format supported
#define ISHMEM_CHECK_GOTO_MSG(ret, lbl, ...)                                                       \
    do {                                                                                           \
        if (ret) {                                                                                 \
            ISHMEM_WARN_MSG(__VA_ARGS__);                                                          \
            goto lbl;                                                                              \
        }                                                                                          \
    } while (0)

#define ISHMEM_CHECK_RETURN_MSG(ret, ...)                                                          \
    do {                                                                                           \
        if (ret) {                                                                                 \
            ISHMEM_WARN_MSG(__VA_ARGS__);                                                          \
            return ret;                                                                            \
        }                                                                                          \
    } while (0)

#define RAISE_ERROR_MSG(...)                                                                       \
    do {                                                                                           \
        ISHMEM_ERROR_MSG(__VA_ARGS__);                                                             \
        exit(1);                                                                                   \
    } while (0)

/* TODO recommend changing this assign to ret with something returning a value */

#define ZE_CHECK(call)                                                                             \
    do {                                                                                           \
        ze_result_t status = call;                                                                 \
        std::string err_name;                                                                      \
        if (status != ZE_RESULT_SUCCESS) {                                                         \
            ZE_ERR_GET_NAME(status, err_name);                                                     \
            ISHMEM_ERROR_MSG("ZE FAIL: call = '%s' result = '0x%x' (%s)\n", #call, status,         \
                             err_name.c_str());                                                    \
            ret = ISHMEMI_ERROR_ZE;                                                                \
        }                                                                                          \
    } while (0)

#define ISHMEMI_CHECK_RESULT(status, pass, label)                                                  \
    do {                                                                                           \
        if (status != pass) {                                                                      \
            ret = status;                                                                          \
            goto label;                                                                            \
        }                                                                                          \
    } while (0)

/* End: Error/warning and debug related macros: */

#ifdef __SYCL_DEVICE_ONLY__
#define ISHMEMI_ADJUST_PTR(TYPENAME, index, p)                                                     \
    ((TYPENAME *) (reinterpret_cast<ptrdiff_t>(p) +                                                \
                   static_cast<ptrdiff_t>(global_info->ipc_buffer_delta[(index)])))
#else
#define ISHMEMI_ADJUST_PTR(TYPENAME, index, p)                                                     \
    ((TYPENAME *) (reinterpret_cast<ptrdiff_t>(p) +                                                \
                   static_cast<ptrdiff_t>(ishmemi_ipc_buffer_delta[(index)])))
#endif

#define ISHMEMI_HOST_ADJUST_PTR(TYPENAME, index, p)                                                \
    ((TYPENAME *) (reinterpret_cast<ptrdiff_t>(p) +                                                \
                   static_cast<ptrdiff_t>(ishmemi_ipc_buffer_delta[(index)])))

#define ISHMEMI_HOST_IN_HEAP(p)                                                                    \
    ((((uintptr_t) p) >= ((uintptr_t) ishmemi_heap_base)) &&                                       \
     (((uintptr_t) p) < (((uintptr_t) ishmemi_heap_base) + ishmemi_heap_length)))

/* common code for pointer arithmetic */

template <typename T>
inline T *pointer_offset(T *p, ptrdiff_t offset)
{
    return ((T *) (((intptr_t) p) + offset));
}
template <typename T>
inline T *pointer_offset(T *p, size_t offset)
{
    return ((T *) (((uintptr_t) p) + offset));
}

inline bool pointer_less_or_equal(void *a, void *b)
{
    return (((uintptr_t) a) <= ((uintptr_t) b));
}

inline bool pointer_greater_or_equal(void *a, void *b)
{
    return (((uintptr_t) a) >= ((uintptr_t) b));
}

/* Used to reduce reliance on macros in function definitions */
#ifdef __SYCL_DEVICE_ONLY__
constexpr bool ishmemi_is_device = true;
#else
constexpr bool ishmemi_is_device = false;
#endif

/* tuning parameters
 * run bandwidth tests in CUTOVER_NEVER and again in CUTOVER_ALWAYS
 * and use the results to choose the CUTOVER_PRODUCTION values
 */
#define CUTOVER_PRODUCTION 1
#define CUTOVER_ALWAYS     0
#define CUTOVER_NEVER      0

#if CUTOVER_NEVER

#define ISHMEM_RMA_CUTOVER             (false)
#define ISHMEM_RMA_GROUP_CUTOVER       (false)
#define ISHMEM_ALLTOALL_CUTOVER        (false)
#define ISHMEM_ALLTOALL_GROUP_CUTOVER  (false)
#define ISHMEM_BROADCAST_CUTOVER       (false)
#define ISHMEM_BROADCAST_GROUP_CUTOVER (false)
#define ISHMEM_COLLECT_CUTOVER         (false)
#define ISHMEM_COLLECT_GROUP_CUTOVER   (false)
#define ISHMEM_FCOLLECT_CUTOVER        (false)
#define ISHMEM_FCOLLECT_GROUP_CUTOVER  (false)

#elif CUTOVER_ALWAYS

#define ISHMEM_RMA_CUTOVER             (true)
#define ISHMEM_RMA_GROUP_CUTOVER       (true)
#define ISHMEM_ALLTOALL_CUTOVER        (true)
#define ISHMEM_ALLTOALL_GROUP_CUTOVER  (true)
#define ISHMEM_BROADCAST_CUTOVER       (true)
#define ISHMEM_BROADCAST_GROUP_CUTOVER (true)
#define ISHMEM_COLLECT_CUTOVER         (true)
#define ISHMEM_COLLECT_GROUP_CUTOVER   (true)
#define ISHMEM_FCOLLECT_CUTOVER        (true)
#define ISHMEM_FCOLLECT_GROUP_CUTOVER  (true)

#else /* CUTOVER_PRODUCTION */

#define ISHMEM_RMA_CUTOVER             (nbytes >= 16384L)
#define ISHMEM_RMA_GROUP_CUTOVER       (nbytes >= 32768L)
#define ISHMEM_ALLTOALL_CUTOVER        (nbytes >= 128L)
#define ISHMEM_ALLTOALL_GROUP_CUTOVER  (nbytes >= 16384L)
#define ISHMEM_BROADCAST_CUTOVER       ((nbytes * ((size_t) info->n_pes)) >= 8192L)
// preferred BROADCAST_GROUP_CUTOVER is nbytes * threads > 512
#define ISHMEM_BROADCAST_GROUP_CUTOVER (nbytes >= 65536L)
#define ISHMEM_FCOLLECT_CUTOVER        (nbytes >= 1024L)
#define ISHMEM_FCOLLECT_GROUP_CUTOVER  (nbytes >= 32768L)
#define ISHMEM_COLLECT_CUTOVER         (total_nbytes >= (1024L * ((size_t) info->n_pes)))
#define ISHMEM_COLLECT_GROUP_CUTOVER   (total_nbytes >= (32768L * ((size_t) info->n_pes)))

#endif /* CUTOVER */
// use the routines below rather than a type specific copy loop
#define USE_VEC_COPY 1
#define VL           16L
#define ALIGNSIZE    (sizeof(T) * VL)
#define ALIGNMASK    (ALIGNSIZE - 1)

template <typename T>
inline void vec_copy_push(T *d, const T *s, size_t count)
{
    if constexpr (ishmemi_is_device) {
#if USE_VEC_COPY
        while ((((uintptr_t) d) & ALIGNMASK) && (count > 0)) {
            *d++ = *s++;
            count -= 1;
        }
        sycl::multi_ptr<T, sycl::access::address_space::global_space, sycl::access::decorated::yes>
            dd;
        sycl::multi_ptr<const T, sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>
            ds;
        dd = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::yes>(d);
        ds = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::yes>(s);
        while (count >= VL) {
            sycl::vec<T, 16> temp;
            temp.load(0, ds);
            temp.store(0, dd);
            ds += VL;
            dd += VL;
            count -= VL;
        }
        while (count > 0) {
            *dd++ = *ds++;
            count -= 1;
        }
#else
        for (size_t i = 0; i < count; i += 1)
            dest[i] = src[i];
#endif
    } else {
        memcpy(d, s, count * sizeof(T));
    }
}

template <typename T>
inline void vec_copy_pull(T *d, const T *s, size_t count)
{
    if constexpr (ishmemi_is_device) {
#if USE_VEC_COPY
        while ((((uintptr_t) s) & ALIGNMASK) && (count > 0)) {
            *d++ = *s++;
            count -= 1;
        }
        sycl::multi_ptr<T, sycl::access::address_space::global_space, sycl::access::decorated::yes>
            dd;
        sycl::multi_ptr<const T, sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>
            ds;
        dd = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::yes>(d);
        ds = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::yes>(s);
        while (count >= VL) {
            sycl::vec<T, VL> temp;
            temp.load(0, ds);
            temp.store(0, dd);
            ds += VL;
            dd += VL;
            count -= VL;
        }
        while (count > 0) {
            *dd++ = *ds++;
            count -= 1;
        }
#else
        for (size_t i = 0; i < count; i += 1)
            dest[i] = src[i];
#endif
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
/* Parameter validation function */
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters(int pe);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters(int pe, void *ptr, size_t src);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters(int pe, void *ptr1, void *ptr2, size_t size);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters(int pe, void *ptr1, void *ptr2, size_t size,
                                                  ptrdiff_t dst, ptrdiff_t sst);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters(int pe_root, void *dest, void *src,
                                                  size_t dest_size, size_t src_size);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters(int pe, void *ptr1, void *ptr2, void *sig_addr,
                                                  size_t size, size_t sig_addr_size);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters(void *ivar, size_t size);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters(void *dest, void *src, size_t size);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters(void *dest, void *src, size_t dest_size,
                                                  size_t src_size);

#if defined(ENABLE_ERROR_CHECKING)
constexpr bool enable_error_checking = true;
#else
constexpr bool enable_error_checking = false;
#endif

/* in cleanup, free an object only if not null, then set it to null */
#define ISHMEMI_FREE(freefn, x)                                                                    \
    if ((x) != nullptr) {                                                                          \
        freefn(x);                                                                                 \
        x = nullptr;                                                                               \
    }

static inline unsigned long rdtsc()
{
    unsigned int hi, lo;

    __asm volatile(
        "xorl %%eax, %%eax\n"
        "cpuid            \n"
        "rdtsc            \n"
        : "=a"(lo), "=d"(hi)
        :
        : "%ebx", "%ecx");

    return ((unsigned long) hi << 32) | lo;
}

#endif /* ISHMEM_INTERNAL_H */
