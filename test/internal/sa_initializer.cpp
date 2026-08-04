/* Copyright (C) 2023-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#if __INTEL_CLANG_COMPILER >= 20250000
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <unistd.h>
#include <stdint.h>
// #include <common.h>

#define NSEC_IN_SEC 1000000000.0
double getduration(const sycl::event &e)
{
    uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    double duration = static_cast<double>(end - start) / NSEC_IN_SEC;
    return (duration);
}

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

typedef struct {
    /* Destination PE */
    int dest_pe;
    /* Root PE used in broadcast */
    int root;
    /* Source, destination, and number of elements */
    const void *src;
    void *dst;
    size_t nelems;
    /* Signal address for put-signal */
    uint64_t *sig_addr;
    /* Attribute used for condition, comparison types, or signal operation types */
    union {
        /* Condition used for compare_swap */
        ishmemi_union_type cond;
        /* Comparison used for pt-pt synch. */
        int cmp;
        /* Signal operation */
        int sig_op;
        /* Destination stride value for iput/iget*/
        ptrdiff_t dst_stride;
    };
    /* Attribute used for values needed for comparison, signal  */
    union {
        ishmemi_union_type value;
        ishmemi_union_type cmp_value;
        uint64_t signal;
        /* Source stride value for iput/iget*/
        ptrdiff_t src_stride;
    };
    /* Operation and data type. */
    ishmemi_op_t op;
    ishmemi_type_t type;
    /* Sequence number and completion for request, This must be last for PCIe ordering */
    uint16_t sequence;
    uint16_t completion;
} ishmemi_request_t;

sycl::ext::oneapi::experimental::device_global<ishmemi_request_t> greq;
sycl::ext::oneapi::experimental::device_global<unsigned int> counter;

long *gdest = (long *) 0x12345;
long *gsrc = (long *) 0x123d5;
ptrdiff_t gdst = 3L;
ptrdiff_t gsst = 5L;
size_t gnelems = 17;
int gpe = 0;

void handle_request(ishmemi_request_t &req)
{
    sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        atomic_counter(counter);
    counter += 1;
    greq = req;
}

template <typename T>
SYCL_EXTERNAL void init_ishmem_iget(T *dest, const T *src, ptrdiff_t dst, ptrdiff_t sst,
                                    size_t nelems, int pe)
{
#ifdef __SYCL_DEVICE_ONLY__
    size_t nbytes = nelems * sizeof(T);

    ishmemi_request_t req = {
        .dest_pe = pe,
        .src = src,
        .dst = dest,
        .nelems = nbytes,
        .dst_stride = dst,
        .src_stride = sst,
        .op = NOP,
        .type = MEM,
    };
    handle_request(req);
#endif
}

SYCL_EXTERNAL void init_long_ishmem_iget(long *dest, const long *src, ptrdiff_t dst, ptrdiff_t sst,
                                         size_t nelems, int pe)
{
    init_ishmem_iget(dest, src, dst, sst, nelems, pe);
}

template <typename T>
SYCL_EXTERNAL void assign_ishmem_iget(T *dest, const T *src, ptrdiff_t dst, ptrdiff_t sst,
                                      size_t nelems, int pe)
{
#ifdef __SYCL_DEVICE_ONLY__
    size_t nbytes = nelems * sizeof(T);

    ishmemi_request_t req;
    req.dest_pe = pe;
    req.src = src;
    req.dst = dest;
    req.nelems = nbytes;
    req.dst_stride = dst;
    req.src_stride = sst;
    req.op = NOP;
    req.type = MEM;

    handle_request(req);
#endif
}

SYCL_EXTERNAL void assign_long_ishmem_iget(long *dest, const long *src, ptrdiff_t dst,
                                           ptrdiff_t sst, size_t nelems, int pe)
{
    assign_ishmem_iget(dest, src, dst, sst, nelems, pe);
}

double runkernel_init_iget(sycl::queue q, size_t count)
{
    long *dest = gdest;
    long *src = gsrc;
    ptrdiff_t dst = gdst;
    ptrdiff_t sst = gsst;
    size_t nelems = gnelems;
    int pe = gpe;

    auto e = q.single_task([=]() {
        for (size_t iter = 0; iter < count; iter += 1) {
            init_long_ishmem_iget(dest, src, dst, sst, nelems, pe);
        }
    });
    e.wait_and_throw();
    return (getduration(e));
}

double runkernel_assign_iget(sycl::queue q, size_t count)
{
    long *dest = gdest;
    long *src = gsrc;
    ptrdiff_t dst = gdst;
    ptrdiff_t sst = gsst;
    size_t nelems = gnelems;
    int pe = gpe;

    auto e = q.single_task([=]() {
        for (size_t iter = 0; iter < count; iter += 1) {
            assign_long_ishmem_iget(dest, src, dst, sst, nelems, pe);
        }
    });
    e.wait_and_throw();
    return (getduration(e));
}

int main(int argc, char **argv)
{
    // ishmem_init();

    setbuf(stdout, NULL);  // set line buffering
    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

    // int my_pe = ishmem_my_pe();
    // int npes = ishmem_n_pes();
    sycl::queue q(prop_list);

    size_t count = 16;
    double duration;
    printf("testing init_ishmem_iget\n");
    while (count < (1 << 24)) {
        duration = runkernel_init_iget(q, count);
        if (duration > 0.02) break;
        count <<= 1;
        printf("init_iget, count %lu, duration %f, calls per second %lu, each call %.1f nsec\n",
               count, duration, (long unsigned) ((double) count / duration),
               (duration * NSEC_IN_SEC) / (double) count);
    }

    count = 16;
    printf("testing assign_ishmem_iget\n");
    while (count < (1 << 24)) {
        duration = runkernel_assign_iget(q, count);
        if (duration > 0.02) break;
        count <<= 1;
        printf("assign_iget, count %lu, duration %f, calls per second %lu, each call %.1f nsec\n",
               count, duration, (long unsigned) ((double) count / duration),
               (duration * NSEC_IN_SEC) / (double) count);
    }

    // ishmem_finalize();
    return 0;
}
