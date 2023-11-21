/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_IMPL_PROXY_H
#define ISHMEM_IMPL_PROXY_H

#include "ishmem.h"
#include "ishmemx.h"
#include "internal.h"
#include "shmem.h"

#include <atomic>

/* eventually this is a constant address */
#define LOAD_PEER_NEXT_RECEIVE() *peer_receive

typedef struct {
    /* Sequence number and completion for request */
    uint16_t sequence;
    uint16_t completion;
    /* Operation and data type */
    ishmemi_op_t op;
    ishmemi_type_t type;
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
} ishmemi_request_t;

typedef struct {
    int lock;               // 0 if free
    unsigned int sequence;  // set by CPU
    ishmemi_union_type ret;
} ishmemi_completion_t;

typedef union {
    ulong8 data;
    ishmemi_completion_t completion;
} ishmemi_ringcompletion_t;

constexpr int RingN = 4096;
constexpr uint16_t UPDATE_RECEIVE_INTERVAL_MASK =
    0x7f;  // used in proxy.cpp for flow control backchannel

class ishmemi_completion {
  public:
    unsigned int next_completion;
    ishmemi_ringcompletion_t *completions;
    uint16_t Allocate()
    {
        ishmemi_ringcompletion_t *comp;
        sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                         sycl::access::address_space::global_space>
            atomic_next_completion(next_completion);
        unsigned int my_index;
        for (;;) {
            /* Truncate my_index to 12 bits */
            my_index = atomic_next_completion.fetch_add(1) & (RingN - 1);
            comp = &completions[my_index];
            sycl::atomic_ref<int, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_lock(comp->completion.lock);
            if (atomic_lock.exchange(1) == 0) break;
        }
        comp->completion.sequence = 0;  // reset sequence

        /* This cast is safe because of the truncation above */
        return static_cast<uint16_t>(my_index);
    }
    void Wait(unsigned int index)
    {
        ishmemi_ringcompletion_t *comp = &completions[index];
        while (comp->completion.sequence != 1) {
            sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        }
    }
    void Free(unsigned int index)
    {
        ishmemi_ringcompletion_t *comp = &completions[index];
        sycl::atomic_ref<int, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                         sycl::access::address_space::global_space>
            atomic_lock(comp->completion.lock);
        atomic_lock.store(0);
    }
};

/* This class is for the GPU end */
class ishmemi_ring {
  public:
    ishmemi_request_t *sendbuf;  // buffer is in host hemory
    unsigned int
        *peer_receive;  // pointer to the place in device memory where the host stores receive index
    unsigned int next_send;

  public:
    /* This function is the device code for sending a message to the host proxy */
    inline void Send(ishmemi_request_t *msgp)
    {
#ifdef __SYCL_DEVICE_ONLY__
        sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                         sycl::access::address_space::global_space>
            atomic_next_send(next_send);
        unsigned int my_send_index = atomic_next_send.fetch_add(1);

        /* Truncate the index to 12 bits */
        ishmemi_request_t *mp = &(sendbuf[my_send_index & (RingN - 1)]);
        ishmemi_request_t rm;
        rm = *msgp;

        /* This cast is safe; truncation is expected. */
        /* The max sequence value should exceed the index into sendbuf above due to information
         * lag between the proxy thread and the GPU. Truncation includes 4 extra bits for safety. */
        rm.sequence = static_cast<uint16_t>(my_send_index);
        // wait for previous uses of the buffer to be complete
        while ((my_send_index - LOAD_PEER_NEXT_RECEIVE()) >= RingN) {
            sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        }
        ucs_ulong8((ulong8 *) mp, *((ulong8 *) &rm));  // could be OOO
        // is this fence actually needed?
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
#endif
    }
};

typedef struct ishmem_info_t {
    ishmemi_ringcompletion_t completions[RingN + 1] __attribute__((aligned(64)));
    //  ishmemi_ringcompletion_t peer_receive __attribute__ ((aligned (64)));
    // the previous line is the intended use of completions[RingN]
    bool is_initialized = false;
    int my_pe;
    int n_pes;
    // stuff for collectives
    int local_rank;
    int n_local_pes;
    int barrier_index;
    long *barrier_all_psync[ISHMEM_SYNC_NUM_PSYNC_ARRS];
    int sync_index;
    long *sync_all_psync[ISHMEM_SYNC_NUM_PSYNC_ARRS];
    ishmemi_team_t reduce;
    size_t *collect_mynelems;  // symmetric, for our collect size
    size_t *collect_nelems;    // symmetric, size of MAX_LOCAL_PEs

    /* IPC stuff */
    void *heap_base;
    size_t heap_length;
    ishmemi_ring ring;
    ishmemi_completion completion;
    ptrdiff_t ipc_buffer_delta[MAX_LOCAL_PES + 1] __attribute__((aligned(64)));
    /* only_intra_node is used to identify if all PEs are run on a single node
     * to enable SYCL atomics */
    bool only_intra_node;
    unsigned int message_buffer_lock[NUM_MESSAGES];
    struct ishmemi_message_t *messages;
    long debug[64];  // development convenience, remove at some point
    uint8_t local_pes[MAX_LOCAL_PES] __attribute__((aligned(64)));
    /* local_pes is a larger array, depending on n_pes at runtime */
    /* local_pes must be LAST in ishmem_info_t */
} ishmem_info_t;

/* the following objects are allocated in host memory */

/* this class is for the CPU end */
class ishmemi_cpu_ring {
  public:
    ishmemi_request_t *recvbuf;
    unsigned int next_receive;
    std::atomic<int> atomic_lock;
    ishmemi_cpu_ring(ishmemi_request_t *recvbuf)
    {
        this->recvbuf = recvbuf;
        next_receive = RingN;
        atomic_lock = 0;
    }
    void Poll();
};

typedef enum {
    READY,
    PROCESSING,
    REQUEST,
    EXIT
} ishmemi_proxy_state_t;

typedef struct ishmem_cpu_info_t {
    int my_pe;
    int n_pes;
    ishmemx_attr_t *attr;
    ishmemi_proxy_state_t proxy_state;
    ishmemi_cpu_ring ring;
    bool use_ipc;
    ishmemi_host_team_t reduce;
} ishmem_cpu_info_t;

/* global to hold the host version of ishmem_cpu_info_t */
extern ishmem_cpu_info_t *ishmemi_cpu_info;

ISHMEM_DEVICE_ATTRIBUTES inline int ishmemi_proxy_get_status(const ishmemi_union_type &field)
{
    return field.i;
}

template <typename T, bool SIGN_MATTERS = false, bool FP_MATTERS = false>
ISHMEM_DEVICE_ATTRIBUTES inline void ishmemi_proxy_set_field_value(ishmemi_union_type &field,
                                                                   const T val)
{
    /* Floating-point types */
    if constexpr (FP_MATTERS) {
        if constexpr (std::is_floating_point_v<T>) {
            if constexpr (std::is_same_v<T, float>) {
                field.f = static_cast<float>(val);
                return;
            } else if constexpr (std::is_same_v<T, double>) {
                field.ld = static_cast<double>(val);
                return;
            } else static_assert(false, "Unknown or unsupported type");
        }
    }

    /* Signed types */
    if constexpr (SIGN_MATTERS) {
        if constexpr (std::is_signed_v<T>) {
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
                field.ull = static_cast<long long>(val);
                return;
            } else static_assert(false, "Unknown or unsupported type");
        }
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

template <typename T, bool SIGN_MATTERS = false, bool FP_MATTERS = false>
ISHMEM_DEVICE_ATTRIBUTES inline T ishmemi_proxy_get_field_value(const ishmemi_union_type &field)
{
    /* Floating-point types */
    if constexpr (FP_MATTERS) {
        if constexpr (std::is_floating_point_v<T>) {
            if constexpr (std::is_same_v<T, float>) return static_cast<T>(field.f);
            else if constexpr (std::is_same_v<T, double>) return static_cast<T>(field.ld);
            else static_assert(false, "Unknown or unsupported type");
        }
    }

    /* Signed types */
    if constexpr (SIGN_MATTERS) {
        if constexpr (std::is_signed_v<T>) {
            if constexpr (sizeof(T) == sizeof(int8_t)) return static_cast<T>(field.i8);
            else if constexpr (sizeof(T) == sizeof(int16_t)) return static_cast<T>(field.i16);
            else if constexpr (sizeof(T) == sizeof(int32_t)) return static_cast<T>(field.i32);
            else if constexpr (sizeof(T) == sizeof(int64_t)) return static_cast<T>(field.i64);
            else if constexpr (sizeof(T) == sizeof(long long)) return static_cast<T>(field.ull);
            else static_assert(false, "Unknown or unsupported type");
        }
    }

    /* Unsigned types */
    if constexpr (sizeof(T) == sizeof(uint8_t)) return static_cast<T>(field.ui8);
    else if constexpr (sizeof(T) == sizeof(uint16_t)) return static_cast<T>(field.ui16);
    else if constexpr (sizeof(T) == sizeof(uint32_t)) return static_cast<T>(field.ui32);
    else if constexpr (sizeof(T) == sizeof(uint64_t)) return static_cast<T>(field.ui64);
    else if constexpr (sizeof(T) == sizeof(unsigned long long)) return static_cast<T>(field.ull);
    else static_assert(false, "Unknown or unsupported type");
}

template <typename T, bool SIGN_MATTERS = false, bool FP_MATTERS = false>
ISHMEM_DEVICE_ATTRIBUTES inline ishmemi_type_t ishmemi_proxy_get_base_type()
{
    /* Floating-point types */
    if constexpr (FP_MATTERS) {
        if constexpr (std::is_floating_point_v<T>) {
            if constexpr (std::is_same_v<T, float>) return FLOAT;
            else if constexpr (std::is_same_v<T, double>) return DOUBLE;
            else static_assert(false, "Unknown or unsupported type");
        }
    }

    /* Signed types */
    if constexpr (SIGN_MATTERS) {
        if constexpr (std::is_signed_v<T>) {
            if constexpr (sizeof(T) == sizeof(int8_t)) return INT8;
            else if constexpr (sizeof(T) == sizeof(int16_t)) return INT16;
            else if constexpr (sizeof(T) == sizeof(int32_t)) return INT32;
            else if constexpr (sizeof(T) == sizeof(int64_t)) return INT64;
            else if constexpr (sizeof(T) == sizeof(long long)) return LONGLONG;
            else static_assert(false, "Unknown or unsupported type");
        }
    }

    /* Unsigned types */
    if constexpr (sizeof(T) == sizeof(uint8_t)) return UINT8;
    else if constexpr (sizeof(T) == sizeof(uint16_t)) return UINT16;
    else if constexpr (sizeof(T) == sizeof(uint32_t)) return UINT32;
    else if constexpr (sizeof(T) == sizeof(uint64_t)) return UINT64;
    else if constexpr (sizeof(T) == sizeof(unsigned long long)) return ULONGLONG;
    else static_assert(false, "Unknown or unsupported type");
}

ISHMEM_DEVICE_ATTRIBUTES inline void ishmemi_proxy_blocking_request(ishmemi_request_t *req)
{
    ishmem_info_t *info = global_info;
    uint16_t comp = info->completion.Allocate();
    req->completion = comp;

    info->ring.Send(req);
    info->completion.Wait(comp);
    info->completion.Free(comp);
}

ISHMEM_DEVICE_ATTRIBUTES inline int ishmemi_proxy_blocking_request_status(ishmemi_request_t *req)
{
    int ret = 0;
    ishmem_info_t *info = global_info;
    uint16_t comp = info->completion.Allocate();
    req->completion = comp;

    info->ring.Send(req);
    info->completion.Wait(comp);

    ret = info->completions[comp].completion.ret.i;

    info->completion.Free(comp);

    return ret;
}

template <typename T>
ISHMEM_DEVICE_ATTRIBUTES inline T ishmemi_proxy_blocking_request_return(ishmemi_request_t *req)
{
    T ret = static_cast<T>(0);
    ishmem_info_t *info = global_info;
    uint16_t comp = info->completion.Allocate();
    req->completion = comp;

    info->ring.Send(req);
    info->completion.Wait(comp);

    ret = ishmemi_proxy_get_field_value<T, true, true>(info->completions[comp].completion.ret);

    info->completion.Free(comp);

    return ret;
}

ISHMEM_DEVICE_ATTRIBUTES inline void ishmemi_proxy_nonblocking_request(ishmemi_request_t *req)
{
    ishmem_info_t *info = global_info;
    req->completion = 0;
    info->ring.Send(req);
}

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

#define USE_VEC_COPY_WORK_GROUP_STRIDED 1

#if USE_USE_VEC_COPY_WORK_GROUP_STRIDED
/* these functions are called collectively by all threads in a work group */
template <typename T, typename Group>
void vec_copy_work_group_push(T *d, const T *s, size_t count, Group grp)
{
    if constexpr (ishmemi_is_device) {
        size_t stride = grp.get_local_linear_range();
        size_t linear_id = grp.get_local_linear_id();
        size_t idx = linear_id;
        T *aligned_d = (T *) sycl::min(((((uintptr_t) d) + ALIGNMASK) & (~ALIGNMASK)),
                                       (uintptr_t) (d + count));
        /* The idea of this loop is that each thread copies every wg_size-th element starting
         * with its own id in the group
         */
        while (((uintptr_t) &d[idx]) < ((uintptr_t) (aligned_d))) {
            d[idx] = s[idx];
            idx += stride;
        }
        count -= (aligned_d - d);  // pointer difference is in units of T
        s += (aligned_d - d);
        /* at this point, if count > 0, then d is aligned, s may not be aligned */
        if (count == 0) return;
        idx = linear_id * VL;
        size_t vstride = stride * VL;

        sycl::multi_ptr<T, sycl::access::address_space::global_space, sycl::access::decorated::yes>
            dd;
        sycl::multi_ptr<const T, sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>
            ds;
        dd = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::yes>(aligned_d);
        ds = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::yes>(s);
        /* In this loop, each thread handles a len VL vector with stride wg_size * VL.  */
        while ((idx + VL) <= count) {
            sycl::vec<T, VL> temp;
            temp.load(0, ds + idx);
            temp.store(0, dd + idx);
            idx += vstride;
        }
        /* at this point, the threads have finished the copy except for the last
         * count & (VL-1) items
         * back to item at a time
         */
        /* idx here should be the postfix index + linear_id */
        idx = linear_id + (count & (~(VL - 1)));
        while (idx < count) {
            dd[idx] = ds[idx];
            idx += stride;
        }
    } else {
        memcpy(d, s, count * sizeof(T));
    }
}

/* this function is called collectively by all threads in a work group */
template <typename T, typename Group>
void vec_copy_work_group_pull(T *d, const T *s, size_t count, Group grp)
{
    if constexpr (ishmemi_is_device) {
        size_t stride = grp.get_local_linear_range();
        size_t linear_id = grp.get_local_linear_id();
        size_t idx = linear_id;
        T *aligned_s = (T *) sycl::min(((((uintptr_t) s) + ALIGNMASK) & (~ALIGNMASK)),
                                       (uintptr_t) (s + count));
        /* The idea of this loop is that each thread copies every wg_size-th element starting
         * with its own id in the group
         */
        while (((uintptr_t) &s[idx]) < ((uintptr_t) (aligned_s))) {
            d[idx] = s[idx];
            idx += stride;
        }
        count -= (aligned_s - s);  // pointer difference is in units of T
        d += (aligned_s - s);
        /* at this point, if count > 0, then d is aligned, s may not be aligned */
        if (count == 0) return;
        idx = linear_id * VL;
        size_t vstride = stride * VL;

        sycl::multi_ptr<T, sycl::access::address_space::global_space, sycl::access::decorated::yes>
            dd;
        sycl::multi_ptr<const T, sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>
            ds;
        dd = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::yes>(d);
        ds = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::yes>(aligned_s);
        /* In this loop, each thread handles a len VL vector with stride wg_size * VL.  */
        while ((idx + VL) <= count) {
            sycl::vec<T, VL> temp;
            temp.load(0, ds + idx);
            temp.store(0, dd + idx);
            idx += vstride;
        }
        /* at this point, the threads have finished the copy except for the last
         * count & (VL-1) items
         * back to item at a time
         */
        idx = linear_id + (count & (~(VL - 1)));
        while (idx < count) {
            dd[idx] = ds[idx];
            idx += stride;
        }
    } else {
        memcpy(d, s, count * sizeof(T));
    }
}
#else  // !USE_VEC_COPY_WORK_GROUP_STRIDED
/* this function is called collectively by all threads in a work group */

template <typename T, typename Group>
void vec_copy_work_group_push(T *d, const T *s, size_t count, Group grp)
{
    if constexpr (ishmemi_is_device) {
        size_t my_nelems_work_item;
        size_t work_item_start_idx;
        ishmemi_work_item_calculate_offset(count, grp, my_nelems_work_item, work_item_start_idx);
        vec_copy_push(d + work_item_start_idx, s + work_item_start_idx, my_nelems_work_item);
    }
}

template <typename T, typename Group>
void vec_copy_work_group_pull(T *d, const T *s, size_t count, Group grp)
{
    if constexpr (ishmemi_is_device) {
        size_t my_nelems_work_item;
        size_t work_item_start_idx;
        ishmemi_work_item_calculate_offset(count, grp, my_nelems_work_item, work_item_start_idx);
        vec_copy_pull(d + work_item_start_idx, s + work_item_start_idx, my_nelems_work_item);
    }
}

#endif

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
            return -1;
    }
}

#define ISHMEMI_RUNTIME_REQUEST_HELPER(T, TYPENAME)                                                \
    /* Basic arguments */                                                                          \
    T *dest __attribute__((unused)) = static_cast<T *>(msg->dst);                                  \
    const T *src __attribute__((unused)) = static_cast<const T *>(msg->src);                       \
    size_t nelems __attribute__((unused)) = msg->nelems;                                           \
    int pe __attribute__((unused)) = msg->dest_pe;                                                 \
    /* Stride arguments */                                                                         \
    ptrdiff_t dst __attribute__((unused)) = msg->dst_stride;                                       \
    ptrdiff_t sst __attribute__((unused)) = msg->src_stride;                                       \
    /* AMO/p arguments */                                                                          \
    T val __attribute__((unused)) = static_cast<T>(msg->value.TYPENAME);                           \
    T cond __attribute__((unused)) = static_cast<T>(msg->cond.TYPENAME);                           \
    /* Signaling arguments */                                                                      \
    uint64_t *sig_addr __attribute__((unused)) = msg->sig_addr;                                    \
    uint64_t signal __attribute__((unused)) = msg->signal;                                         \
    int sig_op __attribute__((unused)) = msg->sig_op;                                              \
    /* Synchronization arguments */                                                                \
    int cmp __attribute__((unused)) = msg->cmp;                                                    \
    T cmp_value __attribute__((unused)) = msg->cmp_value.TYPENAME;                                 \
    /* Broadcast argument */                                                                       \
    int root __attribute__((unused)) = msg->root;                                                  \
    /* Team for collectives */                                                                     \
    shmem_team_t team __attribute__((unused)) = SHMEM_TEAM_WORLD;

#endif  //! ISHMEM_IMPL_H
