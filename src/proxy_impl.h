/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_PROXY_IMPL_H
#define ISHMEM_PROXY_IMPL_H

#include "ishmem.h"
#include "ishmemx.h"
#include "proxy_types.h"
#include "intrinsic.h"
#include "ishmem/err.h"
#include "ishmem/copy.h"
#include "collectives.h"
#include "teams.h"

#include <atomic>

constexpr int RING_SIZE = 4096;

/* Max number of retries to get two consecutive, identical reads of next_send from mmap'd variable
 * in call to ishmemi_drain_ring */
constexpr unsigned int DRAIN_RING_THRESHOLD = 10;

/* For flow control backchannel */
constexpr uint16_t UPDATE_RECEIVE_INTERVAL_MASK = 0x7f;

/* Completion object */
/* The first RING_SIZE completions are "built in"
 * The next RING_SIZE completions are "allocated"
 *
 * A built-in completion is completed when the low 16 bits of its sequence number matches
 * the request sequence number and bit 31 is clear.
 * Completions that return a value have bit 31 set
 *
 * If a return value is present, read it, then clear bit 31 to finish marking the completion as
 * complete
 *
 * The built-in completion sequence numbers are also used for ring flow control.  Before a ring slot
 * can be used for a new message, its previous use must be marked complete, with the sequence field
 * set to the previous sequence number with bit 31 clear (which means any return values in the
 * completion have been read.)
 *
 * The built-in completions do not need to be allocated, they are permanently assigned.  Their
 * status is indicated by the sequence numbers as above.
 *
 * The rest of the completions are used (in <addition> to the built-in one for a particular request)
 * for long-running operations that have return values or require completion signalling. The status
 * of an allocated completion is shown by its lock field (0 means idle, 1 means in use) The lock
 * field is set by completion::allocate, maintained as set by the proxy when writing the completion,
 * and cleared by completion::free
 *
 * In summary, for blocking operations only the built-in completion is used, and provides ring flow
 * control and completion signalling and return values.  Blocking operations can return out of
 * order, and that is fine.  If they are delayed for RING_SIZE other messages, then the subsequent
 * use of the same ring slot will stall For long running ops, both the built-in completion is used
 * (for flow control) and the allocated completion is used (for return results and completion
 * signalling).
 *
 * To reduce memory references, the completion field of the request structure is set to 0 when only
 * the built in completion is used, and set to the index of the allocated completion when both are
 * used.
 */
class ishmemi_completion {
  public:
    unsigned int next_completion;
    ishmemi_ringcompletion_t *completions;

    uint16_t allocate()
    {
        ishmemi_ringcompletion_t *comp;
        sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                         sycl::access::address_space::global_space>
            atomic_next_completion(next_completion);
        unsigned int my_index;
        for (;;) {
            /* Truncate my_index to lg(RING_SIZE) bits */
            my_index = RING_SIZE + (atomic_next_completion.fetch_add(1) & (RING_SIZE - 1));
            comp = &completions[my_index];
            sycl::atomic_ref<int, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_lock(comp->completion.lock);
            if (atomic_lock.exchange(1) == 0) break;
        }
        comp->completion.sequence = 0x10000;  // initialize with a sequence that is never valid

        /* This cast is safe because of the truncation above */
        return static_cast<uint16_t>(my_index);
    }

    void wait(unsigned int index, uint16_t sequence)
    {
        ishmemi_ringcompletion_t *comp = &completions[index];
        sycl::atomic_ref<unsigned int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                         sycl::access::address_space::global_space>
            atomic_comp_sequence(comp->completion.sequence);
        /* masking the sequence number with 1ffff is to handle both automatically allocated
         * completions with indices < RING_SIZE and separately allocated completions with indices
         * >RING_SIZE The former have the low bits of the completion sequence invalid because they
         * are used for flow control. The latter have the completion marked invalid by the 0x10000
         * in allocate above
         */
        while ((atomic_comp_sequence & 0x1ffff) != (uint32_t) sequence)
            ;
    }

    void free(unsigned int index)
    {
        ishmemi_ringcompletion_t *comp = &completions[index];
        sycl::atomic_ref<int, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                         sycl::access::address_space::global_space>
            atomic_lock(comp->completion.lock);
        atomic_lock.store(0);
    }
};

/* Ring objects */
class ishmemi_cpu_ring {
  public:
    ishmemi_cpu_ring() : recvbuf(nullptr), next_receive(0), atomic_lock(0) {}

    /* Initialize the ring */
    inline void init(ishmemi_request_t *_recvbuf, unsigned int _next_receive)
    {
        recvbuf = _recvbuf;
        next_receive = _next_receive;
        atomic_lock = 0;
    }

    /* Cleanup the ring */
    inline void cleanup()
    {
        recvbuf = nullptr;
        next_receive = 0;
        atomic_lock = 0;
    }

    /* Poll for completion */
    void poll(size_t mwait_burst);

    /* Query next receive - for debugging purposes */
    inline unsigned int get_next_receive()
    {
        return next_receive;
    }

    friend void ishmemi_drain_ring();

  private:
    ishmemi_request_t *recvbuf;
    unsigned int next_receive;
    std::atomic<int> atomic_lock;
};

class ishmemi_gpu_ring {
  public:
    ishmemi_gpu_ring() : sendbuf(nullptr), next_send(0), completions(nullptr) {}

    /* Initialize the ring */
    inline void init(ishmemi_request_t *_sendbuf, unsigned int _next_send,
                     ishmemi_completion_t *_completions)
    {
        sendbuf = _sendbuf;
        next_send = _next_send;
        completions = _completions;
    }

    /* Cleanup the ring */
    inline void cleanup()
    {
        sendbuf = nullptr;
        next_send = 0;
        completions = nullptr;
    }

    /* Sends a message to the host proxy. Intended to be called from the device
     * returns sequence number so caller knows where to look for completion
     */
    inline uint32_t send(ishmemi_request_t &msg)
    {
#ifdef __SYCL_DEVICE_ONLY__
        sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_next_send(next_send);
        unsigned int my_send_index = atomic_next_send.fetch_add(1);

        /* Truncate the index to 12 bits */
        ishmemi_request_t *mp = &(sendbuf[my_send_index & (RING_SIZE - 1)]);

        /* This cast is safe; truncation is expected. */
        /* The max sequence value should exceed the index into sendbuf above due to information
         * lag between the proxy thread and the GPU. Truncation includes 4 extra bits for safety. */
        msg.sequence = static_cast<uint16_t>(my_send_index);

        /* Wait for previous use of buffer to be complete */
        ishmemi_completion_t *comp = &completions[my_send_index & (RING_SIZE - 1)];
        uint32_t expected = (my_send_index - RING_SIZE) & 0xffff;
        sycl::atomic_ref<unsigned int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                         sycl::access::address_space::global_space>
            atomic_comp_sequence(comp->sequence);
        /* This is subtle.  A sequence number that is carrying a return value will have the correct
         * low 16 bits, and bit 31 set.  This compare will fail until another thread clears bit 31
         */
        while (atomic_comp_sequence != expected)
            ;
        ucs_ulong8(((ulong8 *) mp), *((ulong8 *) &msg));
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        return (my_send_index & 0xffff);
#else
        assert(0);
        return (0);
#endif
    }

    inline void sendwait(ishmemi_request_t &msg)
    {
#ifdef __SYCL_DEVICE_ONLY__
        sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_next_send(next_send);
        unsigned int my_send_index = atomic_next_send.fetch_add(1);

        /* Truncate the index to 12 bits */
        ishmemi_request_t *mp = &(sendbuf[my_send_index & (RING_SIZE - 1)]);
        ishmemi_request_t rm = msg;
        /* This cast is safe; truncation is expected. */
        /* The max sequence value should exceed the index into sendbuf above due to information
         * lag between the proxy thread and the GPU. Truncation includes 4 extra bits for safety. */
        rm.sequence = static_cast<uint16_t>(my_send_index);
        rm.completion = 0;
        /* Wait for previous use of buffer to be complete */
        ishmemi_completion_t *comp = &completions[my_send_index & (RING_SIZE - 1)];
        uint32_t expected = (my_send_index - RING_SIZE) & 0xffff;
        sycl::atomic_ref<unsigned int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                         sycl::access::address_space::global_space>
            atomic_comp_sequence(comp->sequence);
        /* acquire order makes sure the sequence test happens before the message send */
        /* This is subtle.  A sequence number that is carrying a return value will have the correct
         * low 16 bits, and bit 31 set.  This compare will fail until another thread clears bit 31
         */
        while (atomic_comp_sequence != expected)
            ;
        ucs_ulong8((ulong8 *) mp, *((ulong8 *) &rm));
        expected = my_send_index & 0xffff;
        while (atomic_comp_sequence != expected)
            ;
#endif
    }

    /* Query next send - for debugging purposes */
    inline unsigned int get_next_send()
    {
        return next_send;
    }

    friend void ishmemi_drain_ring();

  private:
    ishmemi_request_t *sendbuf; /* located in host memory */
    unsigned int next_send;
    ishmemi_completion_t *completions; /* location in device memory for completion array */
};

/* Info objects */
typedef struct ishmemi_cpu_info_t {
    /* Basic variables */
    bool is_initialized = false;
    int my_pe;
    int n_pes;
    bool use_ipc;

    /* Proxy variables */
    ishmemi_proxy_state_t proxy_state;
    ishmemi_cpu_ring ring;

    /* Other variables */
    size_t n_teams;
    ishmemi_team_host_t *team_host_pool;
    ishmemx_attr_t *attr;
} ishmemi_cpu_info_t;

extern ishmemi_cpu_info_t *ishmemi_cpu_info;

typedef struct ishmemi_info_t {
    /* The first RING_SIZE completions are "built_in" and paired 1-1 with the send ring
     * The rest are "allocated" and used for long running non-blocking operations
     */
    ishmemi_ringcompletion_t completions[RING_SIZE * 2] __attribute__((aligned(64)));
    ishmemi_completion completion;

    /* Basic variables */
    bool is_initialized = false;
    int my_pe;
    int n_pes;
    /* Teams variables */
    size_t n_teams;
    ishmemi_team_device_t *team_device_pool;

    /* IPC variables */
    void *heap_base;
    size_t heap_length;
    ishmemi_gpu_ring ring;
    ptrdiff_t ipc_buffer_delta[MAX_LOCAL_PES + 1] __attribute__((aligned(64)));
    bool only_intra_node; /* Identifies if all PEs are on a single node (for sycl atomics) */

    /* For printing from the GPU */
    unsigned int message_buffer_lock[NUM_MESSAGES];
    struct ishmemi_message_t *messages;

    /* Other variables */
    size_t sync_last_idx_checked = 0;
    long debug[64]; /* For development convenice. TODO: remove */

    /* NOTE: local_pes MUST BE LAST in ishmemi_info_t */
    /* local_pes is a larger array, depending on n_pes at runtime */
    uint8_t local_pes[MAX_LOCAL_PES] __attribute__((aligned(64)));
} ishmemi_info_t;

inline void ishmemi_drain_ring()
{
    std::atomic<unsigned int> next_send_checkpoint(ishmemi_mmap_gpu_info->ring.next_send);
    std::atomic<unsigned int> temp(ishmemi_mmap_gpu_info->ring.next_send);
    unsigned int iteration = 0;
    while (next_send_checkpoint != temp) {
        if (++iteration > DRAIN_RING_THRESHOLD) {
            ISHMEM_WARN_MSG(
                "Could not obtain consistent read of next_send. Runtime cannot guarantee the "
                "current quiet operation will wait for completion of previously issued ISHMEM "
                "calls on the upcall ring buffer.\n");
            return;
        }

        next_send_checkpoint.store(temp);
        temp.store(ishmemi_mmap_gpu_info->ring.next_send);
    }
    while ((int) next_send_checkpoint - (int) ishmemi_cpu_info->ring.next_receive > 0) {
    }
}

ISHMEM_DEVICE_ATTRIBUTES inline int ishmemi_proxy_get_status(const ishmemi_union_type &field)
{
    return field.i;
}

ISHMEM_DEVICE_ATTRIBUTES inline void ishmemi_proxy_blocking_request(ishmemi_request_t &req)
{
    ishmemi_info_t *info = global_info;
    info->ring.sendwait(req);
}

ISHMEM_DEVICE_ATTRIBUTES inline int ishmemi_proxy_blocking_request_status(ishmemi_request_t &req)
{
    int ret = 0;
    ishmemi_info_t *info = global_info;
    req.completion = 0;
    uint32_t sequence = info->ring.send(req);
    uint32_t completion_index = sequence & (RING_SIZE - 1);
    info->completion.wait(completion_index, static_cast<uint16_t>(sequence));
    ret = info->completions[completion_index].completion.ret.i;
    /* The purpose of this is to clear the 0x80000000 bit, to mark the completion as free */
    info->completions[completion_index].completion.sequence = sequence;
    return ret;
}

template <typename T, ishmemi_op_t OP>
ISHMEM_DEVICE_ATTRIBUTES inline T ishmemi_proxy_blocking_request_return(ishmemi_request_t &req)
{
    T ret = static_cast<T>(0);
    ishmemi_info_t *info = global_info;
    req.completion = 0;
    uint32_t sequence = info->ring.send(req);
    uint32_t completion_index = sequence & (RING_SIZE - 1);
    info->completion.wait(completion_index, static_cast<uint16_t>(sequence));
    ret = ishmemi_union_get_field_value<T, OP>(info->completions[completion_index].completion.ret);
    /* The purpose of this is to clear the 0x80000000 bit, to mark the completion as free */
    info->completions[completion_index].completion.sequence = sequence;
    return ret;
}

ISHMEM_DEVICE_ATTRIBUTES inline void ishmemi_proxy_nonblocking_request(ishmemi_request_t &req)
{
    ishmemi_info_t *info = global_info;
    info->ring.send(req);
}

#define ISHMEMI_RUNTIME_REQUEST_HELPER(T, OP)                                                      \
    /* Basic arguments */                                                                          \
    T *dest __attribute__((unused)) = static_cast<T *>(msg->dst);                                  \
    T *fetch __attribute__((unused)) = static_cast<T *>(msg->fetch);                               \
    const T *src __attribute__((unused)) = static_cast<const T *>(msg->src);                       \
    size_t nelems __attribute__((unused)) = msg->nelems;                                           \
    int pe __attribute__((unused)) = msg->dest_pe;                                                 \
    /* Stride arguments */                                                                         \
    ptrdiff_t dst __attribute__((unused)) = msg->dst_stride;                                       \
    ptrdiff_t sst __attribute__((unused)) = msg->src_stride;                                       \
    size_t bsize __attribute__((unused)) = msg->bsize;                                             \
    /* AMO/p arguments */                                                                          \
    T val __attribute__((unused)) = ishmemi_union_get_field_value<T, OP>(msg->value);              \
    T cond __attribute__((unused)) = ishmemi_union_get_field_value<T, OP>(msg->cond);              \
    /* Signaling arguments */                                                                      \
    uint64_t *sig_addr __attribute__((unused)) = msg->sig_addr;                                    \
    uint64_t signal __attribute__((unused)) = msg->signal;                                         \
    int sig_op __attribute__((unused)) = msg->sig_op;                                              \
    /* Synchronization arguments */                                                                \
    int cmp __attribute__((unused)) = msg->cmp;                                                    \
    T cmp_value __attribute__((unused)) = ishmemi_union_get_field_value<T, OP>(msg->cmp_value);    \
    const T *cmp_values __attribute__((unused)) = static_cast<const T *>(msg->cmp_values);         \
    const int *status __attribute__((unused)) = msg->status;                                       \
    size_t *indices __attribute__((unused)) = msg->indices;                                        \
    /* Broadcast argument */                                                                       \
    int root __attribute__((unused)) = msg->root;                                                  \
    /* Teams/collectives args */                                                                   \
    ishmem_team_t team __attribute__((unused)) = msg->team;                                        \
    ishmemi_team_host_t *team_ptr __attribute__((unused)) =                                        \
        &ishmemi_cpu_info->team_host_pool[msg->team];

#endif /* ISHMEM_PROXY_IMPL_H */
