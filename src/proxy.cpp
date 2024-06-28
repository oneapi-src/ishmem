/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "proxy.h"
#include "ishmem/err.h"
#include "runtime.h"
#include <thread>
#include <immintrin.h>
#include <accelerator.h>
#include <stdio.h>
#include <sched.h>
#include <pthread.h>
#include "proxy_func.h"

static std::thread proxy_thread;

ishmemi_cpu_info_t *ishmemi_cpu_info;

/*  the host has to be able to write the completion array and the peer_receive cell
 *  so those things must be 64 byte aligned and mapped to host memory
 *  They must be in device memory so they can be cached
 *
 *  The setup work happens in proxy_init
 *      allocation of completion array
 *      zeroing it
 *      copying pointer to ishmemi_mmap_gpu_info->ring.completions
 *      mmap the device pointer and store in global here ishmemi_ring_host_completions;
 *
 */
ishmemi_request_t *ishmemi_ring_host_sendbuf;            /* host map of send buffer */
ishmemi_ringcompletion_t *ishmemi_ring_host_completions; /* host map of completions */
ishmemi_message_t *ishmemi_msg_queue;                    /* messages to print from gpu */

#define USE_POLL_AVX 0

void ishmemi_cpu_ring::poll(size_t mwait_burst)
{
    int lockwasbusy = atomic_lock.exchange(1);

    ishmemi_ringcompletion_t comp;
    unsigned completion_index;
    if (lockwasbusy == 0) {
        ishmemi_request_t *mp = &recvbuf[next_receive % RING_SIZE];  // msg
        ishmemi_request_t msg __attribute__((aligned(64)));
        uint16_t matchvalue = (uint16_t) next_receive;
#if USE_POLL_AVX == 1
        __m512i req = _mm512_load_epi64((uint64_t *) mp);
        _mm512_store_epi64((uint64_t *) &msg, req);
        if ((uin16_t) msg.sequence == matchvalue) {
#else
        if ((uint16_t) mp->sequence == matchvalue) {
            _mm_mfence();
            msg = *mp;
#endif
            completion_index = next_receive & (RING_SIZE - 1);
            comp.completion.sequence = next_receive & 0xffff;
            next_receive = next_receive + 1;
            atomic_lock.store(0);      // release lock
            comp.completion.lock = 1;  // it should stay locked until freed at the device
            if (msg.op > DEBUG_TEST) msg.op = DEBUG_TEST;
            if (msg.type == ISHMEMI_TYPE_END) msg.type = MEM;
            // TODO - Enable this with a build flag
            if (0) {
                fprintf(stderr, "[PE %d] proxy seq %d op %s type %s comp %d pe %d\n", ishmemi_my_pe,
                        msg.sequence, ishmemi_op_str[(int) msg.op],
                        ishmemi_type_str[(int) msg.type], msg.completion, msg.dest_pe);
                fprintf(stderr, "[PE %d] target %p source %p size %ld pe %d\n", ishmemi_my_pe,
                        msg.dst, msg.src, msg.nelems, msg.dest_pe);
                fprintf(stderr, "[PE %d] Function being called: %p\n", ishmemi_my_pe,
                        ishmemi_upcall_funcs[msg.op]);
                fprintf(stderr, "[PE %d] Calling function [%d][%d] (%p)\n", ishmemi_my_pe, msg.op,
                        msg.type, (void *) (&ishmemi_upcall_funcs[msg.op][msg.type]));
                fflush(stderr);
            }
            ishmemi_upcall_funcs[msg.op][msg.type](&msg, &comp);
            _movdir64b((void *) &ishmemi_ring_host_completions[completion_index], &comp);
        } else {
            atomic_lock.store(0);  // release lock
            if (mwait_burst) {
                _umonitor(mp);
                long unsigned when = _rdtsc() + 10000L;
                _umwait(1, when);
            }
        }
    }
}

void host_proxy_thread(void *arg)
{
    size_t mwait_burst = ishmemi_params.MWAIT_BURST;
    while (ishmemi_cpu_info->proxy_state != EXIT) {
        ishmemi_cpu_info->ring.poll(mwait_burst);
    }

    ISHMEM_DEBUG_MSG("[proxy_thread] exiting\n");
}

int ishmemi_proxy_init()
{
    /* Type size check for local_pes (defined as uint8_t) */
    static_assert(MAX_LOCAL_PES < 0x100, "ISHMEM max local pe index cannot exceed 255");

    ishmemi_ringcompletion_t device_peer;
    ishmemi_cpu_info->proxy_state = READY;
    /*
     *      allocation of completion array
     *      allocation of sendbuf array
     *      zeroing them
     *      copying pointer to ishmemi_mmap_gpu_info->completions
     *      mmap the device pointer and store in global here ishmemi_ring_host_completions;
     */
    constexpr int CPU_STR_LEN = 4096;
    char str[CPU_STR_LEN];
    int off;

    int ret;
    ret = ishmemi_usm_alloc_host((void **) &ishmemi_ring_host_sendbuf,
                                 RING_SIZE * sizeof(ishmemi_request_t));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    ret = ishmemi_usm_alloc_host((void **) &ishmemi_msg_queue,
                                 NUM_MESSAGES * sizeof(ishmemi_message_t));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    ishmemi_mmap_gpu_info->messages = ishmemi_msg_queue;

    memset(ishmemi_ring_host_sendbuf, 0, RING_SIZE * sizeof(ishmemi_request_t));
    for (int i = 0; i < RING_SIZE; i += 1) {
        ishmemi_ring_host_sendbuf[i].op = G;
    }
    ishmemi_ring_host_completions = &ishmemi_mmap_gpu_info->completions[0];
    memset(ishmemi_ring_host_completions, 0, (RING_SIZE * 2) * sizeof(ishmemi_ringcompletion_t));

    /* Initialize the gpu ring object.  This is a weird operation, calling the ring constructor on
     * the host with a host mmapped pointer, for an object in device memory that will be used only
     * in SYCL code.
     */
    ishmemi_mmap_gpu_info->ring.init(ishmemi_ring_host_sendbuf, RING_SIZE,
                                     &ishmemi_gpu_info->completions[0].completion);

    /* Initialize the gpu completion object */
    ishmemi_mmap_gpu_info->completion.completions = &ishmemi_gpu_info->completions[0];
    ishmemi_mmap_gpu_info->completion.next_completion = 0;

    /* Initialize built-in completions in device memory */
    for (unsigned completion_index = 0; completion_index < RING_SIZE; completion_index += 1) {
        device_peer.completion.sequence = completion_index;
        _movdir64b((void *) &ishmemi_ring_host_completions[completion_index], &device_peer);
    }
    /* Initialize allocated completions in device memory */
    for (unsigned completion_index = 0; completion_index < RING_SIZE; completion_index += 1) {
        device_peer.completion.sequence = 0x10000;
        device_peer.completion.lock = 0;
        _movdir64b((void *) &ishmemi_ring_host_completions[RING_SIZE + completion_index],
                   &device_peer);
    }

    /* Initialized the cpu ring object */
    ishmemi_cpu_info->ring.init(ishmemi_ring_host_sendbuf, RING_SIZE);

    /* Initialize the upcall table.  This is a version of ishmemi_proxy_funcs
     * that has cutover functions replaced by new implementations
     * and also upcall functions that are not implemented by the runtime at all
     */
    ret = ishmemi_proxy_func_init();
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    /* Spawn the proxy thread */
    /* TODO figure out what the proxy_thread affinity should be according to topology
     * the thread should be on the same socket as the PCIe to the device
     */
    /* If we want multiple proxy threads, create them here */
    proxy_thread = std::thread(host_proxy_thread, (void *) NULL);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    ret = pthread_getaffinity_np(proxy_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (ret != 0) {
        ISHMEM_DEBUG_MSG("can't get proxy thread affinity\n");
    } else {
        off = snprintf(str, sizeof(str), "proxy thread affinity: ");
        for (int i = 0; i < sizeof(cpu_set_t) * 8; i += 1) {
            if (CPU_ISSET(i, &cpuset))
                off += snprintf(str + off, sizeof(str) - static_cast<size_t>(off), "%d, ", i);
        }
        ISHMEM_DEBUG_MSG("%s\n", str);
    }
    return 0;
fn_fail:
    return -1;
}

int ishmemi_proxy_fini()
{
    if (ishmemi_cpu_info->proxy_state == READY) {
        ishmemi_cpu_info->proxy_state = EXIT;
        proxy_thread.join();
    }
    /* Smash device's pointer to sendbuf, so we can free it
     * this will prevent any more proxy calls and cause segvs in device code if
     * any kernels are still running
     */
    if (ishmemi_mmap_gpu_info != nullptr) {
        ishmemi_mmap_gpu_info->ring.cleanup();
        ISHMEMI_FREE(ishmemi_usm_free, ishmemi_ring_host_sendbuf);

        if (ishmemi_mmap_gpu_info->messages != nullptr) {
            ISHMEMI_FREE(ishmemi_usm_free, ishmemi_msg_queue);
            ishmemi_mmap_gpu_info->messages = nullptr;
        }
    }
    ishmemi_proxy_func_fini();
    return 0;
}
