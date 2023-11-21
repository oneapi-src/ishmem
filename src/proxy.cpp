/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "proxy.h"
#include "internal.h"
#include "runtime.h"
#include <thread>
#include <immintrin.h>
#include <accelerator.h>
#include <stdio.h>
#include <sched.h>
#include <pthread.h>
#include "proxy_func.h"

static std::thread proxy_thread;

ishmem_cpu_info_t *ishmemi_cpu_info;

/*  the host has to be able to write the completion array and the peer_receive cell
 *  so those things must be 64 byte aligned and mapped to host memory
 *  They must be in device memory so they can be cached
 *
 *  The setup work happens in proxy_init
 *      allocation of completion array
 *      zeroing it
 *      copying pointer to ishmemi_mmap_gpu_info->ring.completions
 *      mmap the device pointer and store in global here ishmemi_completions;
 *
 */
ishmemi_request_t *ishmemi_sendbuf;            /* host address */
ishmemi_ringcompletion_t *ishmemi_completions; /* host map to device */
ishmemi_message_t *ishmemi_msg_queue;

void ishmemi_cpu_ring::Poll()
{
    int lockwasbusy = atomic_lock.exchange(1);

    ishmemi_ringcompletion_t comp;
    ishmemi_ringcompletion_t device_peer;
    if (lockwasbusy == 0) {
        ishmemi_request_t *mp = &recvbuf[next_receive % RingN];  // msg
        if ((mp->sequence & 0xffff) == (next_receive & 0xffff)) {
            ishmemi_request_t msg = *mp;
            next_receive = next_receive + 1;
            if ((next_receive & UPDATE_RECEIVE_INTERVAL_MASK) == 0) {
                device_peer.completion.sequence = next_receive;
                _movdir64b((void *) &ishmemi_completions[RingN], &device_peer);
            }
            atomic_lock.store(0);      // release lock
            comp.completion.lock = 1;  // it should stay locked until freed at the device
            comp.completion.sequence = 1;
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
            if (msg.completion != 0)
                _movdir64b((void *) &ishmemi_completions[msg.completion], &comp);
        } else {
            atomic_lock.store(0);  // release lock
        }
    }
}

void host_proxy_thread(void *arg)
{
    while (ishmemi_cpu_info->proxy_state != EXIT) {
        ishmemi_cpu_info->ring.Poll();
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
     *      mmap the device pointer and store in global here ishmemi_completions;
     */
    constexpr int CPU_STR_LEN = 4096;
    char str[CPU_STR_LEN];
    int off;

    int ret;
    ret = ishmemi_usm_alloc_host((void **) &ishmemi_sendbuf, RingN * sizeof(ishmemi_request_t));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    ret = ishmemi_usm_alloc_host((void **) &ishmemi_msg_queue,
                                 NUM_MESSAGES * sizeof(ishmemi_message_t));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    ishmemi_mmap_gpu_info->messages = ishmemi_msg_queue;

    memset(ishmemi_sendbuf, 0, RingN * sizeof(ishmemi_request_t));
    ishmemi_completions = &ishmemi_mmap_gpu_info->completions[0];
    memset(ishmemi_completions, 0, (RingN + 1) * sizeof(ishmemi_ringcompletion_t));

    /*
     * Initialization of ring and completion for device info; done via mmap copy
     */
    ishmemi_mmap_gpu_info->ring.sendbuf = ishmemi_sendbuf;
    ishmemi_mmap_gpu_info->ring.next_send = RingN;
    ishmemi_mmap_gpu_info->ring.peer_receive =
        &ishmemi_gpu_info->completions[RingN].completion.sequence;
    ishmemi_mmap_gpu_info->completion.completions = &ishmemi_gpu_info->completions[0];
    ishmemi_mmap_gpu_info->completion.next_completion = 0;

    /* initialize peer_receive in device memory */
    device_peer.completion.sequence = RingN;
    _movdir64b((void *) &ishmemi_completions[RingN], &device_peer);

    /* initialize the first completion entry to 0 in device memory */
    device_peer.completion.sequence = 0xffffffff;
    device_peer.completion.lock = 1;
    _movdir64b((void *) &ishmemi_completions[0], &device_peer);

    /* This is the constructor for omeshmemi_cpu_info-> ring
     * should be done as a real object constructor
     * TODO
     */
    ishmemi_cpu_info->ring.recvbuf = ishmemi_sendbuf;
    ishmemi_cpu_info->ring.next_receive = RingN;
    ishmemi_cpu_info->ring.atomic_lock = 0;

    /* initialize the upcall table.  This is a version of ishmemi_proxy_funcs
     * that has cutover functions replaced by new implementations
     * and also upcall functions that are not implemented by the runtime at all
     */
    ret = ishmemi_proxy_func_init();
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    /* Spawn the proxy thread */
    /* TODO figure out what the proxy_thread affinity should be according to topology
     * the thread should be on the same socket as the PCIe to the device
     */
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
        ishmemi_mmap_gpu_info->ring.sendbuf = nullptr;
        ISHMEMI_FREE(ishmemi_usm_free, ishmemi_sendbuf);

        if (ishmemi_mmap_gpu_info->messages != nullptr) {
            ISHMEMI_FREE(ishmemi_usm_free, ishmemi_msg_queue);
            ishmemi_mmap_gpu_info->messages = nullptr;
        }
    }
    ishmemi_proxy_func_fini();
    return 0;
}
