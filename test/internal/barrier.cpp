/* Copyright (C) 2023-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause */

#include <unistd.h>
#include <thread>
#include <ctime>
#include <stdio.h>
#include "proxy_impl.h"
#include "ishmem.h"
#include "ishmem/config.h"
#include "common.h"

double tsc_frequency;
int my_pe;
std::thread timeout_thread;
bool please_return = false;
bool please_restart = false;

unsigned long *sync_psync;
unsigned long *barrier_psync;
extern ishmemi_ringcompletion_t *ishmemi_ring_host_completions; /* host map of completions */
unsigned long *status;

extern void do_timeout()
{
    static int timeout_count = 0;
    int my_pe = ishmem_my_pe();
    printf("[%d] timeout\n", my_pe);
    printf("[%d] sync index %x %016lx %016lx %016lx %016lx\n", my_pe,
           ishmemi_mmap_gpu_info->sync_index, sync_psync[0], sync_psync[1], sync_psync[2],
           sync_psync[3]);
    printf("[%d] barrier index %x %016lx %016lx %016lx %016lx\n", my_pe,
           ishmemi_mmap_gpu_info->barrier_index, barrier_psync[0], barrier_psync[1],
           barrier_psync[2], barrier_psync[3]);
    printf("[%d] next_send %08x next_receive %08x, peer_next_receive %08x\n", my_pe,
           ishmemi_mmap_gpu_info->ring.get_next_send(), ishmemi_cpu_info->ring.get_next_receive(),
           ishmemi_ring_host_completions[RING_SIZE].completion.sequence);
    printf("[%d] status  %016lx %016lx %016lx %016lx\n", my_pe, status[0], status[1], status[2],
           status[3]);
    printf("[%d] status  %lu %lu %lu %lu\n", my_pe, status[0], status[1], status[2], status[3]);
    fflush(stdout);
    timeout_count += 1;
    if (timeout_count > 3) exit(1);
}

void barrier_timeout(void *arg)
{
    clock_t t0 = clock();
    while (!please_return) {
        if (please_restart) t0 = clock();
        clock_t t1 = clock();
        if (difftime(t1, t0) > (8.0 * CLOCKS_PER_SEC)) {
            do_timeout();
            t0 = clock();
        }
    }
}

double runkernel_barrier(sycl::queue q, size_t threads, size_t count)
{
    sycl::event e;
    unsigned long *lstatus = status;
    /* it is possible to use local variables that remain in-scope but you have to work to convice
     * SYCL not to translate the pointers */
    lstatus[2] = __LINE__;
    if (threads == 0) {
        // printf("[%d] launching barrier\n", my_pe);
        e = q.submit([&](sycl::handler &h) {
            h.single_task([=]() {
                lstatus[0] = count;
                for (size_t i = 0; i < count; i += 1) {
                    lstatus[2] = __LINE__;
                    lstatus[1] = i;
                    ishmem_sync_all();
                }
                lstatus[2] = __LINE__;
            });
        });
    } else {
        e = q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(threads), sycl::range<1>(threads)),
                           [=](sycl::nd_item<1> it) {
                               auto grp = it.get_group();
                               lstatus[0] = count;
                               for (size_t i = 0; i < count; i += 1) {
                                   lstatus[2] = __LINE__;
                                   lstatus[1] = i;
                                   ishmemx_sync_all_work_group(grp);
                               }
                               lstatus[2] = __LINE__;
                           });
        });
    }
    e.wait_and_throw();
    return (getduration(e));
}

#define cmd_run   0
#define cmd_print 1
#define cmd_exit  2

// PE 0 tells other PEs what to run
struct CMD {
    long cmd;
    long iter;
    long threads;
};

int main(int argc, char **argv)
{
    ishmem_init();

    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
    my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();
    sycl::queue q(prop_list);
    std::cout << "Host: PE " << my_pe << " of " << npes << std::endl;
    ishmem_barrier_all();

    sync_psync =
        (unsigned long *) (((uintptr_t) ishmemi_mmap_gpu_info->sync_all_psync) -
                           ((uintptr_t) ishmemi_heap_base) + ((uintptr_t) ishmemi_mmap_heap_base));
    barrier_psync =
        (unsigned long *) (((uintptr_t) ishmemi_mmap_gpu_info->barrier_all_psync) -
                           ((uintptr_t) ishmemi_heap_base) + ((uintptr_t) ishmemi_mmap_heap_base));
#if 0
    printf("[%d] heap_base %p mmap_heap_base %p barrier_all_psync %p pbarrier_psync %p\n", my_pe,
           ishmemi_heap_base, ishmemi_mmap_heap_base, ishmemi_mmap_gpu_info->barrier_all_psync,
           barrier_psync);
    printf("[%d] heap_base %p mmap_heap_base %p sync_all_psync %p psync_psync %p\n", my_pe,
           ishmemi_heap_base, ishmemi_mmap_heap_base, ishmemi_mmap_gpu_info->sync_all_psync,
           sync_psync);
#endif
    please_restart = false;
    please_return = false;
    timeout_thread = std::thread(barrier_timeout, (void *) NULL);
    fflush(stdout);
    struct CMD *cmd = (struct CMD *) ishmemi_test_runtime->calloc(2, sizeof(struct CMD));
    status = (unsigned long *) sycl::malloc_host(4096, q);
    double duration = 0;

    if (my_pe == 0) {
        int threads = 0;
        while (threads <= 1024) {
            int count = 1;
            while (count < (1 << 24)) {
                cmd[0].cmd = cmd_run;
                cmd[0].iter = count;
                cmd[0].threads = threads;
                status[3] = __LINE__;
                ishmemi_test_runtime->broadcast(&cmd[1], &cmd[0], sizeof(struct CMD), 0);
                status[3] = __LINE__;
                duration = runkernel_barrier(q, (size_t) threads, (size_t) count);
                if (duration > 0.01) break;
                count <<= 1;
            }
            cmd[0].cmd = cmd_print;
            cmd[0].iter = count;
            cmd[0].threads = threads;
            ishmemi_test_runtime->broadcast(&cmd[1], &cmd[0], sizeof(struct CMD), 0);
            printf("[%d] barrier threads %d count %d duration_us %f bw_mb %f\n", my_pe, threads,
                   count, duration, 1000000 * duration / count);
            fflush(stdout);
            threads = (threads == 0) ? 1 : threads << 1;
        }
        status[3] = __LINE__;
        cmd[0].cmd = cmd_exit;
        ishmemi_test_runtime->broadcast(&cmd[1], &cmd[0], sizeof(struct CMD), 0);
        status[3] = __LINE__;
    } else {
        while (1) {
            status[3] = __LINE__;

            ishmemi_test_runtime->broadcast(&cmd[1], &cmd[0], sizeof(struct CMD), 0);
            status[3] = __LINE__;
            if (cmd[1].cmd == cmd_run) {
                duration = runkernel_barrier(q, (size_t) cmd[1].threads, (size_t) cmd[1].iter);
            } else if (cmd[1].cmd == cmd_print) {
                printf("[%d] barrier threads %ld count %ld duration_us %f bw_mb %f\n", my_pe,
                       cmd[1].threads, cmd[1].iter, duration,
                       (1000000 * duration / (double) cmd[1].iter));
            } else {
                status[3] = __LINE__;
                break;
            }
        }
    }
    printf("[%d] barrier returned\n", my_pe);
    fflush(stdout);
    ishmemi_test_runtime->free(cmd);
    please_return = true;
    timeout_thread.join();
    ishmem_finalize();
    return 0;
}
