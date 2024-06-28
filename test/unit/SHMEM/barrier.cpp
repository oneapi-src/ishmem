/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause */

#include <CL/sycl.hpp>
#include <unistd.h>
#include <thread>
#include <ctime>
#include <stdio.h>
#include <common.h>

double tsc_frequency;
int my_pe;
std::thread timeout_thread;
bool please_return = false;
bool please_restart = false;

long *mypsync;

void do_timeout()
{
    long *psync = mypsync;
    printf("[%d] timeout psync is %lx %lx %lx %lx\n", my_pe, psync[0], psync[1], psync[2],
           psync[3]);
    fflush(stdout);
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
    /* it is possible to use local variables that remain in-scope but you have to work to convice
     * SYCL not to translate the pointers */
    if (threads == 0) {
        // printf("[%d] launching barrier\n", my_pe);
        e = q.submit([&](sycl::handler &h) {
            h.single_task([=]() {
                for (size_t i = 0; i < count; i += 1) {
                    ishmem_barrier_all();
                }
            });
        });
    } else {
        e = q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(threads), sycl::range<1>(threads)),
                           [=](sycl::nd_item<1> it) {
                               auto grp = it.get_group();
                               for (size_t i = 0; i < count; i += 1) {
                                   ishmemx_barrier_all_work_group(grp);
                               }
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
    ishmemx_attr_t attr = {};
    test_init_attr(&attr);
    ishmemx_init_attr(&attr);

    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
    my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();
    sycl::queue q(prop_list);
    std::cout << "Host: PE " << my_pe << " of " << npes << std::endl;
    ishmem_barrier_all();

    /*mypsync = (long *) (((uintptr_t) ishmemi_mmap_gpu_info->barrier_all_psync) -
                        ((uintptr_t) ishmemi_heap_base) + ((uintptr_t) ishmemi_mmap_heap_base));
    printf("[%d] heap_base %p mmap_heap_base %p barrier_all_psync %p mypsync %p\n", my_pe,
           ishmemi_heap_base, ishmemi_mmap_heap_base, ishmemi_mmap_gpu_info->barrier_all_psync,
           mypsync);*/
    please_restart = false;
    please_return = false;
    timeout_thread = std::thread(barrier_timeout, (void *) NULL);
    fflush(stdout);
    struct CMD *cmd = (struct CMD *) runtime_calloc(2, sizeof(struct CMD));
    double duration = 0;

    if (my_pe == 0) {
        int threads = 0;
        while (threads <= 1024) {
            int count = 1;
            while (count < (1 << 24)) {
                cmd[0].cmd = cmd_run;
                cmd[0].iter = count;
                cmd[0].threads = threads;
                runtime_broadcast(&cmd[1], &cmd[0], sizeof(struct CMD), 0);
                duration = runkernel_barrier(q, (size_t) threads, (size_t) count);
                if (duration > 0.01) break;
                count <<= 1;
            }
            cmd[0].cmd = cmd_print;
            cmd[0].iter = count;
            cmd[0].threads = threads;
            // shmem_long_broadcast(SHMEM_TEAM_WORLD, (long *) &cmd[1], (long *) &cmd[0],
            // sizeof(struct CMD)/sizeof(long), 0);
            printf("[%d] barrier threads %d count %d duration %f usec %f\n", my_pe, threads, count,
                   duration, 1000000 * duration / count);
            fflush(stdout);
            threads = (threads == 0) ? 1 : threads << 1;
        }
        cmd[0].cmd = cmd_exit;
        runtime_broadcast(&cmd[1], &cmd[0], sizeof(struct CMD), 0);
    } else {
        while (1) {
            runtime_broadcast(&cmd[1], &cmd[0], sizeof(struct CMD), 0);
            if (cmd[1].cmd == cmd_run) {
                duration = runkernel_barrier(q, (size_t) cmd[1].threads, (size_t) cmd[1].iter);
            } else if (cmd[1].cmd == cmd_print) {
                printf("[%d] barrier threads %ld count %ld duration %f usec %f\n", my_pe,
                       cmd[1].threads, cmd[1].iter, duration,
                       (1000000 * duration / (double) cmd[1].iter));
            } else {
                break;
            }
        }
    }
    printf("[%d] barrier returned\n", my_pe);
    fflush(stdout);
    runtime_free(cmd);
    please_return = true;
    timeout_thread.join();
    ishmem_finalize();
    return 0;
}
