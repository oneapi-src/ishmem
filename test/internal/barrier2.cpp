/* Copyright (C) 2023-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause */

#include <unistd.h>
#include <thread>
#include <ctime>
#include <stdio.h>
#include <common.h>

int my_pe;
std::thread timeout_thread;
bool please_return = false;
bool please_restart = false;

long *mypsync;

void do_timeout()
{
    long psync = *mypsync;
    printf("[%d] timeout psync is %ld %016lx\n", my_pe, psync, psync);
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

double runkernel_barrier(sycl::queue q, size_t threads, int count)
{
    sycl::event e;
    if (threads == 0) {
        // printf("[%d] launching barrier\n", my_pe);
        e = q.submit([&](sycl::handler &h) {
            h.single_task([=]() {
                for (int i = 0; i < count; i += 1) {
                    ishmem_barrier_all();
                }
            });
        });
    } else {
        e = q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(threads), sycl::range<1>(threads)),
                           [=](sycl::nd_item<1> it) {
                               auto grp = it.get_group();
                               for (int i = 0; i < count; i += 1) {
                                   ishmemx_barrier_all_work_group(grp);
                               }
                           });
        });
    }
    e.wait_and_throw();
    return (getduration(e));
}

double my_broadcast(sycl::queue q, int root, int npes, double *duration, double value)
{
    double result;
    ishmem_barrier_all();
    if (root) {
        q.memcpy(duration, &value, sizeof(double)).wait_and_throw();
        sycl::event e = q.submit([&](sycl::handler &h) {
            h.single_task([=]() {
                int my_pe = ishmem_my_pe();
                for (int pe = 0; pe < npes; pe += 1) {
                    if (pe != my_pe) ishmem_double_put(duration, duration, 1, pe);
                }
            });
        });
        e.wait_and_throw();
    }
    ishmem_barrier_all();
    q.memcpy(&result, duration, sizeof(double)).wait_and_throw();
    return (result);
}

int main(int argc, char **argv)
{
    ishmem_init();

    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

    my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();
    sycl::queue q(prop_list);
    std::cout << "Host: PE " << my_pe << " of " << npes << std::endl;
    ishmem_barrier_all();

    /*mypsync = (long *) (((uintptr_t) ishmemi_mmap_gpu_info->barrier_all_psync) -
                        ((uintptr_t) ishmemi_heap_base) + ((uintptr_t) ishmemi_mmap_heap_base));*/
    // printf("[%d] heap_base %p mmap_heap_base %p barrier_all_psync %p mypsync %p\n", my_pe,
    //        ishmemi_heap_base, ishmemi_mmap_heap_base,
    //        ishmemi_mmap_gpu_info->barrier_all_psync, mypsync);
    please_restart = false;
    please_return = false;
    timeout_thread = std::thread(barrier_timeout, (void *) NULL);
    double *devduration = (double *) ishmem_calloc(1, sizeof(double));
    CHECK_ALLOC(devduration);
    size_t threads = 0;
    ishmem_barrier_all();
    while (threads < 256) {
        int count = 1;
        double duration;
        while (count < (1 << 24)) {
            duration = runkernel_barrier(q, threads, count);
            double d0 = my_broadcast(q, my_pe == 0, npes, devduration, duration);
            if (my_pe == 0) assert(d0 == duration);
            if (d0 > 0.01) break;
            count <<= 1;
        }
        if (1) {
            printf("[%d] barrier threads %zu count %d calls per second %f\n", my_pe, threads, count,
                   count / duration);
            fflush(stdout);
        }
        threads = (threads == 0) ? 1 : threads << 1;
    }
    please_return = true;
    timeout_thread.join();
    ishmem_free(devduration);
    // printf("[%d] Calling finalize\n", my_pe);
    ishmem_finalize();
    return 0;
}
