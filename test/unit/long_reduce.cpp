/* Copyright (C) 2023 Intel Corporation
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

double runkernel_reduce(sycl::queue q, size_t threads, long *dest, long *source, size_t nreduce,
                        int count)
{
    auto e = q.submit([&](sycl::handler &h) {
        h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
            grp.parallel_for_work_item([&](sycl::h_item<1> it) {
                for (int i = 0; i < count; i += 1) {
                    ishmem_long_sum_reduce(dest, source, nreduce);
                }
            });
        });
    });
    e.wait_and_throw();
    return (getduration(e));
}

double runkernel_reduce_work_group(sycl::queue q, size_t threads, long *dest, long *source,
                                   size_t nreduce, int count)
{
    auto e = q.submit([&](sycl::handler &h) {
        h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
            grp.parallel_for_work_item([&](sycl::h_item<1> it) {
                for (int i = 0; i < count; i += 1) {
                    ishmemx_long_sum_reduce_work_group(dest, source, nreduce, grp);
                }
            });
        });
    });
    e.wait_and_throw();
    return (getduration(e));
}

#define BUFSIZE (1L << 22)

int main(int argc, char **argv)
{
    ishmem_init();
    validate_runtime();

    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

    my_pe = ishmem_my_pe();
    int local_my_pe = my_pe;
    int npes = ishmem_n_pes();
    int local_npes = npes;
    sycl::queue q(prop_list);

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;

    std::cout << "Host: PE " << my_pe << " of " << npes << std::endl;
    long *source = (long *) ishmem_malloc(BUFSIZE);
    CHECK_ALLOC(source);
    long *dest = (long *) ishmem_malloc(BUFSIZE);
    CHECK_ALLOC(dest);
    long *host = (long *) sycl::aligned_alloc_host(4096, BUFSIZE, q);
    CHECK_ALLOC(host);
    auto e = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range(BUFSIZE / sizeof(long)), [=](sycl::id<1> idx) {
            source[idx] = (1L << (40 + local_my_pe)) + static_cast<long>(idx);
        });
    });
    e.wait_and_throw();
    q.memcpy(host, source, BUFSIZE).wait_and_throw();
    long hostmask = (1L << (40 + local_my_pe));
    for (size_t idx = 0; idx < (BUFSIZE / sizeof(long)); idx += 1) {
        long expected = hostmask + static_cast<long>(idx);
        if (host[idx] != expected) {
            printf("pe %d idx %ld got %08lx expected %08lx\n", my_pe, idx, host[idx], expected);
        }
    }
    memset(host, 0, BUFSIZE);
    ishmem_barrier_all();

    /*mypsync = (long *) (((uintptr_t) ishmemi_mmap_gpu_info->barrier_all_psync) -
                        ((uintptr_t) ishmemi_heap_base) + ((uintptr_t) ishmemi_mmap_heap_base));
    printf("[%d] heap_base %p mmap_heap_base %p barrier_all_psync %p mypsync %p\n", my_pe,
           ishmemi_heap_base, ishmemi_mmap_heap_base, ishmemi_mmap_gpu_info->barrier_all_psync,
           mypsync);*/
    printf("[%d] source %p dest %p \n", my_pe, source, dest);
    please_restart = false;
    please_return = false;
    printf("csv,pe,npes,mode,threads,nreduce,count,calls/second,bwMB\n");
    //    timeout_thread = std::thread(barrier_timeout<, (void *) NULL);
    double *duration = (double *) ishmemi_test_runtime->calloc(1, sizeof(double));
    CHECK_ALLOC(duration);
    double *d0 = (double *) ishmemi_test_runtime->calloc(1, sizeof(double));
    CHECK_ALLOC(d0);
    for (size_t nreduce = 1; nreduce <= (BUFSIZE / sizeof(long)); nreduce <<= 1) {
        for (size_t threads = 1; threads <= 256; threads <<= 1) {
            int count = 1;
            while (count < (1 << 24)) {
                ishmem_barrier_all();
                if (threads > 1) {
                    *duration =
                        runkernel_reduce_work_group(q, threads, dest, source, nreduce, count);
                } else {
                    *duration = runkernel_reduce(q, threads, dest, source, nreduce, count);
                }
                if (count == 1) {
                    q.memcpy(host, dest, sizeof(long) * nreduce).wait_and_throw();
                    long hostmask = ((1L << npes) - 1) << 40;
                    for (size_t idx = 0; idx < nreduce; idx += 1) {
                        long expected = hostmask + ((static_cast<long>(idx)) * local_npes);
                        if (host[idx] != expected) {
                            printf(
                                "pe %d threads %zu nreduce %ld idx %ld got %08lx expected %08lx\n",
                                my_pe, threads, nreduce, idx, host[idx], expected);
                        }
                    }
                }
                ishmemi_test_runtime->broadcast(d0, duration, sizeof(double), 0);
                if (*d0 > 0.01) break;
                count <<= 1;
            }
            if (my_pe == 0) {
                double bw = (double) (static_cast<size_t>(count) * sizeof(long) *
                                      static_cast<size_t>(nreduce)) /
                            (*duration * 1000000.0);
                printf("csv,%d,%d,outofplace,%zu,%ld,%d,%f,%f\n", my_pe, npes, threads, nreduce,
                       count, count / *duration, bw);
                fflush(stdout);
            }
        }
    }
    // now test in place
    for (size_t nreduce = 1; nreduce <= (BUFSIZE / sizeof(long)); nreduce <<= 1) {
        for (size_t threads = 1; threads <= 256; threads <<= 1) {
            int count = 1;
            auto e = q.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::range(BUFSIZE / sizeof(long)), [=](sycl::id<1> idx) {
                    source[idx] = (1L << (40 + local_my_pe)) + static_cast<long>(idx);
                });
            });
            e.wait_and_throw();
            while (count < (1 << 24)) {
                if (threads > 1) {
                    *duration =
                        runkernel_reduce_work_group(q, threads, source, source, nreduce, count);
                } else {
                    *duration = runkernel_reduce(q, threads, source, source, nreduce, count);
                }
                if (count == 1) {
                    q.memcpy(host, dest, sizeof(long) * nreduce).wait_and_throw();
                    long hostmask = ((1L << npes) - 1) << 40;
                    for (size_t idx = 0; idx < nreduce; idx += 1) {
                        long expected = hostmask + (static_cast<long>(idx) * local_npes);
                        if (host[idx] != expected) {
                            printf(
                                "[%d] threads %zu nreduce %ld idx %ld got %08lx expected %08lx\n",
                                my_pe, threads, nreduce, idx, host[idx], expected);
                        }
                    }
                }
                ishmemi_test_runtime->broadcast(d0, duration, sizeof(double), 0);
                if (*d0 > 0.01) break;
                count <<= 1;
            }
            if (my_pe == 0) {
                double bw = (double) (static_cast<size_t>(count) * sizeof(long) *
                                      static_cast<size_t>(nreduce)) /
                            (*duration * 1000000.0);
                printf("csv,%d,%d,inplace,%zu,%ld,%d,%f,%f\n", my_pe, npes, threads, nreduce, count,
                       count / *duration, bw);
                fflush(stdout);
            }
        }
    }
    please_return = true;
    // timeout_thread.join();
    ishmemi_test_runtime->free(duration);
    ishmemi_test_runtime->free(d0);
    ishmem_barrier_all();
    printf("[%d] Calling finalize\n", my_pe);
    fflush(stdout);
    ishmem_finalize();
    return 0;
}
