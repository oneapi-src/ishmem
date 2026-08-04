/* Copyright (C) 2023-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ishmem.h>
#include <unistd.h>
#include <ctime>
#include <stdio.h>
#include <common.h>

int my_pe;
int npes;

double runkernel_fcollect(sycl::queue q, int threads, long *dest, long *source, size_t nelems,
                          int count)
{
    sycl::event e;
    if (threads == 0) {
        e = q.submit([&](sycl::handler &h) {
            h.single_task([=]() {
                for (int i = 0; i < count; i += 1) {
                    ishmem_long_fcollect(dest, source, nelems);
                }
            });
        });
    } else {
        e = q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) threads),
                                             sycl::range<1>((size_t) threads)),
                           [=](sycl::nd_item<1> it) {
                               auto grp = it.get_group();
                               for (int i = 0; i < count; i += 1) {
                                   ishmemx_long_fcollect_work_group(dest, source, nelems, grp);
                               }
                           });
        });
    }
    e.wait_and_throw();
    return (getduration(e));
}

#define BUFSIZE (1L << 20)

int main(int argc, char **argv)
{
    ishmem_init();

    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

    my_pe = ishmem_my_pe();
    int local_my_pe = my_pe;
    npes = ishmem_n_pes();
    sycl::queue q(prop_list);

    std::cout << "Host: PE " << my_pe << " of " << npes << std::endl;
    long nlongs = BUFSIZE / sizeof(long);
    long *source = (long *) ishmem_malloc(BUFSIZE);
    CHECK_ALLOC(source);
    long *dest = (long *) ishmem_malloc(BUFSIZE * (size_t) npes);
    CHECK_ALLOC(dest);
    long *host = (long *) sycl::aligned_alloc_host(4096, BUFSIZE * (size_t) npes, q);
    CHECK_ALLOC(host);
    q.parallel_for(sycl::range((size_t) nlongs), [=](sycl::id<1> idx) {
         source[idx] = (1L << (40 + local_my_pe)) + (long) idx;
     }).wait_and_throw();
    q.memcpy(host, source, BUFSIZE).wait_and_throw();
    size_t hostmask = (1L << (40 + local_my_pe));
    for (size_t idx = 0; idx < nlongs; idx += 1) {
        long expected = (long) (hostmask + idx);
        if (host[idx] != expected) {
            printf("pe %d source idx %ld got %08lx expected %08lx\n", my_pe, idx, host[idx],
                   expected);
        }
    }
    double *devduration = (double *) ishmem_calloc(1, sizeof(double));
    CHECK_ALLOC(devduration);
    printf("[%d] source %p dest %p \n", my_pe, source, dest);
    fflush(stdout);
    if (my_pe == 0) {
        printf("csv,pe,fcollect,threads,nelems,count,duration,calls/sec\n");
    }
    ishmem_barrier_all();

    for (size_t nelems = 1; nelems <= nlongs; nelems <<= 1) {
        int threads = 0;
        while (threads <= 256) {
            int count = 1;
            double duration;
            q.memset(dest, 0xff, nelems * sizeof(long) * (size_t) npes).wait_and_throw();
            ishmem_barrier_all();
            while (count < (1 << 10)) {
                duration = runkernel_fcollect(q, threads, dest, source, nelems, count);
                if (count == 1) {
                    long error = 0;
                    q.memcpy(host, dest, sizeof(long) * nelems * (size_t) npes).wait_and_throw();
                    for (size_t pe = 0; pe < (size_t) npes; pe += 1) {
                        long hostmask = (1L) << (pe + 40);
                        for (size_t idx = 0; idx < nelems; idx += 1) {
                            long expected = hostmask + (long) idx;
                            if (host[(pe * nelems) + idx] != expected) {
                                error += 1;
                                printf(
                                    "[%d] fcollect threads %d nelems %ld idx %ld offset %ld got "
                                    "%08lx expected %08lx\n",
                                    my_pe, threads, nelems, idx, (pe * nelems) + idx,
                                    host[(pe * nelems) + idx], expected);
                                if (error >= 16) break;
                            }
                        }
                    }
                }
                double d0 = duration_broadcast(q, devduration, duration);
                if (d0 > 0.01) break;
                count <<= 1;
            }
            printf("csv,%d,fcollect,%d,%ld,%d,%f,%f\n", my_pe, threads, nelems, count, duration,
                   count / duration);
            fflush(stdout);
            threads = (threads == 0) ? 1 : threads << 1;
        }
    }
    printf("Calling finalize\n");
    fflush(stdout);
    ishmem_free(source);
    ishmem_free(dest);
    ishmem_free(devduration);
    sycl::free(host, q);
    ishmem_finalize();
    return 0;
}
