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

double runkernel_collect(sycl::queue q, int threads, long *dest, long *source, size_t nelems,
                         int count)
{
    sycl::event e;
    if (threads == 0) {
        e = q.single_task([=]() {
            for (int i = 0; i < count; i += 1) {
                ishmem_long_collect(dest, source, nelems);
            }
        });
    } else {
        e = q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) threads), sycl::range<1>((size_t) threads)),
            [=](sycl::nd_item<1> it) {
                auto grp = it.get_group();
                for (int i = 0; i < count; i += 1) {
                    ishmemx_long_collect_work_group(dest, source, nelems, grp);
                }
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
    size_t nlongs = BUFSIZE / sizeof(long);
    long *source = (long *) ishmem_malloc(BUFSIZE);
    CHECK_ALLOC(source);
    long *dest = (long *) ishmem_malloc(BUFSIZE * (size_t) npes);
    CHECK_ALLOC(dest);
    long *host = (long *) sycl::aligned_alloc_host(4096, BUFSIZE * (size_t) npes, q);
    CHECK_ALLOC(host);
    q.parallel_for(sycl::range(nlongs), [=](sycl::id<1> idx) {
         source[idx] = (long) ((1L << (40 + local_my_pe)) + idx);
     }).wait_and_throw();
    q.memcpy(host, source, BUFSIZE).wait_and_throw();
    long hostmask = (1L << (40 + local_my_pe));
    for (size_t idx = 0; idx < nlongs; idx += 1) {
        long expected = hostmask + (long) idx;
        if (host[idx] != expected) {
            printf("pe %d source idx %ld got %08lx expected %08lx\n", my_pe, idx, host[idx],
                   expected);
        }
    }
    double *devduration = (double *) ishmem_calloc(1, sizeof(double));
    CHECK_ALLOC(devduration);

    printf("[%d] source %p dest %p \n", my_pe, source, dest);
    fflush(stdout);
    size_t *extra_pe_nelems =
        (size_t *) sycl::aligned_alloc_host(4096, sizeof(size_t) * (size_t) npes, q);
    CHECK_ALLOC(extra_pe_nelems);
    size_t total_extra_nelems = 0;
    for (size_t pe = 0; pe < (size_t) npes; pe += 1) {
        extra_pe_nelems[pe] = pe + 10L;
        total_extra_nelems += extra_pe_nelems[pe];
    }
    printf("[%d] extra_pe_nelems %ld total_extra_nelems %ld\n", my_pe, extra_pe_nelems[my_pe],
           total_extra_nelems);
    if (my_pe == 0) {
        printf("csv,pe,collect,threads,nelems,count,duration,calls/sec\n");
    }
    /* starting collect */
    for (size_t nelems = 1; nelems <= (BUFSIZE / (2 * sizeof(long))); nelems <<= 1) {
        int threads = 0;
        while (threads <= 256) {
            int count = 1;
            double duration;
            size_t total_size_bytes =
                (nelems * (size_t) npes * sizeof(long)) + (total_extra_nelems * sizeof(long));
            q.memset(dest, 0xff, total_size_bytes).wait_and_throw();
            ishmem_barrier_all();
            while (count < (1 << 24)) {
                duration = runkernel_collect(q, threads, dest, source,
                                             nelems + extra_pe_nelems[my_pe], count);
                if (count == 1) {
                    long error = 0;
                    q.memcpy(host, dest, total_size_bytes).wait_and_throw();
                    size_t offset = 0;
                    for (size_t pe = 0; pe < npes; pe += 1) {
                        long hostmask = (1L) << (pe + 40);
                        for (size_t idx = 0; idx < nelems + extra_pe_nelems[pe]; idx += 1) {
                            long expected = hostmask + (long) idx;
                            if (host[offset] != expected) {
                                error += 1;
                                printf(
                                    "[%d] collect threads %d nelems %ld idx %ld offset %ld got "
                                    "%08lx expected %08lx\n",
                                    my_pe, threads, nelems, idx, offset, host[offset], expected);
                            }
                            if (error >= 16) break;
                            offset += 1;
                        }
                        if (error >= 16) break;
                    }
                    // check_collect_nelems(q, host, nelems);
                }
                double d0 = duration_broadcast(q, devduration, duration);
                if (d0 > 0.01) break;
                count <<= 1;
            }
            printf("csv,%d,collect,%d,%ld,%d,%f,%f\n", my_pe, threads, nelems, count, duration,
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
