/* Copyright (C) 2023-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <unistd.h>
#include <common.h>
#include "proxy_impl.h"

long *gdest = (long *) 0x12345;
long *gsrc = (long *) 0x123d5;
ptrdiff_t gdst = 3L;
ptrdiff_t gsst = 5L;
size_t gnelems = 17;
int gpe = 0;

template <typename T>
SYCL_EXTERNAL void init_ishmem_iget(T *dest, const T *src, ptrdiff_t dst, ptrdiff_t sst,
                                    size_t nelems, int pe)
{
#ifdef __SYCL_DEVICE_ONLY__
    size_t nbytes = nelems * sizeof(T);

    ishmemi_request_t req = {
        .dest_pe = pe,
        .src = src,
        .dst = dest,
        .nelems = nbytes,
        .dst_stride = dst,
        .src_stride = sst,
        .op = NOP,
        .type = MEM,
    };
    ishmemi_proxy_blocking_request(req);
#endif
}

template <typename T>
SYCL_EXTERNAL void assign_ishmem_iget(T *dest, const T *src, ptrdiff_t dst, ptrdiff_t sst,
                                      size_t nelems, int pe)
{
#ifdef __SYCL_DEVICE_ONLY__
    size_t nbytes = nelems * sizeof(T);

    ishmemi_request_t req;
    req.dest_pe = pe;
    req.src = src;
    req.dst = dest;
    req.nelems = nbytes;
    req.dst_stride = dst;
    req.src_stride = sst;
    req.op = NOP;
    req.type = MEM;

    ishmemi_proxy_blocking_request(req);
#endif
}

double runkernel_init_iget(sycl::queue q, size_t count)
{
    long *dest = gdest;
    long *src = gsrc;
    ptrdiff_t dst = gdst;
    ptrdiff_t sst = gsst;
    size_t nelems = gnelems;
    int pe = gpe;

    auto e = q.single_task([=]() {
        for (size_t iter = 0; iter < count; iter += 1) {
            init_ishmem_iget(dest, src, dst, sst, nelems, pe);
        }
    });
    e.wait_and_throw();
    return (getduration(e));
}

double runkernel_assign_iget(sycl::queue q, size_t count)
{
    long *dest = gdest;
    long *src = gsrc;
    ptrdiff_t dst = gdst;
    ptrdiff_t sst = gsst;
    size_t nelems = gnelems;
    int pe = gpe;

    auto e = q.single_task([=]() {
        for (size_t iter = 0; iter < count; iter += 1) {
            assign_ishmem_iget(dest, src, dst, sst, nelems, pe);
        }
    });
    e.wait_and_throw();
    return (getduration(e));
}

int main(int argc, char **argv)
{
    ishmem_init();

    setbuf(stdout, NULL);  // set line buffering
    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

    sycl::queue q(prop_list);

    size_t count = 16;
    double duration;
    printf("testing init_ishmem_iget\n");
    while (count < (1 << 24)) {
        duration = runkernel_init_iget(q, count);
        if (duration > 0.02) break;
        count <<= 1;
        printf("init_iget, count %lu, duration %f, calls per second %lu, each call %.1f nsec\n",
               count, duration, (long unsigned) ((double) count / duration),
               (duration * NSEC_IN_SEC) / (double) count);
    }

    count = 16;
    printf("testing assign_ishmem_iget\n");
    while (count < (1 << 24)) {
        duration = runkernel_assign_iget(q, count);
        if (duration > 0.02) break;
        count <<= 1;
        printf("assign_iget, count %lu, duration %f, calls per second %lu, each call %.1f nsec\n",
               count, duration, (long unsigned) ((double) count / duration),
               (duration * NSEC_IN_SEC) / (double) count);
    }

    ishmem_finalize();
    return 0;
}
