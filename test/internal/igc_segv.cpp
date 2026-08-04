/* Copyright (C) 2023-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause */

#include "ishmemx.h"

void runkernel_reduce_work_group(sycl::queue q, size_t threads, long *dest, long *source,
                                 size_t nreduce, int count)
{
    auto e = q.parallel_for(sycl::nd_range<1>(sycl::range<1>(threads), sycl::range<1>(threads)),
                            [=](sycl::nd_item<1> it) {
                                auto grp = it.get_group();
                                for (int i = 0; i < count; i += 1) {
                                    ishmemx_long_sum_reduce_work_group(dest, source, nreduce, grp);
                                }
                            });
    e.wait_and_throw();
}

#define BUFSIZE (1L << 22)

int main(int argc, char **argv)
{
    sycl::queue q;

    int pe = 0;
    long *source = (long *) sycl::malloc_device(BUFSIZE, q);

    auto e = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range(BUFSIZE / sizeof(long)), [=](sycl::id<1> idx) {
            source[idx] = (1L << (40 + pe)) + static_cast<long>(idx);
        });
    });
    e.wait_and_throw();

    return 0;
}
