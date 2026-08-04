/* Copyright (C) 2023-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <unistd.h>
#include <common.h>

double runkernel_nop(sycl::queue q, size_t threads, size_t count)
{
    size_t iter = count / threads;
    sycl::event e;
    if (threads == 1) {
        e = q.single_task([=]() {
            for (size_t i = 0; i < iter; i += 1) {
                ishmemx_nop();
            }
        });
    } else {
        e = q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) threads), sycl::range<1>((size_t) threads)),
            [=](sycl::nd_item<1> it) {
                for (size_t i = 0; i < iter; i += 1) {
                    ishmemx_nop();
                }
            });
    }
    e.wait_and_throw();
    return (getduration(e));
}

double runkernel_nop_no_r(sycl::queue q, size_t threads, size_t count)
{
    size_t iter = count / threads;
    sycl::event e;
    if (threads == 1) {
        e = q.single_task([=]() {
            for (size_t i = 0; i < iter; i += 1) {
                ishmemx_nop_no_r();
            }
        });
    } else {
        e = q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) threads), sycl::range<1>((size_t) threads)),
            [=](sycl::nd_item<1> it) {
                for (size_t i = 0; i < iter; i += 1) {
                    ishmemx_nop_no_r();
                }
            });
    }
    e.wait_and_throw();
    return (getduration(e));
}

int main(int argc, char **argv)
{
    ishmem_init();

    setbuf(stdout, NULL);  // set line buffering
    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();
    sycl::queue q(prop_list);

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;

    std::cout << "Host: PE " << my_pe << " of " << npes << std::endl;

    ishmem_barrier_all();
    std::cout << "finished barrier" << std::endl;

    auto e2 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            ishmemx_debug_test();
            ishmemx_debug_test();
        });
    });
    e2.wait_and_throw();
    ishmem_barrier_all();
    std::cout << "Returned from ishmem_debug_test kernel" << std::endl;

    for (size_t threads = 1; threads <= 1024; threads <<= 1) {
        double duration;
        int tries = (threads == 1) ? 10 : 1;
        for (int trial = 0; trial < tries; trial += 1) {
            size_t count = threads;
            while (count < (1 << 24)) {
                duration = runkernel_nop(q, threads, count);
                if (duration > 0.02) break;
                count <<= 1;
            }
            printf("nop with %lu threads, count %lu, calls per second %lu, each call %.1f nsec\n",
                   threads, count, (long unsigned) ((double) count / duration),
                   (duration * NSEC_IN_SEC) / (double) count);
        }
    }
#if 0
    for (size_t threads = 1; threads <= 1024; threads <<= 1) {
        size_t count = threads;
        double duration;
        while (count < (1 << 24)) {
            duration = runkernel_nop_no_r(q, threads, count);
            if ((count == 4096) || (duration > 0.2)) {
                printf(
                    "nop_no_r with %lu threads, count %ld, calls per second %lu, each call %.1f "
                    "nsec\n",
                    threads, count, (long unsigned) ((double) count / duration),
                    (duration * NSEC_IN_SEC) / (double) count);
            }
            if (duration > 0.2) break;
            count <<= 1;
        }
    }
#endif
    ishmem_finalize();
    return 0;
}
