/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <unistd.h>
#include <common.h>

/* unit and performance test for timestamp function
 */

constexpr int array_size = 10;
constexpr long CPU_FREQ = 2100000000;

int main(int argc, char **argv)
{
    ishmem_init();

    int my_pe = ishmem_my_pe();
    sycl::queue q;

    std::cout << "Host: PE " << my_pe << std::endl;

    unsigned long *hostbuf = sycl::malloc_host<unsigned long>(array_size, q);
    CHECK_ALLOC(hostbuf);

    auto e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            ishmemx_timestamp((uintptr_t) &hostbuf[0]);
            ishmemx_timestamp((uintptr_t) &hostbuf[1]);
            ishmemx_timestamp((uintptr_t) &hostbuf[2]);
            ishmemx_timestamp((uintptr_t) &hostbuf[3]);
            ishmemx_timestamp((uintptr_t) &hostbuf[4]);
        });
    });
    e1.wait_and_throw();
    for (int i = 0; i < 4; i += 1) {
        unsigned long delta = hostbuf[i + 1] - hostbuf[i];
        double ns = 1000000000.0 * ((double) delta) / ((double) CPU_FREQ);
        printf("with completion delta[%d] = %ld, ns = %f\n", i, delta, ns);
    }
    e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            ishmemx_timestamp_nbi((uintptr_t) &hostbuf[0]);
            ishmemx_timestamp_nbi((uintptr_t) &hostbuf[1]);
            ishmemx_timestamp_nbi((uintptr_t) &hostbuf[2]);
            ishmemx_timestamp_nbi((uintptr_t) &hostbuf[3]);
            ishmemx_timestamp_nbi((uintptr_t) &hostbuf[4]);
            ishmemx_nop();
        });
    });
    e1.wait_and_throw();
    for (int i = 0; i < 4; i += 1) {
        unsigned long delta = hostbuf[i + 1] - hostbuf[i];
        double ns = 1000000000.0 * ((double) delta) / ((double) CPU_FREQ);
        printf("  no completion delta[%d] = %ld, ns = %f\n", i, delta, ns);
    }
    unsigned long start, stop;
    /* it is possible to use local variables that remain in-scope but you have to work to convice
     * SYCL not to translate the pointers */
    ishmemx_ts_handle_t start_handle = ishmemx_ts_handle(&start);
    ishmemx_ts_handle_t stop_handle = ishmemx_ts_handle(&stop);
    fflush(stdout);
    e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            ishmemx_timestamp_nbi(start_handle);
            ishmemx_timestamp_nbi(stop_handle);
            ishmemx_nop();
        });
    });
    e1.wait_and_throw();
    printf("start %ld stop %ld, delta %ld\n", start, stop, stop - start);

    sycl::free(hostbuf, q);
    ishmem_finalize();
    return 0;
}
