/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <unistd.h>
#include <common.h>

/* unit and performance test for timestamp function
 */

constexpr int array_size = 2048;
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
            for (int i = 0; i < array_size; i += 1) {
                ishmemx_timestamp((uintptr_t) &hostbuf[i]);
            }
        });
    });
    e1.wait_and_throw();
    unsigned long total = 0;
    unsigned long min = 1000000000;
    for (int i = 0; i < array_size - 1; i += 1) {
        unsigned long delta = hostbuf[i + 1] - hostbuf[i];
        if (delta < min) min = delta;
        total += delta;
    }
    double ns = 1000000000.0 * ((double) min) / ((double) CPU_FREQ);
    printf("  with completion minimum delta = %lu, ns = %f\n", min, ns);
    double avg = (double) total / (double) (array_size - 1);
    ns = 1000000000.0 * ((double) avg) / ((double) CPU_FREQ);
    printf("  with completion average delta = %f, ns = %f\n", avg, ns);
    e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (int i = 0; i < array_size; i += 1) {
                ishmemx_timestamp_nbi((uintptr_t) &hostbuf[i]);
            }
        });
    });
    e1.wait_and_throw();
    total = 0;
    min = 1000000000;
    for (int i = 0; i < array_size - 1; i += 1) {
        unsigned long delta = hostbuf[i + 1] - hostbuf[i];
        if (delta < min) min = delta;
        total += delta;
    }
    ns = 1000000000.0 * ((double) min) / ((double) CPU_FREQ);
    printf("  without completion minimum delta = %lu, ns = %f\n", min, ns);
    avg = (double) total / (double) (array_size - 1);
    ns = 1000000000.0 * ((double) avg) / ((double) CPU_FREQ);
    printf("  without completion average delta = %f, ns = %f\n", avg, ns);

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
