/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <cstdlib>
#include <iostream>
#include <ishmem.h>
#include <ishmemx.h>

constexpr size_t N = 10;

int main(int argc, char **argv)
{
    // Initialize ISHMEM
    ishmem_init();

    // Each PE is mapped to a device.
    int my_pe = ishmem_my_pe();
    // Get total number of PEs
    int npes = ishmem_n_pes();

    // Print device info for each PE
    sycl::queue q;
    std::cout << "My PE: " << my_pe
              << " , Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    // Allocate memory on the ISHMEM symmetric heap on the device
    int *src = (int *) ishmem_malloc(N * sizeof(int));
    int *dst = (int *) ishmem_malloc(N * sizeof(int));

    // Initialize the src array on each device with the value pe
    q.parallel_for(sycl::range<1>(N), [=](sycl::item<1> id) { src[id] = my_pe; }).wait();

    // Ensure completion of initialization
    ishmem_barrier_all();

    // Get data from the next device
    q.parallel_for(sycl::range<1>(N), [=](sycl::item<1> id) {
         int next_pe = (my_pe + 1) % npes;
         dst[id] = ishmem_int_g(&src[id], next_pe);
     }).wait();

    // Ensure completion of get operations and wait for other PEs
    ishmem_barrier_all();

    // Verify the data
    int *check = (int *) sycl::malloc_host<int>(N, q);
    q.parallel_for(sycl::range<1>(N), [=](sycl::item<1> id) {
         int expected = (my_pe + 1) % npes;
         if (dst[id] != expected)
             check[id] = 1;
         else
             check[id] = 0;
     }).wait();

    bool check_fail = false;
    for (size_t i = 0; i < N; i++)
        if (check[i]) {
            std::cerr << "PE: " << my_pe << " ISHMEM get failed at index: " << i << std::endl;
            check_fail = true;
            continue;
        }

    if (!check_fail)
        std::cout << my_pe << " PE successfully received the data using ISHMEM get." << std::endl;

    // Free the symmetric heap memory and check buffer
    ishmem_free(src);
    ishmem_free(dst);
    free(check, q);

    ishmem_finalize();

    if (check_fail) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
