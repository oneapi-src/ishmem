/* Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * # Launch example
 * export WORLD_SIZE=4
 * export MASTER_ADDR=localhost
 * RANK=0 ${APP} &
 * RANK=1 ${APP} &
 * RANK=2 ${APP} &
 * RANK=3 ${APP} &
 */

#include <cstdlib>
#include <iostream>
#include <ishmem.h>
#include <ishmemx.h>
#include "ishmem/device_utils.h"

constexpr size_t N = 10;

int main(int argc, char **argv)
{
    auto devices = ishmemi_get_visible_devices();
    if (devices.empty()) {
        std::cerr << "No visible Level Zero GPU devices found!" << std::endl;
        return EXIT_FAILURE;
    }

    ishmemx_uniqueid_t uid;
    if (ishmemx_get_uniqueid(&uid) != 0) {
        std::cerr << "Failed to get unique ID for MPCP initialization\n";
        return EXIT_FAILURE;
    }

    // Get world size from environment variable
    const char *world_size_str = getenv("WORLD_SIZE");
    if (!world_size_str) {
        std::cerr << "WORLD_SIZE environment variable not set!" << std::endl;
        return EXIT_FAILURE;
    }
    int world_size = atoi(world_size_str);
    if (world_size <= 0) {
        std::cerr << "Invalid WORLD_SIZE: " << world_size_str << std::endl;
        return EXIT_FAILURE;
    }

    // Get rank from environment variable
    const char *rank_str = getenv("RANK");
    if (!rank_str) {
        std::cerr << "RANK environment variable not set!" << std::endl;
        return EXIT_FAILURE;
    }

    int rank = atoi(rank_str);
    if (rank < 0 || rank >= world_size) {
        std::cerr << "Invalid RANK: " << rank_str << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize ISHMEM
    ishmemx_attr_t attr;
    attr.initialize_runtime = true;
    attr.use_uid = true;
    attr.nranks = world_size;  // Nranks
    attr.rank = rank;          // Rank of this process
    attr.uid = &uid;
    // If more PEs than devices, wrap around device selection based on local rank.
    // This allows testing of explicit device selection in multi-GPU environments
    // even when the number of visible GPUs is less than the number of PEs.
    attr.device_idx = rank % static_cast<int>(devices.size());
    ishmemx_init_attr(&attr);

    // Each PE is mapped to a device.
    int my_pe = ishmem_my_pe();
    // Get total number of PEs
    int npes = ishmem_n_pes();
    // Get device mapped for this PE.
    sycl::device my_device = ishmemx_my_device();

    // Create a queue with my_device
    sycl::queue q(my_device);

    for (int pe = 0; pe < npes; ++pe) {
        if (pe == my_pe) {
            std::cout << "Intel(R) SHMEM initialized successfully! " << "PE " << std::setw(2)
                      << my_pe << " selected device_idx " << std::setw(2) << attr.device_idx
                      << " : " << my_device.get_info<sycl::info::device::name>() << " ("
                      << ishmemi_get_device_uuid(my_device) << ")" << std::endl;
            std::flush(std::cout);
        }
        ishmem_barrier_all();
    }

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
         if (dst[id] != expected) {
             check[id] = 1;
         } else {
             check[id] = 0;
         }
     }).wait();

    bool check_fail = false;
    for (size_t i = 0; i < N; i++)
        if (check[i]) {
            std::cerr << "PE " << my_pe << " ISHMEM get failed at index: " << i << std::endl;
            check_fail = true;
            continue;
        }

    for (int pe = 0; pe < npes; ++pe) {
        if (pe == my_pe && !check_fail) {
            std::cout << "PE " << std::setw(2) << my_pe
                      << " successfully received the data using ISHMEM get." << std::endl;
            std::flush(std::cout);
        }
        ishmem_barrier_all();
    }

    // Free the symmetric heap memory and check buffer
    ishmem_free(src);
    ishmem_free(dst);
    free(check, q);

    ishmem_finalize();

    if (check_fail) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
