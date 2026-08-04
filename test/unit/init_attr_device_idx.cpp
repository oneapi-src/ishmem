/* Copyright (C) 2025-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <common.h>
#include <cstdlib>
#include <vector>
#include "ishmem/device_utils.h"

namespace {
    int get_local_rank()
    {
        constexpr const char *env_names[] = {
            "MPI_LOCALRANKID",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "PMI_LOCAL_RANK",
            "SLURM_LOCALID",
        };

        for (const char *name : env_names) {
            const char *value = std::getenv(name);
            if (value != nullptr) {
                return std::atoi(value);
            }
        }

        return 0;
    }
}  // namespace

int main()
{
    validate_runtime();

    auto devices = ishmemi_get_visible_devices();

    int local_rank = get_local_rank();
    if (local_rank < 0) {
        std::cerr << "Invalid local rank " << local_rank << " for " << devices.size()
                  << " visible GPU devices" << std::endl;
        return EXIT_FAILURE;
    }

    ishmemx_attr_t attr;
    attr.initialize_runtime = true;
    attr.runtime = ishmemi_test_runtime->get_type();
    // If more PEs than devices, wrap around device selection based on local rank.
    // This allows testing of explicit device selection in multi-GPU environments
    // even when the number of visible GPUs is less than the number of PEs.
    attr.device_idx = local_rank % static_cast<int>(devices.size());
    ishmemx_init_attr(&attr);

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();
    int peer = (my_pe + 1) % npes;
    sycl::device my_device = ishmemx_my_device();

    sycl::queue q(my_device);

    if (q.get_device() != my_device) {
        std::cerr << "Queue device does not match ishmemx_my_device() device" << std::endl;
        return EXIT_FAILURE;
    }

    for (int pe = 0; pe < npes; ++pe) {
        if (pe == my_pe) {
            std::cout << "PE " << std::setw(2) << my_pe << " selected device_idx " << std::setw(2)
                      << attr.device_idx << " : " << my_device.get_info<sycl::info::device::name>()
                      << " (" << ishmemi_get_device_uuid(my_device) << ")"
                      << " visible devices: " << devices.size() << std::endl;
            std::flush(std::cout);
        }
        ishmem_barrier_all();
    }

    int *source = (int *) ishmem_malloc(sizeof(int));
    CHECK_ALLOC(source);
    int *target = (int *) ishmem_malloc(sizeof(int));
    CHECK_ALLOC(target);
    int *host_value = sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(host_value);

    q.fill(source, my_pe, 1).wait_and_throw();
    q.fill(target, -1, 1).wait_and_throw();
    ishmem_barrier_all();

    auto e = ishmemx_int_get_on_queue(target, source, 1, peer, q);
    e.wait_and_throw();
    ishmemx_quiet_on_queue(q).wait_and_throw();
    q.copy(target, host_value, 1).wait_and_throw();

    int rc = EXIT_SUCCESS;
    if (*host_value != peer) {
        std::cerr << "PE " << my_pe << " expected " << peer << " but received " << *host_value
                  << std::endl;
        rc = EXIT_FAILURE;
    } else if (my_pe == 0) {
        std::cout << "Test Passed" << std::endl;
    }

    ishmem_barrier_all();
    sycl::free(host_value, q);
    ishmem_free(target);
    ishmem_free(source);
    ishmem_finalize();

    return rc;
}
