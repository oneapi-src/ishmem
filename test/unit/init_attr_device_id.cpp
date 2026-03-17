/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <common.h>
#include <cstdlib>
#include <vector>

namespace {
std::vector<sycl::device> get_visible_level_zero_gpus()
{
    std::vector<sycl::device> devices;

    for (const auto &platform : sycl::platform::get_platforms()) {
        if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;

        for (const auto &device : platform.get_devices()) {
            if (device.is_gpu()) {
                devices.push_back(device);
            }
        }
    }

    return devices;
}

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

    auto devices = get_visible_level_zero_gpus();
    if (devices.size() < 2) {
        std::cout << "Skipping explicit device_id test because fewer than 2 visible Level Zero GPUs "
                     "were detected"
                  << std::endl;
        return 77;
    }

    int local_rank = get_local_rank();
    if ((local_rank < 0) || (static_cast<size_t>(local_rank) >= devices.size())) {
        std::cerr << "Invalid local rank " << local_rank << " for " << devices.size()
                  << " visible GPU devices" << std::endl;
        return EXIT_FAILURE;
    }

    ishmemx_attr_t attr;
    attr.initialize_runtime = true;
    attr.runtime = ishmemi_test_runtime->get_type();
    attr.device_id = local_rank;
    ishmemx_init_attr(&attr);

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();
    int peer = (my_pe + 1) % npes;

    sycl::queue q(devices[static_cast<size_t>(local_rank)]);

    std::cout << "PE " << my_pe << " selected device_id " << local_rank << ": "
              << q.get_device().get_info<sycl::info::device::name>() << std::endl;

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
