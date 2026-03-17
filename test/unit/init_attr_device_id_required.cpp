/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "accelerator.h"
#include <cstdlib>
#include <iostream>
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
}  // namespace

int main()
{
    auto devices = get_visible_level_zero_gpus();
    if (devices.size() < 2) {
        std::cout << "Skipping device_id-required test because fewer than 2 visible Level Zero "
                     "GPUs were detected"
                  << std::endl;
        return 77;
    }

    ishmemx_attr_t attr;
    int ret = ishmemi_accelerator_init(&attr);

    if (ret == 0) {
        std::cerr << "Expected accelerator initialization to fail when multiple GPUs are visible "
                     "and device_id is not set"
                  << std::endl;
        ishmemi_accelerator_fini();
        return EXIT_FAILURE;
    }

    std::cout << "Detected required explicit device selection with " << devices.size()
              << " visible GPUs" << std::endl;
    return EXIT_SUCCESS;
}
