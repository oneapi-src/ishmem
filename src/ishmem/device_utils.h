/* Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_DEVICE_UTILS_H
#define ISHMEM_DEVICE_UTILS_H

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "ishmemx.h"

static inline std::vector<sycl::device> ishmemi_get_visible_devices()
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

static inline std::string ishmemi_get_device_uuid(const sycl::device &dev)
{
    auto uuid = dev.get_info<sycl::ext::intel::info::device::uuid>();
    std::ostringstream oss;
    for (size_t i = 0; i < uuid.size(); ++i) {
        if (i % 2 == 0 && i != 0) oss << "-";
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(uuid[i])
            << std::dec;
    }
    return oss.str();
}

#endif /* ISHMEM_DEVICE_UTILS_H */