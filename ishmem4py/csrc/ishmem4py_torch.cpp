/* Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem4py_runtime.h"

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <functional>
#include <numeric>
#include <vector>

namespace {

size_t ishmem4py_numel(c10::IntArrayRef sizes)
{
    return std::accumulate(sizes.begin(), sizes.end(), static_cast<size_t>(1),
                           std::multiplies<size_t>());
}

at::Tensor ishmem4py_wrap_tensor(void *ptr, c10::IntArrayRef sizes, c10::ScalarType scalar_type,
                                 c10::DeviceIndex device_index)
{
    const auto device = c10::Device(c10::DeviceType::XPU, device_index);
    const auto options = at::TensorOptions().dtype(scalar_type).device(device);
    return at::for_blob(ptr, sizes).options(options).target_device(device).make_tensor();
}

}  // namespace

at::Tensor ishmem4py_alloc_tensor(const std::vector<int64_t> &sizes, int64_t scalar_type,
                                  int64_t device_index)
{
    TORCH_CHECK(device_index >= 0, "device index must be >= 0");
    const auto dtype = static_cast<c10::ScalarType>(scalar_type);
    const auto nbytes = ishmem4py_numel(sizes) * c10::elementSize(dtype);
    if (nbytes == 0) {
        return at::empty(sizes, at::TensorOptions().dtype(dtype).device(c10::DeviceType::XPU,
                                                                        device_index));
    }

    void *ptr = ishmem4py_malloc(nbytes);
    TORCH_CHECK(ptr != nullptr, "ishmem_malloc(", nbytes, ") returned NULL");
    return ishmem4py_wrap_tensor(ptr, sizes, dtype, static_cast<c10::DeviceIndex>(device_index));
}

at::Tensor ishmem4py_tensor_from_ptr(uint64_t ptr, const std::vector<int64_t> &sizes,
                                     int64_t scalar_type, int64_t device_index)
{
    TORCH_CHECK(device_index >= 0, "device index must be >= 0");
    return ishmem4py_wrap_tensor(reinterpret_cast<void *>(ptr), sizes,
                                 static_cast<c10::ScalarType>(scalar_type),
                                 static_cast<c10::DeviceIndex>(device_index));
}

void ishmem4py_free_ptr(uint64_t ptr)
{
    if (ptr == 0) return;
    ishmem4py_free(reinterpret_cast<void *>(ptr));
}

PYBIND11_MODULE(_ishmem4py_torch, m)
{
    m.def("alloc_tensor", &ishmem4py_alloc_tensor);
    m.def("tensor_from_ptr", &ishmem4py_tensor_from_ptr);
    m.def("free_ptr", &ishmem4py_free_ptr);
}
