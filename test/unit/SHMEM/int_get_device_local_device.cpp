/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <common.h>

constexpr int array_size = 10;

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmemx_attr_t attr = {};
    test_init_attr(&attr);
    ishmemx_init_attr(&attr);

    sycl::queue q;

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;

    int *source = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(source);

    int *target = (int *) sycl::malloc_device<int>(array_size * sizeof(int), q);
    CHECK_ALLOC(target);

    int *errors = (int *) sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);

    /* Initialize source data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>{array_size, array_size}, [=](sycl::nd_item<1> idx) {
            size_t i = idx.get_global_id()[0];
            source[i] = static_cast<int>(i);
            target[i] = 0;
        });
    });
    e_init.wait_and_throw();

    ishmem_barrier_all();

    /* Perform put operation */
    auto e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            int my_dev_pe = ishmem_my_pe();
            int my_dev_npes = ishmem_n_pes();

            ishmem_int_get(target, source, array_size, (my_dev_pe + 1) % my_dev_npes);
        });
    });
    e1.wait_and_throw();

    ishmem_barrier_all();
    *errors = 0;
    /* Verify data */
    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (int i = 0; i < array_size; ++i) {
                if (target[i] != i) {
                    *errors = *errors + 1;
                }
            }
        });
    });
    e_verify.wait_and_throw();

    if (*errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
        exit_code = 1;
    } else {
        std::cout << "No errors" << std::endl;
    }

    sycl::free(errors, q);
    ishmem_free(source);

    sycl::free(target, q);

    ishmem_finalize();

    return exit_code;
}
