/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <common.h>

constexpr int array_size = 10;

int main(int argc, char **argv)
{
    int exit_code = 0;

    sycl::queue q;

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;

    ishmem_init();

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    int *source = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(source);
    int *target = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(target);
    int *check = (int *) malloc(array_size * sizeof(int));
    CHECK_ALLOC(check);

    /* Initialize source data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>{array_size, array_size}, [=](sycl::nd_item<1> idx) {
            size_t i = idx.get_global_id()[0];
            source[i] = ((((my_pe + 1) % npes) + 1) << 16) + static_cast<int>(i);
            target[i] = static_cast<int>(0xfeedface);
        });
    });
    e_init.wait_and_throw();

    ishmem_barrier_all();

    /* Perform put operation */
    ishmem_int_put(target, source, array_size, (my_pe + 1) % npes);
    ishmem_barrier_all();

    /* Verify data */
    q.memcpy(check, target, array_size * sizeof(int)).wait_and_throw();
    int errors = 0;
    for (int i = 0; i < array_size; ++i) {
        int expected = ((my_pe + 1) << 16) + i;
        if (check[i] != expected) {
            printf("[%d] index %d expected %08x got %08x\n", my_pe, i, expected, check[1]);
            errors += 1;
        }
    }

    if (errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << errors << std::endl;
        exit_code = 1;
    } else {
        std::cout << "No errors" << std::endl;
    }

    ishmem_free(source);
    ishmem_free(target);

    ishmem_finalize();

    return exit_code;
}
