/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <common.h>

constexpr int array_size = 10;

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmem_init();

    sycl::queue q;

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    int *source = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(source);
    int *target = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(target);
    int *host_target = sycl::malloc_host<int>(array_size, q);
    CHECK_ALLOC(host_target);

    /* Initialize source data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>{array_size, array_size}, [=](sycl::nd_item<1> idx) {
            size_t i = idx.get_global_id()[0];
            source[i] = (my_pe << 16) + static_cast<int>(i);
            target[i] = (my_pe << 16) + 0xface;
            host_target[i] = 0;
        });
    });
    e_init.wait_and_throw();

    ishmem_barrier_all();

    /* Perform put operation */
    auto e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            int my_dev_pe = ishmem_my_pe();
            int my_dev_npes = ishmem_n_pes();

            for (int i = 0; i < array_size; i++)
                ishmem_int_put(&target[i], &source[i], 1, (my_dev_pe + 1) % my_dev_npes);

            ishmem_fence();
            ishmem_int_atomic_set(&target[0], 100, (my_dev_pe + 1) % my_dev_npes);
            ishmem_int_wait_until(&target[0], ISHMEM_CMP_EQ, 100);
            memcpy(host_target, target, array_size * sizeof(int));
        });
    });
    e1.wait_and_throw();

    ishmem_barrier_all();
    /* Verify data */
    for (int i = 1; i < array_size; i++) {
        if (host_target[i] != ((((my_pe - 1 + npes) % npes) << 16) + i)) {
            fprintf(stdout, "[%d] ERROR: index %d expected 0x%08x got 0x%08x\n", i, my_pe,
                    (((my_pe - 1 + npes) % npes) << 16) + i, host_target[i]);
            exit_code = 1;
        }
    }

    fflush(stdout);

    if (!exit_code) std::cout << "[" << my_pe << "] No errors" << std::endl;

    ishmem_free(source);
    ishmem_free(target);
    sycl::free(host_target, q);

    ishmem_finalize();

    return exit_code;
}
