/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <common.h>

constexpr int array_size = 1 << 15;

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
    int *errors = (int *) sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);

    /* Initialize source data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (size_t i = 0; i < array_size; ++i) {
                source[i] = (my_pe << 16) + static_cast<int>(i);
                target[i] = (my_pe << 16) + 0xface;
            }
        });
    });
    e_init.wait_and_throw();

    ishmem_barrier_all();

    /* Perform put operation */
    auto e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            int my_dev_pe = ishmem_my_pe();
            int my_dev_npes = ishmem_n_pes();

            ishmem_int_put_nbi(target, source, array_size, (my_dev_pe + 1) % my_dev_npes);
        });
    });
    e1.wait_and_throw();

    ishmem_barrier_all();
    *errors = 0;
    /* Verify data */
    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (int i = 0; i < array_size; ++i) {
                int source_pe = (my_pe > 0) ? (my_pe - 1) : (npes - 1);
                if (target[i] != (source_pe << 16) + i) {
                    *errors = *errors + 1;
                }
            }
        });
    });
    e_verify.wait_and_throw();

    if (*errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
        int *hosttarget = sycl::malloc_host<int>(array_size, q);
        CHECK_ALLOC(hosttarget);
        q.memcpy(hosttarget, target, sizeof(int) * array_size).wait_and_throw();
        int source_pe = (my_pe > 0) ? (my_pe - 1) : (npes - 1);
        for (int i = 0; i < array_size; i += 1) {
            if (hosttarget[i] != (source_pe << 16) + i) {
                fprintf(stdout, "[%d] index %d expected 0x%08x got 0x%08x\n", my_pe, i,
                        (((my_pe + 1) % npes) << 16) + i, hosttarget[i]);
            }
        }
        sycl::free(hosttarget, q);
        exit_code = 1;
    } else {
        std::cout << "No errors" << std::endl;
    }

    fflush(stdout);
    ishmem_free(source);
    ishmem_free(target);
    sycl::free(errors, q);

    ishmem_finalize();

    return exit_code;
}
