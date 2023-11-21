/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <common.h>

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

    int *source = (int *) ishmem_malloc(sizeof(int));
    CHECK_ALLOC(source);
    int *errors = (int *) sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);

    int expected_value = (my_pe << 16) + ((my_pe - 1 >= 0) ? (my_pe - 1) : (npes - 1));

    /* Initialize source data */
    auto e_init =
        q.submit([&](sycl::handler &h) { h.single_task([=]() { *source = (my_pe << 16); }); });
    e_init.wait_and_throw();

    ishmem_barrier_all();

    /* Perform ishmem_ptr operation */
    auto e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            int my_dev_pe = ishmem_my_pe();
            int my_dev_npes = ishmem_n_pes();

            int *source_next_pe = (int *) ishmem_ptr(source, (my_dev_pe + 1) % my_dev_npes);
            *source_next_pe = (*source_next_pe) + my_dev_pe;
        });
    });
    e1.wait_and_throw();

    ishmem_barrier_all();
    *errors = 0;
    /* Verify data */
    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            if ((*source) != expected_value) {
                *errors = *errors + 1;
            }
        });
    });
    e_verify.wait_and_throw();

    if (*errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
        int *hostsource = sycl::malloc_host<int>(1, q);
        CHECK_ALLOC(hostsource);
        q.memcpy(hostsource, source, sizeof(int)).wait_and_throw();
        fprintf(stdout, "[%d] expected 0x%08x got 0x%08x\n", my_pe, expected_value, *hostsource);
        sycl::free(hostsource, q);
        exit_code = 1;
    } else {
        std::cout << "No errors" << std::endl;
    }

    fflush(stdout);
    ishmem_free(source);
    sycl::free(errors, q);

    ishmem_finalize();

    return exit_code;
}
