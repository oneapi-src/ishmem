/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ishmem.h>
#include <ishmemx.h>
#include <common.h>

int main(int argc, char **argv)
{
    validate_runtime();

    ishmemx_attr_t attr;
    attr.initialize_runtime = true;
    attr.runtime = ishmemi_test_runtime->get_type();
    ishmemx_init_attr(&attr);

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    sycl::queue q;

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;

    int *source = (int *) ishmem_malloc(sizeof(int));
    CHECK_ALLOC(source);
    int *target = (int *) ishmem_malloc(sizeof(int));
    CHECK_ALLOC(target);
    int *errors = sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);

    /* Initialize source data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            *source = my_pe;
            *target = 0;
        });
    });
    e_init.wait_and_throw();

    /* Perform get operation */
    auto e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() { *target = ishmem_int_g(source, my_pe % npes); });
    });
    e1.wait_and_throw();
    ishmem_barrier_all();

    /* Verify data */
    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            if (*target != my_pe % npes) {
                *errors = 1;
            }
        });
    });
    e_verify.wait_and_throw();

    if (*errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
        int *hosttarget = sycl::malloc_host<int>(1, q);
        CHECK_ALLOC(hosttarget);
        q.memcpy(hosttarget, target, sizeof(int)).wait_and_throw();
        if (*hosttarget != (my_pe % npes)) {
            fprintf(stdout, "[%d] expected %d got %d\n", my_pe, ((my_pe) % npes), *hosttarget);
        }
        sycl::free(hosttarget, q);
    } else {
        if (my_pe == 0) {
            std::cout << "Test Passed" << std::endl;
        }
    }

    fflush(stdout);
    ishmem_free(source);
    ishmem_free(target);
    sycl::free(errors, q);

    ishmem_finalize();

    return *errors;
}
