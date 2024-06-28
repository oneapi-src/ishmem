/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <common.h>

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmemx_attr_t attr = {};
    test_init_attr(&attr);
    ishmemx_init_attr(&attr);

    sycl::queue q;

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    int *source = (int *) ishmem_malloc(sizeof(int));
    CHECK_ALLOC(source);
    int *errors = (int *) sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);
    int *is_null = (int *) sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(is_null);

    int expected_value = (my_pe << 16) + ((my_pe - 1 >= 0) ? (my_pe - 1) : (npes - 1));

    /* Initialize source data */
    auto e_init =
        q.submit([&](sycl::handler &h) { h.single_task([=]() { *source = (my_pe << 16); }); });
    e_init.wait_and_throw();

    ishmem_barrier_all();

    /* Perform ishmem_ptr operation */
    *is_null = 1;
    *errors = 0;
    auto e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            int my_dev_pe = ishmem_my_pe();
            int my_dev_npes = ishmem_n_pes();

            int *source_next_pe = (int *) ishmem_ptr(source, (my_dev_pe + 1) % my_dev_npes);
            if (source_next_pe) {
                *source_next_pe = (*source_next_pe) + my_dev_pe;
                *is_null = 0;
            }

            ishmem_barrier_all();

            /* Verify data */
            /* Can the previous pe access me using ishmem_ptr? */
            if (ishmem_ptr(source, (my_dev_pe - 1 + my_dev_npes) % my_dev_npes))
                if ((*source) != expected_value) {
                    *errors = *errors + 1;
                }
        });
    });
    e1.wait_and_throw();

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

    if (*is_null)
        std::cout << "PE: " << (my_pe + 1) % npes
                  << " inaccessible through ishmem_ptr from PE: " << my_pe << std::endl;

    fflush(stdout);
    ishmem_free(source);
    sycl::free(errors, q);
    sycl::free(is_null, q);

    ishmem_finalize();

    return exit_code;
}
