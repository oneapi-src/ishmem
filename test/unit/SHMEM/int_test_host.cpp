/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <common.h>

#define NUM_WORK_ITEMS 4

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmemx_attr_t attr = {};
    test_init_attr(&attr);
    ishmemx_init_attr(&attr);

    int my_pe = ishmem_my_pe();

    sycl::queue q;

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;

    int *source = (int *) ishmem_malloc(sizeof(int));
    CHECK_ALLOC(source);
    int *errors = (int *) sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);

    /* Initialize source data */
    q.submit([&](sycl::handler &h) { h.single_task([=]() { *source = 1; }); }).wait_and_throw();

    /* Perform test operations */
    (*errors) += !ishmem_int_test(source, ISHMEM_CMP_EQ, 1);
    (*errors) += !ishmem_int_test(source, ISHMEM_CMP_NE, 0);
    (*errors) += !ishmem_int_test(source, ISHMEM_CMP_GT, 0);
    (*errors) += !ishmem_int_test(source, ISHMEM_CMP_GE, 1);
    (*errors) += !ishmem_int_test(source, ISHMEM_CMP_LT, 2);
    (*errors) += !ishmem_int_test(source, ISHMEM_CMP_LE, 1);

    // Should return false (0)
    (*errors) += ishmem_int_test(source, ISHMEM_CMP_EQ, 0);
    (*errors) += ishmem_int_test(source, ISHMEM_CMP_NE, 1);
    (*errors) += ishmem_int_test(source, ISHMEM_CMP_GT, 1);
    (*errors) += ishmem_int_test(source, ISHMEM_CMP_GE, 2);
    (*errors) += ishmem_int_test(source, ISHMEM_CMP_LT, 1);
    (*errors) += ishmem_int_test(source, ISHMEM_CMP_LE, 0);

    /* Verify data */
    if (*errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
        exit_code = 1;
    } else {
        std::cout << "[PE " << my_pe << "] No errors" << std::endl;
    }

    sycl::free(errors, q);
    ishmem_free(source);

    ishmem_finalize();

    return exit_code;
}
