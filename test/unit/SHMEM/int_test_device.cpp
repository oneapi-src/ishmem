/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <common.h>

#define INVALID_ISHMEM_CMP_PARAM 777
#define NUM_WORK_ITEMS           4

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmem_init();

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
    auto e_init = q.submit([&](sycl::handler &h) { h.single_task([=]() { *source = 1; }); });
    e_init.wait_and_throw();

    /* Perform test operations */
    *errors = 0;
    auto e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            // Should return true (1)
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
            (*errors) += ishmem_int_test(source, INVALID_ISHMEM_CMP_PARAM, 0);
        });
    });
    e1.wait_and_throw();

    /* Perform test_work_group operations */
    auto e2 = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>{NUM_WORK_ITEMS, NUM_WORK_ITEMS}, [=](sycl::nd_item<1> it) {
            auto grp = it.get_group();

            // Should return true (1)
            (*errors) += !ishmemx_int_test_work_group(source, ISHMEM_CMP_EQ, 1, grp);
            (*errors) += !ishmemx_int_test_work_group(source, ISHMEM_CMP_NE, 0, grp);
            (*errors) += !ishmemx_int_test_work_group(source, ISHMEM_CMP_GT, 0, grp);
            (*errors) += !ishmemx_int_test_work_group(source, ISHMEM_CMP_GE, 1, grp);
            (*errors) += !ishmemx_int_test_work_group(source, ISHMEM_CMP_LT, 2, grp);
            (*errors) += !ishmemx_int_test_work_group(source, ISHMEM_CMP_LE, 1, grp);

            // Should return false (0)
            (*errors) += ishmemx_int_test_work_group(source, ISHMEM_CMP_EQ, 0, grp);
            (*errors) += ishmemx_int_test_work_group(source, ISHMEM_CMP_NE, 1, grp);
            (*errors) += ishmemx_int_test_work_group(source, ISHMEM_CMP_GT, 1, grp);
            (*errors) += ishmemx_int_test_work_group(source, ISHMEM_CMP_GE, 2, grp);
            (*errors) += ishmemx_int_test_work_group(source, ISHMEM_CMP_LT, 1, grp);
            (*errors) += ishmemx_int_test_work_group(source, ISHMEM_CMP_LE, 0, grp);
            (*errors) += ishmemx_int_test_work_group(source, INVALID_ISHMEM_CMP_PARAM, 0, grp);
        });
    });
    e2.wait_and_throw();

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
