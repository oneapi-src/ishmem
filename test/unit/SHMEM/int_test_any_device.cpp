/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <common.h>

#define ARRAY_SIZE     5
#define NUM_WORK_ITEMS 4
#define NUM_TESTS      12

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

    int *source = (int *) ishmem_malloc(ARRAY_SIZE * sizeof(int));
    CHECK_ALLOC(source);
    int *status = sycl::malloc_host<int>(ARRAY_SIZE, q);
    CHECK_ALLOC(status);
    size_t *ret = sycl::malloc_host<size_t>(NUM_TESTS, q);
    CHECK_ALLOC(ret);
    size_t *ret_check = sycl::malloc_host<size_t>(NUM_TESTS, q);
    CHECK_ALLOC(ret_check);
    int *errors = sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);

    /* Initialize source data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (int i = 0; i < ARRAY_SIZE; i++) {
                source[i] = i;
                if (i != 1) status[i] = 1;
            }

            for (size_t i = 0; i < (NUM_TESTS / 2); i++) {
                if (i == 0) ret_check[i] = 0;
                else if (i == 1) ret_check[i] = 1;
                else if (i == 2) ret_check[i] = 2;
                else if (i == 3) ret_check[i] = 3;
                else if (i == 4) ret_check[i] = 0;
                else if (i == 5) ret_check[i] = 1;
            }
            for (size_t i = (NUM_TESTS / 2); i < NUM_TESTS; i++) {
                ret_check[i] = SIZE_MAX;
            }
        });
    });
    e_init.wait_and_throw();

    /* Perform test operations */
    *errors = 0;
    auto e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            // At least one entry in source satisfies the "cmp" condition
            ret[0] = ishmem_int_test_any(source, ARRAY_SIZE, NULL, ISHMEM_CMP_EQ, 0);
            ret[1] = ishmem_int_test_any(source, ARRAY_SIZE, NULL, ISHMEM_CMP_NE, 0);
            ret[2] = ishmem_int_test_any(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GT, 1);
            ret[3] = ishmem_int_test_any(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GE, 1);
            ret[4] = ishmem_int_test_any(source, ARRAY_SIZE, NULL, ISHMEM_CMP_LT, 2);
            ret[5] = ishmem_int_test_any(source, ARRAY_SIZE, NULL, ISHMEM_CMP_LE, 1);

            // No entries satisfy "cmp" condition
            ret[6] = ishmem_int_test_any(source, ARRAY_SIZE, NULL, ISHMEM_CMP_EQ, ARRAY_SIZE);
            ret[7] = ishmem_int_test_any(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GT, ARRAY_SIZE);
            ret[8] = ishmem_int_test_any(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GE, ARRAY_SIZE);
            ret[9] = ishmem_int_test_any(source, ARRAY_SIZE, NULL, ISHMEM_CMP_LT, -1);
            ret[10] = ishmem_int_test_any(source, ARRAY_SIZE, NULL, ISHMEM_CMP_LE, -1);
            ret[11] = ishmem_int_test_any(source, ARRAY_SIZE, status, ISHMEM_CMP_NE, 1);
        });
    });
    e1.wait_and_throw();

    /* Verify data */
    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (size_t i = 0; i < NUM_TESTS; ++i) {
                if (ret[i] != ret_check[i]) {
                    (*errors) += 1;
                }
            }
        });
    });
    e_verify.wait_and_throw();
    if (*errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
        for (size_t i = 0; i < NUM_TESTS; i += 1) {
            if (ret[i] != ret_check[i]) {
                std::cerr << "[" << my_pe << "]: ret[" << i << "] = " << ret[i] << ", expected "
                          << ret_check[i] << std::endl;
            }
        }
        exit_code = 1;
    }

    /* Perform test_work_group operations */
    auto e2 = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>{NUM_WORK_ITEMS, NUM_WORK_ITEMS}, [=](sycl::nd_item<1> it) {
            auto grp = it.get_group();

            // At least one entry in source satisfies the "cmp" condition
            ret[0] =
                ishmemx_int_test_any_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_EQ, 0, grp);
            ret[1] =
                ishmemx_int_test_any_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_NE, 0, grp);
            ret[2] =
                ishmemx_int_test_any_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GT, 1, grp);
            ret[3] =
                ishmemx_int_test_any_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GE, 1, grp);
            ret[4] =
                ishmemx_int_test_any_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_LT, 2, grp);
            ret[5] =
                ishmemx_int_test_any_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_LE, 1, grp);

            // No entries satisfy "cmp" condition
            ret[6] = ishmemx_int_test_any_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_EQ,
                                                     ARRAY_SIZE, grp);
            ret[7] = ishmemx_int_test_any_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GT,
                                                     ARRAY_SIZE, grp);
            ret[8] = ishmemx_int_test_any_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GE,
                                                     ARRAY_SIZE, grp);
            ret[9] =
                ishmemx_int_test_any_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_LT, -1, grp);
            ret[10] =
                ishmemx_int_test_any_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_LE, -1, grp);
            ret[11] =
                ishmemx_int_test_any_work_group(source, ARRAY_SIZE, status, ISHMEM_CMP_NE, 1, grp);
        });
    });
    e2.wait_and_throw();

    /* Verify data */
    *errors = 0;
    auto e_verify_wg = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (size_t i = 0; i < NUM_TESTS; ++i) {
                if (ret[i] != ret_check[i]) {
                    (*errors) += 1;
                }
            }
        });
    });
    e_verify_wg.wait_and_throw();
    if (*errors > 0) {
        std::cerr << "[" << my_pe << "] ERROR, work_group validation check(s) failed: " << *errors
                  << std::endl;
        for (size_t i = 0; i < NUM_TESTS; i += 1) {
            if (ret[i] != ret_check[i]) {
                std::cerr << "[" << my_pe << "]: ret[" << i << "] = " << ret[i] << ", expected "
                          << ret_check[i] << std::endl;
            }
        }
        exit_code = 1;
    }

    if (exit_code) std::cout << "[" << my_pe << "] Test Failed" << std::endl;
    else std::cout << "[" << my_pe << "] Test Passed" << std::endl;

    sycl::free(errors, q);
    sycl::free(status, q);
    sycl::free(ret, q);
    sycl::free(ret_check, q);
    ishmem_free(source);

    ishmem_finalize();

    return exit_code;
}
