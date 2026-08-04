/* Copyright (C) 2023-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <common.h>

#define ARRAY_SIZE     5
#define NUM_WORK_ITEMS 4
#define NUM_TESTS      12

static bool is_valid_test_any_result(size_t test_idx, size_t value)
{
    static const size_t valid_results[NUM_TESTS][ARRAY_SIZE] = {
        {0, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX},
        {1, 2, 3, 4, SIZE_MAX},
        {2, 3, 4, SIZE_MAX, SIZE_MAX},
        {1, 2, 3, 4, SIZE_MAX},
        {0, 1, SIZE_MAX, SIZE_MAX, SIZE_MAX},
        {0, 1, SIZE_MAX, SIZE_MAX, SIZE_MAX},
        {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX},
        {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX},
        {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX},
        {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX},
        {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX},
        {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX},
    };
    static const size_t valid_result_counts[NUM_TESTS] = {
        1, 4, 3, 4, 2, 2, 1, 1, 1, 1, 1, 1,
    };

    if (test_idx >= NUM_TESTS) return false;
    for (size_t i = 0; i < valid_result_counts[test_idx]; ++i) {
        size_t candidate = valid_results[test_idx][i];
        if (candidate == value) return true;
    }
    return false;
}

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

    int *source = (int *) ishmem_malloc(ARRAY_SIZE * sizeof(int));
    CHECK_ALLOC(source);
    int *status = sycl::malloc_host<int>(ARRAY_SIZE, q);
    CHECK_ALLOC(status);
    size_t *ret = sycl::malloc_host<size_t>(NUM_TESTS, q);
    CHECK_ALLOC(ret);
    int *errors = sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);

    /* Initialize source data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (int i = 0; i < ARRAY_SIZE; i++) {
                source[i] = i;
                status[i] = (i != 1) ? 1 : 0;
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
                if (!is_valid_test_any_result(i, ret[i])) {
                    (*errors) += 1;
                }
            }
        });
    });
    e_verify.wait_and_throw();
    if (*errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
        for (size_t i = 0; i < NUM_TESTS; i += 1) {
            if (!is_valid_test_any_result(i, ret[i])) {
                std::cerr << "[" << my_pe << "]: ret[" << i << "] = " << ret[i]
                          << ", expected a valid matching index" << std::endl;
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
                if (!is_valid_test_any_result(i, ret[i])) {
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
            if (!is_valid_test_any_result(i, ret[i])) {
                std::cerr << "[" << my_pe << "]: ret[" << i << "] = " << ret[i]
                          << ", expected a valid matching index" << std::endl;
            }
        }
        exit_code = 1;
    }

    if (exit_code) std::cout << "[" << my_pe << "] Test Failed" << std::endl;
    else std::cout << "[" << my_pe << "] Test Passed" << std::endl;

    sycl::free(errors, q);
    sycl::free(status, q);
    sycl::free(ret, q);
    ishmem_free(source);

    ishmem_finalize();

    return exit_code;
}
