/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <common.h>
#include <algorithm>

#define ARRAY_SIZE     5
#define NUM_WORK_ITEMS 4
#define NUM_TESTS      12

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
    size_t **ret_check = sycl::malloc_host<size_t *>(NUM_TESTS, q);
    CHECK_ALLOC(ret_check);
    size_t *num_matches = sycl::malloc_host<size_t>(NUM_TESTS, q);
    CHECK_ALLOC(num_matches);
    int *errors = sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);

    /* Initialize source data */
    for (size_t i = 0; i < NUM_TESTS; i++) {
        ret_check[i] = sycl::malloc_host<size_t>(ARRAY_SIZE, q);
        CHECK_ALLOC(ret_check[i]);

        for (size_t j = 0; j < ARRAY_SIZE; j++) {
            ret_check[i][j] = SIZE_MAX;
        }
    }
    q.submit([&](sycl::handler &h) {
         h.single_task([=]() {
             for (int i = 0; i < ARRAY_SIZE; i++) {
                 source[i] = i;
                 if (i != 1) status[i] = 1;
             }
         });
     }).wait_and_throw();

    /*
        Use ishmem_int_test_some to identify the range of possible values we can expect to be
        returned by ishmem_int_test_any
    */
    num_matches[0] = ishmem_int_test_some(source, ARRAY_SIZE, ret_check[0], NULL, ISHMEM_CMP_EQ, 0);
    num_matches[1] = ishmem_int_test_some(source, ARRAY_SIZE, ret_check[1], NULL, ISHMEM_CMP_NE, 0);
    num_matches[2] = ishmem_int_test_some(source, ARRAY_SIZE, ret_check[2], NULL, ISHMEM_CMP_GT, 1);
    num_matches[3] = ishmem_int_test_some(source, ARRAY_SIZE, ret_check[3], NULL, ISHMEM_CMP_GE, 1);
    num_matches[4] = ishmem_int_test_some(source, ARRAY_SIZE, ret_check[4], NULL, ISHMEM_CMP_LT, 2);
    num_matches[5] = ishmem_int_test_some(source, ARRAY_SIZE, ret_check[5], NULL, ISHMEM_CMP_LE, 1);

    /* Perform test operations */
    *errors = 0;
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

    /* Verify data */
    for (size_t i = 0; i < NUM_TESTS; ++i) {
        if (i < 6) {
            if (std::find(ret_check[i], ret_check[i] + num_matches[i], ret[i]) ==
                (ret_check[i] + num_matches[i])) {
                (*errors) += 1;
            }
        } else {
            if (ret[i] != SIZE_MAX) (*errors) += 1;
        }
    }
    if (*errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
        for (size_t i = 0; i < NUM_TESTS; i += 1) {
            std::string valid_ret_values;
            if (i < 6) {
                if (std::find(ret_check[i], ret_check[i] + num_matches[i], ret[i]) ==
                    (ret_check[i] + num_matches[i])) {
                    valid_ret_values = "[";
                    for (size_t j = 0; j < num_matches[i]; j++) {
                        valid_ret_values += std::to_string(ret_check[i][j]);
                        if (j != (num_matches[i] - 1)) {
                            valid_ret_values += ", ";
                        }
                    }
                    valid_ret_values += "]";
                    std::cerr << "[" << my_pe << "]: ret[" << i << "] = " << ret[i]
                              << ", expected value from list " << valid_ret_values << std::endl;
                }
            } else {
                if (ret[i] != SIZE_MAX) {
                    valid_ret_values = "[" + std::to_string(SIZE_MAX) + "]";
                    std::cerr << "[" << my_pe << "]: ret[" << i << "] = " << ret[i]
                              << ", expected value from list " << valid_ret_values << std::endl;
                }
            }
        }
        exit_code = 1;
    }

    if (exit_code) std::cout << "[" << my_pe << "] Test Failed" << std::endl;
    else std::cout << "[" << my_pe << "] Test Passed" << std::endl;

    sycl::free(errors, q);
    sycl::free(status, q);
    sycl::free(ret, q);
    for (size_t i = 0; i < NUM_TESTS; i++) {
        sycl::free(ret_check[i], q);
    }
    sycl::free(ret_check, q);
    sycl::free(num_matches, q);
    ishmem_free(source);

    ishmem_finalize();

    return exit_code;
}
