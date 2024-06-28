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
    size_t **indices = sycl::malloc_host<size_t *>(NUM_TESTS, q);
    CHECK_ALLOC(indices);
    int *status = sycl::malloc_host<int>(ARRAY_SIZE, q);
    CHECK_ALLOC(status);
    size_t *ret = sycl::malloc_host<size_t>(NUM_TESTS, q);
    CHECK_ALLOC(ret);
    size_t *ret_check = sycl::malloc_host<size_t>(NUM_TESTS, q);
    CHECK_ALLOC(ret_check);
    int *errors = sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);

    /* Initialize source data */
    for (size_t i = 0; i < NUM_TESTS; i++) {
        indices[i] = sycl::malloc_host<size_t>(ARRAY_SIZE, q);
        for (size_t j = 0; j < ARRAY_SIZE; j++) {
            indices[i][j] = ARRAY_SIZE;
        }
    }
    q.submit([&](sycl::handler &h) {
         h.single_task([=]() {
             for (int i = 0; i < ARRAY_SIZE; i++) {
                 source[i] = i;
                 if (i != 1) status[i] = 1;
             }

             for (size_t i = 0; i < (NUM_TESTS / 2); i++) {
                 if (i == 0) ret_check[i] = 1;
                 else if (i == 1) ret_check[i] = ARRAY_SIZE - 1;
                 else if (i == 2) ret_check[i] = ARRAY_SIZE - 3;
                 else if (i == 3) ret_check[i] = ARRAY_SIZE - 3;
                 else if (i == 4) ret_check[i] = 1;
                 else if (i == 5) ret_check[i] = 1;
             }
             for (size_t i = (NUM_TESTS / 2); i < NUM_TESTS; i++) {
                 ret_check[i] = 0;
             }
         });
     }).wait_and_throw();

    /* Perform test operations */
    *errors = 0;

    ret[0] = ishmem_int_test_some(source, ARRAY_SIZE, indices[0], NULL, ISHMEM_CMP_EQ, 0);
    ret[1] = ishmem_int_test_some(source, ARRAY_SIZE, indices[1], NULL, ISHMEM_CMP_NE, 1);
    ret[2] = ishmem_int_test_some(source, ARRAY_SIZE, indices[2], NULL, ISHMEM_CMP_GT, 2);
    ret[3] = ishmem_int_test_some(source, ARRAY_SIZE, indices[3], NULL, ISHMEM_CMP_GE, 3);
    ret[4] = ishmem_int_test_some(source, ARRAY_SIZE, indices[4], NULL, ISHMEM_CMP_LT, 1);
    ret[5] = ishmem_int_test_some(source, ARRAY_SIZE, indices[5], NULL, ISHMEM_CMP_LE, 0);

    // No entries satisfy "cmp" condition
    ret[6] = ishmem_int_test_some(source, ARRAY_SIZE, indices[6], NULL, ISHMEM_CMP_EQ, ARRAY_SIZE);
    ret[7] = ishmem_int_test_some(source, ARRAY_SIZE, indices[7], NULL, ISHMEM_CMP_GT, ARRAY_SIZE);
    ret[8] = ishmem_int_test_some(source, ARRAY_SIZE, indices[8], NULL, ISHMEM_CMP_GE, ARRAY_SIZE);
    ret[9] = ishmem_int_test_some(source, ARRAY_SIZE, indices[9], NULL, ISHMEM_CMP_LT, -1);
    ret[10] = ishmem_int_test_some(source, ARRAY_SIZE, indices[10], NULL, ISHMEM_CMP_LE, -1);
    ret[11] = ishmem_int_test_some(source, ARRAY_SIZE, indices[11], status, ISHMEM_CMP_NE, 1);

    /* Verify data */
    q.submit([&](sycl::handler &h) {
         h.single_task([=]() {
             for (size_t i = 0; i < NUM_TESTS; ++i) {
                 if (ret[i] != ret_check[i]) {
                     (*errors) += 1;
                     continue;
                 }

                 if (i == 0) {
                     for (size_t j = 0; j < ARRAY_SIZE; j++) {
                         if (((j == 0) && (indices[i][j] != 0)) ||
                             ((j != 0) && (indices[i][j] != ARRAY_SIZE))) {
                             (*errors) += 1;
                             break;
                         }
                     }
                 } else if (i == 1) {
                     for (size_t j = 0; j < ARRAY_SIZE; j++) {
                         if (((j == 0) && (indices[i][j] != 0)) ||
                             ((j > 0) && (j < (ARRAY_SIZE - 1)) && (indices[i][j] != (j + 1))) ||
                             ((j == (ARRAY_SIZE - 1)) && (indices[i][j] != ARRAY_SIZE))) {
                             (*errors) += 1;
                             break;
                         }
                     }
                 } else if ((i == 2) || (i == 3)) {
                     for (size_t j = 0; j < ARRAY_SIZE; j++) {
                         if (((j < 3) && (indices[i][j] != (j + 3))) ||
                             ((j >= 3) && (j < ARRAY_SIZE) && (indices[i][j] != ARRAY_SIZE))) {
                             (*errors) += 1;
                             break;
                         }
                     }
                 } else if ((i == 4) || (i == 5)) {
                     for (size_t j = 0; j < ARRAY_SIZE; j++) {
                         if (((j == 0) && (indices[i][j] != 0)) ||
                             ((j != 0) && (indices[i][j] != ARRAY_SIZE))) {
                             (*errors) += 1;
                             break;
                         }
                     }
                 } else {
                     for (size_t j = 0; j < ARRAY_SIZE; j++) {
                         if (indices[i][j] != ARRAY_SIZE) {
                             (*errors) += 1;
                             break;
                         }
                     }
                 }
             }
         });
     }).wait_and_throw();
    if (*errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
        for (size_t i = 0; i < NUM_TESTS; i += 1) {
            if (ret[i] != ret_check[i]) {
                std::cerr << "[" << my_pe << "]: ret[" << i << "] = " << ret[i] << ", expected "
                          << ret_check[i] << std::endl;
                continue;
            }

            std::string actual_indices = "[";
            for (size_t j = 0; j < ARRAY_SIZE; j++) {
                actual_indices += std::to_string(indices[i][j]);
                if (j < (ARRAY_SIZE - 1)) actual_indices += ", ";
            }
            actual_indices += "]";

            std::string expected_indices = "[";
            for (size_t j = 0; j < ARRAY_SIZE; j++) {
                if (i == 0) {
                    if (j == 0) expected_indices += "0";
                    else expected_indices += std::to_string(ARRAY_SIZE);
                } else if (i == 1) {
                    if (j < (ARRAY_SIZE - 1)) {
                        if (j < 1) expected_indices += std::to_string(j);
                        else expected_indices += std::to_string(j + 1);
                    } else expected_indices += std::to_string(ARRAY_SIZE);
                } else if ((i == 2) || (i == 3)) {
                    if (j < (ARRAY_SIZE - 3)) expected_indices += std::to_string(j + 3);
                    else expected_indices += std::to_string(ARRAY_SIZE);
                } else if ((i == 4) || (i == 5)) {
                    if (j == 0) expected_indices += std::to_string(j);
                    else expected_indices += std::to_string(ARRAY_SIZE);
                } else {
                    expected_indices += std::to_string(ARRAY_SIZE);
                }

                if (j < (ARRAY_SIZE - 1)) expected_indices += ", ";
            }
            expected_indices += "]";

            if (actual_indices != expected_indices) {
                std::cerr << "[" << my_pe << "]: indices[" << i << "] = " << actual_indices
                          << ", expected " << expected_indices << std::endl;
            }
        }
        exit_code = 1;
    }

    if (exit_code) std::cout << "[" << my_pe << "] Test Failed" << std::endl;
    else std::cout << "[" << my_pe << "] Test Passed" << std::endl;

    ishmem_free(source);
    for (size_t i = 0; i < NUM_TESTS; i++) {
        sycl::free(indices[i], q);
    }
    sycl::free(indices, q);
    sycl::free(errors, q);
    sycl::free(status, q);
    sycl::free(ret, q);
    sycl::free(ret_check, q);

    ishmem_finalize();

    return exit_code;
}
