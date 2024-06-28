/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <unistd.h>
#include <common.h>

// One passing and one failing test per conditional
#define NUM_TESTS  7
#define ARRAY_SIZE 5

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmemx_attr_t attr = {};
    test_init_attr(&attr);
    ishmemx_init_attr(&attr);

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    if (npes % 2) {
        std::cerr << "[WARN] int_wait_until_some_host must be run with an even "
                     "number of PEs"
                  << std::endl;
        ishmem_finalize();
        return exit_code;
    }

    sycl::queue q;

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;

    int **source = sycl::malloc_host<int *>(NUM_TESTS, q);
    CHECK_ALLOC(source);
    int **trigger = sycl::malloc_host<int *>(NUM_TESTS, q);
    CHECK_ALLOC(trigger);
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

    for (size_t i = 0; i < NUM_TESTS; i++) {
        source[i] = (int *) ishmem_malloc(ARRAY_SIZE * sizeof(int));
        CHECK_ALLOC(source[i]);

        trigger[i] = (int *) ishmem_malloc(ARRAY_SIZE * sizeof(int));
        CHECK_ALLOC(trigger[i]);

        indices[i] = sycl::malloc_host<size_t>(ARRAY_SIZE, q);
        CHECK_ALLOC(indices[i]);
    }

    /* Initialize source data */
    q.submit([&](sycl::handler &h) {
         h.single_task([=]() {
             for (size_t i = 0; i < NUM_TESTS; i++) {
                 ret[i] = ARRAY_SIZE;
                 for (size_t j = 0; j < ARRAY_SIZE; j++) {
                     source[i][j] = 1;
                     indices[i][j] = ARRAY_SIZE;
                     status[j] = 0;

                     if ((i == 0) || (i == 1) || (i == 4) || (i == 5)) {
                         trigger[i][j] = 1;
                     }

                     if (i == 0 && j == 0) {
                         ret_check[i] = 1;
                         trigger[i][j] = 0;
                     } else if (i == 1 && j == ARRAY_SIZE - 1) {
                         ret_check[i] = 1;
                         trigger[i][j] = 0;
                     } else if (i == 2 && j > 0) {
                         ret_check[i] = ARRAY_SIZE - 1;
                         trigger[i][j] = 3;
                     } else if (i == 3 && j < ARRAY_SIZE - 1) {
                         ret_check[i] = ARRAY_SIZE - 1;
                         trigger[i][j] = 4;
                     } else if (i == 4 && j % 2 == 0) {
                         ret_check[i] = (ARRAY_SIZE + 1) / 2;
                         trigger[i][j] = 0;
                     } else if (i == 5 && j % 2 == 1) {
                         ret_check[i] = ARRAY_SIZE / 2;
                         trigger[i][j] = 0;
                     } else if (i == 6) {
                         /* status[0]/[1] are set to 1 so first two elements of ivars are ignored */
                         ret_check[i] = ARRAY_SIZE - 2;
                         trigger[i][j] = 5;
                     }
                 }
             }
         });
     }).wait_and_throw();
    ishmem_barrier_all();

    /* Perform wait_until operations */
    if (my_pe % 2) {
        /*
        The double-calling of ishmem_int_wait_until_some is intentional. The first time
        the API returns there is no guarantee w.r.t the return value (only that it is
        greater >= 1 if "status" mask is null or its entries do not span every element in
        in the "ivars" argument, otherwise 0 is returned) as it depends on what entry in
        "ivars" is being evaluated by the API when an update to the array's contents are
        performed - after finding an element in "ivars" that meets the "cmp" condition,
        the existing implementation will return after finishing its current iteration
        through the contents of "ivars"

        For the sake of validating the API, we call it a second time with the same
        arguments. For this second invocation we are guaranteed that all elements in
        "ivars" that meet the "cmp" condition will be captured and reflected in the
        return value of the API.
        */

        ret[0] =
            ishmem_int_wait_until_some(source[0], ARRAY_SIZE, indices[0], NULL, ISHMEM_CMP_EQ, 0);
        ishmem_barrier_all();
        ret[0] =
            ishmem_int_wait_until_some(source[0], ARRAY_SIZE, indices[0], NULL, ISHMEM_CMP_EQ, 0);

        ret[1] =
            ishmem_int_wait_until_some(source[1], ARRAY_SIZE, indices[1], NULL, ISHMEM_CMP_NE, 1);
        ishmem_barrier_all();
        ret[1] =
            ishmem_int_wait_until_some(source[1], ARRAY_SIZE, indices[1], NULL, ISHMEM_CMP_NE, 1);

        ret[2] =
            ishmem_int_wait_until_some(source[2], ARRAY_SIZE, indices[2], NULL, ISHMEM_CMP_GT, 2);
        ishmem_barrier_all();
        ret[2] =
            ishmem_int_wait_until_some(source[2], ARRAY_SIZE, indices[2], NULL, ISHMEM_CMP_GT, 2);

        ret[3] =
            ishmem_int_wait_until_some(source[3], ARRAY_SIZE, indices[3], NULL, ISHMEM_CMP_GE, 3);
        ishmem_barrier_all();
        ret[3] =
            ishmem_int_wait_until_some(source[3], ARRAY_SIZE, indices[3], NULL, ISHMEM_CMP_GE, 3);

        ret[4] =
            ishmem_int_wait_until_some(source[4], ARRAY_SIZE, indices[4], NULL, ISHMEM_CMP_LT, 1);
        ishmem_barrier_all();
        ret[4] =
            ishmem_int_wait_until_some(source[4], ARRAY_SIZE, indices[4], NULL, ISHMEM_CMP_LT, 1);

        ret[5] =
            ishmem_int_wait_until_some(source[5], ARRAY_SIZE, indices[5], NULL, ISHMEM_CMP_LE, 0);
        ishmem_barrier_all();
        ret[5] =
            ishmem_int_wait_until_some(source[5], ARRAY_SIZE, indices[5], NULL, ISHMEM_CMP_LE, 0);

        status[0] = status[1] = 1;
        ret[6] =
            ishmem_int_wait_until_some(source[6], ARRAY_SIZE, indices[6], status, ISHMEM_CMP_EQ, 5);
        ishmem_barrier_all();
        ret[6] =
            ishmem_int_wait_until_some(source[6], ARRAY_SIZE, indices[6], status, ISHMEM_CMP_EQ, 5);
    } else {
        for (size_t i = 0; i < NUM_TESTS; i++) {
            q.submit([&](sycl::handler &h) {
                 h.single_task([=]() {
                     for (size_t j = 0; j < ARRAY_SIZE; j++) {
                         ishmem_int_atomic_set(&source[i][j], trigger[i][j], my_pe + 1);
                     }
                 });
             }).wait_and_throw();
            ishmem_barrier_all();
        }
    }

    if (my_pe % 2) {
        *errors = 0;
        /* Verify data */
        q.submit([&](sycl::handler &h) {
             h.single_task([=]() {
                 for (size_t i = 0; i < NUM_TESTS; ++i) {
                     if (ret[i] != ret_check[i]) {
                         (*errors) += 1;
                         continue;
                     }

                     if (i == 0) {
                         if (indices[i][0] != 0) (*errors) += 1;
                     } else if (i == 1) {
                         if (indices[i][0] != (ARRAY_SIZE - 1)) (*errors) += 1;
                     } else if (i == 2) {
                         for (size_t j = 0; j < (ARRAY_SIZE - 1); j++) {
                             if (indices[i][j] != (j + 1)) {
                                 (*errors) += 1;
                                 break;
                             }
                         }
                     } else if (i == 3) {
                         for (size_t j = 0; j < (ARRAY_SIZE - 1); j++) {
                             if (indices[i][j] != j) {
                                 (*errors) += 1;
                                 break;
                             }
                         }
                     } else if (i == 4) {
                         for (size_t j = 0; j < ((ARRAY_SIZE + 1) / 2); j++) {
                             if (indices[i][j] != (j * 2)) {
                                 (*errors) += 1;
                                 break;
                             }
                         }
                     } else if (i == 5) {
                         for (size_t j = 0; j < (ARRAY_SIZE / 2); j++) {
                             if (indices[i][j] != ((j * 2) + 1)) {
                                 (*errors) += 1;
                                 break;
                             }
                         }
                     } else if (i == 6) {
                         for (size_t j = 0; j < (ARRAY_SIZE - 2); j++) {
                             if (indices[i][j] != (j + 2)) {
                                 (*errors) += 1;
                                 break;
                             }
                         }
                     }
                 }
             });
         }).wait_and_throw();
    }

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
                    if (j == 0) expected_indices += std::to_string(ARRAY_SIZE - 1);
                    else expected_indices += std::to_string(ARRAY_SIZE);
                } else if (i == 2) {
                    if (j < (ARRAY_SIZE - 1)) expected_indices += std::to_string(j + 1);
                    else expected_indices += std::to_string(ARRAY_SIZE);
                } else if (i == 3) {
                    if (j < (ARRAY_SIZE - 1)) expected_indices += std::to_string(j);
                    else expected_indices += std::to_string(ARRAY_SIZE);
                } else if (i == 4) {
                    if (j < ((ARRAY_SIZE + 1) / 2)) expected_indices += std::to_string(j * 2);
                    else expected_indices += std::to_string(ARRAY_SIZE);
                } else if (i == 5) {
                    if (j < (ARRAY_SIZE / 2)) expected_indices += std::to_string((j * 2) + 1);
                    else expected_indices += std::to_string(ARRAY_SIZE);
                } else if (i == 6) {
                    if (j < (ARRAY_SIZE - 2)) expected_indices += std::to_string(j + 2);
                    else expected_indices += std::to_string(ARRAY_SIZE);
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

    for (size_t i = 0; i < NUM_TESTS; i++) {
        ishmem_free(source[i]);
        ishmem_free(trigger[i]);
        sycl::free(indices[i], q);
    }
    sycl::free(source, q);
    sycl::free(trigger, q);
    sycl::free(indices, q);
    sycl::free(status, q);
    sycl::free(ret, q);
    sycl::free(ret_check, q);
    sycl::free(errors, q);

    ishmem_finalize();

    return exit_code;
}
