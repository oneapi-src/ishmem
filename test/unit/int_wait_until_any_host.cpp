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

    ishmem_init();

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    if (npes % 2) {
        std::cerr << "[WARN] int_wait_until_any_host must be run with an even "
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
    }

    /* Initialize source data */
    q.submit([&](sycl::handler &h) {
         h.single_task([=]() {
             for (size_t i = 0; i < NUM_TESTS; i++) {
                 ret[i] = ARRAY_SIZE;
                 ret_check[i] = (i % ARRAY_SIZE);

                 for (size_t j = 0; j < ARRAY_SIZE; j++) {
                     source[i][j] = 1;
                     if ((i == 0) || (i == 1) || (i == 4) || (i == 5)) {
                         trigger[i][j] = 1;
                     }
                     status[j] = 0;
                 }

                 if (i == 0) {
                     trigger[i][(i % ARRAY_SIZE)] = 0;
                 } else if (i == 1) {
                     trigger[i][(i % ARRAY_SIZE)] = 0;
                 } else if (i == 2) {
                     trigger[i][(i % ARRAY_SIZE)] = 3;
                 } else if (i == 3) {
                     trigger[i][(i % ARRAY_SIZE)] = 4;
                 } else if (i == 4) {
                     trigger[i][(i % ARRAY_SIZE)] = 0;
                 } else if (i == 5) {
                     trigger[i][(i % ARRAY_SIZE)] = 0;
                 } else if (i == 6) {
                     // status[0]/[1] are set to 1 so first two elements of ivars are ignored
                     ret_check[i] = 2;
                     trigger[i][2] = 5;
                 }
             }
         });
     }).wait_and_throw();
    ishmem_barrier_all();

    /* Perform wait_until operations */
    if (my_pe % 2) {
        ret[0] = ishmem_int_wait_until_any(source[0], ARRAY_SIZE, NULL, ISHMEM_CMP_EQ, 0);
        ret[1] = ishmem_int_wait_until_any(source[1], ARRAY_SIZE, NULL, ISHMEM_CMP_NE, 1);
        ret[2] = ishmem_int_wait_until_any(source[2], ARRAY_SIZE, NULL, ISHMEM_CMP_GT, 2);
        ret[3] = ishmem_int_wait_until_any(source[3], ARRAY_SIZE, NULL, ISHMEM_CMP_GE, 3);
        ret[4] = ishmem_int_wait_until_any(source[4], ARRAY_SIZE, NULL, ISHMEM_CMP_LT, 1);
        ret[5] = ishmem_int_wait_until_any(source[5], ARRAY_SIZE, NULL, ISHMEM_CMP_LE, 0);
        status[0] = status[1] = 1;
        ret[6] = ishmem_int_wait_until_any(source[6], ARRAY_SIZE, status, ISHMEM_CMP_EQ, 5);
    } else {
        for (size_t i = 0; i < NUM_TESTS; i++) {
            q.submit([&](sycl::handler &h) {
                 h.single_task([=]() {
                     for (size_t j = 0; j < ARRAY_SIZE; j++) {
                         ishmem_int_atomic_set(&source[i][j], trigger[i][j], my_pe + 1);
                     }
                 });
             }).wait_and_throw();
        }
    }

    ishmem_barrier_all();
    if (my_pe % 2) {
        *errors = 0;
        /* Verify data */
        q.submit([&](sycl::handler &h) {
             h.single_task([=]() {
                 for (size_t i = 0; i < NUM_TESTS; ++i) {
                     if (ret[i] != ret_check[i]) {
                         (*errors) += 1;
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
            }
        }
        exit_code = 1;
    }

    if (exit_code) std::cout << "[" << my_pe << "] Test Failed" << std::endl;
    else std::cout << "[" << my_pe << "] Test Passed" << std::endl;

    for (size_t i = 0; i < NUM_TESTS; i++) {
        ishmem_free(source[i]);
        ishmem_free(trigger[i]);
    }
    sycl::free(source, q);
    sycl::free(trigger, q);
    sycl::free(status, q);
    sycl::free(ret, q);
    sycl::free(ret_check, q);
    sycl::free(errors, q);

    ishmem_finalize();

    return exit_code;
}
