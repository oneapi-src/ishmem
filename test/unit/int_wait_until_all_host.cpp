/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <unistd.h>
#include <common.h>

// Using a reduced array size until a resolution is reached as to why
// repeated calls to non-fetching atomic operations eventually lead to an
// "Operation retry limit exceeded issue" when using CXI provider to
// facilitate communication.
// #define ARRAY_SIZE 5096

#define ARRAY_SIZE 32
#define NUM_TESTS  7

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmem_init();

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    if (npes % 2) {
        std::cerr << "[WARN] int_wait_until_all_host must be run with an even "
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
                 for (size_t j = 0; j < ARRAY_SIZE; j++) {
                     source[i][j] = 1;

                     if (i == 0) trigger[i][j] = 0;
                     else if (i == 1) trigger[i][j] = 0;
                     else if (i == 2) trigger[i][j] = 3;
                     else if (i == 3) trigger[i][j] = 4;
                     else if (i == 4) trigger[i][j] = 0;
                     else if (i == 5) trigger[i][j] = 0;
                     else if (i == 6) {
                         status[j] = 0;
                         trigger[i][j] = 5;
                     }
                 }
             }
         });
     }).wait_and_throw();
    ishmem_barrier_all();

    /* Perform wait_until_all operations */
    if (my_pe % 2) {
        ishmem_int_wait_until_all(source[0], ARRAY_SIZE, NULL, ISHMEM_CMP_EQ, 0);
        ishmem_int_wait_until_all(source[1], ARRAY_SIZE, NULL, ISHMEM_CMP_NE, 1);
        ishmem_int_wait_until_all(source[2], ARRAY_SIZE, NULL, ISHMEM_CMP_GT, 2);
        ishmem_int_wait_until_all(source[3], ARRAY_SIZE, NULL, ISHMEM_CMP_GE, 3);
        ishmem_int_wait_until_all(source[4], ARRAY_SIZE, NULL, ISHMEM_CMP_LT, 1);
        ishmem_int_wait_until_all(source[5], ARRAY_SIZE, NULL, ISHMEM_CMP_LE, 0);
        status[0] = status[1] = 1;
        ishmem_int_wait_until_all(source[6], ARRAY_SIZE, status, ISHMEM_CMP_EQ, 5);
    } else {
        q.submit([&](sycl::handler &h) {
             h.single_task([=]() {
                 for (size_t i = 0; i < NUM_TESTS; i++)
                     for (size_t j = 0; j < ARRAY_SIZE; j++) {
                         ishmem_int_atomic_set(&source[i][j], trigger[i][j], my_pe + 1);
                     }
             });
         }).wait_and_throw();
    }

    std::cout << "[PE " << my_pe << "] No errors" << std::endl;

    for (size_t i = 0; i < NUM_TESTS; i++) {
        ishmem_free(source[i]);
        ishmem_free(trigger[i]);
    }
    sycl::free(source, q);
    sycl::free(trigger, q);
    sycl::free(status, q);

    ishmem_finalize();

    return exit_code;
}
