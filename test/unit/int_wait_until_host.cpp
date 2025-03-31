/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <unistd.h>
#include <common.h>

// One passing and one failing test per conditional
#define NUM_TESTS      6
#define NUM_WORK_ITEMS 4

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmem_init();

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    if (npes % 2) {
        std::cerr << "[WARN] int_wait_until_host must be run with an even "
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

    int *source = (int *) ishmem_malloc(NUM_TESTS * sizeof(int));
    CHECK_ALLOC(source);
    int *trigger = (int *) ishmem_malloc(NUM_TESTS * sizeof(int));
    CHECK_ALLOC(trigger);

    /* Initialize source data */
    q.submit([&](sycl::handler &h) {
         h.single_task([=]() {
             for (int i = 0; i < NUM_TESTS; i++) {
                 source[i] = 1;

                 if (i == 0) trigger[i] = 0;
                 else if (i == 1) trigger[i] = 0;
                 else if (i == 2) trigger[i] = 3;
                 else if (i == 3) trigger[i] = 4;
                 else if (i == 4) trigger[i] = 0;
                 else if (i == 5) trigger[i] = 0;
             }
         });
     }).wait_and_throw();
    ishmem_barrier_all();

    /* Perform wait_until operations */
    if (my_pe % 2) {
        ishmem_int_wait_until(&source[0], ISHMEM_CMP_EQ, 0);
        ishmem_int_wait_until(&source[1], ISHMEM_CMP_NE, 1);
        ishmem_int_wait_until(&source[2], ISHMEM_CMP_GT, 2);
        ishmem_int_wait_until(&source[3], ISHMEM_CMP_GE, 3);
        ishmem_int_wait_until(&source[4], ISHMEM_CMP_LT, 1);
        ishmem_int_wait_until(&source[5], ISHMEM_CMP_LE, 0);
    } else {
        q.submit([&](sycl::handler &h) {
             h.single_task([=]() {
                 for (size_t i = 0; i < NUM_TESTS; i++) {
                     ishmem_int_atomic_set(&source[i], trigger[i], my_pe + 1);
                 }
             });
         }).wait_and_throw();
    }

    std::cout << "[PE " << my_pe << "] No errors" << std::endl;

    ishmem_free(source);
    ishmem_free(trigger);

    ishmem_finalize();

    return exit_code;
}
