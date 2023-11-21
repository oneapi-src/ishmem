/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
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
        std::cerr << "[ERROR] int_wait_until_device must be run with an even "
                     "number of PEs"
                  << std::endl;
        exit_code = 1;
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
    int *trigger = (int *) sycl::malloc_host(NUM_TESTS * sizeof(int), q);
    CHECK_ALLOC(trigger);

    /* Initialize source data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (int i = 0; i < NUM_TESTS; i++) {
                source[i] = 1;

                if (i == 0)
                    trigger[i] = 0;
                else if (i == 1)
                    trigger[i] = 0;
                else if (i == 2)
                    trigger[i] = 3;
                else if (i == 3)
                    trigger[i] = 4;
                else if (i == 4)
                    trigger[i] = 0;
                else if (i == 5)
                    trigger[i] = 0;
            }
        });
    });
    e_init.wait_and_throw();
    ishmem_barrier_all();

    /* Perform wait_until operations */
    if (my_pe % 2) {
        auto e1 = q.submit([&](sycl::handler &h) {
            sycl::stream out(2048, 256, h);
            h.parallel_for(sycl::nd_range<1>{NUM_TESTS, NUM_TESTS}, [=](sycl::nd_item<1> it) {
                size_t id = it.get_local_id(0);

                if (id == 0)
                    ishmem_int_wait_until(&source[0], ISHMEM_CMP_EQ, 0);
                else if (id == 1)
                    ishmem_int_wait_until(&source[1], ISHMEM_CMP_NE, 1);
                else if (id == 2)
                    ishmem_int_wait_until(&source[2], ISHMEM_CMP_GT, 2);
                else if (id == 3)
                    ishmem_int_wait_until(&source[3], ISHMEM_CMP_GE, 3);
                else if (id == 4)
                    ishmem_int_wait_until(&source[4], ISHMEM_CMP_LT, 1);
                else if (id == 5)
                    ishmem_int_wait_until(&source[5], ISHMEM_CMP_LE, 0);
            });
        });
        e1.wait_and_throw();
    } else {
        q.submit([&](sycl::handler &h) {
             // FIXME: Use atomic operations?
             sleep(1);
             h.single_task([=]() { ishmem_int_put(source, trigger, NUM_TESTS, my_pe + 1); });
         }).wait_and_throw();
    }

    /* Reinitialize source data for work group wait_until */
    auto e_reinit = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (int i = 0; i < NUM_TESTS; i++) {
                source[i] = 1;
            }
        });
    });
    e_reinit.wait_and_throw();
    ishmem_barrier_all();

    /* Perform wait_until_work_group operations */
    if (my_pe % 2) {
        auto e1 = q.submit([&](sycl::handler &h) {
            sycl::stream out(2048, 256, h);
            h.parallel_for(
                sycl::nd_range<1>{NUM_TESTS * NUM_WORK_ITEMS, NUM_WORK_ITEMS},
                [=](sycl::nd_item<1> it) {
                    auto grp = it.get_group();
                    size_t grp_id = it.get_group_linear_id();

                    if (grp_id == 0)
                        ishmemx_int_wait_until_work_group(&source[0], ISHMEM_CMP_EQ, 0, grp);
                    else if (grp_id == 1)
                        ishmemx_int_wait_until_work_group(&source[1], ISHMEM_CMP_NE, 1, grp);
                    else if (grp_id == 2)
                        ishmemx_int_wait_until_work_group(&source[2], ISHMEM_CMP_GT, 2, grp);
                    else if (grp_id == 3)
                        ishmemx_int_wait_until_work_group(&source[3], ISHMEM_CMP_GE, 3, grp);
                    else if (grp_id == 4)
                        ishmemx_int_wait_until_work_group(&source[4], ISHMEM_CMP_LT, 1, grp);
                    else if (grp_id == 5)
                        ishmemx_int_wait_until_work_group(&source[5], ISHMEM_CMP_LE, 0, grp);
                });
        });
        e1.wait_and_throw();
    } else {
        sleep(1);
        q.submit([&](sycl::handler &h) {
             // FIXME: Use atomic operations?
             h.single_task([=]() { ishmem_int_put(source, trigger, NUM_TESTS, my_pe + 1); });
         }).wait_and_throw();
    }

    std::cout << "[PE " << my_pe << "] No errors" << std::endl;

    ishmem_free(source);
    sycl::free(trigger, q);

    ishmem_finalize();

    return exit_code;
}
