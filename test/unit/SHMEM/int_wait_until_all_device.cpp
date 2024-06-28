/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <unistd.h>
#include <common.h>

#define ARRAY_SIZE 5096
#define NUM_TESTS  7
#define X_SIZE     8
#define Y_SIZE     8
#define Z_SIZE     8

#define INITIALIZE_SOURCE()                                                                        \
    do {                                                                                           \
        q.submit([&](sycl::handler &h) {                                                           \
             h.single_task([=]() {                                                                 \
                 for (size_t i = 0; i < NUM_TESTS; i++) {                                          \
                     for (size_t j = 0; j < ARRAY_SIZE; j++)                                       \
                         source[i][j] = 1;                                                         \
                 }                                                                                 \
             });                                                                                   \
         }).wait_and_throw();                                                                      \
        ishmem_barrier_all();                                                                      \
    } while (false);

#define RUN_WORK_GROUP_TESTS()                                                                     \
    do {                                                                                           \
        if (grp_id == 0)                                                                           \
            ishmemx_int_wait_until_all_work_group(source[0], ARRAY_SIZE, NULL, ISHMEM_CMP_EQ, 0,   \
                                                  grp);                                            \
        else if (grp_id == 1)                                                                      \
            ishmemx_int_wait_until_all_work_group(source[1], ARRAY_SIZE, NULL, ISHMEM_CMP_NE, 1,   \
                                                  grp);                                            \
        else if (grp_id == 2)                                                                      \
            ishmemx_int_wait_until_all_work_group(source[2], ARRAY_SIZE, NULL, ISHMEM_CMP_GT, 2,   \
                                                  grp);                                            \
        else if (grp_id == 3)                                                                      \
            ishmemx_int_wait_until_all_work_group(source[3], ARRAY_SIZE, NULL, ISHMEM_CMP_GE, 3,   \
                                                  grp);                                            \
        else if (grp_id == 4)                                                                      \
            ishmemx_int_wait_until_all_work_group(source[4], ARRAY_SIZE, NULL, ISHMEM_CMP_LT, 1,   \
                                                  grp);                                            \
        else if (grp_id == 5)                                                                      \
            ishmemx_int_wait_until_all_work_group(source[5], ARRAY_SIZE, NULL, ISHMEM_CMP_LE, 0,   \
                                                  grp);                                            \
        else if (grp_id == 6) {                                                                    \
            status[0] = status[1] = 1;                                                             \
            ishmemx_int_wait_until_all_work_group(source[6], ARRAY_SIZE, status, ISHMEM_CMP_EQ, 5, \
                                                  grp);                                            \
        }                                                                                          \
    } while (false);

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmemx_attr_t attr = {};
    test_init_attr(&attr);
    ishmemx_init_attr(&attr);

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    if (npes % 2) {
        std::cerr << "[WARN] int_wait_until_all_device must be run with an even "
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

    /* Perform wait_until_all operations - Single Thread*/
    if (my_pe % 2) {
        q.submit([&](sycl::handler &h) {
             h.parallel_for(sycl::nd_range<1>{NUM_TESTS, NUM_TESTS}, [=](sycl::nd_item<1> it) {
                 size_t id = it.get_local_id(0);

                 if (id == 0)
                     ishmem_int_wait_until_all(source[0], ARRAY_SIZE, NULL, ISHMEM_CMP_EQ, 0);
                 else if (id == 1)
                     ishmem_int_wait_until_all(source[1], ARRAY_SIZE, NULL, ISHMEM_CMP_NE, 1);
                 else if (id == 2)
                     ishmem_int_wait_until_all(source[2], ARRAY_SIZE, NULL, ISHMEM_CMP_GT, 2);
                 else if (id == 3)
                     ishmem_int_wait_until_all(source[3], ARRAY_SIZE, NULL, ISHMEM_CMP_GE, 3);
                 else if (id == 4)
                     ishmem_int_wait_until_all(source[4], ARRAY_SIZE, NULL, ISHMEM_CMP_LT, 1);
                 else if (id == 5)
                     ishmem_int_wait_until_all(source[5], ARRAY_SIZE, NULL, ISHMEM_CMP_LE, 0);
                 else if (id == 6) {
                     status[0] = status[1] = 1;
                     ishmem_int_wait_until_all(source[6], ARRAY_SIZE, status, ISHMEM_CMP_EQ, 5);
                 }
             });
         }).wait_and_throw();
    } else {
        q.submit([&](sycl::handler &h) {
             h.single_task([=]() {
                 for (size_t i = 0; i < NUM_TESTS; i++)
                     ishmem_int_put(source[i], trigger[i], ARRAY_SIZE, my_pe + 1);
             });
         }).wait_and_throw();
    }

    /* Reinitialize source data */
    INITIALIZE_SOURCE();

    /* Perform wait_until_all_work_group operations - 1-D Group */
    if (my_pe % 2) {
        q.submit([&](sycl::handler &h) {
             h.parallel_for(sycl::nd_range<1>{NUM_TESTS * X_SIZE, X_SIZE},
                            [=](sycl::nd_item<1> it) {
                                auto grp = it.get_group();
                                size_t grp_id = it.get_group_linear_id();

                                RUN_WORK_GROUP_TESTS();
                            });
         }).wait_and_throw();
    } else {
        q.submit([&](sycl::handler &h) {
             h.single_task([=]() {
                 for (size_t i = 0; i < NUM_TESTS; i++)
                     ishmem_int_put(source[i], trigger[i], ARRAY_SIZE, my_pe + 1);
             });
         }).wait_and_throw();
    }

    // Reinitialize source data
    INITIALIZE_SOURCE();

    /* Perform wait_until_all_work_group operations - 2-D Group */
    if (my_pe % 2) {
        q.submit([&](sycl::handler &h) {
             h.parallel_for(sycl::nd_range<2>{sycl::range<2>(NUM_TESTS * X_SIZE, Y_SIZE),
                                              sycl::range<2>(X_SIZE, Y_SIZE)},
                            [=](sycl::nd_item<2> it) {
                                auto grp = it.get_group();
                                size_t grp_id = it.get_group_linear_id();

                                RUN_WORK_GROUP_TESTS();
                            });
         }).wait_and_throw();
    } else {
        q.submit([&](sycl::handler &h) {
             h.single_task([=]() {
                 for (size_t i = 0; i < NUM_TESTS; i++)
                     ishmem_int_put(source[i], trigger[i], ARRAY_SIZE, my_pe + 1);
             });
         }).wait_and_throw();
    }

    // Reinitialize source data
    INITIALIZE_SOURCE();

    /* Perform wait_until_all_work_group operations - 3-D Group */
    if (my_pe % 2) {
        q.submit([&](sycl::handler &h) {
             h.parallel_for(sycl::nd_range<3>{sycl::range<3>(NUM_TESTS * X_SIZE, Y_SIZE, Z_SIZE),
                                              sycl::range<3>(X_SIZE, Y_SIZE, Z_SIZE)},
                            [=](sycl::nd_item<3> it) {
                                auto grp = it.get_group();
                                size_t grp_id = it.get_group_linear_id();

                                RUN_WORK_GROUP_TESTS();
                            });
         }).wait_and_throw();
    } else {
        q.submit([&](sycl::handler &h) {
             h.single_task([=]() {
                 for (size_t i = 0; i < NUM_TESTS; i++)
                     ishmem_int_put(source[i], trigger[i], ARRAY_SIZE, my_pe + 1);
             });
         }).wait_and_throw();
    }

    // Reinitialize source data
    INITIALIZE_SOURCE();

    /* Perform wait_until_all_work_group operations - Sub-group */
    if (my_pe % 2) {
        q.submit([&](sycl::handler &h) {
             h.parallel_for(sycl::nd_range<3>{sycl::range<3>(NUM_TESTS * X_SIZE, Y_SIZE, Z_SIZE),
                                              sycl::range<3>(X_SIZE, Y_SIZE, Z_SIZE)},
                            [=](sycl::nd_item<3> it) {
                                auto grp = it.get_sub_group();
                                size_t grp_id = grp.get_group_linear_id();

                                RUN_WORK_GROUP_TESTS();
                            });
         }).wait_and_throw();
    } else {
        q.submit([&](sycl::handler &h) {
             h.single_task([=]() {
                 for (size_t i = 0; i < NUM_TESTS; i++)
                     ishmem_int_put(source[i], trigger[i], ARRAY_SIZE, my_pe + 1);
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
