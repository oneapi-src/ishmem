/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <common.h>

#define ARRAY_SIZE 60
#define NUM_TESTS  12
#define X_SIZE     4
#define Y_SIZE     4
#define Z_SIZE     3

#define VALIDATE_RESULTS(err_msg)                                                                  \
    do {                                                                                           \
        *errors = 0;                                                                               \
        for (size_t i = 0; i < NUM_TESTS; ++i) {                                                   \
            if (ret[i] != ret_check[i]) {                                                          \
                (*errors) += 1;                                                                    \
            }                                                                                      \
        }                                                                                          \
        if (*errors > 0) {                                                                         \
            std::cerr << "[" << my_pe << "] ERROR, " << err_msg << ": " << *errors << std::endl;   \
            for (size_t i = 0; i < NUM_TESTS; i += 1) {                                            \
                if (ret[i] != ret_check[i]) {                                                      \
                    std::cerr << "[" << my_pe << "]: ret[" << i << "] = " << ret[i]                \
                              << ", expected " << ret_check[i] << std::endl;                       \
                }                                                                                  \
            }                                                                                      \
            exit_code = 1;                                                                         \
        }                                                                                          \
    } while (false);

#define RUN_WORK_GROUP_TESTS()                                                                     \
    do {                                                                                           \
        /* Should return true (1) */                                                               \
        ret[0] = ishmemx_int_test_all_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_EQ, 1, grp); \
        ret[1] = ishmemx_int_test_all_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_NE, 0, grp); \
        ret[2] = ishmemx_int_test_all_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GT, 0, grp); \
        ret[3] = ishmemx_int_test_all_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GE, 1, grp); \
        ret[4] = ishmemx_int_test_all_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_LT, 2, grp); \
        source[0] = 2;                                                                             \
        ret[5] =                                                                                   \
            ishmemx_int_test_all_work_group(source, ARRAY_SIZE, status, ISHMEM_CMP_LE, 1, grp);    \
        source[0] = 1;                                                                             \
                                                                                                   \
        /* Should return false (0) */                                                              \
        ret[6] = ishmemx_int_test_all_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_EQ, 0, grp); \
        ret[7] = ishmemx_int_test_all_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_NE, 1, grp); \
        ret[8] = ishmemx_int_test_all_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GT, 1, grp); \
        ret[9] = ishmemx_int_test_all_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GE, 2, grp); \
        ret[10] =                                                                                  \
            ishmemx_int_test_all_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_LT, 1, grp);      \
        ret[11] =                                                                                  \
            ishmemx_int_test_all_work_group(source, ARRAY_SIZE, NULL, ISHMEM_CMP_LE, 0, grp);      \
    } while (false);

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
    int *ret = sycl::malloc_host<int>(NUM_TESTS, q);
    CHECK_ALLOC(ret);
    int *ret_check = sycl::malloc_host<int>(NUM_TESTS, q);
    CHECK_ALLOC(ret_check);
    int *errors = sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);

    /* Initialize source data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            status[0] = 1;

            for (size_t i = 0; i < ARRAY_SIZE; i++) {
                source[i] = 1;
            }

            for (int i = 0; i < NUM_TESTS; i++) {
                if (i < (NUM_TESTS / 2)) ret_check[i] = 1;
                else ret_check[i] = 0;
            }
        });
    });
    e_init.wait_and_throw();

    /* Perform test_all operations - Single Thread */
    auto e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            // Should return true (1)
            ret[0] = ishmem_int_test_all(source, ARRAY_SIZE, NULL, ISHMEM_CMP_EQ, 1);
            ret[1] = ishmem_int_test_all(source, ARRAY_SIZE, NULL, ISHMEM_CMP_NE, 0);
            ret[2] = ishmem_int_test_all(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GT, 0);
            ret[3] = ishmem_int_test_all(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GE, 1);
            ret[4] = ishmem_int_test_all(source, ARRAY_SIZE, NULL, ISHMEM_CMP_LT, 2);
            source[0] = 2;
            ret[5] = ishmem_int_test_all(source, ARRAY_SIZE, status, ISHMEM_CMP_LE, 1);
            source[0] = 1;

            // Should return false (0)
            ret[6] = ishmem_int_test_all(source, ARRAY_SIZE, NULL, ISHMEM_CMP_EQ, 0);
            ret[7] = ishmem_int_test_all(source, ARRAY_SIZE, NULL, ISHMEM_CMP_NE, 1);
            ret[8] = ishmem_int_test_all(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GT, 1);
            ret[9] = ishmem_int_test_all(source, ARRAY_SIZE, NULL, ISHMEM_CMP_GE, 2);
            ret[10] = ishmem_int_test_all(source, ARRAY_SIZE, NULL, ISHMEM_CMP_LT, 1);
            ret[11] = ishmem_int_test_all(source, ARRAY_SIZE, NULL, ISHMEM_CMP_LE, 0);
        });
    });
    e1.wait_and_throw();

    /* Verify data - Single Thread */
    VALIDATE_RESULTS("single-thread validation check(s) failed");

    /* Perform test_all_work_group operations - 1-D Group */
    auto e2 = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>{X_SIZE, X_SIZE}, [=](sycl::nd_item<1> it) {
            auto grp = it.get_group();

            RUN_WORK_GROUP_TESTS();
        });
    });
    e2.wait_and_throw();

    /* Verify data - 1-D Group */
    VALIDATE_RESULTS("work_group (1-D) validation check(s) failed");

    /* Perform test_all_work_group operations - 2-D Group */
    auto e3 = q.submit([&](sycl::handler &h) {
        h.parallel_for(
            sycl::nd_range<2>{sycl::range<2>(X_SIZE, Y_SIZE), sycl::range<2>(X_SIZE, Y_SIZE)},
            [=](sycl::nd_item<2> it) {
                auto grp = it.get_group();

                RUN_WORK_GROUP_TESTS();
            });
    });
    e3.wait_and_throw();

    /* Verify data - 2-D Group */
    VALIDATE_RESULTS("work_group (2-D) validation check(s) failed");

    /* Perform test_all_work_group operations - 3-D Group*/
    auto e4 = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<3>{sycl::range<3>(X_SIZE, Y_SIZE, Z_SIZE),
                                         sycl::range<3>(X_SIZE, Y_SIZE, Z_SIZE)},
                       [=](sycl::nd_item<3> it) {
                           auto grp = it.get_group();

                           RUN_WORK_GROUP_TESTS();
                       });
    });
    e4.wait_and_throw();

    /* Verify data - 3-D Group */
    VALIDATE_RESULTS("work_group (3-D) validation check(s) failed");

    /* Perform test_all_work_group operations - Sub-group */
    auto e5 = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<3>{sycl::range<3>(X_SIZE, Y_SIZE, Z_SIZE),
                                         sycl::range<3>(X_SIZE, Y_SIZE, Z_SIZE)},
                       [=](sycl::nd_item<3> it) {
                           auto grp = it.get_sub_group();
                           if (grp.get_group_linear_id() == 0) {
                               RUN_WORK_GROUP_TESTS();
                           }
                       });
    });
    e5.wait_and_throw();

    /* Verify data - Sub-group */
    VALIDATE_RESULTS("work_group (sub-group) validation check(s) failed");

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
