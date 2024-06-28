/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <unistd.h>
#include <common.h>

#define NUM_TESTS      6
#define NUM_WORK_ITEMS 4

#define SIGNAL_WAIT_UNTIL_TEST_SINGLE(cmp)                                                         \
    /* Perform wait_until operations */                                                            \
    if (my_pe % 2) {                                                                               \
        q.single_task([=]() {                                                                      \
             *ret = ishmem_signal_wait_until(sig_addr, cmp, cmp_values[cmp - 1]);                  \
         }).wait_and_throw();                                                                      \
    } else {                                                                                       \
        q.submit([&](sycl::handler &h) {                                                           \
             h.single_task([=]() {                                                                 \
                 ishmem_int_put_signal(source, dest, 1, sig_addr, trigger[cmp - 1],                \
                                       ISHMEM_SIGNAL_SET, my_pe + 1);                              \
             });                                                                                   \
         }).wait_and_throw();                                                                      \
    }                                                                                              \
    if (my_pe % 2) {                                                                               \
        /* Verify data */                                                                          \
        if (*ret != trigger[cmp - 1]) {                                                            \
            std::cerr << "[" << my_pe << "] ERROR, " << #cmp << " validation check failed"         \
                      << std::endl;                                                                \
            std::cerr << "[" << my_pe << "]: ret = " << *ret << ", expected " << trigger[cmp - 1]  \
                      << std::endl;                                                                \
            exit_code = 1;                                                                         \
        }                                                                                          \
    }                                                                                              \
    ishmem_barrier_all();

#define SIGNAL_WAIT_UNTIL_TEST_WORK_GROUP(cmp)                                                     \
    /* Perform wait_until operations */                                                            \
    if (my_pe % 2) {                                                                               \
        q.submit([&](sycl::handler &h) {                                                           \
             h.parallel_for(sycl::nd_range<1>{NUM_WORK_ITEMS, NUM_WORK_ITEMS},                     \
                            [=](sycl::nd_item<1> it) {                                             \
                                auto grp = it.get_group();                                         \
                                *ret = ishmemx_signal_wait_until_work_group(                       \
                                    sig_addr, cmp, cmp_values[cmp - 1], grp);                      \
                            });                                                                    \
         }).wait_and_throw();                                                                      \
    } else {                                                                                       \
        q.submit([&](sycl::handler &h) {                                                           \
             h.single_task([=]() {                                                                 \
                 ishmem_int_put_signal(source, dest, 1, sig_addr, trigger[cmp - 1],                \
                                       ISHMEM_SIGNAL_SET, my_pe + 1);                              \
             });                                                                                   \
         }).wait_and_throw();                                                                      \
    }                                                                                              \
    if (my_pe % 2) {                                                                               \
        /* Verify data */                                                                          \
        if (*ret != trigger[cmp - 1]) {                                                            \
            std::cerr << "[" << my_pe << "] ERROR, " << #cmp                                       \
                      << " work_group validation check failed" << std::endl;                       \
            std::cerr << "[" << my_pe << "]: ret = " << *ret << ", expected " << trigger[cmp - 1]  \
                      << std::endl;                                                                \
            exit_code = 1;                                                                         \
        }                                                                                          \
    }                                                                                              \
    ishmem_barrier_all();

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmemx_attr_t attr = {};
    test_init_attr(&attr);
    ishmemx_init_attr(&attr);

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    if (npes % 2) {
        std::cerr << "[WARN] signal_wait_until_device must be run with an even "
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
    int *dest = (int *) ishmem_malloc(NUM_TESTS * sizeof(int));
    CHECK_ALLOC(dest);
    uint64_t *sig_addr = (uint64_t *) ishmem_malloc(NUM_TESTS * sizeof(uint64_t));
    CHECK_ALLOC(sig_addr);
    uint64_t *trigger = sycl::malloc_host<uint64_t>(NUM_TESTS, q);
    CHECK_ALLOC(trigger);
    uint64_t *cmp_values = sycl::malloc_host<uint64_t>(NUM_TESTS, q);
    CHECK_ALLOC(cmp_values);
    uint64_t *ret = sycl::malloc_host<uint64_t>(1, q);
    CHECK_ALLOC(ret);

    /* Initialize source data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (int i = 0; i < NUM_TESTS; i++) {
                source[i] = 1;
                dest[i] = 0;
                *sig_addr = 1;
                *ret = UINT64_MAX;

                if (i == 0) {
                    trigger[i] = 0;
                    cmp_values[i] = 0;
                } else if (i == 1) {
                    trigger[i] = 1;
                    cmp_values[i] = 0;
                } else if (i == 2) {
                    trigger[i] = 3;
                    cmp_values[i] = 2;
                } else if (i == 3) {
                    trigger[i] = 4;
                    cmp_values[i] = 4;
                } else if (i == 4) {
                    trigger[i] = 1;
                    cmp_values[i] = 2;
                } else if (i == 5) {
                    trigger[i] = 0;
                    cmp_values[i] = 0;
                }
            }
        });
    });
    e_init.wait_and_throw();
    ishmem_barrier_all();

    SIGNAL_WAIT_UNTIL_TEST_SINGLE(ISHMEM_CMP_EQ)
    SIGNAL_WAIT_UNTIL_TEST_SINGLE(ISHMEM_CMP_NE)
    SIGNAL_WAIT_UNTIL_TEST_SINGLE(ISHMEM_CMP_GT)
    SIGNAL_WAIT_UNTIL_TEST_SINGLE(ISHMEM_CMP_GE)
    SIGNAL_WAIT_UNTIL_TEST_SINGLE(ISHMEM_CMP_LT)
    SIGNAL_WAIT_UNTIL_TEST_SINGLE(ISHMEM_CMP_LE)

    /* Reinitialize source data for work group wait_until */
    auto e_reinit = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (int i = 0; i < NUM_TESTS; i++) {
                *sig_addr = 1;
                *ret = UINT64_MAX;
            }
        });
    });
    e_reinit.wait_and_throw();
    ishmem_barrier_all();

    SIGNAL_WAIT_UNTIL_TEST_WORK_GROUP(ISHMEM_CMP_EQ)
    SIGNAL_WAIT_UNTIL_TEST_WORK_GROUP(ISHMEM_CMP_NE)
    SIGNAL_WAIT_UNTIL_TEST_WORK_GROUP(ISHMEM_CMP_GT)
    SIGNAL_WAIT_UNTIL_TEST_WORK_GROUP(ISHMEM_CMP_GE)
    SIGNAL_WAIT_UNTIL_TEST_WORK_GROUP(ISHMEM_CMP_LT)
    SIGNAL_WAIT_UNTIL_TEST_WORK_GROUP(ISHMEM_CMP_LE)

    if (exit_code) std::cout << "[" << my_pe << "] Test Failed" << std::endl;
    else std::cout << "[" << my_pe << "] Test Passed" << std::endl;

    ishmem_free(source);
    ishmem_free(dest);
    ishmem_free(sig_addr);
    sycl::free(trigger, q);
    sycl::free(cmp_values, q);
    sycl::free(ret, q);

    ishmem_finalize();

    return exit_code;
}
