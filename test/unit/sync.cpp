/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <common.h>

int main(int argc, char *argv[])
{
    int exit_code = 0;

    ishmem_init();

    sycl::queue q;

    const int my_pe = ishmem_my_pe();

    if (my_pe == 0) {
        std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
                  << std::endl;
        std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
                  << std::endl;
    }

    int *val_host = (int *) sycl::malloc_host(sizeof(int), q);
    CHECK_ALLOC(val_host);
    int *val_dev = (int *) ishmem_malloc(sizeof(int));
    CHECK_ALLOC(val_dev);
    q.single_task([=]() { *val_dev = 0; }).wait_and_throw();
    ishmem_barrier_all();

    // ishmem_sync_all(), host-initiated
    if (my_pe == 0) {
        q.memset(val_dev, 1, 1).wait_and_throw();
        ishmem_sync_all();
    } else {
        ishmem_sync_all();
        q.single_task([=]() { *val_dev = ishmem_int_g(val_dev, 0); }).wait_and_throw();
    }
    q.memcpy(val_host, val_dev, sizeof(int)).wait_and_throw();
    if (*val_host != 1) {
        std::cout << "[" << my_pe << "]: [ERROR] host-inititated ishmem_sync_all test failed"
                  << std::endl;
        exit_code = 1;
    }
    ishmem_sync_all();

    // ishmem_sync_all(), device-initiated
    if (my_pe == 0) {
        q.single_task([=]() {
             *val_dev = 2;
             ishmem_sync_all();
         }).wait_and_throw();
    } else {
        q.single_task([=]() {
             ishmem_sync_all();
             *val_dev = ishmem_int_g(val_dev, 0);
         }).wait_and_throw();
    }
    q.memcpy(val_host, val_dev, sizeof(int)).wait_and_throw();
    if (*val_host != 2) {
        std::cout << "[" << my_pe << "]: [ERROR] device-inititated ishmem_sync_all test failed"
                  << std::endl;
        exit_code = 1;
    }
    ishmem_sync_all();

    // ishmemx_sync_all_work_group(), device-initiated
    if (my_pe == 0) {
        q.parallel_for(sycl::nd_range<1>{1024, 1024}, [=](sycl::nd_item<1> idx) {
             auto grp = idx.get_group();
             if (grp.leader()) *val_dev = 3;
             ishmemx_sync_all_work_group(grp);
         }).wait_and_throw();
    } else {
        q.parallel_for(sycl::nd_range<1>{1024, 1024}, [=](sycl::nd_item<1> idx) {
             auto grp = idx.get_group();
             ishmemx_sync_all_work_group(grp);
             if (grp.leader()) *val_dev = ishmem_int_g(val_dev, 0);
         }).wait_and_throw();
    }
    q.memcpy(val_host, val_dev, sizeof(int)).wait_and_throw();
    q.memcpy(val_host, val_dev, sizeof(int)).wait_and_throw();
    if (*val_host != 3) {
        std::cout << "[" << my_pe
                  << "]: [ERROR] device-inititated ishmemx_sync_all_work_group test failed"
                  << std::endl;
        exit_code = 1;
    }

    sycl::free(val_host, q);
    ishmem_free(val_dev);
    ishmem_finalize();

    if (exit_code) printf("[%d] Test Failed\n", my_pe);
    else printf("[%d] Test Passed\n", my_pe);

    return exit_code;
}
