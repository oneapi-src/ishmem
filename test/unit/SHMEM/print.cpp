/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <unistd.h>
#include <common.h>

/* TODO
 * Test for the printf feature of the ring communications
 */

void print2(sycl::queue q, char *hostbuf, char *devbuf, const char *format, int val)
{
    int size = sprintf(hostbuf, format, val);
    q.memcpy(devbuf, hostbuf, static_cast<size_t>(size));
    try {
        q.single_task([=]() {
             ishmemx_print(devbuf, ishmemx_print_msg_type_t::DEBUG);
         }).wait_and_throw();
    } catch (sycl::exception &e) {
        std::cout << "print raises SYCL exception ";
        std::cout << e.what();
        std::cout << std::endl;
    }
}

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmem_init();
    sycl::queue q;
    char *hostbuf = sycl::malloc_host<char>(4096, q);
    CHECK_ALLOC(hostbuf);
    char *devbuf = sycl::malloc_device<char>(4096, q);
    ishmemx_print("Host print\n", ishmemx_print_msg_type_t::DEBUG);
    for (int i = 0; i < 10; i += 1) {
        print2(q, hostbuf, devbuf, "Hello %d\n", i);
    }

    sycl::free(hostbuf, q);
    sycl::free(devbuf, q);
    ishmem_finalize();
    return exit_code;
}
