/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <iostream>
#include <CL/sycl.hpp>
#include <string.h>
#include <common.h>

using std::cerr, std::endl;

int main()
{
    ishmem_init();

    // Ensure length of name is less than ISHMEM_MAX_NAME_LEN
    std::string vendor_string(ISHMEM_VENDOR_STRING);
    if (vendor_string.length() >= ISHMEM_MAX_NAME_LEN) {
        cerr << "ishmem_info_get_name() failed" << endl;
        cerr << "Length of name" << vendor_string << " exceeds " << ISHMEM_MAX_NAME_LEN << endl;
        ishmem_finalize();
        return EXIT_FAILURE;
    }

    // get name from device
    sycl::queue q;
    char *dev_name = (char *) sycl::malloc_host<char>(ISHMEM_MAX_NAME_LEN, q);
    CHECK_ALLOC(dev_name);
    auto e1 = q.submit(
        [&](sycl::handler &h) { h.single_task([=]() { ishmem_info_get_name(dev_name); }); });
    e1.wait_and_throw();

    // get name from host
    char host_name[ISHMEM_MAX_NAME_LEN];
    ishmem_info_get_name(host_name);

    // Test
    if (strcmp(dev_name, ISHMEM_VENDOR_STRING)) {
        cerr << "ishmem_info_get_name() from device failed" << endl;
        cerr << "Received: " << dev_name << endl;
        cerr << "Expected: " << ISHMEM_VENDOR_STRING << endl;
        goto failure;
    }
    if (strcmp(host_name, ISHMEM_VENDOR_STRING)) {
        cerr << "ishmem_info_get_name() from host failed" << endl;
        cerr << "Received: " << host_name << endl;
        cerr << "Expected: " << ISHMEM_VENDOR_STRING << endl;
        goto failure;
    }

    sycl::free(dev_name, q);
    ishmem_finalize();
    return EXIT_SUCCESS;

failure:
    sycl::free(dev_name, q);
    ishmem_finalize();
    return EXIT_FAILURE;
}
