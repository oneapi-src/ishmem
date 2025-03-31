/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <iostream>
#include <common.h>

using std::cerr, std::endl;

int main()
{
    ishmem_init();

    // get version from host
    int maj, min;
    ishmem_info_get_version(&maj, &min);

    // get version from device
    sycl::queue q;
    int *dev_ver = (int *) sycl::malloc_host<int>(2, q);
    CHECK_ALLOC(dev_ver);
    auto e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() { ishmem_info_get_version(dev_ver, dev_ver + 1); });
    });
    e1.wait_and_throw();

    // Test
    if (dev_ver[0] != ISHMEM_MAJOR_VERSION || dev_ver[1] != ISHMEM_MINOR_VERSION) {
        cerr << "get_version() from device failed" << endl;
        cerr << "Received: " << dev_ver[0] << "." << dev_ver[1] << endl;
        goto failure;
    }

    if (maj != ISHMEM_MAJOR_VERSION || min != ISHMEM_MINOR_VERSION) {
        cerr << "get_version() from host failed" << endl;
        cerr << "Received: " << maj << "." << min << endl;
        goto failure;
    }

    sycl::free(dev_ver, q);
    ishmem_finalize();
    return EXIT_SUCCESS;

failure:
    cerr << "Expected version: " << ISHMEM_MAJOR_VERSION << "." << ISHMEM_MINOR_VERSION << endl;
    cerr << "ishmem_info_get_version() failed!" << endl;
    sycl::free(dev_ver, q);
    ishmem_finalize();
    return EXIT_FAILURE;
}
