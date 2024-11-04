/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <cstdlib>
#include <iostream>
#include <ishmem.h>
#include <ishmemx.h>

int main(int argc, char **argv)
{
    // Initialize ISHMEM
    ishmem_init();

    // Get PE ID
    int my_pe = ishmem_my_pe();
    // Get total number of PEs
    int npes = ishmem_n_pes();

    std::cout << "I'm PE " << my_pe << " out of " << npes << " PEs." << std::endl;

    // Finalize before exit
    ishmem_finalize();
    return EXIT_SUCCESS;
}
