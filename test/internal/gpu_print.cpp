/* Copyright (C) 2023-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ishmem/err.h>
#include <common.h>

void usage()
{
    std::cout << "remember to run with ISHMEM_DEBUG=1" << std::endl;
    std::cout << "remember to build with cmake -DENABLE_ERROR_CHECKING" << std::endl;
    printf(" run with one integer argument: \n");
    printf("1 try validate_parameter(pe=17)\n");
    printf("2 try validate_parameter(heap_base, 128)\n");
    printf("3 try validate_parameter(heap_base, bignum)\n");
    printf("4 try ishmem_int_g(heap_base, 17)\n");
    printf("5 try ishmem_int_get(goodbuf, heap_base - 128, 10, 0)\n");
    exit(1);
}

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmem_init();

    sycl::queue q;

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;

    // ISHMEM_DEBUG must be set for DEBUG and WARNING messages to be printed
    std::cout << "About to print parameter messages" << std::endl;
    void *heap_base = ishmem_malloc(64); /* kinda */
    if (argc != 2) usage();
    int cmd = atoi(argv[1]);
    if ((cmd < 1) || (cmd > 5)) usage();
    int *goodbuf = (int *) ishmem_calloc(sizeof(int), 256);
    try {
        q.single_task([=]() {
             ishmemx_print(__FILE__, __LINE__, __func__, "print to STDOUT\n",
                           ishmemx_print_msg_type_t::STDOUT);
             ishmemx_print(__FILE__, __LINE__, __func__, "print to STDERR\n",
                           ishmemx_print_msg_type_t::STDERR);
             ishmemx_print("print DEBUG message\n");
             if (cmd == 1) {
                 ishmemx_print(__FILE__, __LINE__, __func__, "check validate_parameters(17);\n",
                               ishmemx_print_msg_type_t::STDOUT);
                 validate_parameters(17);
             }
             if (cmd == 2) {
                 ishmemx_print(__FILE__, __LINE__, __func__,
                               "check validate_parameters(base,128);\n",
                               ishmemx_print_msg_type_t::STDOUT);
                 validate_parameters(heap_base, 128);
             }
             if (cmd == 3) {
                 ishmemx_print(__FILE__, __LINE__, __func__,
                               "check validate_parameters(base,1<<28);\n",
                               ishmemx_print_msg_type_t::STDOUT);
                 validate_parameters(heap_base, 1L << 29);
             }
             if (cmd == 4) {
                 ishmemx_print(__FILE__, __LINE__, __func__, "calling ishmem_int_g with bad pe\n",
                               ishmemx_print_msg_type_t::STDOUT);
                 ishmem_int_g(goodbuf, 17);
             }
             if (cmd == 5) {
                 ishmemx_print(__FILE__, __LINE__, __func__, "calling ishmem_int_g with bad pe\n",
                               ishmemx_print_msg_type_t::STDOUT);
                 ishmem_int_get(goodbuf, (int *) (((uintptr_t) heap_base) - 128), 16, 0);
             }
         }).wait_and_throw();
    } catch (sycl::exception &e) {
        exit_code = 1;
        std::cout << "print raises SYCL exception ";
        std::cout << e.what();
        std::cout << std::endl;
    }

    std::cout << "Done parameter messages, exit code " << exit_code << std::endl;

    ishmem_finalize();
    return exit_code;
}
