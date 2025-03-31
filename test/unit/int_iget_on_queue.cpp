/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <common.h>

constexpr size_t array_size = 20;

// Constraint: max(dst * elems_to_copy, sst * elems_to_copy) <= array_size
constexpr size_t elems_to_copy = 5;
constexpr ptrdiff_t dst = 3;
constexpr ptrdiff_t sst = 2;

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmem_init();

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    sycl::queue q;

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;

    int *source = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(source);
    int *target = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(target);
    int *errors = sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);

    printf("[%d] source = %p target = %p\n", my_pe, source, target);

    /* Initialize source data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>{array_size, array_size}, [=](sycl::nd_item<1> idx) {
            size_t i = idx.get_global_id()[0];
            source[i] = (my_pe << 16) + static_cast<int>(i);
            target[i] = (my_pe << 16) + 0xface;
        });
    });
    e_init.wait_and_throw();

    ishmem_barrier_all();

    /* Perform iget operation */
    ishmemx_int_iget_on_queue(target, source, dst, sst, elems_to_copy, (my_pe + 1) % npes, q);
    q.wait_and_throw();

    ishmem_barrier_all();
    *errors = 0;
    /* Verify data */
    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            int source_idx = 0;
            for (size_t i = 0; i < array_size; ++i) {
                if (((i % dst) == 0) && (i < (dst * elems_to_copy))) {
                    if (target[i] != (((my_pe + 1) % npes) << 16) + source_idx) {
                        *errors = *errors + 1;
                    }
                    source_idx += sst;
                } else {
                    if (target[i] != ((my_pe << 16) + 0xface)) {
                        *errors = *errors + 1;
                    }
                }
            }
        });
    });
    e_verify.wait_and_throw();

    if (*errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
        int *hosttarget = sycl::malloc_host<int>(array_size, q);
        CHECK_ALLOC(hosttarget);
        q.memcpy(hosttarget, target, sizeof(int) * array_size).wait_and_throw();
        int source_idx = 0;
        for (int i = 0; i < array_size; i += 1) {
            if (((i % dst) == 0) && (i < (dst * elems_to_copy))) {
                if (hosttarget[i] != (((my_pe + 1) % npes) << 16) + source_idx) {
                    fprintf(stdout, "[%d] index %d expected 0x%08x got 0x%08x\n", my_pe, i,
                            (uint) ((((my_pe + 1) % npes) << 16) + source_idx), hosttarget[i]);
                }
                source_idx += sst;
            } else {
                if (hosttarget[i] != ((my_pe << 16) + 0xface)) {
                    fprintf(stdout, "[%d] index %d expected 0x%08x got 0x%08x\n", my_pe, i,
                            ((my_pe << 16) + 0xface), hosttarget[i]);
                }
            }
        }
        sycl::free(hosttarget, q);
        exit_code = 1;
    } else {
        std::cout << "No errors" << std::endl;
    }

    fflush(stdout);
    sycl::free(errors, q);
    ishmem_free(source);
    ishmem_free(target);

    ishmem_finalize();

    return exit_code;
}
