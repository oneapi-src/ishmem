/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <common.h>

constexpr int array_size = 10;

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmem_init();

    sycl::queue q;

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    int *source = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(source);
    int *target = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(target);
    uint64_t *sig_addr = (uint64_t *) ishmem_calloc(1, sizeof(uint64_t));
    CHECK_ALLOC(sig_addr);
    int *errors = (int *) sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);

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
    int dest_pe = (my_pe + 1) % npes;

    /* Perform put-signal operation with SIGNAL_SET */
    if (my_pe == 0) {
        ishmemx_int_put_signal_on_queue(target, source, array_size, sig_addr,
                                        static_cast<uint64_t>(dest_pe + 1), ISHMEM_SIGNAL_SET,
                                        dest_pe, q);
        ishmemx_uint64_wait_until_on_queue(sig_addr, ISHMEM_CMP_EQ,
                                           static_cast<uint64_t>(my_pe + 1), q);
    } else {
        ishmemx_uint64_wait_until_on_queue(sig_addr, ISHMEM_CMP_EQ,
                                           static_cast<uint64_t>(my_pe + 1), q);
        ishmemx_int_put_signal_on_queue(target, source, array_size, sig_addr,
                                        static_cast<uint64_t>(dest_pe + 1), ISHMEM_SIGNAL_SET,
                                        dest_pe, q);
    }
    q.wait_and_throw();

    ishmem_barrier_all();
    *errors = 0;
    /* Verify data */
    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (int i = 0; i < array_size; ++i) {
                if (target[i] != (((my_pe - 1 + npes) % npes) << 16) + i) {
                    *errors = *errors + 1;
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
        for (int i = 0; i < array_size; i += 1) {
            if (hosttarget[i] != (((my_pe - 1 + npes) % npes) << 16) + i) {
                fprintf(stdout, "[%d] index %d expected 0x%08x got 0x%08x\n", my_pe, i,
                        (((my_pe - 1 + npes) % npes) << 16) + i, hosttarget[i]);
            }
        }
        sycl::free(hosttarget, q);
        exit_code = 1;
    } else {
        std::cout << "[" << my_pe << "] No errors" << std::endl;
    }

    fflush(stdout);
    ishmem_free(source);
    ishmem_free(target);
    ishmem_free(sig_addr);
    sycl::free(errors, q);

    ishmem_finalize();

    return exit_code;
}
