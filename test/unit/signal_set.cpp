/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <common.h>
#include <limits.h>

int main(int argc, char *argv[])
{
    ishmem_init();

    int rc = EXIT_SUCCESS;

    sycl::queue q;
    const int mype = ishmem_my_pe();
    const int npes = ishmem_n_pes();

    uint64_t *sgnl = (uint64_t *) ishmem_calloc(1, sizeof(uint64_t));
    CHECK_ALLOC(sgnl);
    long *src = (long *) ishmem_calloc(1, sizeof(long));
    CHECK_ALLOC(src);
    long *dest = (long *) ishmem_calloc(1, sizeof(long));
    CHECK_ALLOC(dest);
    int *errors = sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);
    uint64_t sigval;

    auto e_init = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            /* This value is send to the neighboring PE (mype+1)%npes. Therefore, at destination,
             * this is same as 123456 + destpe */
            *src = (long) 123456 + (long) (mype + 1) % npes;
            *sgnl = std::numeric_limits<uint64_t>::max();
        });
    });
    e_init.wait_and_throw();
    ishmem_barrier_all();

    /* Test signal_set from device*/
    auto e_run = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            ishmem_long_put(dest, src, 1, (mype + 1) % npes);
            ishmem_fence();
            ishmemx_signal_set(sgnl, (uint64_t) ((mype + 1) % npes), (mype + 1) % npes);
        });
    });
    e_run.wait_and_throw();

    auto e_wait = q.submit([&](sycl::handler &h) {
        h.single_task([=]() { ishmem_uint64_wait_until(sgnl, ISHMEM_CMP_EQ, (uint64_t) mype); });
    });
    e_wait.wait_and_throw();

    *errors = 0;
    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            if (dest[0] != 123456 + (long) (mype)) *errors = *errors + 1;
        });
    });
    e_verify.wait_and_throw();
    if (*errors > 0) {
        std::cerr << "Expected: " << 123456 + mype << "Received: " << dest[0] << std::endl;
        rc = EXIT_FAILURE;
    }

    ishmem_barrier_all();

    /* Test signal_set from host */
    ishmemx_signal_set(sgnl, 123, (mype + 1) % npes);
    ishmem_barrier_all();
    sigval = ishmem_uint64_atomic_fetch(sgnl, mype);
    if (sigval != 123) {
        std::cerr << "signal_set failed" << std::endl;
        std::cerr << "Expected: " << 123 << "Received: " << sigval << std::endl;
        rc = EXIT_FAILURE;
    }

    sycl::free(errors, q);
    ishmem_free(src);
    ishmem_free(dest);
    ishmem_free(sgnl);
    ishmem_finalize();

    if (rc) std::cout << mype << ": Test Failed" << std::endl;
    else std::cout << mype << ": Test Passed" << std::endl;
    return rc;
}
