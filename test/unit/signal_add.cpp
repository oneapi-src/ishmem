/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <common.h>

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

    auto e_init = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            /* This value is send to the neighboring PE (mype+1)%npes. Therefore, at destination,
             * this is same as 123456 + destpe */
            *src = (long) 123456 + (long) (mype + 1) % npes;
            *sgnl = 123;
        });
    });
    e_init.wait_and_throw();
    ishmem_barrier_all();

    /* Test signal add from device */
    auto e_run = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            ishmem_long_put(dest, src, 1, (mype + 1) % npes);
            ishmem_fence();
            ishmemx_signal_add(sgnl, 123, (mype + 1) % npes);
        });
    });
    e_run.wait_and_throw();

    /* Initialized value + signal from device*/
    auto e_wait = q.submit([&](sycl::handler &h) {
        h.single_task([=]() { ishmem_uint64_wait_until(sgnl, ISHMEM_CMP_EQ, 246); });
    });
    e_wait.wait_and_throw();

    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            if (dest[0] != 123456 + (long) (mype)) *errors = *errors + 1;
        });
    });
    e_verify.wait_and_throw();
    if (*errors > 0) {
        std::cerr << "Expected: " << 123456 + mype << "Recieved: " << dest[0] << std::endl;
        rc = EXIT_FAILURE;
    }

    ishmem_barrier_all();

    /* Test signal add from host */
    ishmemx_signal_add(sgnl, 123, (mype + 1) % npes);
    ishmem_barrier_all();

    /* Initialized value + signal from device + signal from host */
    if (ishmem_uint64_atomic_fetch(sgnl, mype) != 123 * 3) {
        std::cerr << "ishmem_signal_add failed" << std::endl;
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
