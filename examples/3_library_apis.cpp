/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <ishmem.h>
#include <ishmemx.h>

constexpr int array_size = 10; /* num_threads = array_size / chunk_size */
constexpr int chunk_size = 2;  /* data partition/chunk size per thread */

int main()
{
    sycl::queue q;
    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    /* Initialize ISHMEM
     * The ISHMEM launch script will set things up so that ishmem uses
     * the same GPU device as the SYCL queue above
     */
    ishmem_init();

    /* Query APIs */
    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    std::cout << "Hello from PE " << my_pe << std::endl;

    int num_threads = array_size / chunk_size;

    /* Symmetric data object allocation APIs */
    int *src = (int *) ishmem_malloc(array_size * sizeof(int));
    int *dst_put = (int *) ishmem_calloc(array_size, sizeof(int));
    int *dst_get = (int *) ishmem_calloc(array_size, sizeof(int));
    int32_t *dst_fadd = (int32_t *) ishmem_calloc(1, sizeof(int32_t));
    int32_t *dst_xor = (int32_t *) ishmem_calloc(1, sizeof(int32_t));
    int *dst_sum = (int *) ishmem_calloc(1, sizeof(int));
    int *dst_wait = (int *) ishmem_calloc(1, sizeof(int));
    int *src_bcast = (int *) ishmem_malloc(sizeof(int));
    int *dst_bcast = (int *) ishmem_malloc(sizeof(int));
    int *fadd_ret = (int *) ishmem_calloc(1, sizeof(int));
    int *reduce_src = (int *) ishmem_calloc(1, sizeof(int));

    /* Error for validation. This memory is accessible to both host
     * code and device code, but is not remotely accessible
     */
    int *errors = sycl::malloc_host<int>(1, q);
    *errors = 0;

    /* Local data */
    int32_t fadd_val = 5;
    int32_t xor_val = 2;
    int32_t xor_init = 3;

    /* Random data to initialize */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            dst_fadd[0] = my_pe;
            reduce_src[0] = my_pe;
            dst_xor[0] = xor_init;

            /* set src buffers in this fasion:
             *   PE0-> [1, 2, 4, ...];
             *   PE1-> [2, 4, 8, ...];
             *   PE2-> [3, 6, 12, ...];
             */
            for (int i = 0; i < array_size; i++) {
                src[i] = (my_pe + 1) * (1 << i);
            }

            if (my_pe == 0) {
                src_bcast[0] = 42;
            }
        });
    });
    e_init.wait_and_throw();

    /* This is a host call that will make sure all PEs have gotten to
     * this point before any of the PEs proceed beyond.
     */
    ishmem_barrier_all();

    /* Run a test SYCL kernel (on every PE). The nd_range sets up a
     * single SYCL work_group that has num_threads workers each of
     * which will get one index value
     */
    auto e_test = q.parallel_for(
        sycl::nd_range<1>{static_cast<size_t>(num_threads), static_cast<size_t>(num_threads)},
        [=](sycl::nd_item<1> idx) {
            /* Device query APIs */
            int my_dev_pe = ishmem_my_pe();
            int my_dev_npes = ishmem_n_pes();
            /* Get the work_group handle; will be used in the device extension APIs */
            auto grp = idx.get_group();
            /* Get the work_item index */
            size_t i = idx.get_global_id()[0];

            /* The operations below operate on either the "next" pe around a ring of PEs
             * or the "previous" pe around the ring of PEs
             */
            int next_pe = (my_dev_pe + 1) % my_dev_npes;
            int prev_pe = (my_dev_pe == 0 ? my_dev_npes - 1 : my_dev_pe - 1);

            /* Put - ring */
            /* each of the num_threads workers will make a non-blocking put call to copy a
             * block of the src buffer on the local pe to the corresponding location in the
             * dst_put buffer on the remote pe
             */
            ishmem_putmem_nbi(&dst_put[i * chunk_size], &src[i * chunk_size],
                              sizeof(int) * chunk_size, next_pe);

            /* Device extension Get - ring */
            /* the above individual operation can be done directly by using the ishmem work_group
             * apis, as follows.  All the threads work together to accomplish the entire copy
             */
            ishmemx_getmem_nbi_work_group(dst_get, src, sizeof(int) * array_size, prev_pe, grp);

            /* This code waits until all GPU threads have reached this point, and then the group
             * leader calls ishmem_barrier_all to make sure that all PEs have reached this point
             * In addition, barrier will wait until all non-blocking operations have finished
             */
            sycl::group_barrier(grp);
            if (grp.leader()) ishmem_barrier_all();
            sycl::group_barrier(grp);
            /* This is equivalent to */
            ishmemx_barrier_all_work_group(grp);

            /* Broadcast */
            /* copy src_bcast on PE 0 to dst_bcast on all PEs */
            if (grp.leader()) ishmem_broadcastmem(dst_bcast, src_bcast, sizeof(int), 0);
            /* or, equivalently, we can use device extension Broadcast */
            ishmemx_broadcastmem_work_group(dst_bcast, src_bcast, sizeof(int), 0, grp);

            /* Atomic fetch add */
            *fadd_ret = ishmem_int32_atomic_fetch_add(dst_fadd, fadd_val, next_pe);

            /* Atomic XOR */
            ishmem_int32_atomic_xor(dst_xor, xor_val, 0);

            ishmemx_barrier_all_work_group(grp);

            /* Sum reduction */
            /* Sum of all the reduce_src values on all PEs are sent to the dst_sum on every PE */
            if (grp.leader()) ishmem_int_sum_reduce(dst_sum, reduce_src, 1);
            /* or, equivalently Device extension Sum reduction */
            ishmemx_int_sum_reduce_work_group(dst_sum, reduce_src, 1, grp);

            /* Atomic Increment + Wait Until */
            /* each of the GPU threads will separately increment dst_wait on next_pe */
            ishmem_int_atomic_inc(dst_wait, next_pe);
            /* leading thread will wait until the local dst_wait reached the desired value */
            if (grp.leader()) ishmem_int_wait_until(dst_wait, ISHMEM_CMP_GE, num_threads);
        });
    e_test.wait_and_throw();

    /* back on the host again, and will wait until all PEs have finished their GPU kernels */
    ishmem_barrier_all();

    /* Verify results */
    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            int init_val = (my_pe == 0 ? npes : my_pe);

            /* Put/Get rings */
            for (int i = 0; i < array_size; i++) {
                if ((dst_put[i] != dst_get[i]) || (dst_put[i] != init_val * pow(2, i))) {
                    *errors = *errors + 1;
                    break;
                }
            }

            /* Broadcast */
            if (*dst_bcast != 42) {
                *errors = *errors + 1;
            }

            /* Atomic XOR */
            if (my_pe == 0) {
                int xor_expected = xor_init;
                for (int i = 0; i < npes * num_threads; i++) {
                    xor_expected ^= xor_val;
                }

                if (*dst_xor != xor_expected) {
                    *errors = *errors + 1;
                }
            }

            /* Fetch-Add symmetric variable */
            if (*dst_fadd != my_pe + fadd_val * num_threads) {
                *errors = *errors + 1;
            }

            /* Sum reduction */
            if (my_pe == 0) {
                if (*dst_sum != (npes * (npes - 1) / 2)) {
                    *errors = *errors + 1;
                }
            }
        });
    });
    e_verify.wait_and_throw();

    if (*errors == 0) {
        std::cout << "PE#" << my_pe << " SUCCESS - verified put/get/amo/broadcast/reduce/sync"
                  << std::endl;
    } else {
        std::cout << "PE#" << my_pe << " FAILURE - Error count: " << *errors << std::endl;
    }

    /* Free memory */
    ishmem_free(src);
    ishmem_free(dst_put);
    ishmem_free(dst_get);
    ishmem_free(dst_fadd);
    ishmem_free(dst_xor);
    ishmem_free(dst_sum);
    ishmem_free(dst_wait);
    ishmem_free(src_bcast);
    ishmem_free(dst_bcast);
    ishmem_free(fadd_ret);
    ishmem_free(reduce_src);
    sycl::free(errors, q);

    /* Finalize */
    ishmem_finalize();

    return 0;
}
