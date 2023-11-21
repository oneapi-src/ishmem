/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <shmem.h>
#include <common.h>

constexpr int array_size = 10; /* num_threads = array_size / chunk_size */
constexpr int chunk_sz = 2;    /* data partition/chunk size per thread */

int main()
{
    sycl::queue q;
    ishmem_init();

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    std::cout << "Hello from PE " << my_pe << std::endl;

    auto platforms = sycl::platform::get_platforms();

    for (auto &platform : platforms) {
        std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << "\n";
        auto devices = platform.get_devices();
        for (auto &device : devices) {
            std::cout << "  Device: " << device.get_info<sycl::info::device::name>() << "\n";
        }
    }

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    sycl::range<1> num_threads{array_size / chunk_sz};

    int *src = (int *) sycl::malloc_shared<int>(array_size, q);
    CHECK_ALLOC(src);
    int *dst = (int *) sycl::malloc_shared<int>(array_size, q);
    CHECK_ALLOC(dst);

    // TODO: should use ishmem_calloc instead of shmem_calloc
    std::cerr << "Warning: Using shmem_calloc instead of ishmem_calloc\n";
    int *dst_put = (int *) shmem_calloc(array_size, sizeof(int));
    // int *dst_put = (int *) ishmem_calloc(array_size, sizeof(int));
    CHECK_ALLOC(dst_put);

    /* set src buffers in this fasion:
     *   PE0-> [1, 2, 3, ...];
     *   PE1-> [2, 3, 4, ...];
     *   PE2-> [3, 4, 5, ...]       */
    for (int i = 0; i < array_size; i++) {
        src[i] = shmem_my_pe() + i + 1;
    }

    shmem_barrier_all();

    auto e1 =
        q.parallel_for(sycl::nd_range<1>{num_threads, num_threads}, [=](sycl::nd_item<1> idx) {
            int my_dev_pe = ishmem_my_pe();
            int my_dev_npes = ishmem_n_pes();

            size_t i = idx.get_global_id()[0];

            /* Put ring */
            ishmem_int_put(&dst_put[i * chunk_sz], &src[i * chunk_sz], chunk_sz,
                           (my_dev_pe + 1) % my_dev_npes);

            /* Get ring */
            ishmem_int_get(&dst[i * chunk_sz], &dst_put[i * chunk_sz], chunk_sz,
                           (my_dev_pe + 1) % my_dev_npes);
        });
    q.wait();

    shmem_barrier_all();

    /* Verify results */
    int errors = 0;
    int src_pe = (my_pe == 0 ? npes - 1 : my_pe - 1);
    for (int i = 0; i < array_size; i++) {
        if (src[i] != dst[i] || dst_put[i] != src_pe + i + 1) {
            std::cerr << "element " << i << " on PE " << my_pe << " incorrect! "
                      << "src " << src[i] << " dst " << dst[i] << " dst_put " << dst_put[i]
                      << " exp " << src_pe + i + 1 << std::endl;
            errors++;
        }
    }

    if (errors == 0)
        std::cout << "PE#" << my_pe << " SUCCESS - verified get/put" << std::endl;
    else
        std::cout << "Verification FAILED" << std::endl;

    free(src);
    free(dst);

    // TODO: change to ishmem_free
    shmem_free(dst_put);

    ishmem_finalize();
    shmem_finalize();
    return 0;
}
