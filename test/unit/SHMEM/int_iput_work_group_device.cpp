/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <common.h>

constexpr int array_size = 393;

constexpr size_t x_size = 10;
constexpr size_t y_size = 2;
constexpr size_t z_size = 2;

// Constraint: max(dst * elems_to_copy, sst * elems_to_copy) <= array_size
constexpr size_t elems_to_copy = 80;
constexpr ptrdiff_t dst = 3;
constexpr ptrdiff_t sst = 2;

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmemx_attr_t attr = {};
    test_init_attr(&attr);
    ishmemx_init_attr(&attr);

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    sycl::queue q;

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;

    int *source = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(source);

    int *dest_group_1d = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(dest_group_1d);
    int *dest_group_2d = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(dest_group_2d);
    int *dest_group_3d = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(dest_group_3d);
    int *dest_sub_group = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(dest_sub_group);

    int *errors = (int *) sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);

    /* Initialize source data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>{array_size, array_size}, [=](sycl::nd_item<1> idx) {
            size_t i = idx.get_global_linear_id();
            source[i] =
                0x4000000 + (((my_pe + 1) % npes) << 24) + (my_pe << 20) + static_cast<int>(i);
            dest_group_1d[i] = 0;
            dest_group_2d[i] = 0;
            dest_group_3d[i] = 0;
            dest_sub_group[i] = 0;
        });
    });
    e_init.wait_and_throw();

    ishmem_barrier_all();

    /* Perform get work group operations */
    auto e1 = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(x_size), sycl::range<1>(x_size)),
                       [=](sycl::nd_item<1> it) {
                           int my_dev_pe = ishmem_my_pe();
                           int my_dev_npes = ishmem_n_pes();

                           auto grp = it.get_group();
                           ishmemx_int_iput_work_group(dest_group_1d, source, dst, sst,
                                                       elems_to_copy, (my_dev_pe + 1) % my_dev_npes,
                                                       grp);
                       });
    });
    auto e2 = q.submit([&](sycl::handler &h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(x_size, y_size), sycl::range<2>(x_size, y_size)),
            [=](sycl::nd_item<2> it) {
                int my_dev_pe = ishmem_my_pe();
                int my_dev_npes = ishmem_n_pes();

                auto grp = it.get_group();
                ishmemx_int_iput_work_group(dest_group_2d, source, dst, sst, elems_to_copy,
                                            (my_dev_pe + 1) % my_dev_npes, grp);
            });
    });
    auto e3 = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(x_size, y_size, z_size),
                                         sycl::range<3>(x_size, y_size, z_size)),
                       [=](sycl::nd_item<3> it) {
                           int my_dev_pe = ishmem_my_pe();
                           int my_dev_npes = ishmem_n_pes();

                           auto grp = it.get_group();
                           ishmemx_int_iput_work_group(dest_group_3d, source, dst, sst,
                                                       elems_to_copy, (my_dev_pe + 1) % my_dev_npes,
                                                       grp);
                       });
    });
    size_t *debug = (size_t *) sycl::malloc_host(5 * sizeof(size_t), q);
    CHECK_ALLOC(debug);
    auto e4 = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(x_size, y_size, z_size),
                                         sycl::range<3>(x_size, y_size, z_size)),
                       [=](sycl::nd_item<3> it) {
                           int my_dev_pe = ishmem_my_pe();
                           int my_dev_npes = ishmem_n_pes();

                           auto grp = it.get_sub_group();
                           size_t sub_group_id = grp.get_group_linear_id();
                           size_t num_sub_groups = grp.get_group_linear_range();
                           size_t min_nelems_sub_group = elems_to_copy / num_sub_groups;
                           size_t my_nelems_sub_group = min_nelems_sub_group;
                           size_t carry_over = elems_to_copy % num_sub_groups;

                           size_t x = (carry_over < sub_group_id) ? carry_over : sub_group_id;
                           if (sub_group_id < carry_over) my_nelems_sub_group += 1;

                           size_t sub_group_start_idx = (x * (min_nelems_sub_group + 1)) +
                                                        ((sub_group_id - x) * min_nelems_sub_group);

                           ishmemx_int_iput_work_group(dest_sub_group + (sub_group_start_idx * dst),
                                                       source + (sub_group_start_idx * sst), dst,
                                                       sst, my_nelems_sub_group,
                                                       (my_dev_pe + 1) % my_dev_npes, grp);
                       });
    });
    e1.wait_and_throw();
    e2.wait_and_throw();
    e3.wait_and_throw();
    e4.wait_and_throw();

    ishmem_barrier_all();

    /* Verify data */
    *errors = 0;
    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            int source_pe = (my_pe > 0) ? (my_pe - 1) : (npes - 1);
            int source_idx = 0;
            int expected_value = 0;
            for (int i = 0; i < array_size; ++i) {
                if (((i % dst) == 0) && (i < (dst * elems_to_copy))) {
                    expected_value = 0x4000000 + (((source_pe + 1) % npes) << 24) +
                                     ((source_pe % npes) << 20) + (source_idx);
                    source_idx += sst;
                } else {
                    expected_value = 0;
                }

                if (dest_group_1d[i] != expected_value) {
                    *errors = *errors + 1;
                }
                if (dest_group_2d[i] != expected_value) {
                    *errors = *errors + 1;
                }
                if (dest_group_3d[i] != expected_value) {
                    *errors = *errors + 1;
                }
                if (dest_sub_group[i] != expected_value) {
                    *errors = *errors + 1;
                }
            }
        });
    });
    e_verify.wait_and_throw();

    if (*errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
        exit_code = 1;
    } else {
        std::cout << "No errors" << std::endl;
    }

    sycl::free(errors, q);

    ishmem_free(source);
    ishmem_free(dest_group_1d);
    ishmem_free(dest_group_2d);
    ishmem_free(dest_group_3d);
    ishmem_free(dest_sub_group);

    ishmem_finalize();

    return exit_code;
}
