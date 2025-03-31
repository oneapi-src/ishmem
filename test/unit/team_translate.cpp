/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portions derived from Sandia OpenSHMEM (https://github.com/Sandia-OpenSHMEM/SOS)
 * For license and copyright information, see the LICENSE and third-party-programs.txt
 * files in the top level directory of this distribution.
 *
 * This test validates operations on/between the two teams formed by even
 * ranked PEs from ISHMEM_TEAM_WORLD and by multiple-of-3 ranked PEs from
 * ISHMEM_TEAM_WORLD. The teams are created using the ishmem_team_split_strided
 * operation, then ishmem_team_translate_pe and ishmem_int_sum_reduction are
 * checked for consistency and correctness.
 */

#include <ishmem.h>
#include <ishmemx.h>
#include <stdio.h>
#include <common.h>

constexpr int array_size = 1 << 10;

int main(void)
{
    int my_pe, npes;
    int t_2_size, t_3_size, t_pe_2, t_pe_3, t_pe_2_to_3, t_pe_3_to_2;
    ishmem_team_t team_2s, team_3s;
    ishmem_team_config_t *config;

    ishmem_init();

    my_pe = ishmem_my_pe();
    npes = ishmem_n_pes();

    sycl::queue q;

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;
    config = NULL;

    ishmem_team_split_strided(ISHMEM_TEAM_WORLD, 0, 2, ((npes - 1) / 2) + 1, config, 0, &team_2s);
    ishmem_team_split_strided(ISHMEM_TEAM_WORLD, 0, 3, ((npes - 1) / 3) + 1, config, 0, &team_3s);

    t_2_size = ishmem_team_n_pes(team_2s);
    t_3_size = ishmem_team_n_pes(team_3s);

    t_pe_3 = ishmem_team_my_pe(team_3s);
    t_pe_2 = ishmem_team_my_pe(team_2s);
    t_pe_3_to_2 = ishmem_team_translate_pe(team_3s, t_pe_3, team_2s);
    t_pe_2_to_3 = ishmem_team_translate_pe(team_2s, t_pe_2, team_3s);

    int *errors = sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);
    *errors = 0;

    if (my_pe % 2 == 0 && my_pe % 3 == 0) {
        if (t_pe_2 == -1 || t_pe_3 == -1 || t_pe_2_to_3 == -1 || t_pe_3_to_2 == -1) {
            printf("ERROR: PE %d, t_pe_2=%d, t_pe_3=%d, t_pe_3_to_2=%d, t_pe_2_to_3=%d\n", my_pe,
                   t_pe_2, t_pe_3, t_pe_3_to_2, t_pe_2_to_3);
            ++errors;
        }
    } else if (my_pe % 2 == 0) {
        if (t_pe_2 == -1 || t_pe_3 != -1 || t_pe_2_to_3 != -1 || t_pe_3_to_2 != -1) {
            printf("ERROR: PE %d, t_pe_2=%d, t_pe_3=%d, t_pe_3_to_2=%d, t_pe_2_to_3=%d\n", my_pe,
                   t_pe_2, t_pe_3, t_pe_3_to_2, t_pe_2_to_3);
            ++errors;
        }
    } else if (my_pe % 3 == 0) {
        if (t_pe_2 != -1 || t_pe_3 == -1 || t_pe_2_to_3 != -1 || t_pe_3_to_2 != -1) {
            printf("ERROR: PE %d, t_pe_2=%d, t_pe_3=%d, t_pe_3_to_2=%d, t_pe_2_to_3=%d\n", my_pe,
                   t_pe_2, t_pe_3, t_pe_3_to_2, t_pe_2_to_3);
            ++errors;
        }
    } else {
        if (t_pe_2 != -1 || t_pe_3 != -1 || t_pe_2_to_3 != -1 || t_pe_3_to_2 != -1) {
            printf("ERROR: PE %d, t_pe_2=%d, t_pe_3=%d, t_pe_3_to_2=%d, t_pe_2_to_3=%d\n", my_pe,
                   t_pe_2, t_pe_3, t_pe_3_to_2, t_pe_2_to_3);
            ++errors;
        }
    }

    fflush(stdout);

    int *source = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(source);
    int *target = (int *) ishmem_calloc(array_size, sizeof(int));
    CHECK_ALLOC(target);
    int *source_3s = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(source_3s);
    int *target_3s = (int *) ishmem_calloc(array_size, sizeof(int));
    CHECK_ALLOC(target_3s);

    /* Initialize source data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>{array_size, array_size}, [=](sycl::nd_item<1> idx) {
            int i = static_cast<int>(idx.get_global_id()[0]);
            source[i] = ((i % (t_pe_2 + 2)) << 16) + i;
            source_3s[i] = ((i % (t_pe_3 + 2)) << 16) + i;
        });
    });
    e_init.wait_and_throw();

    ishmem_barrier_all();

    /* Perform translation and reduction consistency checks */
    auto e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            int dev_pe = ishmem_my_pe();

            /* This is not a recommended method; it is a consistency test */
            int d_pe_2 = ishmem_team_translate_pe(ISHMEM_TEAM_WORLD, dev_pe, team_2s);
            int d_pe_3 = ishmem_team_translate_pe(ISHMEM_TEAM_WORLD, dev_pe, team_3s);
            int d_pe_3_to_2 = ishmem_team_translate_pe(team_3s, t_pe_3, team_2s);
            int d_pe_2_to_3 = ishmem_team_translate_pe(team_2s, t_pe_2, team_3s);

            if (d_pe_2 == -1 && team_2s != ISHMEM_TEAM_INVALID) {
                *errors += 1;
                ishmemx_print("Inconsistent team validity\n", ishmemx_print_msg_type_t::DEBUG);
            }

            if (d_pe_2 != -1 && team_2s == ISHMEM_TEAM_INVALID) {
                *errors += 1;
                ishmemx_print("Inconsistent team invalidity\n", ishmemx_print_msg_type_t::DEBUG);
            }

            if (dev_pe % 2 == 0 && dev_pe % 3 == 0) {
                if (d_pe_2 == -1 || d_pe_3 == -1 || d_pe_2_to_3 == -1 || d_pe_3_to_2 == -1) {
                    *errors += 1;
                    ishmemx_print("bad multiple of 6 team ID \n", ishmemx_print_msg_type_t::DEBUG);
                }
            } else if (dev_pe % 2 == 0) {
                if (d_pe_2 == -1 || d_pe_3 != -1 || d_pe_2_to_3 != -1 || d_pe_3_to_2 != -1) {
                    *errors += 1;
                    ishmemx_print("bad multiple of 2 team ID \n", ishmemx_print_msg_type_t::DEBUG);
                }
            } else if (dev_pe % 3 == 0) {
                if (d_pe_2 != -1 || d_pe_3 == -1 || d_pe_2_to_3 != -1 || d_pe_3_to_2 != -1) {
                    *errors += 1;
                    ishmemx_print("bad multiple of 3 team ID \n", ishmemx_print_msg_type_t::DEBUG);
                }
            } else {
                if (d_pe_2 != -1 || d_pe_3 != -1 || d_pe_2_to_3 != -1 || d_pe_3_to_2 != -1) {
                    *errors += 1;
                    ishmemx_print("bad non-multiple team ID \n", ishmemx_print_msg_type_t::DEBUG);
                }
            }

            if (team_2s != ISHMEM_TEAM_INVALID) {
                ishmem_int_sum_reduce(team_2s, target, source, array_size);
            }

            if (team_3s != ISHMEM_TEAM_INVALID) {
                ishmem_int_sum_reduce(team_3s, target_3s, source_3s, array_size);
            }
        });
    });
    e1.wait_and_throw();

    ishmem_barrier_all();

    /* Verify data */
    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            if (team_2s != ISHMEM_TEAM_INVALID) {
                for (int i = 0; i < array_size; ++i) {
                    int cur_sum = 0;
                    for (int j = 0; j < t_2_size; ++j) {
                        cur_sum += ((i % (j + 2)) << 16) + i;
                    }
                    if (target[i] != cur_sum) {
                        *errors += 1;
                    }
                }
            }
            if (team_3s != ISHMEM_TEAM_INVALID) {
                for (int i = 0; i < array_size; ++i) {
                    int cur_sum = 0;
                    for (int j = 0; j < t_3_size; ++j) {
                        cur_sum += ((i % (j + 2)) << 16) + i;
                    }
                    if (target_3s[i] != cur_sum) {
                        *errors += 1;
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
        int *hosttarget_3s = sycl::malloc_host<int>(array_size, q);
        CHECK_ALLOC(hosttarget_3s);
        q.memcpy(hosttarget, target, sizeof(int) * array_size).wait_and_throw();
        q.memcpy(hosttarget_3s, target_3s, sizeof(int) * array_size).wait_and_throw();
        for (int i = 0; i < array_size; ++i) {
            int cur_sum = 0;
            if (team_2s != ISHMEM_TEAM_INVALID) {
                for (int j = 0; j < t_2_size; ++j) {
                    cur_sum += ((i % (j + 2)) << 16) + i;
                }
                if (hosttarget[i] != cur_sum) {
                    fprintf(stdout, "2s [%d] index %d expected 0x%08x got 0x%08x\n", my_pe, i,
                            cur_sum, hosttarget[i]);
                }
            }
            if (team_3s != ISHMEM_TEAM_INVALID) {
                cur_sum = 0;
                for (int j = 0; j < t_3_size; ++j) {
                    cur_sum += ((i % (j + 2)) << 16) + i;
                }
                if (hosttarget_3s[i] != cur_sum) {
                    fprintf(stdout, "3s [%d] index %d expected 0x%08x got 0x%08x\n", my_pe, i,
                            cur_sum, hosttarget_3s[i]);
                }
            }
        }
        sycl::free(hosttarget, q);
        sycl::free(hosttarget_3s, q);
    } else {
        std::cout << my_pe << ": No errors" << std::endl;
    }

    fflush(stdout);
    sycl::free(errors, q);
    ishmem_free(source);
    ishmem_free(target);
    ishmem_free(source_3s);
    ishmem_free(target_3s);

    ishmem_finalize();
    return *errors != 0;
}
