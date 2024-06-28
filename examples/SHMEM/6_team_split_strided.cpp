/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * This program is derived from the shmem_team_split_strided example in the
 * OpenSHMEM v1.5 specification.
 */

#include <ishmem.h>
#include <ishmemx.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int rank, npes;
    int t_pe, t_size;
    int ret;
    ishmem_team_t even_team;
    ishmem_team_config_t *config;

    ishmem_init();
    config = NULL;
    rank = ishmem_my_pe();
    npes = ishmem_n_pes();

    if (npes < 2) {
        fprintf(stderr, "ERR - Requires at least 2 PEs\n");
        ishmem_finalize();
        return 0;
    }

    ret = ishmem_team_split_strided(ISHMEM_TEAM_WORLD, 0, 2, (npes + 1) / 2, config, 0, &even_team);
    if (ret != 0) {
        ishmem_finalize();
        return EXIT_FAILURE;
    }

    t_size = ishmem_team_n_pes(even_team);
    t_pe = ishmem_team_my_pe(even_team);

    sycl::queue q;
    int *errors = sycl::malloc_host<int>(1, q);
    *errors = 0;

    if (even_team != ISHMEM_TEAM_INVALID) {
        if ((rank % 2 != 0) || (rank / 2 != t_pe) || ((npes + 1) / 2 != t_size)) {
            printf("[%d] Error on even_team\n", rank);
            return EXIT_FAILURE;
        }
    } else {
        if ((rank % 2 == 0) || (t_pe != -1) || (t_size != -1)) {
            printf("[%d] Error on non even_team\n", rank);
            return EXIT_FAILURE;
        }
    }

    int *dev_buf = (int *) ishmem_malloc(sizeof(int));
    int *dev_sum = (int *) ishmem_calloc(1, sizeof(int));

    /* Do a simple sum reduction and broadcast on the even_team from within a sycl kernel */
    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            *dev_buf = t_pe;

            if (even_team != ISHMEM_TEAM_INVALID) {
                ishmem_int_sum_reduce(even_team, dev_sum, dev_buf, 1);

                if (*dev_sum != (t_size * (t_size - 1) / 2)) {
                    *errors += 1;
                    ishmemx_print("Wrong reduce on even_team (device)\n",
                                  ishmemx_print_msg_type_t::ERROR);
                }

                ishmem_int_broadcast(even_team, dev_buf, dev_sum, 1, 0);

                if (*dev_buf != (t_size * (t_size - 1) / 2)) {
                    *errors += 1;
                    ishmemx_print("Wrong broadcast on even_team (device)\n",
                                  ishmemx_print_msg_type_t::ERROR);
                }

                *dev_sum = 123;
            }
        });
    });
    e_verify.wait_and_throw();

    /* A simple broadcast on the even_team using device memory from the host: */
    if (even_team != ISHMEM_TEAM_INVALID) {
        ishmem_int_broadcast(even_team, dev_buf, dev_sum, 1, 0);

        e_verify = q.submit([&](sycl::handler &h) {
            h.single_task([=]() {
                if (*dev_buf != 123) {
                    *errors += 1;
                    ishmemx_print("Wrong broadcast on even_team (host)\n",
                                  ishmemx_print_msg_type_t::ERROR);
                }
            });
        });
    }

    ishmem_barrier_all();

    ishmem_team_t world_team_copy;
    ret = ishmem_team_split_strided(ISHMEM_TEAM_WORLD, 0, 1, npes, config, 0, &world_team_copy);
    if (ret != 0) {
        ishmem_finalize();
        return EXIT_FAILURE;
    }

    ishmem_team_sync(world_team_copy);

    /* Now do a simple sum reduction on ISHMEM_TEAM_WORLD from within a sycl kernel */
    e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            *dev_sum = 0;
            *dev_buf = rank;
            int my_team_pe = ishmem_team_my_pe(world_team_copy);
            if (my_team_pe != rank) *errors += 1;

            /* These 3 calls perform the same operation */
            ishmem_team_sync(ISHMEM_TEAM_WORLD);
            // ishmem_team_sync(world_team_copy);
            // ishmem_sync_all();

            ishmem_int_sum_reduce(world_team_copy, dev_sum, dev_buf, 1);

            ishmem_barrier_all();
            if (*dev_sum != (npes * (npes - 1) / 2)) {
                *errors += 1;
                ishmemx_print("Wrong reduce on world_team (device)\n",
                              ishmemx_print_msg_type_t::ERROR);
            }
        });
    });
    e_verify.wait_and_throw();

    if (*errors == 0) {
        std::cout << "PE#" << rank << " SUCCESS - verified" << std::endl;
    } else {
        std::cout << "PE#" << rank << " FAILURE - Error count: " << *errors << std::endl;
    }

    ishmem_team_destroy(even_team);
    ishmem_team_destroy(world_team_copy);

    ishmem_free(dev_buf);
    ishmem_free(dev_sum);
    sycl::free(errors, q);

    ishmem_finalize();
    return 0;
}
