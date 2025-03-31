/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portions derived from Sandia OpenSHMEM (https://github.com/Sandia-OpenSHMEM/SOS)
 * For license and copyright information, see the LICENSE and third-party-programs.txt
 * files in the top level directory of this distribution.
 *
 * This test first does a local sum reduction of an arbitrary buffer across all
 * PEs in ISHMEMX_TEAM_NODE. Then each PE nominiates a leader PE from from
 * ISHMEMX_TEAM_NODE to accumulate the buffer across nodes in a hierarchical
 * manner.  This test will fail if:
 *     1) It is executed with fewer than 2 PEs.
 *     1) Instances of ISHMEMX_TEAM_NODE have differing sizes (e.g. on heterogeneous platforms).
 *     2) leader PEs do not have a regular stride between them (e.g, for irregular PE numberings).
 */

#include <common.h>
#include <stdio.h>

constexpr int array_size = 1 << 10;
constexpr size_t num_threads = 1 << 8;

int main(void)
{
    int my_pe, npes, team_node_n_pes;

    ishmem_init();

    my_pe = ishmem_my_pe();
    npes = ishmem_n_pes();

    if (npes < 2) {
        std::cerr << "ERR - Requires at least 2 PEs" << std::endl;
        ishmem_finalize();
        return 0;
    }

    team_node_n_pes = ishmem_team_n_pes(ISHMEMX_TEAM_NODE);

    int *peers = (int *) malloc(static_cast<size_t>(team_node_n_pes) * sizeof(int));
    int num_peers = 0;

    sycl::queue q;

    /* Print the team members on ISHMEMX_TEAM_NODE. */
    /* Synchronize each iteration for clean output. */
    for (int pe = 0; pe < npes; pe++) {
        if (pe == my_pe) {
            std::cout << "[PE: " << my_pe << "] ISHMEMX_TEAM_NODE peers: { ";
            for (int i = 0; i < npes; i++) {
                if (ishmem_team_translate_pe(ISHMEM_TEAM_WORLD, i, ISHMEMX_TEAM_NODE) != -1) {
                    peers[num_peers++] = i;
                    std::cout << i << " ";
                }
            }
            std::cout << "} (num_peers: " << num_peers << ")" << std::endl << std::flush;
            std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
                      << std::endl
                      << std::flush;
        }
        ishmem_sync_all();
    }

    if (num_peers != team_node_n_pes) {
        std::cerr << "Inconsistent number of peers in ISHMEMX_TEAM_NODE" << std::endl;
        ishmem_finalize();
        return 1;
    }

    free(peers);

    int *source = (int *) ishmem_malloc(array_size * sizeof(int));
    CHECK_ALLOC(source);
    int *target = (int *) ishmem_calloc(array_size, sizeof(int));
    CHECK_ALLOC(target);
    int *leader_pe = (int *) ishmem_calloc(1, sizeof(int));
    CHECK_ALLOC(leader_pe);
    int *leader_pes = (int *) ishmem_calloc(static_cast<size_t>(npes), sizeof(int));
    CHECK_ALLOC(leader_pes);
    int *team_size = (int *) ishmem_calloc(1, sizeof(int));
    CHECK_ALLOC(team_size);
    int *team_stride = (int *) ishmem_calloc(1, sizeof(int));
    CHECK_ALLOC(team_stride);
    int *team_sizes_collect = (int *) ishmem_calloc(static_cast<size_t>(npes), sizeof(int));
    CHECK_ALLOC(team_sizes_collect);
    int *team_strides_collect = (int *) ishmem_calloc(static_cast<size_t>(npes), sizeof(int));
    CHECK_ALLOC(team_strides_collect);

    /* Initialize source data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>{num_threads, num_threads}, [=](sycl::nd_item<1> idx) {
            int i = static_cast<int>(idx.get_global_id()[0]);
            assert(array_size > num_threads);
            int chunk_size = array_size / num_threads;
            int start = i * chunk_size;
            int end = (i == num_threads - 1) ? array_size : start + chunk_size;
            for (int j = start; j < end; j++)
                source[j] = ((j % (my_pe + 2)) << 16) + j;
        });
    });
    e_init.wait_and_throw();

    ishmem_barrier_all();

    int *errors = sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);
    *errors = 0;

    int *leader_stride_host = sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(leader_stride_host);
    *leader_stride_host = 0;

    e_init = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            /* Local sum reduce across ISHMEMX_TEAM_NODE: */
            ishmem_int_sum_reduce(ISHMEMX_TEAM_NODE, target, source, array_size);

            /* Check the reduction result: */
            for (int i = 0; i < array_size; i++) {
                int cur_sum = 0;
                for (int j = 0; j < team_node_n_pes; j++) {
                    int world_pe =
                        ishmem_team_translate_pe(ISHMEMX_TEAM_NODE, j, ISHMEM_TEAM_WORLD);
                    cur_sum += ((i % (world_pe + 2)) << 16) + i;
                }
                if (target[i] != cur_sum) {
                    *errors += 1;
                }
            }

            *leader_pe = my_pe;

            ishmem_team_sync(ISHMEMX_TEAM_NODE);

            /* The leader PE is the one with the smallest world PE ID */
            ishmem_int_min_reduce(ISHMEMX_TEAM_NODE, leader_pe, leader_pe, 1);

            for (int i = 0; i < npes; i++)
                leader_pes[i] = -1;

            /* Verify that all ISHMEMX_TEAM_NODE team instances have the same number of PEs */
            team_size[0] = team_node_n_pes;

            ishmem_sync_all();

            ishmem_int_fcollect(team_sizes_collect, team_size, 1);

            for (int i = 0; i < npes; i++)
                if (team_sizes_collect[i] != team_node_n_pes) *errors += 1;

            ishmem_sync_all();

            /* fcollect all the leader PEs: */
            ishmem_int_fcollect(leader_pes, leader_pe, 1);

            /* Check that the leader PEs have a regular stride */
            assert(npes > 1 && leader_pes[team_node_n_pes] != -1);
            int curr_stride = leader_pes[team_node_n_pes] - leader_pes[0];
            for (int i = team_node_n_pes; i + team_node_n_pes < npes; i += team_node_n_pes) {
                int stride = 0;
                if (leader_pes[i + team_node_n_pes] != -1) {
                    stride = leader_pes[i + team_node_n_pes] - leader_pes[i];
                } else {
                    break;
                }
                if (stride != curr_stride) {
                    ishmemx_print("no valid leader stride\n", ishmemx_print_msg_type_t::ERROR);
                }
            }
            *leader_stride_host = curr_stride;

            team_stride[0] = curr_stride;

            ishmem_int_fcollect(team_strides_collect, team_stride, 1);

            for (int i = 0; i < npes; i++)
                if (team_strides_collect[i] != curr_stride) *errors += 1;
        });
    });
    e_init.wait_and_throw();

    /* Verify and print ISHMEMX_TEAM_NODE sum reduction result: */
    if (*errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
        int *hosttarget = sycl::malloc_host<int>(array_size, q);
        CHECK_ALLOC(hosttarget);
        q.memcpy(hosttarget, target, sizeof(int) * array_size).wait_and_throw();
        for (int i = 0; i < array_size; ++i) {
            int cur_sum = 0;
            for (int j = 0; j < team_node_n_pes; ++j) {
                int world_pe = ishmem_team_translate_pe(ISHMEMX_TEAM_NODE, j, ISHMEM_TEAM_WORLD);
                cur_sum += ((i % (world_pe + 2)) << 16) + i;
            }
            if (hosttarget[i] != cur_sum) {
                fprintf(stderr, "[%d] index %d expected 0x%08x got 0x%08x\n", my_pe, i, cur_sum,
                        hosttarget[i]);
            }
        }
        sycl::free(hosttarget, q);
    } else {
        std::cout << my_pe << ": No errors" << std::endl;
    }

    assert(npes % team_node_n_pes == 0);

    /* Create the leader PE team: */
    ishmem_team_t leader_team;
    ishmem_team_split_strided(ISHMEM_TEAM_WORLD, 0, *leader_stride_host, npes / team_node_n_pes,
                              NULL, 0, &leader_team);

    /* The Leader PEs now sum reduce the results of the previous ISHMEMX_TEAM_NODE sum reduction: */
    if (leader_team != ISHMEM_TEAM_INVALID) {
        int leader_errors = 0;

        ishmem_int_sum_reduce(leader_team, target, target, array_size);

        /* Verify the result: */
        int *hosttarget = sycl::malloc_host<int>(array_size, q);
        CHECK_ALLOC(hosttarget);
        q.memcpy(hosttarget, target, sizeof(int) * array_size).wait_and_throw();
        for (int i = 0; i < array_size; ++i) {
            int cur_sum = 0;
            for (int j = 0; j < npes; ++j) {
                cur_sum += ((i % (j + 2)) << 16) + i;
            }
            if (hosttarget[i] != cur_sum) {
                fprintf(stderr, "[%d] index %d expected 0x%08x got 0x%08x\n", my_pe, i, cur_sum,
                        hosttarget[i]);
                leader_errors += 1;
            }
        }
        sycl::free(hosttarget, q);

        if (leader_errors > 0) {
            std::cerr << "[ERROR] Leader validation check(s) failed: " << *errors << std::endl;
            *errors += leader_errors;
        }
    }

    ishmem_free(source);
    ishmem_free(target);
    ishmem_free(leader_pe);
    ishmem_free(leader_pes);
    ishmem_free(team_size);
    ishmem_free(team_stride);
    ishmem_free(team_sizes_collect);
    ishmem_free(team_strides_collect);
    ishmem_team_destroy(leader_team);
    ishmem_finalize();
    return *errors;
}
