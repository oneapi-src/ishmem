/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portions derived from Sandia OpenSHMEM (https://github.com/Sandia-OpenSHMEM/SOS)
 * For license and copyright information, see the LICENSE and third-party-programs.txt
 * files in the top level directory of this distribution.
 */

#include <CL/sycl.hpp>
#include <common.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int me, npes;
    sycl::queue q;
    int *errors = sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);
    *errors = 0;

    ishmemx_attr_t attr = {};
    test_init_attr(&attr);
    ishmemx_init_attr(&attr);

    me = ishmem_my_pe();
    npes = ishmem_n_pes();

    if (npes < 2) {
        if (me == 0) printf("Test requires 2 or more PEs\n");
        ishmem_finalize();
        return 0;
    }

    int *src = (int *) ishmem_malloc(sizeof(int));
    int *dst = (int *) ishmem_malloc((size_t) npes * sizeof(int));

    /* Initialize the data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            *src = me;
            for (int i = 0; i < npes; i++) {
                dst[i] = -1;
            }
        });
    });
    e_init.wait_and_throw();

    if (me == 0) printf("-1 stride:\n");

    ishmem_team_t new_team;
    ishmem_team_split_strided(ISHMEM_TEAM_WORLD, npes - 1, -1, npes, NULL, 0, &new_team);

    if (new_team != ISHMEM_TEAM_INVALID) {
        int new_team_id = ishmem_team_translate_pe(ISHMEM_TEAM_WORLD, me, new_team);

        if (new_team_id != -1 && new_team_id == abs(me - npes + 1)) {
            printf("world team ID = %d, new team ID = %d\n", me, new_team_id);
        } else {
            *errors += 1;
        }

        ishmem_int_fcollect(new_team, dst, src, 1);

        /* Verify the destination data */
        auto e_init = q.submit([&](sycl::handler &h) {
            h.single_task([=]() {
                for (int i = 0; i < npes; i++) {
                    if (dst[i] != npes - 1 - i) {
                        *errors += 1;
                    }
                }
            });
        });
        e_init.wait_and_throw();
    }

    ishmem_team_destroy(new_team);

    if (*errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
        int *hosttarget = sycl::malloc_host<int>((size_t) npes, q);
        CHECK_ALLOC(hosttarget);
        q.memcpy(hosttarget, dst, sizeof(int) * (size_t) npes).wait_and_throw();
        for (int i = 0; i < npes; ++i) {
            if (hosttarget[i] != npes - 1 - i) {
                fprintf(stderr, "[%d] index %d expected %d got %d\n", me, i, npes - 1 - i,
                        hosttarget[i]);
            }
        }
        sycl::free(hosttarget, q);
    } else {
        std::cout << me << ": No stride of -1 errors" << std::endl;
    }

    /* Re-initialize destination data */
    e_init = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (int i = 0; i < npes; i++) {
                dst[i] = -1;
            }
        });
    });
    e_init.wait_and_throw();

    if (me == 0) printf("-2 stride:\n");

    ishmem_team_split_strided(ISHMEM_TEAM_WORLD, npes - 1, -2, npes / 2, NULL, 0, &new_team);

    if (new_team != ISHMEM_TEAM_INVALID) {
        int new_team_id = ishmem_team_translate_pe(ISHMEM_TEAM_WORLD, me, new_team);

        if (new_team_id != -1 && new_team_id == abs((npes - me - 1) / 2)) {
            printf("world team ID = %d, new team ID = %d\n", me, new_team_id);
        } else {
            *errors += 1;
        }

        ishmem_int_fcollect(new_team, dst, src, 1);

        /* Verify the destination data */
        auto e_init = q.submit([&](sycl::handler &h) {
            h.single_task([=]() {
                for (int i = 0; i < ishmem_team_n_pes(new_team); i++) {
                    if (dst[i] != npes - 1 - 2 * i) {
                        *errors += 1;
                    }
                }
            });
        });
        e_init.wait_and_throw();

        if (*errors > 0) {
            std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
            int *hosttarget = sycl::malloc_host<int>((size_t) npes, q);
            CHECK_ALLOC(hosttarget);
            q.memcpy(hosttarget, dst, sizeof(int) * (size_t) npes).wait_and_throw();
            for (int i = 0; i < ishmem_team_n_pes(new_team); ++i) {
                if (hosttarget[i] != npes - 1 - 2 * i) {
                    fprintf(stderr, "[%d] dst[%d] expected %d got %d\n", me, i, npes - 1 - 2 * i,
                            hosttarget[i]);
                }
            }
            sycl::free(hosttarget, q);
        } else {
            std::cout << me << ": No stride of -2 errors" << std::endl;
        }
    }

    ishmem_team_destroy(new_team);
    ishmem_free(src);
    ishmem_free(dst);
    ishmem_finalize();

    return *errors != 0;
}
