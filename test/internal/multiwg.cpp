/* Copyright (C) 2023-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ishmem.h>
#include <ishmemx.h>
#include <stdlib.h>
#include <oneapi/dpl/random>
#include <assert.h>

int main(int argc, char **argv)
{
    ishmem_init();
    sycl::queue q;

    constexpr size_t work_group_size = 1024;
    constexpr size_t num_work_groups = 4;
    constexpr size_t nitems = work_group_size * num_work_groups;

    ishmem_team_t work_group_teams[num_work_groups];

    // make clones of ISHMEM_TEAM_SHARED
    for (size_t wg = 0; wg < num_work_groups; wg += 1) {
        int team_n_pes = ishmem_team_n_pes(ISHMEM_TEAM_SHARED);
        int res = ishmem_team_split_strided(ISHMEM_TEAM_SHARED, 0, 1, team_n_pes, NULL, 0,
                                            &work_group_teams[wg]);
        assert(res == 0);
        printf("team %ld %d\n", wg, work_group_teams[wg]);
    }

    double *inbuf = (double *) ishmem_calloc(nitems, sizeof(double));
    assert(inbuf);
    double *outbuf = (double *) ishmem_calloc(nitems, sizeof(double));
    assert(outbuf);
    std::uint32_t seed = 777;
    printf("calling kernel\n");
    auto e = q.parallel_for(sycl::nd_range<1>(nitems, work_group_size), [=](sycl::nd_item<1> it) {
        auto grp = it.get_group();
        auto group_id = grp.get_group_linear_id();
        int idx = it.get_global_id(0);
    oneapi:
        dpl::minstd_rand engine(seed, idx);
        oneapi::dpl::normal_distribution<double> distr;
        double rand = distr(engine);
        inbuf[idx] = rand;
        size_t start = group_id * work_group_size;
        ishmemx_double_sum_reduce_work_group(work_group_teams[group_id], &outbuf[start],
                                             &inbuf[start], work_group_size, grp);
    });
    e.wait_and_throw();
    printf("return kernel\n");

    /* Host code here to do something with the results */
    ishmem_free(inbuf);
    ishmem_free(outbuf);
    ishmem_finalize();

    return (0);
}
