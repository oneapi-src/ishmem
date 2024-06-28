/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <vector>
#include <CL/sycl.hpp>
#include <shmem.h>
#include <ishmem.h>
#include <ishmemx.h>

typedef float real;
constexpr real tol = 1.0e-4f;
constexpr int max_iter_max = 10000000;

using std::sin, std::sinh;
using std::chrono::high_resolution_clock, std::chrono::microseconds;
const real PI = static_cast<real>(2.0 * std::asin(1.0));

template <typename T>
T get_argval(char **begin, char **end, const std::string &arg, const T default_val)
{
    T argval = default_val;
    char **itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

void init_boundaries(real *__restrict__ const a, real *__restrict__ const a_new, const real pi,
                     const int offset, const int nx, const int my_ny, const int ny, sycl::queue &q)
{
    q.parallel_for(static_cast<size_t>(my_ny), [=](sycl::id<1> iy) {
         const real y0 =
             sycl::sin(2.0f * pi * (static_cast<real>(offset + iy) / static_cast<real>(ny - 1)));
         a[(iy + 1) * nx + 0] = y0;
         a[(iy + 1) * nx + (nx - 1)] = y0;
         a_new[(iy + 1) * nx + 0] = y0;
         a_new[(iy + 1) * nx + (nx - 1)] = y0;
     }).wait_and_throw();
}

sycl::event jacobi_kernel(real *__restrict__ const a_new, real *__restrict__ const a,
                          real *__restrict__ const l2_norm_d, const int iy_start, const int iy_end,
                          const int nx, const int top_pe, const int top_iy, const int bottom_pe,
                          const int bottom_iy, sycl::queue &q, std::vector<sycl::event> event_waits,
                          int global_range)
{
    // TODO: try-catch added to resolve issue detected by Coverity. More robust
    // error handling to be implemented later.
    try {
        auto sum_red = sycl::reduction(l2_norm_d, sycl::plus<>());
        auto ret = q.parallel_for(
            sycl::range<1>{static_cast<size_t>(global_range)}, sum_red,
            [=](sycl::id<1> idx, auto &sumr) {
                int iy = static_cast<int>(idx) / nx + iy_start;
                int ix = static_cast<int>(idx) % nx + 1;
                real local_norm = static_cast<real>(0.0);

                if (iy < iy_end && ix < (nx - 1)) {
                    const real new_val =
                        static_cast<real>(0.25) * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                                   a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
                    a_new[iy * nx + ix] = new_val;

                    // apply boundary conditions
                    if (iy_start == iy) {
                        ishmem_float_p((real *) (a_new + top_iy * nx + ix), new_val, top_pe);
                    }

                    if (iy_end - 1 == iy) {
                        ishmem_float_p((real *) (a_new + bottom_iy * nx + ix), new_val, bottom_pe);
                    }

                    real residue = a[iy * nx + ix] - new_val;
                    local_norm = residue * residue;
                    sumr += local_norm;
                }
            });
        return ret;
    } catch (sycl::exception &e) {
        std::cout << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Exact solution to the laplace equation for the given boundary condition
// Assuming length in x direction = length in y direction, which is true for
// only square grids
void check_solution(real *const A, const int nx, const int ny, const int iy_start_global,
                    const int iy_end_global, const int pe)
{
    // If the grid is not fine enough, numerical errors are significant
    // Disable the check in that case.
    const real check_tol = static_cast<real>(0.05);
    if (nx < 64 || ny < 64 || nx != ny) return;
    int ny_in = ny - 2;
    for (int iy = iy_start_global; iy < iy_end_global; iy++)
        for (int ix = 0; ix < nx; ix++) {
            const real x = (static_cast<real>(ix) / static_cast<real>(nx - 1));
            const real y = (static_cast<real>(iy) / static_cast<real>(ny_in - 1));
            const real val =
                sin(2 * PI * y) * (sinh(2 * PI * x) + sinh(2 * PI * (1 - x))) / sinh(2 * PI);

            if (std::fabs(A[(iy + 1) * nx + ix] - val) > check_tol) {
                std::cerr << "Numerical solution " << A[(iy + 1) * nx + ix]
                          << " does not match the exact solution " << val << " at row = " << iy
                          << "and column = " << ix << std::endl;
                return;
            }
        }

    if (pe == 0) std::cout << "Numerical solution matches the exact solution" << std::endl;
}

int main(int argc, char *argv[])
{
    int iter_max = get_argval<int>(argv, argv + argc, "-niter", 10000);
    int nx = get_argval<int>(argv, argv + argc, "-nx", 128);
    int ny = get_argval<int>(argv, argv + argc, "-ny", 128);
    // TODO: use a size of 16384 in performance testing

    if (iter_max < 1 || iter_max > max_iter_max) {
        std::cout << "Invalid input: iter_max";
        return EXIT_FAILURE;
    }

    if (nx < 1 || ny < 1) {
        std::cout << "Invalid input: grid_size";
        return EXIT_FAILURE;
    }

    const int nccheck = 1;
    sycl::queue Q;

    ishmem_init();

    int mype = shmem_my_pe();
    int npes = shmem_n_pes();

    // ny - 2 rows are distributed amongst `size` ranks in such a way
    // that each rank gets either (ny - 2) / size or (ny - 2) / size + 1 rows.
    // This optimizes load balancing when (ny - 2) % size != 0
    int chunk_size;
    int chunk_size_low = (ny - 2) / npes;
    int chunk_size_high = chunk_size_low + 1;
    // To calculate the number of ranks that need to compute an extra row,
    // the following formula is derived from this equation:
    // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
    int num_ranks_low =
        npes * static_cast<int>(chunk_size_low) + npes -
        static_cast<int>(ny - 2);  // Number of ranks with chunk_size = chunk_size_low

    if (mype < num_ranks_low) chunk_size = chunk_size_low;
    else chunk_size = chunk_size_high;

    sycl::event compute_done[2], compute_barrier_done[2], reset_l2_norm_done[2],
        copy_l2_norm_done[2];
    real *a_ref_h = sycl::malloc_host<real>(static_cast<size_t>(nx * ny), Q);
    real *a_h = sycl::malloc_host<real>(static_cast<size_t>(nx * ny), Q);
    real *a = (real *) ishmem_calloc(static_cast<size_t>(nx * (chunk_size_high + 2)), sizeof(real));
    real *a_new =
        (real *) ishmem_calloc(static_cast<size_t>(nx * (chunk_size_high + 2)), sizeof(real));
    real *l2_norms_d = sycl::malloc_device<real>(2, Q);
    real *l2_norms_h = (real *) shmem_malloc(2 * sizeof(real));
    real *l2_norms = (real *) shmem_malloc(2 * sizeof(real));

    Q.memset(a_ref_h, 0, static_cast<size_t>(nx * ny) * sizeof(real));
    Q.memset(a_h, 0, static_cast<size_t>(nx * ny) * sizeof(real));
    Q.memset(l2_norms_d, 0, 2 * sizeof(real));
    l2_norms_h[0] = 1;
    l2_norms_h[1] = 1;
    Q.wait();

    // Calculate local domain boundaries
    int iy_start_global;  // My start index in the global array
    if (mype < num_ranks_low) {
        iy_start_global = mype * chunk_size_low + 1;
    } else {
        iy_start_global =
            num_ranks_low * chunk_size_low + (mype - num_ranks_low) * chunk_size_high + 1;
    }

    int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array
    // do not process boundaries
    iy_end_global = std::min(iy_end_global, ny - 3);

    int iy_start = (mype == 0) ? 2 : 1;
    int iy_end = iy_end_global - iy_start_global + 2;
    // calculate boundary indices for top and bottom boundaries
    int top_pe = mype > 0 ? mype - 1 : (npes - 1);
    int bottom_pe = (mype + 1) % npes;

    int iy_end_top = (top_pe < num_ranks_low) ? chunk_size_low + 1 : chunk_size_high + 1;
    int iy_start_bottom = 0;

    // Set Dirichlet boundary conditions on left and right boundary
    init_boundaries(a, a_new, PI, iy_start_global - 1, nx, chunk_size, ny - 2, Q);

    int iter = 0;

    real err = 0;
    bool l2_norm_greater_than_tol = true;
    constexpr int local_x = 32;
    constexpr int local_y = 32;

    int global_x = ((nx + local_x - 1) / local_x) * local_x;
    int global_y = ((chunk_size + local_y - 1) / local_y) * local_y;
    int global_range = global_x * global_y;

    shmem_barrier_all();

    auto time_start = high_resolution_clock::now();

    while (l2_norm_greater_than_tol && iter < iter_max) {
        // on new iteration: old current vars are now previous vars, old
        // previous vars are no longer needed
        int prev = iter % 2;
        int curr = (iter + 1) % 2;
        compute_done[curr] =
            jacobi_kernel(a_new, a, &l2_norms_d[curr], iy_start, iy_end, nx, top_pe, iy_end_top,
                          bottom_pe, iy_start_bottom, Q,
                          {compute_barrier_done[prev], reset_l2_norm_done[curr]}, global_range);

        compute_barrier_done[curr] = Q.submit([&](sycl::handler &h) {
            h.depends_on(compute_done[curr]);
            h.single_task([=]() { ishmem_barrier_all(); });
        });

        if ((iter % nccheck) == 0) {
            copy_l2_norm_done[curr] = Q.submit([&](sycl::handler &h) {
                h.depends_on(compute_done[curr]);
                h.memcpy(&l2_norms_h[curr], &l2_norms_d[curr], sizeof(real));
            });

            copy_l2_norm_done[prev].wait();

            shmem_float_sum_reduce(SHMEM_TEAM_WORLD, &l2_norms[prev], &l2_norms_h[prev], 1);

            l2_norms[prev] = std::sqrt(l2_norms[prev]);
            l2_norm_greater_than_tol = (l2_norms[prev] > tol);

            if ((iter % 100) == 0) {
                if (!mype) printf("%5d, %0.6f\n", iter, l2_norms[prev]);
            }
            // reset everything for next iteration
            err = l2_norms[prev];
            l2_norms[prev] = static_cast<real>(0.0);
            l2_norms_h[prev] = static_cast<real>(0.0);
            reset_l2_norm_done[prev] = Q.memset(&l2_norms_d[prev], 0, sizeof(real));
        }

        std::swap(a_new, a);
        iter++;
    }

    Q.wait();
    shmem_barrier_all();

    auto time_end = high_resolution_clock::now();

    if (mype == 0 && !l2_norm_greater_than_tol)
        std::cout << "Solution converged within tolerance " << tol << std::endl;
    else if (mype == 0) std::cout << "Solution unconverged!" << std::endl;

    auto duration = std::chrono::duration_cast<microseconds>(time_end - time_start);
    if (mype == 0)
        std::cout << "Time per iteration for " << nx << " x " << ny
                  << " grid = " << (static_cast<int>(duration.count()) / iter) << "us/iteration"
                  << std::endl;

    Q.memcpy(
         a_h + iy_start_global * nx, a + nx,
         static_cast<size_t>(std::min(ny - 2 - iy_start_global, chunk_size) * nx) * sizeof(real))
        .wait();

    if (!mype) {
        printf("l2_norms %16.15f iter %d\n", err, iter);
        printf("Num GPUs: %d\n", npes);
        printf("%d x %d: 1 GPU\n", ny, nx);
    }

    check_solution(a_h, nx, ny, iy_start_global, iy_end_global, mype);

    free(a_ref_h, Q);
    free(a_h, Q);
    ishmem_free(a_new);
    ishmem_free(a);
    free(l2_norms_d, Q);
    shmem_free(l2_norms_h);
    shmem_free(l2_norms);

    ishmem_finalize();
    shmem_finalize();
    return EXIT_SUCCESS;
}
