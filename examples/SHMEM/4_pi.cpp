/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <ishmem.h>
#include <ishmemx.h>
#include <oneapi/dpl/random>

constexpr size_t npoints = 100000;
const double pi = 4 * std::atan(1);

/*
Consider a unit square with its center on (0.5, 0.5) and the portion of a circle x^2+y^2=1 in the
first quadrant.

Area of unit circle within the first quadrant = 0.25 * pi.
Area of the unit square = 1.
The probability of point drawn within the square to be inside the circle is the ratio of their areas
= 0.25 * pi.

We compute this probability numerically, and use it to estimate the value of pi.
*/

int main(int argc, char **argv)
{
    /* Initialize ISHMEM */
    ishmem_init();
    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    /* Seed for the pseudo-random number generator */
    /* Each PE needs to be seeded uniquely */
    uint32_t seed = static_cast<uint32_t>(my_pe) + 1;
    /* Number of points inside the quarter circle */
    size_t *inside = (size_t *) ishmem_malloc(sizeof(size_t));

    sycl::queue q;
    /* To perform a sycl reduction of within a GPU */
    auto sumr = sycl::reduction(inside, sycl::ext::oneapi::plus<>());
    /* Launch a kernel to perform the experiment */
    q.parallel_for(sycl::range<1>(npoints), sumr, [=](sycl::item<1> id, auto &sum) {
         std::uint64_t offset = id.get_linear_id();
         /* Create minstd_rand engine (LCG) */
         oneapi::dpl::minstd_rand engine(seed, offset);
         /* Create double uniform_real_distribution distribution */
         oneapi::dpl::uniform_real_distribution<double> distr;

         /* Draw a point */
         double x = distr(engine);
         double y = distr(engine);

         /* Check if the point is inside the quarter circle */
         if (x * x + y * y < 1) sum += 1;
     }).wait();

    /* Ensure that all PEs completed the kernel before collecting results */
    ishmem_barrier_all();

    /* Add up the results from all PEs on PE 0 */
    /* inside_h resides in the host */
    size_t inside_h = 0;
    if (my_pe == 0) {
        size_t i_inside;
        for (int i = 0; i < npes; i++) {
            ishmem_size_get(&i_inside, inside, 1, i);
            inside_h += i_inside;
        }
    }

    if (my_pe == 0) {
        /* Each PE does a simulation with npoints */
        size_t total_points = npoints * static_cast<size_t>(npes);

        double pi_appx = 4 * static_cast<double>(inside_h) / static_cast<double>(total_points);
        std::cout << "Value of pi from this experiment = " << pi_appx << std::endl;
        double err = (pi_appx - pi) * 100 / pi;
        std::cout << "Relative error (%) = " << err << " %" << std::endl;
    }

    /* Free memory */
    ishmem_free(inside);

    /* Finalize ISHMEM */
    ishmem_finalize();

    return EXIT_SUCCESS;
}
