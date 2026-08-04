/* Copyright (C) 2024-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <unistd.h>
#include <common.h>
#include <float.h>
#include <sys/time.h>

// Sizes are in bytes
#define RUN_HOST_INITIATIED true

#define MSG_SIZE_MIN 1
#define MSG_SIZE_MAX 8 * 1024 * 1024

#define LARGE_MSG_SIZE 65536

#define WARMUP       100
#define WARMUP_LARGE 10

#define TRIALS       1000
#define TRIALS_LARGE 100

#define RUN_RMA_TEST_HOST(uut)                                                                     \
    do {                                                                                           \
        size_t warmup_iters = (sz >= LARGE_MSG_SIZE) ? WARMUP_LARGE : WARMUP;                      \
        size_t iters = (sz >= LARGE_MSG_SIZE) ? TRIALS_LARGE : TRIALS;                             \
        static double pe_bw_sum, bw;                                                               \
        pe_bw_sum = 0.0;                                                                           \
        bw = 0.0;                                                                                  \
        static double start_time, end_time, start_time_min, end_time_max;                          \
        start_time = DBL_MAX;                                                                      \
        end_time = 0.0;                                                                            \
        start_time_min = 0.0, end_time_max = 0.0;                                                  \
        if (my_pe < (npes / 2)) {                                                                  \
            for (int i = 0; i < warmup_iters; i++) {                                               \
                uut(dest, src, sz, my_pe + (npes / 2));                                            \
            }                                                                                      \
            ishmem_barrier_all();                                                                  \
                                                                                                   \
            /* Timed iterations */                                                                 \
            struct timeval start, end;                                                             \
            gettimeofday(&start, NULL);                                                            \
            for (int i = 0; i < iters; i++) {                                                      \
                uut(dest, src, sz, my_pe + (npes / 2));                                            \
            }                                                                                      \
            ishmem_quiet();                                                                        \
            gettimeofday(&end, NULL);                                                              \
                                                                                                   \
            /* Calculate start/end/total times in microseconds */                                  \
            double start_time = start.tv_sec * 1e6;                                                \
            start_time += (double) start.tv_usec;                                                  \
            double end_time = end.tv_sec * 1e6;                                                    \
            end_time += (double) end.tv_usec;                                                      \
                                                                                                   \
            ishmem_double_min_reduce(&start_time_min, &start_time, 1);                             \
            ishmem_sync_all();                                                                     \
            ishmem_double_max_reduce(&end_time_max, &end_time, 1);                                 \
            double wtime = end_time_max - start_time_min;                                          \
                                                                                                   \
            /* Calculate BW in MBs/second (1e6 bytes) */                                           \
            bw = ((sz * iters * (npes / 2)) / (wtime / 1e6)) / 1e6;                                \
                                                                                                   \
            if (my_pe == 0) {                                                                      \
                std::cout << "Size: " << sz << ", BW: " << bw << " MB/s" << std::endl;             \
            }                                                                                      \
        } else {                                                                                   \
            ishmem_barrier_all();                                                                  \
            ishmem_double_min_reduce(&start_time_min, &start_time, 1);                             \
            ishmem_sync_all();                                                                     \
            ishmem_double_max_reduce(&end_time_max, &end_time, 1);                                 \
        }                                                                                          \
    } while (0)

#define RUN_RMA_TEST_DEVICE(uut)                                                                   \
    do {                                                                                           \
        size_t warmup_iters = (sz >= LARGE_MSG_SIZE) ? WARMUP_LARGE : WARMUP;                      \
        size_t iters = (sz >= LARGE_MSG_SIZE) ? TRIALS_LARGE : TRIALS;                             \
        static double pe_bw_sum, bw;                                                               \
        pe_bw_sum = 0.0;                                                                           \
        bw = 0.0;                                                                                  \
        static double start_time, end_time, start_time_min, end_time_max;                          \
        start_time = DBL_MAX;                                                                      \
        end_time = 0.0;                                                                            \
        start_time_min = 0.0, end_time_max = 0.0;                                                  \
        if (my_pe < (npes / 2)) {                                                                  \
            q.single_task([=]() {                                                                  \
                 for (int i = 0; i < warmup_iters; i++) {                                          \
                     uut(dest, src, sz, my_pe + (npes / 2));                                       \
                 }                                                                                 \
             }).wait_and_throw();                                                                  \
            ishmem_barrier_all();                                                                  \
                                                                                                   \
            /* Timed iterations */                                                                 \
            struct timeval start, end;                                                             \
            gettimeofday(&start, NULL);                                                            \
            q.single_task([=]() {                                                                  \
                 for (int i = 0; i < iters; i++) {                                                 \
                     uut(dest, src, sz, my_pe + (npes / 2));                                       \
                 }                                                                                 \
             }).wait_and_throw();                                                                  \
            /*ishmem_quiet();*/                                                                    \
            gettimeofday(&end, NULL);                                                              \
                                                                                                   \
            /* Calculate start/end/total times in microseconds */                                  \
            double start_time = start.tv_sec * 1e6;                                                \
            start_time += (double) start.tv_usec;                                                  \
            double end_time = end.tv_sec * 1e6;                                                    \
            end_time += (double) end.tv_usec;                                                      \
                                                                                                   \
            ishmem_double_min_reduce(&start_time_min, &start_time, 1);                             \
            ishmem_sync_all();                                                                     \
            ishmem_double_max_reduce(&end_time_max, &end_time, 1);                                 \
            double wtime = end_time_max - start_time_min;                                          \
                                                                                                   \
            /* Calculate BW in MBs/second (1e6 bytes) */                                           \
            bw = ((sz * iters * (npes / 2)) / (wtime / 1e6)) / 1e6;                                \
                                                                                                   \
            if (my_pe == 0) {                                                                      \
                std::cout << "Size: " << sz << ", BW: " << bw << " MB/s" << std::endl;             \
            }                                                                                      \
        } else {                                                                                   \
            ishmem_barrier_all();                                                                  \
            ishmem_double_min_reduce(&start_time_min, &start_time, 1);                             \
            ishmem_sync_all();                                                                     \
            ishmem_double_max_reduce(&end_time_max, &end_time, 1);                                 \
        }                                                                                          \
    } while (0)

int main(int argc, char **argv)
{
    ishmem_init();
    // sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    if (npes % 2) {
        if (my_pe == 0) {
            std::cerr << "[WARN]: Throughput test must be run with an even number of PEs"
                      << std::endl;
        }
        exit(0);
    }

    // sycl::queue q(prop_list);
    sycl::queue q;

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;

    std::cout << "Host: PE " << my_pe << " of " << npes << std::endl;

    char *dest = (char *) ishmem_malloc(MSG_SIZE_MAX);
    char *src = (char *) ishmem_malloc(MSG_SIZE_MAX);

    ishmem_barrier_all();

    for (size_t sz = MSG_SIZE_MIN; sz <= MSG_SIZE_MAX; sz *= 2) {
        if (RUN_HOST_INITIATIED) {
            RUN_RMA_TEST_HOST(ishmem_char_put_nbi);
        } else {
            RUN_RMA_TEST_DEVICE(ishmem_char_put_nbi);
        }
    }

    ishmem_finalize();
    return 0;
}
