/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef COMMON_H
#define COMMON_H

#include <getopt.h>
#include <iomanip>
#include <type_traits>
#include <cmath>
#include <ishmem.h>
#include <ishmemx.h>
#include <ishmem/config.h>
#include "runtime.h"

static std::once_flag validate_once;
ishmemi_test_runtime_type *ishmemi_test_runtime = nullptr;

static inline void validate_runtime()
{
    std::call_once(validate_once, []() {
        const char *env_val = getenv("ISHMEM_RUNTIME");
        if (env_val && strcasecmp(env_val, "OPENSHMEM") == 0) {
#if defined(ENABLE_OPENSHMEM)
            ishmemi_test_runtime = new ishmemi_test_runtime_openshmem();
#else
            fprintf(stderr, "ERROR: Runtime OpenSHMEM selected but it is not configured\n");
#endif
        } else if (env_val && strcasecmp(env_val, "MPI") == 0) {
#if defined(ENABLE_MPI)
            ishmemi_test_runtime = new ishmemi_test_runtime_mpi();
#else
            fprintf(stderr, "ERROR: Runtime MPI selected but it is not configured\n");
#endif
        } else if (env_val && strcasecmp(env_val, "PMI") == 0) {
#if defined(ENABLE_PMI)
            fprintf(stderr, "ERROR: Runtime PMI selected but is not yet supported\n");
#else
            fprintf(stderr, "ERROR: Runtime PMI selected but it is not configured\n");
#endif
        } else {
            /* Default value */
#if defined(ENABLE_OPENSHMEM)
            ishmemi_test_runtime = new ishmemi_test_runtime_openshmem();
            return;
#endif
#if defined(ENABLE_MPI)
            ishmemi_test_runtime = new ishmemi_test_runtime_mpi();
            return;
#endif
#if defined(ENABLE_PMI)
            fprintf(stderr, "ERROR: Runtime PMI selected but is not yet supported\n");
            return;
#endif
        }
    });
}

/* Max memory size is limited to 8M to avoid reaching the Level Zero memory
 * allocation unsupported size threshold
 */
#define MAX_MSG_SIZE        (1L << 23)
#define MAX_SCHAR_VALUE     127
#define MAX_WORK_GROUP_SIZE 1024

#define NSEC_IN_SEC 1000000000.0
double getduration(const sycl::event &e)
{
    uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    double duration = static_cast<double>(end - start) / NSEC_IN_SEC;
    return (duration);
}

enum Operation {
    put = 0,
    get = 1
};

enum Pair {
    same,
    tile,
    xe
};

extern const char *ishmemi_op_str[];
extern const char *ishmemi_type_str[];

std::string pair_to_string(Pair pair)
{
    switch (pair) {
        case Pair::same:
            return "same";
        case Pair::tile:
            return "tile";
        case Pair::xe:
            return "xe";
        default:
            return "";
    }
}

int get_pe_for_pair(Pair pair)
{
    if (pair == Pair::same) {
        return 0;
    } else if (pair == Pair::tile) {
        return 1;
    } else { /* pair == Pair::xe */
        return 2;
    }
}

template <typename TYPE, int OPERATION>
SYCL_EXTERNAL void call_ishmem_rma(TYPE *dst, TYPE *src, size_t count, int pe)
{
    // Put operations
    if constexpr (std::is_same<TYPE, float>::value && OPERATION == Operation::put)
        ishmem_float_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, double>::value && OPERATION == Operation::put)
        ishmem_double_put(dst, src, count, pe);
    // else if constexpr(std::is_same<TYPE, char>::value && OPERATION == Operation::put)
    // ishmem_longdouble_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, char>::value && OPERATION == Operation::put)
        ishmem_char_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, signed char>::value && OPERATION == Operation::put)
        ishmem_schar_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, short>::value && OPERATION == Operation::put)
        ishmem_short_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, int>::value && OPERATION == Operation::put)
        ishmem_int_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, long>::value && OPERATION == Operation::put)
        ishmem_long_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, long long>::value && OPERATION == Operation::put)
        ishmem_longlong_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, unsigned char>::value && OPERATION == Operation::put)
        ishmem_uchar_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, unsigned short>::value && OPERATION == Operation::put)
        ishmem_ushort_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, unsigned int>::value && OPERATION == Operation::put)
        ishmem_uint_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, unsigned long>::value && OPERATION == Operation::put)
        ishmem_ulong_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, unsigned long long>::value && OPERATION == Operation::put)
        ishmem_ulonglong_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, int8_t>::value && OPERATION == Operation::put)
        ishmem_int8_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, int16_t>::value && OPERATION == Operation::put)
        ishmem_int16_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, int32_t>::value && OPERATION == Operation::put)
        ishmem_int32_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, int64_t>::value && OPERATION == Operation::put)
        ishmem_int64_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, uint8_t>::value && OPERATION == Operation::put)
        ishmem_uint8_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, uint16_t>::value && OPERATION == Operation::put)
        ishmem_uint16_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, uint32_t>::value && OPERATION == Operation::put)
        ishmem_uint32_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, uint64_t>::value && OPERATION == Operation::put)
        ishmem_uint64_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, size_t>::value && OPERATION == Operation::put)
        ishmem_size_put(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, ptrdiff_t>::value && OPERATION == Operation::put)
        ishmem_ptrdiff_put(dst, src, count, pe);

    // Get operations
    else if constexpr (std::is_same<TYPE, float>::value && OPERATION == Operation::get)
        ishmem_float_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, double>::value && OPERATION == Operation::get)
        ishmem_double_get(dst, src, count, pe);
    // else if constexpr(std::is_same<TYPE, char>::value && OPERATION == Operation::get)
    // ishmem_longdouble_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, char>::value && OPERATION == Operation::get)
        ishmem_char_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, signed char>::value && OPERATION == Operation::get)
        ishmem_schar_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, short>::value && OPERATION == Operation::get)
        ishmem_short_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, int>::value && OPERATION == Operation::get)
        ishmem_int_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, long>::value && OPERATION == Operation::get)
        ishmem_long_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, long long>::value && OPERATION == Operation::get)
        ishmem_longlong_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, unsigned char>::value && OPERATION == Operation::get)
        ishmem_uchar_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, unsigned short>::value && OPERATION == Operation::get)
        ishmem_ushort_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, unsigned int>::value && OPERATION == Operation::get)
        ishmem_uint_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, unsigned long>::value && OPERATION == Operation::get)
        ishmem_ulong_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, unsigned long long>::value && OPERATION == Operation::get)
        ishmem_ulonglong_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, int8_t>::value && OPERATION == Operation::get)
        ishmem_int8_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, int16_t>::value && OPERATION == Operation::get)
        ishmem_int16_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, int32_t>::value && OPERATION == Operation::get)
        ishmem_int32_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, int64_t>::value && OPERATION == Operation::get)
        ishmem_int64_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, uint8_t>::value && OPERATION == Operation::get)
        ishmem_uint8_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, uint16_t>::value && OPERATION == Operation::get)
        ishmem_uint16_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, uint32_t>::value && OPERATION == Operation::get)
        ishmem_uint32_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, uint64_t>::value && OPERATION == Operation::get)
        ishmem_uint64_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, size_t>::value && OPERATION == Operation::get)
        ishmem_size_get(dst, src, count, pe);
    else if constexpr (std::is_same<TYPE, ptrdiff_t>::value && OPERATION == Operation::get)
        ishmem_ptrdiff_get(dst, src, count, pe);
}

template <typename TYPE, int OPERATION>
SYCL_EXTERNAL void call_ishmem_work_group_rma(TYPE *dst, TYPE *src, size_t count, int pe,
                                              sycl::group<1> grp)
{
    // Put operations
    if constexpr (std::is_same<TYPE, float>::value && OPERATION == Operation::put)
        ishmemx_float_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, double>::value && OPERATION == Operation::put)
        ishmemx_double_put_work_group(dst, src, count, pe, grp);
    // else if constexpr(std::is_same<TYPE, char>::value && OPERATION == Operation::put)
    // ishmemx_longdouble_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, char>::value && OPERATION == Operation::put)
        ishmemx_char_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, signed char>::value && OPERATION == Operation::put)
        ishmemx_schar_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, short>::value && OPERATION == Operation::put)
        ishmemx_short_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, int>::value && OPERATION == Operation::put)
        ishmemx_int_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, long>::value && OPERATION == Operation::put)
        ishmemx_long_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, long long>::value && OPERATION == Operation::put)
        ishmemx_longlong_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, unsigned char>::value && OPERATION == Operation::put)
        ishmemx_uchar_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, unsigned short>::value && OPERATION == Operation::put)
        ishmemx_ushort_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, unsigned int>::value && OPERATION == Operation::put)
        ishmemx_uint_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, unsigned long>::value && OPERATION == Operation::put)
        ishmemx_ulong_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, unsigned long long>::value && OPERATION == Operation::put)
        ishmemx_ulonglong_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, int8_t>::value && OPERATION == Operation::put)
        ishmemx_int8_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, int16_t>::value && OPERATION == Operation::put)
        ishmemx_int16_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, int32_t>::value && OPERATION == Operation::put)
        ishmemx_int32_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, int64_t>::value && OPERATION == Operation::put)
        ishmemx_int64_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, uint8_t>::value && OPERATION == Operation::put)
        ishmemx_uint8_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, uint16_t>::value && OPERATION == Operation::put)
        ishmemx_uint16_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, uint32_t>::value && OPERATION == Operation::put)
        ishmemx_uint32_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, uint64_t>::value && OPERATION == Operation::put)
        ishmemx_uint64_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, size_t>::value && OPERATION == Operation::put)
        ishmemx_size_put_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, ptrdiff_t>::value && OPERATION == Operation::put)
        ishmemx_ptrdiff_put_work_group(dst, src, count, pe, grp);

    // Get operations
    else if constexpr (std::is_same<TYPE, float>::value && OPERATION == Operation::get)
        ishmemx_float_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, double>::value && OPERATION == Operation::get)
        ishmemx_double_get_work_group(dst, src, count, pe, grp);
    // else if constexpr(std::is_same<TYPE, char>::value && OPERATION == Operation::get)
    // ishmemx_longdouble_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, char>::value && OPERATION == Operation::get)
        ishmemx_char_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, signed char>::value && OPERATION == Operation::get)
        ishmemx_schar_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, short>::value && OPERATION == Operation::get)
        ishmemx_short_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, int>::value && OPERATION == Operation::get)
        ishmemx_int_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, long>::value && OPERATION == Operation::get)
        ishmemx_long_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, long long>::value && OPERATION == Operation::get)
        ishmemx_longlong_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, unsigned char>::value && OPERATION == Operation::get)
        ishmemx_uchar_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, unsigned short>::value && OPERATION == Operation::get)
        ishmemx_ushort_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, unsigned int>::value && OPERATION == Operation::get)
        ishmemx_uint_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, unsigned long>::value && OPERATION == Operation::get)
        ishmemx_ulong_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, unsigned long long>::value && OPERATION == Operation::get)
        ishmemx_ulonglong_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, int8_t>::value && OPERATION == Operation::get)
        ishmemx_int8_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, int16_t>::value && OPERATION == Operation::get)
        ishmemx_int16_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, int32_t>::value && OPERATION == Operation::get)
        ishmemx_int32_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, int64_t>::value && OPERATION == Operation::get)
        ishmemx_int64_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, uint8_t>::value && OPERATION == Operation::get)
        ishmemx_uint8_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, uint16_t>::value && OPERATION == Operation::get)
        ishmemx_uint16_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, uint32_t>::value && OPERATION == Operation::get)
        ishmemx_uint32_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, uint64_t>::value && OPERATION == Operation::get)
        ishmemx_uint64_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, size_t>::value && OPERATION == Operation::get)
        ishmemx_size_get_work_group(dst, src, count, pe, grp);
    else if constexpr (std::is_same<TYPE, ptrdiff_t>::value && OPERATION == Operation::get)
        ishmemx_ptrdiff_get_work_group(dst, src, count, pe, grp);
}

struct PerfTestArgs {
    sycl::queue q;
    int my_pe;
    size_t starting_msg_size;
    size_t max_msg_size;
    size_t iterations;
    size_t skip;
    size_t starting_work_group_size;
    size_t max_work_group_size;
    size_t starting_work_groups;
    size_t max_work_groups;
};

template <typename TYPE, int OPERATION>
void ishmem_rma_perf_test(Pair pair, PerfTestArgs &perfTestArgs)
{
    // Unpack perfTestArgs parameter
    sycl::queue q = perfTestArgs.q;
    int my_pe = perfTestArgs.my_pe;
    size_t starting_msg_size = perfTestArgs.starting_msg_size;
    size_t max_msg_size = perfTestArgs.max_msg_size;
    size_t iterations = perfTestArgs.iterations;
    size_t skip = perfTestArgs.skip;

    double *durations = (double *) malloc(iterations * sizeof(double));
    double *latency = (double *) malloc(sizeof(double));
    double total_kernel_duration = 0;
    TYPE *host_array = (TYPE *) malloc(max_msg_size);
    TYPE *src = (TYPE *) ishmem_malloc(max_msg_size);
    TYPE *dst = (TYPE *) ishmem_malloc(max_msg_size);

    int dest_pe = 0;
    int source_pe = 0;
    std::string direction = "";
    if (OPERATION == Operation::put) { /* put operation */
        dest_pe = get_pe_for_pair(pair);
        source_pe = 0;
        direction = "push";
    } else { /* get operation */
        dest_pe = 0;
        source_pe = get_pe_for_pair(pair);
        direction = "pull";
    }

    size_t array_size = max_msg_size / sizeof(TYPE);
    q.parallel_for(sycl::range<1>(array_size), [=](sycl::id<1> i) {
         src[i] = ((TYPE) ((my_pe << 16) + i));
     }).wait_and_throw();
    ishmem_barrier_all();

    for (size_t size = starting_msg_size; size <= max_msg_size; size *= 2) {
        size_t num_elems = size / sizeof(TYPE);
        auto e_init =
            q.parallel_for(sycl::range<1>(array_size), [=](sycl::id<1> i) { dst[i] = 0; });
        e_init.wait_and_throw();
        ishmem_barrier_all();

        if (my_pe == 0) {
            int remote_pe = (direction == "pull") ? source_pe : dest_pe;
            for (size_t iter = 0; iter < iterations + skip; iter++) {
                auto e_run = q.single_task([=]() {
                    call_ishmem_rma<TYPE, OPERATION>(dst, src, num_elems, remote_pe);
                    atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                });
                e_run.wait_and_throw();

                if (iter >= skip) durations[iter - skip] = getduration(e_run);
            }

            for (size_t i = 0; i < iterations; ++i) {
                total_kernel_duration += durations[i];
            }
            latency[0] = total_kernel_duration;
            total_kernel_duration = 0;
        }
        ishmem_barrier_all();

        if (my_pe == 0) {
            double bandwidth = 0;
            double message_rate = 0;

            std::cout << "Message Size" << std::setw(25) << "Bandwidth" << std::setw(25)
                      << "Message Rate" << std::endl
                      << "(in bytes)" << std::setw(25) << "(in MB/sec)" << std::setw(26)
                      << "(in messages/sec)" << std::endl;

            bandwidth = ((double) (size * iterations) / latency[0]) / (1000 * 1000);
            message_rate = bandwidth * 1000 * 1000 / ((double) size);
            std::cout << std::setw(12) << size << std::setw(18) << bandwidth << std::setw(26)
                      << message_rate << std::endl;

            printf("\n");
            bandwidth = ((double) (size * iterations) / latency[0]) / (1000 * 1000);
            printf("csv,%s,%s,%s,%lu,%ld,%f,%f\n", pair_to_string(pair).c_str(), direction.c_str(),
                   "vec_ulong", sizeof(TYPE), size, latency[0], bandwidth);
            fflush(stdout);
        }

        /* Verify results */
        int errors = 0;
        bool failure_detected = false;
        size_t first_failure_index = 0;
        TYPE first_failure_value;
        q.memcpy(host_array, dst, size).wait_and_throw();
        if (my_pe == dest_pe) {
            for (size_t i = 0; i < num_elems; i++) {
                if (host_array[i] !=
                    ((TYPE) ((static_cast<unsigned long long>(source_pe) << 16) + i))) {
                    errors += 1;
                    if (!failure_detected) {
                        first_failure_index = i;
                        first_failure_value = host_array[i];
                        failure_detected = true;
                    }
                }
            }

            std::cout << "Test Completed with " << errors << " errors!" << std::endl;
            if (errors > 0) {
                std::cout << "First occurence of unexpected value (index " << first_failure_index
                          << "): " << " Received (" << first_failure_value << ") Expected ("
                          << static_cast<size_t>(source_pe << 16) + first_failure_index << ")"
                          << std::endl;
            }
        }
    }
    free(durations);
    free(latency);
    free(host_array);
    ishmem_free(src);
    ishmem_free(dst);
}

template <typename TYPE, int OPERATION>
void ishmem_rma_work_group_perf_test(Pair pair, PerfTestArgs &perfTestArgs)
{
    // Unpack perfTestArgs parameter
    sycl::queue q = perfTestArgs.q;
    int my_pe = perfTestArgs.my_pe;
    size_t starting_msg_size = perfTestArgs.starting_msg_size;
    size_t max_msg_size = perfTestArgs.max_msg_size;
    size_t starting_work_group_size = perfTestArgs.starting_work_group_size;
    size_t max_work_group_size = perfTestArgs.max_work_group_size;
    size_t starting_work_groups = perfTestArgs.starting_work_groups;
    size_t max_work_groups = perfTestArgs.max_work_groups;
    // size_t iterations = perfTestArgs.iterations;
    // size_t skip = perfTestArgs.skip;

    size_t num_work_item_sizes =
        static_cast<size_t>(
            log2(static_cast<double>(max_work_group_size / starting_work_group_size))) +
        1;
    size_t *total_iterations = (size_t *) malloc(num_work_item_sizes * sizeof(size_t));
    double *latency = (double *) malloc(num_work_item_sizes * sizeof(double));
    int latency_counter = 0;
    double total_kernel_duration = 0;
    TYPE *host_array = (TYPE *) malloc(max_msg_size);
    TYPE *src = (TYPE *) ishmem_malloc(max_msg_size);
    TYPE *dst = (TYPE *) ishmem_malloc(max_msg_size);

    int dest_pe = 0;
    int source_pe = 0;
    std::string direction = "";
    if (OPERATION == Operation::put) { /* put operation */
        dest_pe = get_pe_for_pair(pair);
        source_pe = 0;
        direction = "push";
    } else { /* get operation */
        dest_pe = 0;
        source_pe = get_pe_for_pair(pair);
        direction = "pull";
    }

    size_t array_size = max_msg_size / sizeof(TYPE);
    q.parallel_for(sycl::range<1>(array_size), [=](sycl::id<1> i) {
         src[i] = ((TYPE) ((my_pe << 16) + i));
     }).wait_and_throw();
    ishmem_barrier_all();

    for (size_t size = starting_msg_size; size <= max_msg_size; size *= 2) {
        size_t num_elems = size / sizeof(TYPE);
        for (size_t work_groups = starting_work_groups;
             work_groups <= ((max_work_groups > 0)
                                 ? ((max_work_groups <= num_elems) ? max_work_groups : num_elems)
                                 : num_elems);
             work_groups *= 2) {
            size_t loc_loop_wg = size / (work_groups * sizeof(TYPE));
            latency_counter = 0;
            auto e_init =
                q.parallel_for(sycl::range<1>(array_size), [=](sycl::id<1> i) { dst[i] = 0; });
            e_init.wait_and_throw();
            ishmem_barrier_all();

            if (my_pe == 0) {
                int remote_pe = (direction == "pull") ? source_pe : dest_pe;
                for (size_t work_group_size = starting_work_group_size;
                     (work_group_size <= (num_elems / work_groups)) &&
                     (work_group_size <= max_work_group_size);
                     work_group_size *= 2) {
                    size_t iterations = 1;
                    total_iterations[latency_counter] = 0;

                    while (total_kernel_duration < 0.01) {
                        auto e_run = q.parallel_for(
                            sycl::nd_range<1>{work_group_size * work_groups, work_group_size},
                            [=](sycl::nd_item<1> idx) {
                                auto grp = idx.get_group();
                                size_t work_group_start_idx =
                                    loc_loop_wg * grp.get_group_linear_id();

                                call_ishmem_work_group_rma<TYPE, OPERATION>(
                                    &dst[work_group_start_idx], &src[work_group_start_idx],
                                    loc_loop_wg, remote_pe, grp);
                                atomic_fence(sycl::memory_order::acquire,
                                             sycl::memory_scope::system);
                            });
                        e_run.wait_and_throw();

                        total_kernel_duration += getduration(e_run);
                        total_iterations[latency_counter] += iterations;
                        iterations <<= 1;
                    }

                    latency[latency_counter] = total_kernel_duration;
                    ++latency_counter;
                    total_kernel_duration = 0;
                }
            }
            ishmem_barrier_all();

            latency_counter = 0;
            if (my_pe == 0) {
                double bandwidth = 0;
                double message_rate = 0;

                std::cout << "Message Size" << std::setw(25) << "Num. Work Groups" << std::setw(29)
                          << "Num. Work Items" << std::setw(25) << "Bandwidth" << std::setw(25)
                          << "Message Rate" << std::endl
                          << "(in bytes)" << std::setw(56) << "(per work group)" << std::setw(26)
                          << "(in MB/sec)" << std::setw(26) << "(in messages/sec)" << std::endl;

                for (size_t work_group_size = starting_work_group_size;
                     (work_group_size <= (num_elems / work_groups)) &&
                     (work_group_size <= max_work_group_size);
                     work_group_size *= 2) {
                    bandwidth = ((double) (size * total_iterations[latency_counter]) /
                                 latency[latency_counter]) /
                                (1000 * 1000);
                    latency_counter += 1;
                    message_rate = bandwidth * 1000 * 1000 / ((double) size);
                    std::cout << std::setw(12) << size << std::setw(18) << work_groups
                              << std::setw(30) << work_group_size << std::setw(32) << bandwidth
                              << std::setw(26) << message_rate << std::endl;
                }
                latency_counter = 0;

                printf("\n");
                for (size_t work_group_size = starting_work_group_size;
                     (work_group_size <= (num_elems / work_groups)) &&
                     (work_group_size <= max_work_group_size);
                     work_group_size *= 2) {
                    bandwidth = ((double) (size * total_iterations[latency_counter]) /
                                 latency[latency_counter]) /
                                (1000 * 1000);
                    printf("csv,%s,%s,%s,%lu,%lu,%lu,%lu,%lu,%f,%f\n", pair_to_string(pair).c_str(),
                           direction.c_str(), "vec_ulong", sizeof(TYPE), size, work_groups,
                           work_group_size, size / (work_groups * sizeof(TYPE) * work_group_size),
                           latency[latency_counter++], bandwidth);
                    fflush(stdout);
                }
            }

            /* Verify results */
            int errors = 0;
            bool failure_detected = false;
            size_t first_failure_index = 0;
            TYPE first_failure_value;
            q.memcpy(host_array, dst, size).wait_and_throw();
            if (my_pe == dest_pe) {
                for (size_t i = 0; i < num_elems; i++) {
                    if (host_array[i] !=
                        ((TYPE) ((static_cast<unsigned long long>(source_pe) << 16) + i))) {
                        errors += 1;
                        if (!failure_detected) {
                            first_failure_index = i;
                            first_failure_value = host_array[i];
                            failure_detected = true;
                        }
                    }
                }

                std::cout << "Test Completed with " << errors << " errors!" << std::endl;
                if (errors > 0) {
                    std::cout << "First occurence of unexpected value (index "
                              << first_failure_index << "): " << " Received ("
                              << first_failure_value << ") Expected ("
                              << static_cast<size_t>(source_pe << 16) + first_failure_index << ")"
                              << std::endl;
                }
            }
        }
    }
    free(total_iterations);
    free(latency);
    free(host_array);
    ishmem_free(src);
    ishmem_free(dst);
}

static int isPowerOfTwo(size_t n)
{
    if (n == 0) return 0;
    if ((n & (n - 1)) == 0) return 1;

    return 0;
}

/* For making a mutual decision about when to exit the self-timing loop */
double duration_broadcast(sycl::queue q, double *duration, double value)
{
    double result;
    q.memcpy(duration, &value, sizeof(double)).wait_and_throw();
    q.single_task([=]() { ishmem_double_broadcast(duration, duration, 1, 0); }).wait_and_throw();
    q.memcpy(&result, duration, sizeof(double)).wait_and_throw();
    return (result);
}

int opt;
int verify = 1;

typedef enum {
    OP_FETCH = 0,
    OP_SET,
    OP_CSWAP,
    OP_SWAP,
    OP_FINC,
    OP_INC,
    OP_FADD,
    OP_ADD,
    OP_FAND,
    OP_AND,
    OP_FOR,
    OP_OR,
    OP_FXOR,
    OP_XOR
} atomic_op_type;

const atomic_op_type atomic_op_list[] = {OP_FETCH, OP_SET,  OP_CSWAP, OP_SWAP, OP_FINC,
                                         OP_INC,   OP_FADD, OP_ADD,   OP_FAND, OP_AND,
                                         OP_FOR,   OP_OR,   OP_FXOR,  OP_XOR};
const char *atomic_op_name[] = {"fetch", "set",  "cswap", "swap", "finc", "inc",  "fadd",
                                "add",   "fand", "and",   "for",  "or",   "fxor", "xor"};

static void print_usage()
{
    std::cerr << "Usage: \n";
    std::cerr << "  <launcher> -n 2 [launcher-options] ./scripts/ishmrun <test> [test-options]\n";
    std::cerr << "test-options: \n";
    std::cerr << "  --msg-size, -m    Set the max message size that will be"
                 " operated on\n"
                 "  --work-group-size, -w  Set the dimensions of the device kernel's"
                 " entire index space (only used in"
                 " multithreaded tests)\n"
                 "  --groups, -w  Set the number of work-groups (only used in"
                 " multithreaded tests)\n"
                 "  --iterations, -i  Set the number of times the operation will"
                 " be run\n"
                 "  --no-verify       Skip the verification step\n"
                 "  --skip, -s        Set the number of iterations to be 'burned'"
                 " before measurements begin\n"
                 "  --help,  -h       Print usage message\n";
    exit(1);
}

void parse_args(int argc, char *argv[], size_t *max_msg_size, size_t *iterations, size_t *skip)
{
    char *endptr = NULL;

    static struct option long_opts[] = {{"msg-size", required_argument, nullptr, 'm'},
                                        {"iterations", required_argument, nullptr, 'i'},
                                        {"no-verify", no_argument, &verify, 0},
                                        {"skip", required_argument, nullptr, 's'},
                                        {"help", required_argument, nullptr, 'h'},
                                        {0, 0, nullptr, 0}};

    while ((opt = getopt_long(argc, argv, "m:i:s:h", long_opts, nullptr)) != -1) {
        long val = strtol(optarg, &endptr, 0);
        switch (opt) {
            case 'm':
                if (isPowerOfTwo(static_cast<size_t>(val)) && val > 0)
                    *max_msg_size = static_cast<size_t>(val);
                else {
                    std::cerr << "Warning: The argument for max message size must be a "
                                 "positive integer and a power of two, therefore, the "
                                 "default value of "
                              << *max_msg_size << " will be used." << std::endl;
                }
                break;
            case 'i':
                if (val > 0) *iterations = static_cast<size_t>(val);
                else {
                    std::cerr << "Warning: The argument for number of iterations must be "
                                 "a positive integer, therefore, the default value of "
                              << *iterations << " will be used." << std::endl;
                }
                break;
            case 's':
                if (val >= 0) *skip = static_cast<size_t>(val);
                else {
                    std::cerr << "Warning: The argument for number of skipped iterations "
                                 "must be a positive integer, therefore, the default "
                                 "value of "
                              << *skip << " will be used." << std::endl;
                }
                break;
            case 'h':
            case '?':
                print_usage();
                break;
            default:
                break;
        }
    }
}

void parse_args(int argc, char *argv[], size_t *max_msg_size, int *work_group_size, int *max_groups,
                size_t *iterations, size_t *skip)
{
    char *endptr = NULL;

    static struct option long_opts[] = {{"msg-size", required_argument, nullptr, 'm'},
                                        {"work-group-size", required_argument, nullptr, 'w'},
                                        {"max_groups", required_argument, nullptr, 'g'},
                                        {"iterations", required_argument, nullptr, 'i'},
                                        {"no-verify", no_argument, &verify, 0},
                                        {"skip", required_argument, nullptr, 's'},
                                        {"help", required_argument, nullptr, 'h'},
                                        {0, 0, nullptr, 0}};

    while ((opt = getopt_long(argc, argv, "m:w:g:i:s:h", long_opts, nullptr)) != -1) {
        long val = strtol(optarg, &endptr, 0);
        switch (opt) {
            case 'm':
                if (isPowerOfTwo(static_cast<size_t>(val)) && val > 0)
                    *max_msg_size = static_cast<size_t>(val);
                else {
                    std::cerr << "Warning: The argument for max message size must be a "
                                 "positive integer and a power of two, therefore, the "
                                 "default value of "
                              << *max_msg_size << " will be used." << std::endl;
                }
                break;
            case 'w':
                if (val > 0 && val <= 1024)
                    *work_group_size = static_cast<int>(strtol(optarg, &endptr, 2));
                else {
                    std::cerr << "Warning: The argument for the max work group size must be "
                                 "a positive integer that is no greater than 1024, "
                                 "therefore, the default value of "
                              << *work_group_size << " will be used." << std::endl;
                }
                break;
            case 'g':
                if (val > 0 && val <= (long) work_group_size && (long) work_group_size % val == 0)
                    *max_groups = static_cast<int>(strtol(optarg, &endptr, 2));
                else {
                    std::cerr << "Warning: The argument for number of max work groups must be "
                                 "a positive integer that is both no greater than and divisible by "
                                 "the number of work-items, "
                                 "therefore, the default value of "
                              << *max_groups << " will be used." << std::endl;
                }
                break;
            case 'i':
                if (val > 0) *iterations = static_cast<size_t>(val);
                else {
                    std::cerr << "Warning: The argument for number of iterations must be "
                                 "a positive integer, therefore, the default value of "
                              << *iterations << " will be used." << std::endl;
                }
                break;
            case 's':
                if (val > 0) *skip = static_cast<size_t>(val);
                else {
                    std::cerr << "Warning: The argument for number of skipped iterations "
                                 "must be a positive integer, therefore, the default "
                                 "value of "
                              << *skip << " will be used." << std::endl;
                }
                break;
            case 'h':
            case '?':
                print_usage();
                break;
            default:
                break;
        }
    }
}

#define CHECK_ALLOC(ptr)                                                                           \
    if (ptr == nullptr) fprintf(stderr, "Could not allocate " #ptr "\n");

static inline unsigned long rdtsc()
{
    unsigned int hi, lo;

    __asm volatile(
        "xorl %%eax, %%eax\n"
        "cpuid            \n"
        "rdtsc            \n"
        : "=a"(lo), "=d"(hi)
        :
        : "%ebx", "%ecx");

    return ((unsigned long) hi << 32) | lo;
}

double measure_tsc_frequency()
{
    struct timespec start, stop;
    unsigned long rstart, rstop;
    clock_gettime(CLOCK_REALTIME, &start);
    rstart = rdtsc();
    struct timespec wait = {0, 100000000};
    nanosleep(&wait, nullptr);
    clock_gettime(CLOCK_REALTIME, &stop);
    rstop = rdtsc();
    double seconds = ((double) (stop.tv_sec - start.tv_sec)) +
                     ((double) ((long) stop.tv_nsec - (long) start.tv_nsec) / NSEC_IN_SEC);
    double cycles = static_cast<double>(rstop - rstart);
    return (cycles / seconds);
}

#endif /* COMMON_H */
