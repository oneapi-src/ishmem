/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <common.h>

constexpr int N = 5;

#define TEST_SHMEM_FETCH_AND_SINGLE_TASK(TYPE, TYPENAME)                                           \
    do {                                                                                           \
        TYPE *remote = (TYPE *) ishmem_calloc(1, sizeof(TYPE));                                    \
        TYPE *local = sycl::malloc_host<TYPE>(static_cast<size_t>(npes), q);                       \
        int *errors = sycl::malloc_host<int>(1, q);                                                \
        auto e_init =                                                                              \
            q.submit([&](sycl::handler &h) { h.single_task([=]() { remote[0] = ~(TYPE) 0; }); });  \
        e_init.wait_and_throw();                                                                   \
        ishmem_barrier_all();                                                                      \
        auto e_run = q.submit([&](sycl::handler &h) {                                              \
            h.single_task([=]() {                                                                  \
                for (int i = 0; i < npes; i++)                                                     \
                    ishmem_##TYPENAME##_atomic_fetch_and_nbi(&local[i], remote,                    \
                                                             ~(TYPE) (1LLU << mype), i);           \
                ishmem_barrier_all();                                                              \
            });                                                                                    \
        });                                                                                        \
        e_run.wait_and_throw();                                                                    \
        *errors = 0;                                                                               \
        auto e_verify = q.submit([&](sycl::handler &h) {                                           \
            h.single_task([=]() {                                                                  \
                for (int i = 0; i < npes; i++) {                                                   \
                    if ((local[i] & (TYPE) (1LLU << mype)) == 0) *errors = *errors + 1;            \
                }                                                                                  \
                if (remote[0] != ~(TYPE) ((1LLU << npes) - 1LLU)) *errors = *errors + 1;           \
            });                                                                                    \
        });                                                                                        \
        e_verify.wait_and_throw();                                                                 \
        if (*errors > 0) rc = EXIT_FAILURE;                                                        \
        sycl::free(errors, q);                                                                     \
        sycl::free(local, q);                                                                      \
        ishmem_free(remote);                                                                       \
    } while (false)

#define TEST_SHMEM_FETCH_AND_PARALLEL_FOR(TYPE, TYPENAME)                                          \
    do {                                                                                           \
        TYPE *remote = (TYPE *) ishmem_malloc(N * sizeof(TYPE));                                   \
        TYPE *val = sycl::malloc_host<TYPE>(N * static_cast<size_t>(npes), q);                     \
        int *errors = sycl::malloc_host<int>(1, q);                                                \
        ishmem_barrier_all();                                                                      \
        auto e_init = q.submit([&](sycl::handler &h) {                                             \
            h.parallel_for(sycl::nd_range<1>{N, N}, [=](sycl::nd_item<1> idx) {                    \
                size_t i = idx.get_global_id()[0];                                                 \
                remote[i] = ~(TYPE) 0;                                                             \
                for (size_t j = 0; j < npes; j++)                                                  \
                    val[j + i * (size_t) npes] = (TYPE) 0;                                         \
            });                                                                                    \
        });                                                                                        \
        e_init.wait_and_throw();                                                                   \
        ishmem_barrier_all();                                                                      \
        auto e_run = q.parallel_for(sycl::nd_range<1>{N, N}, [=](sycl::nd_item<1> idx) {           \
            size_t i = idx.get_global_id(0);                                                       \
            auto grp = idx.get_group();                                                            \
            for (size_t j = 0; j < npes; j++)                                                      \
                ishmem_##TYPENAME##_atomic_fetch_and_nbi(&val[i * (size_t) npes + j], &remote[i],  \
                                                         ~(TYPE) (1LLU << mype),                   \
                                                         static_cast<int>(j));                     \
            ishmemx_barrier_all_work_group(grp);                                                   \
        });                                                                                        \
        e_run.wait_and_throw();                                                                    \
        *errors = 0;                                                                               \
        auto e_verify = q.submit([&](sycl::handler &h) {                                           \
            h.single_task([=]() {                                                                  \
                for (int i = 0; i < N * npes; i++) {                                               \
                    if (val[i] == ~(TYPE) ((1LLU << npes) - 1LLU)) *errors = *errors + 1;          \
                }                                                                                  \
                for (int i = 0; i < N; i++) {                                                      \
                    if (remote[i] != ~(TYPE) ((1LLU << npes) - 1LLU)) *errors = *errors + 1;       \
                }                                                                                  \
            });                                                                                    \
        });                                                                                        \
        e_verify.wait_and_throw();                                                                 \
        if (*errors > 0) rc = EXIT_FAILURE;                                                        \
        sycl::free(errors, q);                                                                     \
        sycl::free(val, q);                                                                        \
        ishmem_free(remote);                                                                       \
    } while (false)

int main(int argc, char *argv[])
{
    ishmem_init();

    sycl::queue q;

    const int mype = ishmem_my_pe();
    const int npes = ishmem_n_pes();

    if (mype == 0) {
        std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
                  << std::endl;
        std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
                  << std::endl;
    }

    int rc = EXIT_SUCCESS;

    TEST_SHMEM_FETCH_AND_SINGLE_TASK(unsigned int, uint);
    TEST_SHMEM_FETCH_AND_SINGLE_TASK(unsigned long, ulong);
    TEST_SHMEM_FETCH_AND_SINGLE_TASK(unsigned long long, ulonglong);
    TEST_SHMEM_FETCH_AND_SINGLE_TASK(int32_t, int32);
    TEST_SHMEM_FETCH_AND_SINGLE_TASK(int64_t, int64);
    TEST_SHMEM_FETCH_AND_SINGLE_TASK(uint32_t, uint32);
    TEST_SHMEM_FETCH_AND_SINGLE_TASK(uint64_t, uint64);

    TEST_SHMEM_FETCH_AND_PARALLEL_FOR(unsigned int, uint);
    TEST_SHMEM_FETCH_AND_PARALLEL_FOR(unsigned long, ulong);
    TEST_SHMEM_FETCH_AND_PARALLEL_FOR(unsigned long long, ulonglong);
    TEST_SHMEM_FETCH_AND_PARALLEL_FOR(int32_t, int32);
    TEST_SHMEM_FETCH_AND_PARALLEL_FOR(int64_t, int64);
    TEST_SHMEM_FETCH_AND_PARALLEL_FOR(uint32_t, uint32);
    TEST_SHMEM_FETCH_AND_PARALLEL_FOR(uint64_t, uint64);

    ishmem_finalize();
    if (rc) std::cout << mype << ": Test Failed" << std::endl;
    else std::cout << mype << ": Test Passed" << std::endl;
    return rc;
}
