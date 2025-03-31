/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <common.h>

constexpr int N = 5;

#define TEST_SHMEM_FETCH_OR_SINGLE_TASK(TYPE, TYPENAME)                                            \
    do {                                                                                           \
        TYPE *remote = (TYPE *) ishmem_calloc(1, sizeof(TYPE));                                    \
        TYPE *fetched_val = (TYPE *) ishmem_calloc(static_cast<size_t>(npes), sizeof(TYPE));       \
        TYPE *host_fetched_val = (TYPE *) sycl::malloc_host<TYPE>(static_cast<size_t>(npes), q);   \
        TYPE *host_remote = (TYPE *) sycl::malloc_host<TYPE>(1, q);                                \
        int *errors = sycl::malloc_host<int>(1, q);                                                \
        ishmem_barrier_all();                                                                      \
        auto e_run = q.submit([&](sycl::handler &h) {                                              \
            h.single_task([=]() {                                                                  \
                for (int i = 0; i < npes; i++) {                                                   \
                    fetched_val[i] =                                                               \
                        ishmem_##TYPENAME##_atomic_fetch_or(remote, (TYPE) (1LLU << mype), i);     \
                }                                                                                  \
            });                                                                                    \
        });                                                                                        \
        e_run.wait_and_throw();                                                                    \
        ishmem_barrier_all();                                                                      \
        *errors = 0;                                                                               \
        auto e_verify = q.submit([&](sycl::handler &h) {                                           \
            h.single_task([=]() {                                                                  \
                for (int i = 0; i < npes; i++) {                                                   \
                    if (fetched_val[i] == (TYPE) ((1LLU << npes) - 1LLU)) *errors = *errors + 1;   \
                }                                                                                  \
                if (remote[0] != (TYPE) ((1LLU << npes) - 1LLU)) *errors = *errors + 1;            \
            });                                                                                    \
        });                                                                                        \
        e_verify.wait_and_throw();                                                                 \
        if (*errors > 0) {                                                                         \
            rc = EXIT_FAILURE;                                                                     \
            q.memcpy(host_fetched_val, fetched_val, static_cast<size_t>(npes) * sizeof(TYPE))      \
                .wait_and_throw();                                                                 \
            q.memcpy(host_remote, remote, sizeof(TYPE)).wait_and_throw();                          \
            for (int i = 0; i < npes; i++) {                                                       \
                std::cout << "[" << mype << "]: fetched " << host_fetched_val[i] << std::endl;     \
            }                                                                                      \
            std::cout << "[" << mype << "]: remote " << host_remote[0] << std::endl;               \
        }                                                                                          \
        sycl::free(errors, q);                                                                     \
        sycl::free(host_fetched_val, q);                                                           \
        sycl::free(host_remote, q);                                                                \
        ishmem_free(remote);                                                                       \
        ishmem_free(fetched_val);                                                                  \
    } while (false)

#define TEST_SHMEM_FETCH_OR_PARALLEL_FOR(TYPE, TYPENAME)                                           \
    do {                                                                                           \
        TYPE *remote = (TYPE *) ishmem_calloc(N, sizeof(TYPE));                                    \
        TYPE *fetched_val = (TYPE *) ishmem_calloc(N * static_cast<size_t>(npes), sizeof(TYPE));   \
        TYPE *host_fetched_val =                                                                   \
            (TYPE *) sycl::malloc_host<TYPE>(N * static_cast<size_t>(npes), q);                    \
        TYPE *host_remote = (TYPE *) sycl::malloc_host<TYPE>(N, q);                                \
        int *errors = sycl::malloc_host<int>(1, q);                                                \
        ishmem_barrier_all();                                                                      \
        auto e_run = q.parallel_for(sycl::nd_range<1>{N, N}, [=](sycl::nd_item<1> idx) {           \
            size_t i = idx.get_global_id(0);                                                       \
            for (size_t j = 0; j < npes; j++)                                                      \
                fetched_val[j + i * (size_t) npes] = ishmem_##TYPENAME##_atomic_fetch_or(          \
                    &remote[i], (TYPE) (1LLU << mype), static_cast<int>(j));                       \
        });                                                                                        \
        e_run.wait_and_throw();                                                                    \
        ishmem_barrier_all();                                                                      \
        *errors = 0;                                                                               \
        auto e_verify = q.submit([&](sycl::handler &h) {                                           \
            h.single_task([=]() {                                                                  \
                for (int i = 0; i < N * npes; i++) {                                               \
                    if (fetched_val[i] == (TYPE) ((1LLU << npes) - 1LLU)) *errors = *errors + 1;   \
                }                                                                                  \
                for (int i = 0; i < N; i++) {                                                      \
                    if (remote[i] != (TYPE) ((1LLU << npes) - 1LLU)) *errors = *errors + 1;        \
                }                                                                                  \
            });                                                                                    \
        });                                                                                        \
        e_verify.wait_and_throw();                                                                 \
        if (*errors > 0) {                                                                         \
            rc = EXIT_FAILURE;                                                                     \
            q.memcpy(host_fetched_val, fetched_val, N *static_cast<size_t>(npes) * sizeof(TYPE))   \
                .wait_and_throw();                                                                 \
            q.memcpy(host_remote, remote, N * sizeof(TYPE)).wait_and_throw();                      \
            for (int i = 0; i < N * npes; ++i) {                                                   \
                std::cout << "[" << mype << "]: fetched " << host_fetched_val[i] << std::endl;     \
            }                                                                                      \
            for (int i = 0; i < N; ++i) {                                                          \
                std::cout << "[" << mype << "]: remote " << host_remote[i] << std::endl;           \
            }                                                                                      \
        }                                                                                          \
        sycl::free(errors, q);                                                                     \
        ishmem_free(fetched_val);                                                                  \
        ishmem_free(remote);                                                                       \
        sycl::free(host_fetched_val, q);                                                           \
        sycl::free(host_remote, q);                                                                \
    } while (false)

int main(int argc, char *argv[])
{
    ishmem_init();

    sycl::queue q;

    int mype = ishmem_my_pe();
    int npes = ishmem_n_pes();

    if (mype == 0) {
        std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
                  << std::endl;
        std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
                  << std::endl;
    }

    int rc = EXIT_SUCCESS;

    TEST_SHMEM_FETCH_OR_SINGLE_TASK(unsigned int, uint);
    TEST_SHMEM_FETCH_OR_SINGLE_TASK(unsigned long, ulong);
    TEST_SHMEM_FETCH_OR_SINGLE_TASK(unsigned long long, ulonglong);
    TEST_SHMEM_FETCH_OR_SINGLE_TASK(int32_t, int32);
    TEST_SHMEM_FETCH_OR_SINGLE_TASK(int64_t, int64);
    TEST_SHMEM_FETCH_OR_SINGLE_TASK(uint32_t, uint32);
    TEST_SHMEM_FETCH_OR_SINGLE_TASK(uint64_t, uint64);

    TEST_SHMEM_FETCH_OR_PARALLEL_FOR(unsigned int, uint);
    TEST_SHMEM_FETCH_OR_PARALLEL_FOR(unsigned long, ulong);
    TEST_SHMEM_FETCH_OR_PARALLEL_FOR(unsigned long long, ulonglong);
    TEST_SHMEM_FETCH_OR_PARALLEL_FOR(int32_t, int32);
    TEST_SHMEM_FETCH_OR_PARALLEL_FOR(int64_t, int64);
    TEST_SHMEM_FETCH_OR_PARALLEL_FOR(uint32_t, uint32);
    TEST_SHMEM_FETCH_OR_PARALLEL_FOR(uint64_t, uint64);

    ishmem_finalize();

    if (rc) std::cout << "PE " << mype << ": Test Failed" << std::endl;
    else std::cout << "PE " << mype << ": Test Passed" << std::endl;

    return rc;
}
