/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "internal.h"
#include "impl_proxy.h"

/* Test */
template <typename T>
int ishmem_test(T *ivar, int cmp, T cmp_value)
{
    if constexpr (ishmemi_is_device) {
        if constexpr (enable_error_checking) {
            validate_parameters((void *) ivar, sizeof(T));
        }
        sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                         sycl::access::address_space::global_space>
            atomic_ivar(*ivar);
        T atomic_val = atomic_ivar.load(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        return (ishmemi_comparison<T>(atomic_val, cmp_value, cmp) > 0);
    } else {
        RAISE_ERROR_MSG("ishmem_test not callable from host\n");
    }
}

/* Test (work-group) */
template <typename T, typename Group>
int ishmemx_test_work_group(T *ivar, int cmp, T cmp_value, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        int result = 0;
        sycl::group_barrier(grp);
        if (grp.leader()) {
            if constexpr (enable_error_checking) {
                validate_parameters((void *) ivar, sizeof(T));
            }
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_ivar(*ivar);
            T atomic_val =
                atomic_ivar.load(sycl::memory_order::seq_cst, sycl::memory_scope::system);
            result = (ishmemi_comparison<T>(atomic_val, cmp_value, cmp) > 0);
        }
        result = sycl::group_broadcast(grp, result);
        sycl::group_barrier(grp); /* TODO not needed if group_broadcast actually works */
        return result;
    } else {
        RAISE_ERROR_MSG("ishmemx_test_work_group not callable from host\n");
    }
}

/* clang-format off */
#define ISHMEMI_API_IMPL_TEST(TYPENAME, TYPE)                                                                                             \
    int ishmem_##TYPENAME##_test(TYPE *ivar, int cmp, TYPE cmp_value) { return ishmem_test(ivar, cmp, cmp_value); }                       \
    template int ishmemx_##TYPENAME##_test_work_group<sycl::group<1>>(TYPE *ivar, int cmp, TYPE cmp_value, const sycl::group<1> &grp);    \
    template int ishmemx_##TYPENAME##_test_work_group<sycl::group<2>>(TYPE *ivar, int cmp, TYPE cmp_value, const sycl::group<2> &grp);    \
    template int ishmemx_##TYPENAME##_test_work_group<sycl::group<3>>(TYPE *ivar, int cmp, TYPE cmp_value, const sycl::group<3> &grp);    \
    template int ishmemx_##TYPENAME##_test_work_group<sycl::sub_group>(TYPE *ivar, int cmp, TYPE cmp_value, const sycl::sub_group &grp);  \
    template <typename Group> int ishmemx_##TYPENAME##_test_work_group(TYPE *ivar, int cmp, TYPE cmp_value, const Group &grp) { return ishmemx_test_work_group(ivar, cmp, cmp_value, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_TEST(int, int)
ISHMEMI_API_IMPL_TEST(long, long)
ISHMEMI_API_IMPL_TEST(longlong, long long)
ISHMEMI_API_IMPL_TEST(uint, unsigned int)
ISHMEMI_API_IMPL_TEST(ulong, unsigned long)
ISHMEMI_API_IMPL_TEST(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEST(int32, int32_t)
ISHMEMI_API_IMPL_TEST(int64, int64_t)
ISHMEMI_API_IMPL_TEST(uint32, uint32_t)
ISHMEMI_API_IMPL_TEST(uint64, uint64_t)
ISHMEMI_API_IMPL_TEST(size, size_t)
ISHMEMI_API_IMPL_TEST(ptrdiff, ptrdiff_t)

/* Wait-Until */
template <typename T>
void ishmem_wait_until(T *ivar, int cmp, T cmp_value)
{
    if constexpr (ishmemi_is_device) {
        if constexpr (enable_error_checking) {
            validate_parameters((void *) ivar, sizeof(T));
        }
        sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                         sycl::access::address_space::global_space>
            atomic_ivar(*ivar);
        T atomic_val = 0;
        do {
            atomic_val = atomic_ivar.load(sycl::memory_order::seq_cst, sycl::memory_scope::system);
            if (ishmemi_comparison<T>(atomic_val, cmp_value, cmp)) return;
            while (atomic_ivar.load(sycl::memory_order::seq_cst, sycl::memory_scope::system) ==
                   atomic_val) {
            }
        } while (true);
    } else {
        RAISE_ERROR_MSG("ishmem_wait_until not callable from host\n");
    }
}

/* Wait-Until (work-group) */
template <typename T, typename Group>
void ishmemx_wait_until_work_group(T *ivar, int cmp, T cmp_value, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        sycl::group_barrier(grp);
        if (grp.leader()) {
            if constexpr (enable_error_checking) {
                validate_parameters((void *) ivar, sizeof(T));
            }
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_ivar(*ivar);
            T atomic_val = 0;
            do {
                atomic_val =
                    atomic_ivar.load(sycl::memory_order::seq_cst, sycl::memory_scope::system);
                if (ishmemi_comparison<T>(atomic_val, cmp_value, cmp)) return;
                while (atomic_ivar.load(sycl::memory_order::seq_cst, sycl::memory_scope::system) ==
                       atomic_val) {
                }
            } while (true);
        }
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_wait_until_work_group not callable from host\n");
    }
}

/* clang-format off */
#define ISHMEMI_API_IMPL_WAIT_UNTIL(TYPENAME, TYPE)                                                                                              \
    void ishmem_##TYPENAME##_wait_until(TYPE *ivar, int cmp, TYPE cmp_value) { ishmem_wait_until(ivar, cmp, cmp_value); }                        \
    template void ishmemx_##TYPENAME##_wait_until_work_group<sycl::group<1>>(TYPE *ivar, int cmp, TYPE cmp_value, const sycl::group<1> &grp);    \
    template void ishmemx_##TYPENAME##_wait_until_work_group<sycl::group<2>>(TYPE *ivar, int cmp, TYPE cmp_value, const sycl::group<2> &grp);    \
    template void ishmemx_##TYPENAME##_wait_until_work_group<sycl::group<3>>(TYPE *ivar, int cmp, TYPE cmp_value, const sycl::group<3> &grp);    \
    template void ishmemx_##TYPENAME##_wait_until_work_group<sycl::sub_group>(TYPE *ivar, int cmp, TYPE cmp_value, const sycl::sub_group &grp);  \
    template <typename Group> void ishmemx_##TYPENAME##_wait_until_work_group(TYPE *ivar, int cmp, TYPE cmp_value, const Group &grp) { ishmemx_wait_until_work_group(ivar, cmp, cmp_value, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_WAIT_UNTIL(int, int)
ISHMEMI_API_IMPL_WAIT_UNTIL(long, long)
ISHMEMI_API_IMPL_WAIT_UNTIL(longlong, long long)
ISHMEMI_API_IMPL_WAIT_UNTIL(uint, unsigned int)
ISHMEMI_API_IMPL_WAIT_UNTIL(ulong, unsigned long)
ISHMEMI_API_IMPL_WAIT_UNTIL(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_WAIT_UNTIL(int32, int32_t)
ISHMEMI_API_IMPL_WAIT_UNTIL(int64, int64_t)
ISHMEMI_API_IMPL_WAIT_UNTIL(uint32, uint32_t)
ISHMEMI_API_IMPL_WAIT_UNTIL(uint64, uint64_t)
ISHMEMI_API_IMPL_WAIT_UNTIL(size, size_t)
ISHMEMI_API_IMPL_WAIT_UNTIL(ptrdiff, ptrdiff_t)
