/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/err.h"
#include "proxy_impl.h"
#include "runtime.h"
#include "ishmem/util.h"
#include "on_queue.h"

/* Test */
template <typename T>
int ishmem_test(T *ivar, int cmp, T cmp_value)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) ivar, sizeof(T));
    }
#ifdef __SYCL_DEVICE_ONLY__
    sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        atomic_ivar(*ivar);
    T atomic_val = atomic_ivar.load();
    return (ishmemi_comparison<T>(atomic_val, cmp_value, cmp) > 0);
#else
    int ret = 0;
    ishmemi_request_t req;
    req.dst = ivar;
    req.nelems = 1;
    req.cmp = cmp;
    ishmemi_union_set_field_value<T, TEST>(req.cmp_value, cmp_value);
    req.op = TEST;
    req.type = ishmemi_union_get_base_type<T, TEST>();
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = comp.completion.ret.i;
    return ret;
#endif
}

/* Test (work-group) */
template <typename T, typename Group>
int ishmemx_test_work_group(T *ivar, int cmp, T cmp_value, const Group &grp)
{
#ifdef __SYCL_DEVICE_ONLY__
    int result = 0;
    sycl::group_barrier(grp);
    if (grp.leader()) {
        result = ishmem_test(ivar, cmp, cmp_value);
    }
    result = sycl::group_broadcast(grp, result);
    sycl::group_barrier(grp); /* TODO not needed if group_broadcast actually works */
    return result;
#else
    RAISE_ERROR_MSG("ISHMEMX_TEST_WORK_GROUP routines are not callable from host\n");
#endif
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

/* Test-All */
template <typename T>
int ishmem_test_all(T *ivars, size_t nelems, const int *status, int cmp, T cmp_value)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) ivars, (void *) status, nelems * sizeof(T),
                            nelems * sizeof(int), ishmemi_op_t::TEST_ALL);
    }
#ifdef __SYCL_DEVICE_ONLY__
    T atomic_val = 0;
    for (size_t i = 0; i < nelems; i++) {
        if (!status || !status[i]) {
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_ivar(ivars[i]);
            atomic_val = atomic_ivar.load();
            if (ishmemi_comparison<T>(atomic_val, cmp_value, cmp) == 0) return 0;
        }
    }
    return 1;
#else
    int ret = 0;
    ishmemi_request_t req;
    req.dst = ivars;
    req.nelems = nelems;
    req.status = status;
    req.cmp = cmp;
    ishmemi_union_set_field_value<T, TEST_ALL>(req.cmp_value, cmp_value);
    req.op = TEST_ALL;
    req.type = ishmemi_union_get_base_type<T, TEST_ALL>();
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = comp.completion.ret.i;
    return ret;
#endif
}

/* Test-All (work-group) */
template <typename T, typename Group>
int ishmemx_test_all_work_group(T *ivars, size_t nelems, const int *status, int cmp, T cmp_value,
                                const Group &grp)
{
    if constexpr (enable_error_checking) {
        if (grp.leader())
            validate_parameters((void *) ivars, (void *) status, nelems * sizeof(T),
                                nelems * sizeof(int), ishmemi_op_t::TEST_ALL);
    }
#ifdef __SYCL_DEVICE_ONLY__
    int result = 1;
    sycl::group_barrier(grp);
    size_t id = grp.get_local_linear_id();
    size_t num_threads = grp.get_local_linear_range();
    for (size_t i = id; i < nelems; i += num_threads) {
        if (!status || !status[i]) result &= ishmem_test(&ivars[i], cmp, cmp_value);
    }
    result = sycl::reduce_over_group(grp, result, sycl::bit_and<>());
    return result;
#else
    RAISE_ERROR_MSG("ISHMEMX_TEST_ALL_WORK_GROUP routines are not callable from host\n");
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_TEST_ALL(TYPENAME, TYPE)                                                                                                                                \
    int ishmem_##TYPENAME##_test_all(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value) { return ishmem_test_all(ivars, nelems, status, cmp, cmp_value); }    \
    template int ishmemx_##TYPENAME##_test_all_work_group<sycl::group<1>>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::group<1> &grp);    \
    template int ishmemx_##TYPENAME##_test_all_work_group<sycl::group<2>>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::group<2> &grp);    \
    template int ishmemx_##TYPENAME##_test_all_work_group<sycl::group<3>>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::group<3> &grp);    \
    template int ishmemx_##TYPENAME##_test_all_work_group<sycl::sub_group>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::sub_group &grp);  \
    template <typename Group> int ishmemx_##TYPENAME##_test_all_work_group(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const Group &grp) { return ishmemx_test_all_work_group(ivars, nelems, status, cmp, cmp_value, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_TEST_ALL(int, int)
ISHMEMI_API_IMPL_TEST_ALL(long, long)
ISHMEMI_API_IMPL_TEST_ALL(longlong, long long)
ISHMEMI_API_IMPL_TEST_ALL(uint, unsigned int)
ISHMEMI_API_IMPL_TEST_ALL(ulong, unsigned long)
ISHMEMI_API_IMPL_TEST_ALL(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEST_ALL(int32, int32_t)
ISHMEMI_API_IMPL_TEST_ALL(int64, int64_t)
ISHMEMI_API_IMPL_TEST_ALL(uint32, uint32_t)
ISHMEMI_API_IMPL_TEST_ALL(uint64, uint64_t)
ISHMEMI_API_IMPL_TEST_ALL(size, size_t)
ISHMEMI_API_IMPL_TEST_ALL(ptrdiff, ptrdiff_t)

/* Test-Any */
template <typename T>
size_t ishmem_test_any(T *ivars, size_t nelems, const int *status, int cmp, T cmp_value)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) ivars, (void *) status, nelems * sizeof(T),
                            nelems * sizeof(int), ishmemi_op_t::TEST_ANY);
    }
#ifdef __SYCL_DEVICE_ONLY__
    T atomic_val = 0;
    ishmemi_info_t *info = global_info;
    size_t cur_idx = info->sync_last_idx_checked;
    for (size_t i = 0; i < nelems; i++) {
        cur_idx += 1;
        if (cur_idx >= nelems) cur_idx = 0;
        if (!status || !status[cur_idx]) {
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_ivar(ivars[cur_idx]);
            atomic_val = atomic_ivar.load();
            if (ishmemi_comparison<T>(atomic_val, cmp_value, cmp) > 0) {
                info->sync_last_idx_checked = cur_idx;
                return cur_idx;
            }
        }
    }
    return SIZE_MAX;
#else
    size_t ret = 0;
    ishmemi_request_t req;
    req.dst = ivars;
    req.nelems = nelems;
    req.status = status;
    req.cmp = cmp;
    ishmemi_union_set_field_value<T, TEST_ANY>(req.cmp_value, cmp_value);
    req.op = TEST_ANY;
    req.type = ishmemi_union_get_base_type<T, TEST_ANY>();
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = comp.completion.ret.szt;
    return ret;
#endif
}

/* Test-Any (work-group) */
template <typename T, typename Group>
size_t ishmemx_test_any_work_group(T *ivars, size_t nelems, const int *status, int cmp, T cmp_value,
                                   const Group &grp)
{
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(grp);
    size_t result = SIZE_MAX;
    if (grp.leader()) {
        result = ishmem_test_any(ivars, nelems, status, cmp, cmp_value);
    }
    result = sycl::group_broadcast(grp, result);
    return result;
#else
    RAISE_ERROR_MSG("ISHMEMX_TEST_ANY_WORK_GROUP routines are not callable from host\n");
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_TEST_ANY(TYPENAME, TYPE)                                                                                                                                  \
    size_t ishmem_##TYPENAME##_test_any(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value) { return ishmem_test_any(ivars, nelems, status, cmp, cmp_value); }    \
    template size_t ishmemx_##TYPENAME##_test_any_work_group<sycl::group<1>>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::group<1> &grp);    \
    template size_t ishmemx_##TYPENAME##_test_any_work_group<sycl::group<2>>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::group<2> &grp);    \
    template size_t ishmemx_##TYPENAME##_test_any_work_group<sycl::group<3>>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::group<3> &grp);    \
    template size_t ishmemx_##TYPENAME##_test_any_work_group<sycl::sub_group>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::sub_group &grp);  \
    template <typename Group> size_t ishmemx_##TYPENAME##_test_any_work_group(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const Group &grp) { return ishmemx_test_any_work_group(ivars, nelems, status, cmp, cmp_value, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_TEST_ANY(int, int)
ISHMEMI_API_IMPL_TEST_ANY(long, long)
ISHMEMI_API_IMPL_TEST_ANY(longlong, long long)
ISHMEMI_API_IMPL_TEST_ANY(uint, unsigned int)
ISHMEMI_API_IMPL_TEST_ANY(ulong, unsigned long)
ISHMEMI_API_IMPL_TEST_ANY(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEST_ANY(int32, int32_t)
ISHMEMI_API_IMPL_TEST_ANY(int64, int64_t)
ISHMEMI_API_IMPL_TEST_ANY(uint32, uint32_t)
ISHMEMI_API_IMPL_TEST_ANY(uint64, uint64_t)
ISHMEMI_API_IMPL_TEST_ANY(size, size_t)
ISHMEMI_API_IMPL_TEST_ANY(ptrdiff, ptrdiff_t)

/* Test-Some */
template <typename T>
size_t ishmem_test_some(T *ivars, size_t nelems, size_t *indices, const int *status, int cmp,
                        T cmp_value)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) ivars, (void *) indices, (void *) status, nelems * sizeof(T),
                            nelems * sizeof(size_t), nelems * sizeof(int));
    }
#ifdef __SYCL_DEVICE_ONLY__
    size_t cmp_counter = 0;
    T atomic_val = 0;
    for (size_t i = 0; i < nelems; i++) {
        if (!status || !status[i]) {
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_ivar(ivars[i]);
            atomic_val = atomic_ivar.load();
            if (ishmemi_comparison<T>(atomic_val, cmp_value, cmp) > 0) {
                indices[cmp_counter++] = i;
            }
        }
    }
    return cmp_counter;
#else
    size_t ret = 0;
    ishmemi_request_t req;
    req.dst = ivars;
    req.nelems = nelems;
    req.indices = indices;
    req.status = status;
    req.cmp = cmp;
    ishmemi_union_set_field_value<T, TEST_SOME>(req.cmp_value, cmp_value);
    req.op = TEST_SOME;
    req.type = ishmemi_union_get_base_type<T, TEST_SOME>();
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = comp.completion.ret.szt;
    return ret;
#endif
}

/* Test-Some (work-group) */
template <typename T, typename Group>
size_t ishmemx_test_some_work_group(T *ivars, size_t nelems, size_t *indices, const int *status,
                                    int cmp, T cmp_value, const Group &grp)
{
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(grp);
    size_t cmp_counter = 0;
    if (grp.leader()) {
        cmp_counter = ishmem_test_some(ivars, nelems, indices, status, cmp, cmp_value);
    }
    cmp_counter = sycl::group_broadcast(grp, cmp_counter);
    return cmp_counter;
#else
    RAISE_ERROR_MSG("ISHMEMX_TEST_SOME_WORK_GROUP routines are not callable from host\n");
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_TEST_SOME(TYPENAME, TYPE)                                                                                                                                  \
    size_t ishmem_##TYPENAME##_test_some(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value) { return ishmem_test_some(ivars, nelems, indices, status, cmp, cmp_value); }    \
    template size_t ishmemx_##TYPENAME##_test_some_work_group<sycl::group<1>>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, const sycl::group<1> &grp);    \
    template size_t ishmemx_##TYPENAME##_test_some_work_group<sycl::group<2>>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, const sycl::group<2> &grp);    \
    template size_t ishmemx_##TYPENAME##_test_some_work_group<sycl::group<3>>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, const sycl::group<3> &grp);    \
    template size_t ishmemx_##TYPENAME##_test_some_work_group<sycl::sub_group>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, const sycl::sub_group &grp);  \
    template <typename Group> size_t ishmemx_##TYPENAME##_test_some_work_group(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, const Group &grp) { return ishmemx_test_some_work_group(ivars, nelems, indices, status, cmp, cmp_value, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_TEST_SOME(int, int)
ISHMEMI_API_IMPL_TEST_SOME(long, long)
ISHMEMI_API_IMPL_TEST_SOME(longlong, long long)
ISHMEMI_API_IMPL_TEST_SOME(uint, unsigned int)
ISHMEMI_API_IMPL_TEST_SOME(ulong, unsigned long)
ISHMEMI_API_IMPL_TEST_SOME(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEST_SOME(int32, int32_t)
ISHMEMI_API_IMPL_TEST_SOME(int64, int64_t)
ISHMEMI_API_IMPL_TEST_SOME(uint32, uint32_t)
ISHMEMI_API_IMPL_TEST_SOME(uint64, uint64_t)
ISHMEMI_API_IMPL_TEST_SOME(size, size_t)
ISHMEMI_API_IMPL_TEST_SOME(ptrdiff, ptrdiff_t)

/* Test-All-Vector */
template <typename T>
int ishmem_test_all_vector(T *ivars, size_t nelems, const int *status, int cmp, const T *cmp_values)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) ivars, (void *) status, nelems * sizeof(T),
                            nelems * sizeof(int), ishmemi_op_t::TEST_ALL_VECTOR);
    }
#ifdef __SYCL_DEVICE_ONLY__
    T atomic_val = 0;
    for (size_t i = 0; i < nelems; i++) {
        if (!status || !status[i]) {
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_ivar(ivars[i]);
            atomic_val = atomic_ivar.load();
            if (ishmemi_comparison<T>(atomic_val, cmp_values[i], cmp) == 0) return 0;
        }
    }
    return 1;
#else
    int ret = 0;
    ishmemi_request_t req;
    req.dst = ivars;
    req.nelems = nelems;
    req.status = status;
    req.cmp = cmp;
    req.cmp_values = cmp_values;
    req.op = TEST_ALL_VECTOR;
    req.type = ishmemi_union_get_base_type<T, TEST_ALL_VECTOR>();
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = comp.completion.ret.i;
    return ret;
#endif
}

/* Test-All-Vector (work-group) */
template <typename T, typename Group>
int ishmemx_test_all_vector_work_group(T *ivars, size_t nelems, const int *status, int cmp,
                                       const T *cmp_values, const Group &grp)
{
    if constexpr (enable_error_checking) {
        if (grp.leader())
            validate_parameters((void *) ivars, (void *) status, nelems * sizeof(T),
                                nelems * sizeof(int), ishmemi_op_t::TEST_ALL_VECTOR);
    }
#ifdef __SYCL_DEVICE_ONLY__
    int result = 1;
    sycl::group_barrier(grp);
    size_t id = grp.get_local_linear_id();
    size_t num_threads = grp.get_local_linear_range();
    for (size_t i = id; i < nelems; i += num_threads) {
        if (!status || !status[i]) result &= ishmem_test(&ivars[i], cmp, cmp_values[i]);
    }
    result = sycl::reduce_over_group(grp, result, sycl::bit_and<>());
    return result;
#else
    RAISE_ERROR_MSG("ISHMEMX_TEST_ALL_VECTOR_WORK_GROUP routines are not callable from host\n");
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_TEST_ALL_VECTOR(TYPENAME, TYPE)                                                                                                                                         \
    int ishmem_##TYPENAME##_test_all_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values) { return ishmem_test_all_vector(ivars, nelems, status, cmp, cmp_values); } \
    template int ishmemx_##TYPENAME##_test_all_vector_work_group<sycl::group<1>>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<1> &grp);           \
    template int ishmemx_##TYPENAME##_test_all_vector_work_group<sycl::group<2>>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<2> &grp);           \
    template int ishmemx_##TYPENAME##_test_all_vector_work_group<sycl::group<3>>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<3> &grp);           \
    template int ishmemx_##TYPENAME##_test_all_vector_work_group<sycl::sub_group>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::sub_group &grp);         \
    template <typename Group> int ishmemx_##TYPENAME##_test_all_vector_work_group(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const Group &grp) { return ishmemx_test_all_vector_work_group(ivars, nelems, status, cmp, cmp_values, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_TEST_ALL_VECTOR(int, int)
ISHMEMI_API_IMPL_TEST_ALL_VECTOR(long, long)
ISHMEMI_API_IMPL_TEST_ALL_VECTOR(longlong, long long)
ISHMEMI_API_IMPL_TEST_ALL_VECTOR(uint, unsigned int)
ISHMEMI_API_IMPL_TEST_ALL_VECTOR(ulong, unsigned long)
ISHMEMI_API_IMPL_TEST_ALL_VECTOR(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEST_ALL_VECTOR(int32, int32_t)
ISHMEMI_API_IMPL_TEST_ALL_VECTOR(int64, int64_t)
ISHMEMI_API_IMPL_TEST_ALL_VECTOR(uint32, uint32_t)
ISHMEMI_API_IMPL_TEST_ALL_VECTOR(uint64, uint64_t)
ISHMEMI_API_IMPL_TEST_ALL_VECTOR(size, size_t)
ISHMEMI_API_IMPL_TEST_ALL_VECTOR(ptrdiff, ptrdiff_t)

/* Test-Any-Vector */
template <typename T>
size_t ishmem_test_any_vector(T *ivars, size_t nelems, const int *status, int cmp,
                              const T *cmp_values)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) ivars, (void *) status, nelems * sizeof(T),
                            nelems * sizeof(int), ishmemi_op_t::TEST_ANY_VECTOR);
    }
#ifdef __SYCL_DEVICE_ONLY__
    T atomic_val = 0;
    ishmemi_info_t *info = global_info;
    size_t cur_idx = info->sync_last_idx_checked;
    for (size_t i = 0; i < nelems; i++) {
        cur_idx += 1;
        if (cur_idx >= nelems) cur_idx = 0;
        if (!status || !status[cur_idx]) {
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_ivar(ivars[cur_idx]);
            atomic_val = atomic_ivar.load();
            if (ishmemi_comparison<T>(atomic_val, cmp_values[cur_idx], cmp) > 0) {
                info->sync_last_idx_checked = cur_idx;
                return cur_idx;
            }
        }
    }
    return SIZE_MAX;
#else
    size_t ret = 0;
    ishmemi_request_t req;
    req.dst = ivars;
    req.nelems = nelems;
    req.status = status;
    req.cmp = cmp;
    req.cmp_values = cmp_values;
    req.op = TEST_ANY_VECTOR;
    req.type = ishmemi_union_get_base_type<T, TEST_ANY_VECTOR>();
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = comp.completion.ret.szt;
    return ret;
#endif
}

/* Test-Any-Vector (work-group) */
template <typename T, typename Group>
size_t ishmemx_test_any_vector_work_group(T *ivars, size_t nelems, const int *status, int cmp,
                                          const T *cmp_values, const Group &grp)
{
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(grp);
    size_t result = SIZE_MAX;
    if (grp.leader()) {
        result = ishmem_test_any_vector(ivars, nelems, status, cmp, cmp_values);
    }
    result = sycl::group_broadcast(grp, result);
    return result;
#else
    RAISE_ERROR_MSG("ISHMEMX_TEST_ANY_VECTOR_WORK_GROUP routines are not callable from host\n");
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_TEST_ANY_VECTOR(TYPENAME, TYPE)                                                                                                                                            \
    size_t ishmem_##TYPENAME##_test_any_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values) { return ishmem_test_any_vector(ivars, nelems, status, cmp, cmp_values); } \
    template size_t ishmemx_##TYPENAME##_test_any_vector_work_group<sycl::group<1>>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<1> &grp);           \
    template size_t ishmemx_##TYPENAME##_test_any_vector_work_group<sycl::group<2>>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<2> &grp);           \
    template size_t ishmemx_##TYPENAME##_test_any_vector_work_group<sycl::group<3>>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<3> &grp);           \
    template size_t ishmemx_##TYPENAME##_test_any_vector_work_group<sycl::sub_group>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::sub_group &grp);         \
    template <typename Group> size_t ishmemx_##TYPENAME##_test_any_vector_work_group(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const Group &grp) { return ishmemx_test_any_vector_work_group(ivars, nelems, status, cmp, cmp_values, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_TEST_ANY_VECTOR(int, int)
ISHMEMI_API_IMPL_TEST_ANY_VECTOR(long, long)
ISHMEMI_API_IMPL_TEST_ANY_VECTOR(longlong, long long)
ISHMEMI_API_IMPL_TEST_ANY_VECTOR(uint, unsigned int)
ISHMEMI_API_IMPL_TEST_ANY_VECTOR(ulong, unsigned long)
ISHMEMI_API_IMPL_TEST_ANY_VECTOR(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEST_ANY_VECTOR(int32, int32_t)
ISHMEMI_API_IMPL_TEST_ANY_VECTOR(int64, int64_t)
ISHMEMI_API_IMPL_TEST_ANY_VECTOR(uint32, uint32_t)
ISHMEMI_API_IMPL_TEST_ANY_VECTOR(uint64, uint64_t)
ISHMEMI_API_IMPL_TEST_ANY_VECTOR(size, size_t)
ISHMEMI_API_IMPL_TEST_ANY_VECTOR(ptrdiff, ptrdiff_t)

/* Test-Some-Vector */
template <typename T>
size_t ishmem_test_some_vector(T *ivars, size_t nelems, size_t *indices, const int *status, int cmp,
                               const T *cmp_values)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) ivars, (void *) indices, (void *) status, nelems * sizeof(T),
                            nelems * sizeof(size_t), nelems * sizeof(int));
    }
#ifdef __SYCL_DEVICE_ONLY__
    size_t cmp_counter = 0;
    T atomic_val = 0;
    for (size_t i = 0; i < nelems; i++) {
        if (!status || !status[i]) {
            sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_ivar(ivars[i]);
            atomic_val = atomic_ivar.load();
            if (ishmemi_comparison<T>(atomic_val, cmp_values[i], cmp) > 0) {
                indices[cmp_counter++] = i;
            }
        }
    }
    return cmp_counter;
#else
    size_t ret = 0;
    ishmemi_request_t req;
    req.dst = ivars;
    req.nelems = nelems;
    req.indices = indices;
    req.status = status;
    req.cmp = cmp;
    req.cmp_values = cmp_values;
    req.op = TEST_SOME_VECTOR;
    req.type = ishmemi_union_get_base_type<T, TEST_SOME_VECTOR>();
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = comp.completion.ret.szt;
    return ret;
#endif
}

/* Test-Some-Vector (work-group) */
template <typename T, typename Group>
size_t ishmemx_test_some_vector_work_group(T *ivars, size_t nelems, size_t *indices,
                                           const int *status, int cmp, const T *cmp_values,
                                           const Group &grp)
{
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(grp);
    size_t cmp_counter = 0;
    if (grp.leader()) {
        cmp_counter = ishmem_test_some_vector(ivars, nelems, indices, status, cmp, cmp_values);
    }
    cmp_counter = sycl::group_broadcast(grp, cmp_counter);
    return cmp_counter;
#else
    RAISE_ERROR_MSG("ISHMEMX_TEST_SOME_VECTOR_WORK_GROUP routines are not callable from host\n");
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_TEST_SOME_VECTOR(TYPENAME, TYPE)                                                                                                                                                                       \
    size_t ishmem_##TYPENAME##_test_some_vector(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, const TYPE *cmp_values) { return ishmem_test_some_vector(ivars, nelems, indices, status, cmp, cmp_values); } \
    template size_t ishmemx_##TYPENAME##_test_some_vector_work_group<sycl::group<1>>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<1> &grp);                     \
    template size_t ishmemx_##TYPENAME##_test_some_vector_work_group<sycl::group<2>>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<2> &grp);                     \
    template size_t ishmemx_##TYPENAME##_test_some_vector_work_group<sycl::group<3>>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<3> &grp);                     \
    template size_t ishmemx_##TYPENAME##_test_some_vector_work_group<sycl::sub_group>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, const TYPE *cmp_values, const sycl::sub_group &grp);                   \
    template <typename Group> size_t ishmemx_##TYPENAME##_test_some_vector_work_group(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, const TYPE *cmp_values, const Group &grp) { return ishmemx_test_some_vector_work_group(ivars, nelems, indices, status, cmp, cmp_values, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_TEST_SOME_VECTOR(int, int)
ISHMEMI_API_IMPL_TEST_SOME_VECTOR(long, long)
ISHMEMI_API_IMPL_TEST_SOME_VECTOR(longlong, long long)
ISHMEMI_API_IMPL_TEST_SOME_VECTOR(uint, unsigned int)
ISHMEMI_API_IMPL_TEST_SOME_VECTOR(ulong, unsigned long)
ISHMEMI_API_IMPL_TEST_SOME_VECTOR(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_TEST_SOME_VECTOR(int32, int32_t)
ISHMEMI_API_IMPL_TEST_SOME_VECTOR(int64, int64_t)
ISHMEMI_API_IMPL_TEST_SOME_VECTOR(uint32, uint32_t)
ISHMEMI_API_IMPL_TEST_SOME_VECTOR(uint64, uint64_t)
ISHMEMI_API_IMPL_TEST_SOME_VECTOR(size, size_t)
ISHMEMI_API_IMPL_TEST_SOME_VECTOR(ptrdiff, ptrdiff_t)

/* Wait-Until */
template <typename T>
void ishmem_wait_until(T *ivar, int cmp, T cmp_value)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) ivar, sizeof(T));
    }
#ifdef __SYCL_DEVICE_ONLY__
    sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        atomic_ivar(*ivar);
    T atomic_val = 0;
    do {
        atomic_val = atomic_ivar.load();
        if (ishmemi_comparison<T>(atomic_val, cmp_value, cmp) > 0) return;
        while (atomic_ivar.load(sycl::memory_order::seq_cst, sycl::memory_scope::system) ==
               atomic_val) {
        }
    } while (true);
#else
    ishmemi_request_t req;
    req.dst = ivar;
    req.nelems = 1;
    req.cmp = cmp;
    ishmemi_union_set_field_value<T, WAIT>(req.cmp_value, cmp_value);
    req.op = WAIT;
    req.type = ishmemi_union_get_base_type<T, WAIT>();
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

template <typename T>
sycl::event ishmemx_wait_until_on_queue(T *ivar, int cmp, T cmp_value, sycl::queue &q,
                                        const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        cgh.single_task([=]() { ishmem_wait_until(ivar, cmp, cmp_value); });
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

/* Wait-Until (work-group) */
template <typename T, typename Group>
void ishmemx_wait_until_work_group(T *ivar, int cmp, T cmp_value, const Group &grp)
{
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(grp);
    if (grp.leader()) {
        ishmem_wait_until(ivar, cmp, cmp_value);
    }
    sycl::group_barrier(grp);
#else
    RAISE_ERROR_MSG("ISHMEMX_WAIT_UNTIL_WORK_GROUP routines are not callable from host\n");
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_WAIT_UNTIL(TYPENAME, TYPE)                                                                                                   \
    void ishmem_##TYPENAME##_wait_until(TYPE *ivar, int cmp, TYPE cmp_value) { ishmem_wait_until(ivar, cmp, cmp_value); }                             \
    sycl::event ishmemx_##TYPENAME##_wait_until_on_queue(TYPE *ivar, int cmp, TYPE cmp_value, sycl::queue &q, const std::vector<sycl::event> &deps) { \
        return ishmemx_wait_until_on_queue(ivar, cmp, cmp_value, q, deps);                                                                            \
    }                                                                                                                                                 \
    template void ishmemx_##TYPENAME##_wait_until_work_group<sycl::group<1>>(TYPE *ivar, int cmp, TYPE cmp_value, const sycl::group<1> &grp);         \
    template void ishmemx_##TYPENAME##_wait_until_work_group<sycl::group<2>>(TYPE *ivar, int cmp, TYPE cmp_value, const sycl::group<2> &grp);         \
    template void ishmemx_##TYPENAME##_wait_until_work_group<sycl::group<3>>(TYPE *ivar, int cmp, TYPE cmp_value, const sycl::group<3> &grp);         \
    template void ishmemx_##TYPENAME##_wait_until_work_group<sycl::sub_group>(TYPE *ivar, int cmp, TYPE cmp_value, const sycl::sub_group &grp);       \
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

/* Wait-Until-All */
template <typename T>
void ishmem_wait_until_all(T *ivars, size_t nelems, const int *status, int cmp, T cmp_value)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) ivars, (void *) status, nelems * sizeof(T),
                            nelems * sizeof(int), ishmemi_op_t::WAIT_ALL);
    }
#ifdef __SYCL_DEVICE_ONLY__
    for (size_t i = 0; i < nelems; i++) {
        if (!status || !status[i]) {
            ishmem_wait_until(&ivars[i], cmp, cmp_value);
        }
    }
#else
    ishmemi_request_t req;
    req.dst = ivars;
    req.nelems = nelems;
    req.status = status;
    req.cmp = cmp;
    ishmemi_union_set_field_value<T, WAIT_ALL>(req.cmp_value, cmp_value);
    req.op = WAIT_ALL;
    req.type = ishmemi_union_get_base_type<T, WAIT_ALL>();
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

template <typename T>
sycl::event ishmemx_wait_until_all_on_queue(T *ivars, size_t nelems, const int *status, int cmp,
                                            T cmp_value, sycl::queue &q,
                                            const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        cgh.single_task([=]() { ishmem_wait_until_all(ivars, nelems, status, cmp, cmp_value); });
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

/* Wait-Until-All (work-group) */
template <typename T, typename Group>
void ishmemx_wait_until_all_work_group(T *ivars, size_t nelems, const int *status, int cmp,
                                       T cmp_value, const Group &grp)
{
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(grp);
    if constexpr (enable_error_checking) {
        if (grp.leader())
            validate_parameters((void *) ivars, (void *) status, nelems * sizeof(T),
                                nelems * sizeof(int), ishmemi_op_t::WAIT_ALL);
    }
    size_t id = grp.get_local_linear_id();
    size_t num_threads = grp.get_local_linear_range();
    for (size_t idx = id; idx < nelems; idx += num_threads) {
        if (!status || !status[idx]) ishmem_wait_until(&ivars[idx], cmp, cmp_value);
    }
    sycl::group_barrier(grp);
#else
    RAISE_ERROR_MSG("ISHMEMX_WAIT_UNTIL_ALL_WORK_GROUP routines are not callable from host\n");
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_WAIT_UNTIL_ALL(TYPENAME, TYPE)                                                                                                                                      \
    void ishmem_##TYPENAME##_wait_until_all(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value) { ishmem_wait_until_all(ivars, nelems, status, cmp, cmp_value); }        \
    sycl::event ishmemx_##TYPENAME##_wait_until_all_on_queue(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, sycl::queue &q, const std::vector<sycl::event> &deps) { \
        return ishmemx_wait_until_all_on_queue(ivars, nelems, status, cmp, cmp_value, q, deps);                                                                                              \
    }                                                                                                                                                                                        \
    template void ishmemx_##TYPENAME##_wait_until_all_work_group<sycl::group<1>>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::group<1> &grp);         \
    template void ishmemx_##TYPENAME##_wait_until_all_work_group<sycl::group<2>>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::group<2> &grp);         \
    template void ishmemx_##TYPENAME##_wait_until_all_work_group<sycl::group<3>>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::group<3> &grp);         \
    template void ishmemx_##TYPENAME##_wait_until_all_work_group<sycl::sub_group>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::sub_group &grp);       \
    template <typename Group> void ishmemx_##TYPENAME##_wait_until_all_work_group(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const Group &grp) { ishmemx_wait_until_all_work_group(ivars, nelems, status, cmp, cmp_value, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_WAIT_UNTIL_ALL(int, int)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL(long, long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL(longlong, long long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL(uint, unsigned int)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL(ulong, unsigned long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL(int32, int32_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL(int64, int64_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL(uint32, uint32_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL(uint64, uint64_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL(size, size_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL(ptrdiff, ptrdiff_t)

/* Wait-Until-Any */
template <typename T>
size_t ishmem_wait_until_any(T *ivars, size_t nelems, const int *status, int cmp, T cmp_value)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) ivars, (void *) status, nelems * sizeof(T),
                            nelems * sizeof(int), ishmemi_op_t::WAIT_ANY);
    }
#ifdef __SYCL_DEVICE_ONLY__
    if (nelems != 0) {
        T atomic_val = 0;
        ishmemi_info_t *info = global_info;
        size_t cur_idx = info->sync_last_idx_checked;
        do {
            for (size_t i = 0; i < nelems; i++) {
                cur_idx += 1;
                if (cur_idx >= nelems) cur_idx = 0;
                if (!status || !status[cur_idx]) {
                    sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                                     sycl::access::address_space::global_space>
                        atomic_ivar(ivars[cur_idx]);
                    atomic_val =
                        atomic_ivar.load(sycl::memory_order::seq_cst, sycl::memory_scope::system);
                    if (ishmemi_comparison<T>(atomic_val, cmp_value, cmp) > 0) {
                        info->sync_last_idx_checked = cur_idx;
                        return cur_idx;
                    }
                }
            }
        } while (true);
    }
    return SIZE_MAX;
#else
    size_t ret = 0;
    ishmemi_request_t req;
    req.dst = ivars;
    req.nelems = nelems;
    req.status = status;
    req.cmp = cmp;
    ishmemi_union_set_field_value<T, WAIT_ANY>(req.cmp_value, cmp_value);
    req.op = WAIT_ANY;
    req.type = ishmemi_union_get_base_type<T, WAIT_ANY>();
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = comp.completion.ret.szt;
    return ret;
#endif
}

template <typename T>
sycl::event ishmemx_wait_until_any_on_queue(T *ivars, size_t nelems, const int *status, int cmp,
                                            T cmp_value, size_t *ret, sycl::queue &q,
                                            const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        cgh.single_task(
            [=]() { *ret = ishmem_wait_until_any(ivars, nelems, status, cmp, cmp_value); });
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

/* Wait-Until-Any (work-group) */
template <typename T, typename Group>
size_t ishmemx_wait_until_any_work_group(T *ivars, size_t nelems, const int *status, int cmp,
                                         T cmp_value, const Group &grp)
{
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(grp);
    size_t result = SIZE_MAX;
    if (grp.leader()) {
        result = ishmem_wait_until_any(ivars, nelems, status, cmp, cmp_value);
    }
    result = sycl::group_broadcast(grp, result);
    return result;
#else
    RAISE_ERROR_MSG("ISHMEMX_WAIT_UNTIL_ANY_WORK_GROUP routines are not callable from host\n");
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_WAIT_UNTIL_ANY(TYPENAME, TYPE)                                                                                                                                                   \
    size_t ishmem_##TYPENAME##_wait_until_any(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value) { return ishmem_wait_until_any(ivars, nelems, status, cmp, cmp_value); }            \
    sycl::event ishmemx_##TYPENAME##_wait_until_any_on_queue(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, size_t *ret, sycl::queue &q, const std::vector<sycl::event> &deps) { \
        return ishmemx_wait_until_any_on_queue(ivars, nelems, status, cmp, cmp_value, ret, q, deps);                                                                                                      \
    }                                                                                                                                                                                                     \
    template size_t ishmemx_##TYPENAME##_wait_until_any_work_group<sycl::group<1>>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::group<1> &grp);                    \
    template size_t ishmemx_##TYPENAME##_wait_until_any_work_group<sycl::group<2>>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::group<2> &grp);                    \
    template size_t ishmemx_##TYPENAME##_wait_until_any_work_group<sycl::group<3>>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::group<3> &grp);                    \
    template size_t ishmemx_##TYPENAME##_wait_until_any_work_group<sycl::sub_group>(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const sycl::sub_group &grp);                  \
    template <typename Group> size_t ishmemx_##TYPENAME##_wait_until_any_work_group(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, const Group &grp) { return ishmemx_wait_until_any_work_group(ivars, nelems, status, cmp, cmp_value, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_WAIT_UNTIL_ANY(int, int)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY(long, long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY(longlong, long long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY(uint, unsigned int)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY(ulong, unsigned long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY(int32, int32_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY(int64, int64_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY(uint32, uint32_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY(uint64, uint64_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY(size, size_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY(ptrdiff, ptrdiff_t)

/* Wait-Until-Some */
template <typename T>
size_t ishmem_wait_until_some(T *ivars, size_t nelems, size_t *indices, const int *status, int cmp,
                              T cmp_value)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) ivars, (void *) indices, (void *) status, nelems * sizeof(T),
                            nelems * sizeof(size_t), nelems * sizeof(int));
    }
#ifdef __SYCL_DEVICE_ONLY__
    size_t cmp_counter = 0;
    if (nelems != 0) {
        T atomic_val = 0;
        do {
            for (size_t i = 0; i < nelems; i++) {
                if (!status || !status[i]) {
                    sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                                     sycl::access::address_space::global_space>
                        atomic_ivar(ivars[i]);
                    atomic_val =
                        atomic_ivar.load(sycl::memory_order::seq_cst, sycl::memory_scope::system);
                    if (ishmemi_comparison<T>(atomic_val, cmp_value, cmp) > 0) {
                        indices[cmp_counter++] = i;
                    }
                }
            }
            if (cmp_counter) break;
        } while (true);
    }
    return cmp_counter;
#else
    size_t ret = 0;
    ishmemi_request_t req;
    req.dst = ivars;
    req.nelems = nelems;
    req.indices = indices;
    req.status = status;
    req.cmp = cmp;
    ishmemi_union_set_field_value<T, WAIT_SOME>(req.cmp_value, cmp_value);
    req.op = WAIT_SOME;
    req.type = ishmemi_union_get_base_type<T, WAIT_SOME>();
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = comp.completion.ret.szt;
    return ret;
#endif
}

template <typename T>
sycl::event ishmemx_wait_until_some_on_queue(T *ivars, size_t nelems, size_t *indices,
                                             const int *status, int cmp, T cmp_value, size_t *ret,
                                             sycl::queue &q, const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        cgh.single_task([=]() {
            *ret = ishmem_wait_until_some(ivars, nelems, indices, status, cmp, cmp_value);
        });
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

/* Wait-Until-Some (work-group) */
template <typename T, typename Group>
size_t ishmemx_wait_until_some_work_group(T *ivars, size_t nelems, size_t *indices,
                                          const int *status, int cmp, T cmp_value, const Group &grp)
{
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(grp);
    size_t cmp_counter = 0;
    if (grp.leader()) {
        cmp_counter = ishmem_wait_until_some(ivars, nelems, indices, status, cmp, cmp_value);
    }
    cmp_counter = sycl::group_broadcast(grp, cmp_counter);
    return cmp_counter;
#else
    RAISE_ERROR_MSG("ISHMEMX_WAIT_UNTIL_SOME_WORK_GROUP routines are not callable from host\n");
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_WAIT_UNTIL_SOME(TYPENAME, TYPE)                                                                                                                                                                    \
    size_t ishmem_##TYPENAME##_wait_until_some(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value) { return ishmem_wait_until_some(ivars, nelems, indices, status, cmp, cmp_value); }  \
    sycl::event ishmemx_##TYPENAME##_wait_until_some_on_queue(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, size_t *ret, sycl::queue &q, const std::vector<sycl::event> &deps) { \
        return ishmemx_wait_until_some_on_queue(ivars, nelems, indices, status, cmp, cmp_value, ret, q, deps);                                                                                                              \
    }                                                                                                                                                                                                                       \
    template size_t ishmemx_##TYPENAME##_wait_until_some_work_group<sycl::group<1>>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, const sycl::group<1> &grp);                    \
    template size_t ishmemx_##TYPENAME##_wait_until_some_work_group<sycl::group<2>>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, const sycl::group<2> &grp);                    \
    template size_t ishmemx_##TYPENAME##_wait_until_some_work_group<sycl::group<3>>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, const sycl::group<3> &grp);                    \
    template size_t ishmemx_##TYPENAME##_wait_until_some_work_group<sycl::sub_group>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, const sycl::sub_group &grp);                  \
    template <typename Group> size_t ishmemx_##TYPENAME##_wait_until_some_work_group(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value, const Group &grp) { return ishmemx_wait_until_some_work_group(ivars, nelems, indices, status, cmp, cmp_value, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_WAIT_UNTIL_SOME(int, int)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME(long, long)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME(longlong, long long)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME(uint, unsigned int)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME(ulong, unsigned long)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME(int32, int32_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME(int64, int64_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME(uint32, uint32_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME(uint64, uint64_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME(size, size_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME(ptrdiff, ptrdiff_t)

/* Wait-Until-All-Vector */
template <typename T>
void ishmem_wait_until_all_vector(T *ivars, size_t nelems, const int *status, int cmp,
                                  const T *cmp_values)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) ivars, (void *) status, nelems * sizeof(T),
                            nelems * sizeof(int), ishmemi_op_t::WAIT_ALL_VECTOR);
    }
#ifdef __SYCL_DEVICE_ONLY__
    for (size_t i = 0; i < nelems; i++) {
        if (!status || !status[i]) {
            ishmem_wait_until(&ivars[i], cmp, cmp_values[i]);
        }
    }
#else
    ishmemi_request_t req;
    req.dst = ivars;
    req.nelems = nelems;
    req.status = status;
    req.cmp = cmp;
    req.cmp_values = cmp_values;
    req.op = WAIT_ALL_VECTOR;
    req.type = ishmemi_union_get_base_type<T, WAIT_ALL_VECTOR>();
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
#endif
}

template <typename T>
sycl::event ishmemx_wait_until_all_vector_on_queue(T *ivars, size_t nelems, const int *status,
                                                   int cmp, const T *cmp_values, sycl::queue &q,
                                                   const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        cgh.single_task(
            [=]() { ishmem_wait_until_all_vector(ivars, nelems, status, cmp, cmp_values); });
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

/* Wait-Until-All-Vector (work-group) */
template <typename T, typename Group>
void ishmemx_wait_until_all_vector_work_group(T *ivars, size_t nelems, const int *status, int cmp,
                                              const T *cmp_values, const Group &grp)
{
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(grp);
    if constexpr (enable_error_checking) {
        if (grp.leader())
            validate_parameters((void *) ivars, (void *) status, nelems * sizeof(T),
                                nelems * sizeof(int), ishmemi_op_t::WAIT_ALL_VECTOR);
    }
    size_t id = grp.get_local_linear_id();
    size_t num_threads = grp.get_local_linear_range();
    for (size_t idx = id; idx < nelems; idx += num_threads) {
        if (!status || !status[idx]) ishmem_wait_until(&ivars[idx], cmp, cmp_values[idx]);
    }
    sycl::group_barrier(grp);
#else
    RAISE_ERROR_MSG(
        "ISHMEMX_WAIT_UNTIL_ALL_VECTOR_WORK_GROUP routines are not callable from host\n");
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_WAIT_UNTIL_ALL_VECTOR(TYPENAME, TYPE)                                                                                                                                         \
    void ishmem_##TYPENAME##_wait_until_all_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values) { ishmem_wait_until_all_vector(ivars, nelems, status, cmp, cmp_values); } \
    sycl::event ishmemx_##TYPENAME##_wait_until_all_vector_on_queue(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, sycl::queue &q, const std::vector<sycl::event> &deps) {  \
        return ishmemx_wait_until_all_vector_on_queue(ivars, nelems, status, cmp, cmp_values, q, deps);                                                                                                \
    }                                                                                                                                                                                                  \
    template void ishmemx_##TYPENAME##_wait_until_all_vector_work_group<sycl::group<1>>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<1> &grp);          \
    template void ishmemx_##TYPENAME##_wait_until_all_vector_work_group<sycl::group<2>>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<2> &grp);          \
    template void ishmemx_##TYPENAME##_wait_until_all_vector_work_group<sycl::group<3>>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<3> &grp);          \
    template void ishmemx_##TYPENAME##_wait_until_all_vector_work_group<sycl::sub_group>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::sub_group &grp);        \
    template <typename Group> void ishmemx_##TYPENAME##_wait_until_all_vector_work_group(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const Group &grp) { ishmemx_wait_until_all_vector_work_group(ivars, nelems, status, cmp, cmp_values, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_WAIT_UNTIL_ALL_VECTOR(int, int)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL_VECTOR(long, long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL_VECTOR(longlong, long long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL_VECTOR(uint, unsigned int)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL_VECTOR(ulong, unsigned long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL_VECTOR(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL_VECTOR(int32, int32_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL_VECTOR(int64, int64_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL_VECTOR(uint32, uint32_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL_VECTOR(uint64, uint64_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL_VECTOR(size, size_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ALL_VECTOR(ptrdiff, ptrdiff_t)

/* Wait-Until-Any-Vector */
template <typename T>
size_t ishmem_wait_until_any_vector(T *ivars, size_t nelems, const int *status, int cmp,
                                    const T *cmp_values)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) ivars, (void *) status, nelems * sizeof(T),
                            nelems * sizeof(int), ishmemi_op_t::WAIT_ANY_VECTOR);
    }
#ifdef __SYCL_DEVICE_ONLY__
    if (nelems != 0) {
        T atomic_val = 0;
        ishmemi_info_t *info = global_info;
        size_t cur_idx = info->sync_last_idx_checked;
        do {
            for (size_t i = 0; i < nelems; i++) {
                cur_idx += 1;
                if (cur_idx >= nelems) cur_idx = 0;
                if (!status || !status[cur_idx]) {
                    sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                                     sycl::access::address_space::global_space>
                        atomic_ivar(ivars[cur_idx]);
                    atomic_val =
                        atomic_ivar.load(sycl::memory_order::seq_cst, sycl::memory_scope::system);
                    if (ishmemi_comparison<T>(atomic_val, cmp_values[cur_idx], cmp) > 0) {
                        info->sync_last_idx_checked = cur_idx;
                        return cur_idx;
                    }
                }
            }
        } while (true);
    }
    return SIZE_MAX;
#else
    size_t ret = 0;
    ishmemi_request_t req;
    req.dst = ivars;
    req.nelems = nelems;
    req.status = status;
    req.cmp = cmp;
    req.cmp_values = cmp_values;
    req.op = WAIT_ANY_VECTOR;
    req.type = ishmemi_union_get_base_type<T, WAIT_ANY_VECTOR>();
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = comp.completion.ret.szt;
    return ret;
#endif
}

template <typename T>
sycl::event ishmemx_wait_until_any_vector_on_queue(T *ivars, size_t nelems, const int *status,
                                                   int cmp, const T *cmp_values, size_t *ret,
                                                   sycl::queue &q,
                                                   const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        cgh.single_task(
            [=]() { *ret = ishmem_wait_until_any_vector(ivars, nelems, status, cmp, cmp_values); });
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

/* Wait-Until-Any (work-group) */
template <typename T, typename Group>
size_t ishmemx_wait_until_any_vector_work_group(T *ivars, size_t nelems, const int *status, int cmp,
                                                const T *cmp_values, const Group &grp)
{
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(grp);
    size_t result = SIZE_MAX;
    if (grp.leader()) {
        result = ishmem_wait_until_any_vector(ivars, nelems, status, cmp, cmp_values);
    }
    result = sycl::group_broadcast(grp, result);
    return result;
#else
    RAISE_ERROR_MSG(
        "ISHMEMX_WAIT_UNTIL_ANY_VECTOR_WORK_GROUP routines are not callable from host\n");
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_WAIT_UNTIL_ANY_VECTOR(TYPENAME, TYPE)                                                                                                                                                     \
    size_t ishmem_##TYPENAME##_wait_until_any_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values) { return ishmem_wait_until_any_vector(ivars, nelems, status, cmp, cmp_values); }    \
    sycl::event ishmemx_##TYPENAME##_wait_until_any_vector_on_queue(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, size_t *ret, sycl::queue &q, const std::vector<sycl::event> &deps) { \
        return ishmemx_wait_until_any_vector_on_queue(ivars, nelems, status, cmp, cmp_values, ret, q, deps);                                                                                                       \
    }                                                                                                                                                                                                              \
    template size_t ishmemx_##TYPENAME##_wait_until_any_vector_work_group<sycl::group<1>>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<1> &grp);                    \
    template size_t ishmemx_##TYPENAME##_wait_until_any_vector_work_group<sycl::group<2>>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<2> &grp);                    \
    template size_t ishmemx_##TYPENAME##_wait_until_any_vector_work_group<sycl::group<3>>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<3> &grp);                    \
    template size_t ishmemx_##TYPENAME##_wait_until_any_vector_work_group<sycl::sub_group>(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const sycl::sub_group &grp);                  \
    template <typename Group> size_t ishmemx_##TYPENAME##_wait_until_any_vector_work_group(TYPE *ivars, size_t nelems, const int *status, int cmp, const TYPE *cmp_values, const Group &grp) { return ishmemx_wait_until_any_vector_work_group(ivars, nelems, status, cmp, cmp_values, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_WAIT_UNTIL_ANY_VECTOR(int, int)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY_VECTOR(long, long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY_VECTOR(longlong, long long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY_VECTOR(uint, unsigned int)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY_VECTOR(ulong, unsigned long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY_VECTOR(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY_VECTOR(int32, int32_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY_VECTOR(int64, int64_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY_VECTOR(uint32, uint32_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY_VECTOR(uint64, uint64_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY_VECTOR(size, size_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_ANY_VECTOR(ptrdiff, ptrdiff_t)

/* Wait-Until-Some-Vector */
template <typename T>
size_t ishmem_wait_until_some_vector(T *ivars, size_t nelems, size_t *indices, const int *status,
                                     int cmp, const T *cmp_values)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) ivars, (void *) indices, (void *) status, nelems * sizeof(T),
                            nelems * sizeof(size_t), nelems * sizeof(int));
    }
#ifdef __SYCL_DEVICE_ONLY__
    size_t cmp_counter = 0;
    if (nelems != 0) {
        T atomic_val = 0;
        do {
            for (size_t i = 0; i < nelems; i++) {
                if (!status || !status[i]) {
                    sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                                     sycl::access::address_space::global_space>
                        atomic_ivar(ivars[i]);
                    atomic_val =
                        atomic_ivar.load(sycl::memory_order::seq_cst, sycl::memory_scope::system);
                    if (ishmemi_comparison<T>(atomic_val, cmp_values[i], cmp) > 0) {
                        indices[cmp_counter++] = i;
                    }
                }
            }
            if (cmp_counter) break;
        } while (true);
    }
    return cmp_counter;
#else
    size_t ret = 0;
    ishmemi_request_t req;
    req.dst = ivars;
    req.nelems = nelems;
    req.indices = indices;
    req.status = status;
    req.cmp = cmp;
    req.cmp_values = cmp_values;
    req.op = WAIT_SOME_VECTOR;
    req.type = ishmemi_union_get_base_type<T, WAIT_SOME_VECTOR>();
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = comp.completion.ret.szt;
    return ret;
#endif
}

template <typename T>
sycl::event ishmemx_wait_until_some_vector_on_queue(T *ivars, size_t nelems, size_t *indices,
                                                    const int *status, int cmp, const T *cmp_values,
                                                    size_t *ret, sycl::queue &q,
                                                    const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        cgh.single_task([=]() {
            *ret = ishmem_wait_until_some_vector(ivars, nelems, indices, status, cmp, cmp_values);
        });
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

/* Wait-Until-Some-Vector (work-group) */
template <typename T, typename Group>
size_t ishmemx_wait_until_some_vector_work_group(T *ivars, size_t nelems, size_t *indices,
                                                 const int *status, int cmp, const T *cmp_values,
                                                 const Group &grp)
{
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(grp);
    size_t cmp_counter = 0;
    if (grp.leader()) {
        cmp_counter =
            ishmem_wait_until_some_vector(ivars, nelems, indices, status, cmp, cmp_values);
    }
    cmp_counter = sycl::group_broadcast(grp, cmp_counter);
    return cmp_counter;
#else
    RAISE_ERROR_MSG(
        "ISHMEMX_WAIT_UNTIL_SOME_VECTOR_WORK_GROUP routines are not callable from host\n");
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_WAIT_UNTIL_SOME_VECTOR(TYPENAME, TYPE)                                                                                                                                                                             \
    size_t ishmem_##TYPENAME##_wait_until_some_vector(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, const TYPE *cmp_values) { return ishmem_wait_until_some_vector(ivars, nelems, indices, status, cmp, cmp_values); } \
    sycl::event ishmemx_##TYPENAME##_wait_until_some_vector_on_queue(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, const TYPE *cmp_values, size_t *ret, sycl::queue &q, const std::vector<sycl::event> &deps) {        \
        return ishmemx_wait_until_some_vector_on_queue(ivars, nelems, indices, status, cmp, cmp_values, ret, q, deps);                                                                                                                      \
    }                                                                                                                                                                                                                                       \
    template size_t ishmemx_##TYPENAME##_wait_until_some_vector_work_group<sycl::group<1>>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<1> &grp);                           \
    template size_t ishmemx_##TYPENAME##_wait_until_some_vector_work_group<sycl::group<2>>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<2> &grp);                           \
    template size_t ishmemx_##TYPENAME##_wait_until_some_vector_work_group<sycl::group<3>>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, const TYPE *cmp_values, const sycl::group<3> &grp);                           \
    template size_t ishmemx_##TYPENAME##_wait_until_some_vector_work_group<sycl::sub_group>(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, const TYPE *cmp_values, const sycl::sub_group &grp);                         \
    template <typename Group> size_t ishmemx_##TYPENAME##_wait_until_some_vector_work_group(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, const TYPE *cmp_values, const Group &grp) { return ishmemx_wait_until_some_vector_work_group(ivars, nelems, indices, status, cmp, cmp_values, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_WAIT_UNTIL_SOME_VECTOR(int, int)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME_VECTOR(long, long)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME_VECTOR(longlong, long long)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME_VECTOR(uint, unsigned int)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME_VECTOR(ulong, unsigned long)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME_VECTOR(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME_VECTOR(int32, int32_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME_VECTOR(int64, int64_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME_VECTOR(uint32, uint32_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME_VECTOR(uint64, uint64_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME_VECTOR(size, size_t)
ISHMEMI_API_IMPL_WAIT_UNTIL_SOME_VECTOR(ptrdiff, ptrdiff_t)

/* Signal-Wait-Until */
uint64_t ishmem_signal_wait_until(uint64_t *sig_addr, int cmp, uint64_t cmp_value)
{
    if constexpr (enable_error_checking) {
        validate_parameters((void *) sig_addr, sizeof(uint64_t));
    }
#ifdef __SYCL_DEVICE_ONLY__
    sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        atomic_sig_addr(*sig_addr);
    uint64_t atomic_val = UINT64_MAX;
    do {
        atomic_val = atomic_sig_addr.load();
        if (ishmemi_comparison<uint64_t>(atomic_val, cmp_value, cmp) > 0) return atomic_val;
        while (atomic_sig_addr.load() == atomic_val)
            ;
    } while (true);
#else
    uint64_t ret = 0;
    ishmemi_request_t req;
    req.sig_addr = sig_addr;
    req.cmp = cmp;
    req.cmp_value.ui64 = cmp_value;
    req.op = SIGNAL_WAIT_UNTIL;
    req.type = UINT64;
    ishmemi_ringcompletion_t comp;
    ishmemi_runtime->proxy_funcs[req.op][req.type](&req, &comp);
    ret = comp.completion.ret.ui64;
    return ret;
#endif
}

sycl::event ishmemx_signal_wait_until_on_queue(uint64_t *sig_addr, int cmp, uint64_t cmp_value,
                                               uint64_t *ret, sycl::queue &q,
                                               const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        cgh.single_task([=]() {
            uint64_t tmp_ret = ishmem_signal_wait_until(sig_addr, cmp, cmp_value);
            if (ret) *ret = tmp_ret;
        });
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

/* Signal-Wait-Until (work-group) */
/* clang-format off */
template uint64_t ishmemx_signal_wait_until_work_group<sycl::group<1>>(uint64_t *sig_addr, int cmp, uint64_t cmp_value, const sycl::group<1> &grp);
template uint64_t ishmemx_signal_wait_until_work_group<sycl::group<2>>(uint64_t *sig_addr, int cmp, uint64_t cmp_value, const sycl::group<2> &grp);
template uint64_t ishmemx_signal_wait_until_work_group<sycl::group<3>>(uint64_t *sig_addr, int cmp, uint64_t cmp_value, const sycl::group<3> &grp);
template uint64_t ishmemx_signal_wait_until_work_group<sycl::sub_group>(uint64_t *sig_addr, int cmp, uint64_t cmp_value, const sycl::sub_group &grp);
/* clang-format on */
template <typename Group>
uint64_t ishmemx_signal_wait_until_work_group(uint64_t *sig_addr, int cmp, uint64_t cmp_value,
                                              const Group &grp)
{
#ifdef __SYCL_DEVICE_ONLY__
    sycl::group_barrier(grp);
    uint64_t result = UINT64_MAX;
    if (grp.leader()) {
        result = ishmem_signal_wait_until(sig_addr, cmp, cmp_value);
    }
    result = sycl::group_broadcast(grp, result);
    return result;
#else
    RAISE_ERROR_MSG("ishmemx_signal_wait_until_work_group not callable from host\n");
#endif
}
