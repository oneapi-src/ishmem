/* Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem4py_runtime.h"

#include <ishmem.h>
#include <ishmemx.h>

#include <cstring>

namespace {

template <typename T>
uint64_t ishmem4py_to_bits(T value)
{
    uint64_t bits = 0;
    static_assert(sizeof(T) <= sizeof(bits));
    std::memcpy(&bits, &value, sizeof(T));
    return bits;
}

template <typename T>
T ishmem4py_from_bits(uint64_t bits)
{
    T value{};
    std::memcpy(&value, &bits, sizeof(T));
    return value;
}

const ishmem_team_config_t *ishmem4py_convert_team_config(const ishmem4py_team_config_t *config,
                                                          ishmem_team_config_t *tmp)
{
    if (config == nullptr) return nullptr;
    tmp->num_contexts = config->num_contexts;
    return tmp;
}

#define ISHMEM4PY_ATOMIC_FETCH_CASE(DTYPE_ENUM, CPP_TYPE, SUFFIX)                                   \
    case DTYPE_ENUM:                                                                                 \
        return ishmem4py_to_bits(                                                                    \
            ishmem_##SUFFIX##_atomic_fetch(reinterpret_cast<CPP_TYPE *>(source), pe))

#define ISHMEM4PY_ATOMIC_SET_CASE(DTYPE_ENUM, CPP_TYPE, SUFFIX)                                     \
    case DTYPE_ENUM:                                                                                 \
        ishmem_##SUFFIX##_atomic_set(reinterpret_cast<CPP_TYPE *>(dest),                             \
                                     ishmem4py_from_bits<CPP_TYPE>(value_bits), pe);                \
        return

#define ISHMEM4PY_ATOMIC_SWAP_CASE(DTYPE_ENUM, CPP_TYPE, SUFFIX)                                    \
    case DTYPE_ENUM:                                                                                 \
        return ishmem4py_to_bits(                                                                    \
            ishmem_##SUFFIX##_atomic_swap(reinterpret_cast<CPP_TYPE *>(dest),                        \
                                          ishmem4py_from_bits<CPP_TYPE>(value_bits), pe))

#define ISHMEM4PY_ATOMIC_COMPARE_SWAP_CASE(DTYPE_ENUM, CPP_TYPE, SUFFIX)                            \
    case DTYPE_ENUM:                                                                                 \
        return ishmem4py_to_bits(                                                                    \
            ishmem_##SUFFIX##_atomic_compare_swap(                                                   \
                reinterpret_cast<CPP_TYPE *>(dest), ishmem4py_from_bits<CPP_TYPE>(cond_bits),       \
                ishmem4py_from_bits<CPP_TYPE>(value_bits), pe))

#define ISHMEM4PY_ATOMIC_FETCH_INC_CASE(DTYPE_ENUM, CPP_TYPE, SUFFIX)                               \
    case DTYPE_ENUM:                                                                                 \
        return ishmem4py_to_bits(                                                                    \
            ishmem_##SUFFIX##_atomic_fetch_inc(reinterpret_cast<CPP_TYPE *>(dest), pe))

#define ISHMEM4PY_ATOMIC_INC_CASE(DTYPE_ENUM, CPP_TYPE, SUFFIX)                                     \
    case DTYPE_ENUM:                                                                                 \
        ishmem_##SUFFIX##_atomic_inc(reinterpret_cast<CPP_TYPE *>(dest), pe);                       \
        return

#define ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(DTYPE_ENUM, CPP_TYPE, SUFFIX, OP)                        \
    case DTYPE_ENUM:                                                                                 \
        return ishmem4py_to_bits(                                                                    \
            ishmem_##SUFFIX##_atomic_fetch_##OP(reinterpret_cast<CPP_TYPE *>(dest),                  \
                                                ishmem4py_from_bits<CPP_TYPE>(value_bits), pe))

#define ISHMEM4PY_ATOMIC_BINARY_CASE(DTYPE_ENUM, CPP_TYPE, SUFFIX, OP)                              \
    case DTYPE_ENUM:                                                                                 \
        ishmem_##SUFFIX##_atomic_##OP(reinterpret_cast<CPP_TYPE *>(dest),                            \
                                      ishmem4py_from_bits<CPP_TYPE>(value_bits), pe);               \
        return

#define ISHMEM4PY_REDUCE_CASE(DTYPE_ENUM, CPP_TYPE, SUFFIX, OP)                                     \
    case DTYPE_ENUM:                                                                                 \
        return ishmem_##SUFFIX##_##OP##_reduce(                                                      \
            static_cast<ishmem_team_t>(team), reinterpret_cast<CPP_TYPE *>(dest),                   \
            reinterpret_cast<const CPP_TYPE *>(src), count)

}  // namespace

extern "C" {

void ishmem4py_init(void)
{
    ishmem_init();
}

void ishmem4py_init_with_device(int device_id)
{
    ishmemx_attr_t attr{};
    attr.device_id = device_id;
    ishmemx_init_attr(&attr);
}

void ishmem4py_finalize(void)
{
    ishmem_finalize();
}

int ishmem4py_my_pe(void)
{
    return ishmem_my_pe();
}

int ishmem4py_n_pes(void)
{
    return ishmem_n_pes();
}

void ishmem4py_info_get_version(int *major, int *minor)
{
    ishmem_info_get_version(major, minor);
}

void ishmem4py_info_get_name(char *name)
{
    ishmem_info_get_name(name);
}

void ishmem4py_vendor_get_version(int *major, int *minor, int *patch)
{
    if (major != nullptr) *major = ISHMEM_MAJOR_VERSION;
    if (minor != nullptr) *minor = ISHMEM_MINOR_VERSION;
    if (patch != nullptr) *patch = ISHMEM_PATCH_VERSION;
}

void ishmem4py_barrier_all(void)
{
    ishmem_barrier_all();
}

void ishmem4py_sync_all(void)
{
    ishmem_sync_all();
}

void ishmem4py_fence(void)
{
    ishmem_fence();
}

void ishmem4py_quiet(void)
{
    ishmem_quiet();
}

void *ishmem4py_malloc(size_t size)
{
    return ishmem_malloc(size);
}

void *ishmem4py_calloc(size_t count, size_t size)
{
    return ishmem_calloc(count, size);
}

void ishmem4py_free(void *ptr)
{
    ishmem_free(ptr);
}

void ishmem4py_putmem(void *dest, const void *src, size_t nbytes, int pe)
{
    ishmem_putmem(dest, src, nbytes, pe);
}

void ishmem4py_getmem(void *dest, const void *src, size_t nbytes, int pe)
{
    ishmem_getmem(dest, src, nbytes, pe);
}

void ishmem4py_putmem_on_queue(void *dest, const void *src, size_t nbytes, int pe, void *queue)
{
    auto *q = reinterpret_cast<sycl::queue *>(queue);
    ishmemx_putmem_on_queue(dest, src, nbytes, pe, *q, {});
}

void ishmem4py_getmem_on_queue(void *dest, const void *src, size_t nbytes, int pe, void *queue)
{
    auto *q = reinterpret_cast<sycl::queue *>(queue);
    ishmemx_getmem_on_queue(dest, src, nbytes, pe, *q, {});
}

void ishmem4py_quiet_on_queue(void *queue)
{
    auto *q = reinterpret_cast<sycl::queue *>(queue);
    ishmemx_quiet_on_queue(*q, {});
}

void ishmem4py_queue_sync(void *queue)
{
    auto *q = reinterpret_cast<sycl::queue *>(queue);
    q->wait_and_throw();
}

void *ishmem4py_ptr(const void *dest, int pe)
{
    return ishmem_ptr(dest, pe);
}

int ishmem4py_team_my_pe(int team)
{
    return ishmem_team_my_pe(static_cast<ishmem_team_t>(team));
}

int ishmem4py_team_n_pes(int team)
{
    return ishmem_team_n_pes(static_cast<ishmem_team_t>(team));
}

int ishmem4py_team_translate_pe(int src_team, int src_pe, int dest_team)
{
    return ishmem_team_translate_pe(static_cast<ishmem_team_t>(src_team), src_pe,
                                    static_cast<ishmem_team_t>(dest_team));
}

int ishmem4py_team_sync(int team)
{
    return ishmem_team_sync(static_cast<ishmem_team_t>(team));
}

int ishmem4py_team_split_strided(int parent_team, int start, int stride, int size,
                                 const ishmem4py_team_config_t *config, long config_mask,
                                 int *new_team)
{
    ishmem_team_t result_team = ISHMEM_TEAM_INVALID;
    ishmem_team_config_t tmp_config{};
    const ishmem_team_config_t *converted_config =
        ishmem4py_convert_team_config(config, &tmp_config);
    int result = ishmem_team_split_strided(static_cast<ishmem_team_t>(parent_team), start, stride,
                                           size, converted_config, config_mask, &result_team);
    if (new_team != nullptr) *new_team = result_team;
    return result;
}

int ishmem4py_team_split_2d(int parent_team, int xrange,
                            const ishmem4py_team_config_t *xaxis_config, long xaxis_mask,
                            int *xaxis_team, const ishmem4py_team_config_t *yaxis_config,
                            long yaxis_mask, int *yaxis_team)
{
    ishmem_team_t xaxis_result = ISHMEM_TEAM_INVALID;
    ishmem_team_t yaxis_result = ISHMEM_TEAM_INVALID;
    ishmem_team_config_t xaxis_tmp{};
    ishmem_team_config_t yaxis_tmp{};
    const ishmem_team_config_t *converted_xaxis =
        ishmem4py_convert_team_config(xaxis_config, &xaxis_tmp);
    const ishmem_team_config_t *converted_yaxis =
        ishmem4py_convert_team_config(yaxis_config, &yaxis_tmp);

    int result = ishmem_team_split_2d(static_cast<ishmem_team_t>(parent_team), xrange,
                                      converted_xaxis, xaxis_mask, &xaxis_result, converted_yaxis,
                                      yaxis_mask, &yaxis_result);
    if (xaxis_team != nullptr) *xaxis_team = xaxis_result;
    if (yaxis_team != nullptr) *yaxis_team = yaxis_result;
    return result;
}

void ishmem4py_team_destroy(int team)
{
    ishmem_team_destroy(static_cast<ishmem_team_t>(team));
}

int ishmem4py_broadcastmem(int team, void *dest, const void *src, size_t nbytes, int root)
{
    return ishmem_broadcastmem(static_cast<ishmem_team_t>(team), dest, src, nbytes, root);
}

int ishmem4py_collectmem(int team, void *dest, const void *src, size_t nbytes)
{
    return ishmem_collectmem(static_cast<ishmem_team_t>(team), dest, src, nbytes);
}

int ishmem4py_fcollectmem(int team, void *dest, const void *src, size_t nbytes)
{
    return ishmem_fcollectmem(static_cast<ishmem_team_t>(team), dest, src, nbytes);
}

int ishmem4py_alltoallmem(int team, void *dest, const void *src, size_t nbytes)
{
    return ishmem_alltoallmem(static_cast<ishmem_team_t>(team), dest, src, nbytes);
}

int ishmem4py_reduce(int op, int dtype, int team, void *dest, const void *src, size_t count)
{
    switch (op) {
        case ISHMEM4PY_REDUCE_SUM:
            switch (dtype) {
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, sum);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, sum);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, sum);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, sum);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_FLOAT32, float, float, sum);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_FLOAT64, double, double, sum);
            }
            break;
        case ISHMEM4PY_REDUCE_PROD:
            switch (dtype) {
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, prod);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, prod);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, prod);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, prod);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_FLOAT32, float, float, prod);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_FLOAT64, double, double, prod);
            }
            break;
        case ISHMEM4PY_REDUCE_AND:
            switch (dtype) {
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, and);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, and);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, and);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, and);
            }
            break;
        case ISHMEM4PY_REDUCE_OR:
            switch (dtype) {
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, or);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, or);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, or);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, or);
            }
            break;
        case ISHMEM4PY_REDUCE_XOR:
            switch (dtype) {
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, xor);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, xor);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, xor);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, xor);
            }
            break;
        case ISHMEM4PY_REDUCE_MIN:
            switch (dtype) {
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, min);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, min);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, min);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, min);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_FLOAT32, float, float, min);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_FLOAT64, double, double, min);
            }
            break;
        case ISHMEM4PY_REDUCE_MAX:
            switch (dtype) {
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, max);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, max);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, max);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, max);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_FLOAT32, float, float, max);
                ISHMEM4PY_REDUCE_CASE(ISHMEM4PY_DTYPE_FLOAT64, double, double, max);
            }
            break;
    }
    return -1;
}

uint64_t ishmem4py_atomic_fetch(int dtype, void *source, int pe)
{
    switch (dtype) {
        ISHMEM4PY_ATOMIC_FETCH_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32);
        ISHMEM4PY_ATOMIC_FETCH_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64);
        ISHMEM4PY_ATOMIC_FETCH_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32);
        ISHMEM4PY_ATOMIC_FETCH_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64);
        ISHMEM4PY_ATOMIC_FETCH_CASE(ISHMEM4PY_DTYPE_FLOAT32, float, float);
        ISHMEM4PY_ATOMIC_FETCH_CASE(ISHMEM4PY_DTYPE_FLOAT64, double, double);
    }
    return 0;
}

void ishmem4py_atomic_set(int dtype, void *dest, uint64_t value_bits, int pe)
{
    switch (dtype) {
        ISHMEM4PY_ATOMIC_SET_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32);
        ISHMEM4PY_ATOMIC_SET_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64);
        ISHMEM4PY_ATOMIC_SET_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32);
        ISHMEM4PY_ATOMIC_SET_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64);
        ISHMEM4PY_ATOMIC_SET_CASE(ISHMEM4PY_DTYPE_FLOAT32, float, float);
        ISHMEM4PY_ATOMIC_SET_CASE(ISHMEM4PY_DTYPE_FLOAT64, double, double);
    }
}

uint64_t ishmem4py_atomic_swap(int dtype, void *dest, uint64_t value_bits, int pe)
{
    switch (dtype) {
        ISHMEM4PY_ATOMIC_SWAP_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32);
        ISHMEM4PY_ATOMIC_SWAP_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64);
        ISHMEM4PY_ATOMIC_SWAP_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32);
        ISHMEM4PY_ATOMIC_SWAP_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64);
        ISHMEM4PY_ATOMIC_SWAP_CASE(ISHMEM4PY_DTYPE_FLOAT32, float, float);
        ISHMEM4PY_ATOMIC_SWAP_CASE(ISHMEM4PY_DTYPE_FLOAT64, double, double);
    }
    return 0;
}

uint64_t ishmem4py_atomic_compare_swap(int dtype, void *dest, uint64_t cond_bits,
                                       uint64_t value_bits, int pe)
{
    switch (dtype) {
        ISHMEM4PY_ATOMIC_COMPARE_SWAP_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32);
        ISHMEM4PY_ATOMIC_COMPARE_SWAP_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64);
        ISHMEM4PY_ATOMIC_COMPARE_SWAP_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32);
        ISHMEM4PY_ATOMIC_COMPARE_SWAP_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64);
    }
    return 0;
}

uint64_t ishmem4py_atomic_fetch_inc(int dtype, void *dest, int pe)
{
    switch (dtype) {
        ISHMEM4PY_ATOMIC_FETCH_INC_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32);
        ISHMEM4PY_ATOMIC_FETCH_INC_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64);
        ISHMEM4PY_ATOMIC_FETCH_INC_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32);
        ISHMEM4PY_ATOMIC_FETCH_INC_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64);
    }
    return 0;
}

void ishmem4py_atomic_inc(int dtype, void *dest, int pe)
{
    switch (dtype) {
        ISHMEM4PY_ATOMIC_INC_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32);
        ISHMEM4PY_ATOMIC_INC_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64);
        ISHMEM4PY_ATOMIC_INC_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32);
        ISHMEM4PY_ATOMIC_INC_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64);
    }
}

uint64_t ishmem4py_atomic_fetch_add(int dtype, void *dest, uint64_t value_bits, int pe)
{
    switch (dtype) {
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, add);
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, add);
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, add);
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, add);
    }
    return 0;
}

void ishmem4py_atomic_add(int dtype, void *dest, uint64_t value_bits, int pe)
{
    switch (dtype) {
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, add);
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, add);
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, add);
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, add);
    }
}

uint64_t ishmem4py_atomic_fetch_and(int dtype, void *dest, uint64_t value_bits, int pe)
{
    switch (dtype) {
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, and);
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, and);
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, and);
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, and);
    }
    return 0;
}

void ishmem4py_atomic_and(int dtype, void *dest, uint64_t value_bits, int pe)
{
    switch (dtype) {
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, and);
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, and);
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, and);
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, and);
    }
}

uint64_t ishmem4py_atomic_fetch_or(int dtype, void *dest, uint64_t value_bits, int pe)
{
    switch (dtype) {
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, or);
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, or);
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, or);
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, or);
    }
    return 0;
}

void ishmem4py_atomic_or(int dtype, void *dest, uint64_t value_bits, int pe)
{
    switch (dtype) {
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, or);
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, or);
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, or);
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, or);
    }
}

uint64_t ishmem4py_atomic_fetch_xor(int dtype, void *dest, uint64_t value_bits, int pe)
{
    switch (dtype) {
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, xor);
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, xor);
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, xor);
        ISHMEM4PY_ATOMIC_FETCH_BINARY_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, xor);
    }
    return 0;
}

void ishmem4py_atomic_xor(int dtype, void *dest, uint64_t value_bits, int pe)
{
    switch (dtype) {
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_INT32, int32_t, int32, xor);
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_INT64, int64_t, int64, xor);
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_UINT32, uint32_t, uint32, xor);
        ISHMEM4PY_ATOMIC_BINARY_CASE(ISHMEM4PY_DTYPE_UINT64, uint64_t, uint64, xor);
    }
}

uintptr_t ishmem4py_cast_ptr_to_uintptr(const void *ptr)
{
    return reinterpret_cast<uintptr_t>(ptr);
}

void *ishmem4py_cast_uintptr_to_ptr(uintptr_t value)
{
    return reinterpret_cast<void *>(value);
}

}  // extern "C"
