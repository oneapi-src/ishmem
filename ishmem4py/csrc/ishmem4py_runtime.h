/* Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM4PY_RUNTIME_H
#define ISHMEM4PY_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__)
#define ISHMEM4PY_EXPORT __attribute__((visibility("default")))
#else
#define ISHMEM4PY_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ishmem4py_dtype_t {
    ISHMEM4PY_DTYPE_INT32 = 0,
    ISHMEM4PY_DTYPE_INT64 = 1,
    ISHMEM4PY_DTYPE_UINT32 = 2,
    ISHMEM4PY_DTYPE_UINT64 = 3,
    ISHMEM4PY_DTYPE_FLOAT32 = 4,
    ISHMEM4PY_DTYPE_FLOAT64 = 5,
} ishmem4py_dtype_t;

typedef enum ishmem4py_reduce_op_t {
    ISHMEM4PY_REDUCE_SUM = 0,
    ISHMEM4PY_REDUCE_PROD = 1,
    ISHMEM4PY_REDUCE_AND = 2,
    ISHMEM4PY_REDUCE_OR = 3,
    ISHMEM4PY_REDUCE_XOR = 4,
    ISHMEM4PY_REDUCE_MIN = 5,
    ISHMEM4PY_REDUCE_MAX = 6,
} ishmem4py_reduce_op_t;

typedef struct ishmem4py_team_config_t {
    int num_contexts;
} ishmem4py_team_config_t;

ISHMEM4PY_EXPORT void ishmem4py_init(void);
ISHMEM4PY_EXPORT void ishmem4py_init_with_device(int device_id);
ISHMEM4PY_EXPORT void ishmem4py_finalize(void);
ISHMEM4PY_EXPORT int ishmem4py_my_pe(void);
ISHMEM4PY_EXPORT int ishmem4py_n_pes(void);
ISHMEM4PY_EXPORT void ishmem4py_info_get_version(int *major, int *minor);
ISHMEM4PY_EXPORT void ishmem4py_info_get_name(char *name);
ISHMEM4PY_EXPORT void ishmem4py_vendor_get_version(int *major, int *minor, int *patch);
ISHMEM4PY_EXPORT void ishmem4py_barrier_all(void);
ISHMEM4PY_EXPORT void ishmem4py_sync_all(void);
ISHMEM4PY_EXPORT void ishmem4py_fence(void);
ISHMEM4PY_EXPORT void ishmem4py_quiet(void);
ISHMEM4PY_EXPORT void *ishmem4py_malloc(size_t size);
ISHMEM4PY_EXPORT void *ishmem4py_calloc(size_t count, size_t size);
ISHMEM4PY_EXPORT void ishmem4py_free(void *ptr);
ISHMEM4PY_EXPORT void ishmem4py_putmem(void *dest, const void *src, size_t nbytes, int pe);
ISHMEM4PY_EXPORT void ishmem4py_getmem(void *dest, const void *src, size_t nbytes, int pe);
ISHMEM4PY_EXPORT void ishmem4py_putmem_on_queue(void *dest, const void *src, size_t nbytes,
                                                int pe, void *queue);
ISHMEM4PY_EXPORT void ishmem4py_getmem_on_queue(void *dest, const void *src, size_t nbytes,
                                                int pe, void *queue);
ISHMEM4PY_EXPORT void ishmem4py_quiet_on_queue(void *queue);
ISHMEM4PY_EXPORT void ishmem4py_queue_sync(void *queue);
ISHMEM4PY_EXPORT void *ishmem4py_ptr(const void *dest, int pe);
ISHMEM4PY_EXPORT int ishmem4py_team_my_pe(int team);
ISHMEM4PY_EXPORT int ishmem4py_team_n_pes(int team);
ISHMEM4PY_EXPORT int ishmem4py_team_translate_pe(int src_team, int src_pe, int dest_team);
ISHMEM4PY_EXPORT int ishmem4py_team_sync(int team);
ISHMEM4PY_EXPORT int ishmem4py_team_split_strided(int parent_team, int start, int stride, int size,
                                                  const ishmem4py_team_config_t *config,
                                                  long config_mask, int *new_team);
ISHMEM4PY_EXPORT int ishmem4py_team_split_2d(int parent_team, int xrange,
                                             const ishmem4py_team_config_t *xaxis_config,
                                             long xaxis_mask, int *xaxis_team,
                                             const ishmem4py_team_config_t *yaxis_config,
                                             long yaxis_mask, int *yaxis_team);
ISHMEM4PY_EXPORT void ishmem4py_team_destroy(int team);
ISHMEM4PY_EXPORT int ishmem4py_broadcastmem(int team, void *dest, const void *src, size_t nbytes,
                                            int root);
ISHMEM4PY_EXPORT int ishmem4py_collectmem(int team, void *dest, const void *src, size_t nbytes);
ISHMEM4PY_EXPORT int ishmem4py_fcollectmem(int team, void *dest, const void *src, size_t nbytes);
ISHMEM4PY_EXPORT int ishmem4py_alltoallmem(int team, void *dest, const void *src, size_t nbytes);
ISHMEM4PY_EXPORT int ishmem4py_reduce(int op, int dtype, int team, void *dest, const void *src,
                                      size_t count);
ISHMEM4PY_EXPORT uint64_t ishmem4py_atomic_fetch(int dtype, void *source, int pe);
ISHMEM4PY_EXPORT void ishmem4py_atomic_set(int dtype, void *dest, uint64_t value_bits, int pe);
ISHMEM4PY_EXPORT uint64_t ishmem4py_atomic_swap(int dtype, void *dest, uint64_t value_bits, int pe);
ISHMEM4PY_EXPORT uint64_t ishmem4py_atomic_compare_swap(int dtype, void *dest, uint64_t cond_bits,
                                                        uint64_t value_bits, int pe);
ISHMEM4PY_EXPORT uint64_t ishmem4py_atomic_fetch_inc(int dtype, void *dest, int pe);
ISHMEM4PY_EXPORT void ishmem4py_atomic_inc(int dtype, void *dest, int pe);
ISHMEM4PY_EXPORT uint64_t ishmem4py_atomic_fetch_add(int dtype, void *dest, uint64_t value_bits,
                                                     int pe);
ISHMEM4PY_EXPORT void ishmem4py_atomic_add(int dtype, void *dest, uint64_t value_bits, int pe);
ISHMEM4PY_EXPORT uint64_t ishmem4py_atomic_fetch_and(int dtype, void *dest, uint64_t value_bits,
                                                     int pe);
ISHMEM4PY_EXPORT void ishmem4py_atomic_and(int dtype, void *dest, uint64_t value_bits, int pe);
ISHMEM4PY_EXPORT uint64_t ishmem4py_atomic_fetch_or(int dtype, void *dest, uint64_t value_bits,
                                                    int pe);
ISHMEM4PY_EXPORT void ishmem4py_atomic_or(int dtype, void *dest, uint64_t value_bits, int pe);
ISHMEM4PY_EXPORT uint64_t ishmem4py_atomic_fetch_xor(int dtype, void *dest, uint64_t value_bits,
                                                     int pe);
ISHMEM4PY_EXPORT void ishmem4py_atomic_xor(int dtype, void *dest, uint64_t value_bits, int pe);
ISHMEM4PY_EXPORT uintptr_t ishmem4py_cast_ptr_to_uintptr(const void *ptr);
ISHMEM4PY_EXPORT void *ishmem4py_cast_uintptr_to_ptr(uintptr_t value);

#ifdef __cplusplus
}
#endif

#endif
