/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_WRAPPER_SHMEM_H
#define ISHMEM_RUNTIME_WRAPPER_SHMEM_H

#include <shmem.h>
/* shmemx header needed for SHMEMX_TEAM_NODE and heap preinit */
#include <shmemx.h>

extern void (*shmem_WRAPPER_init)(void);
extern void (*shmemx_WRAPPER_heap_preinit)();
extern int (*shmemx_WRAPPER_heap_preinit_thread)(int, int *);
extern void (*shmemx_WRAPPER_heap_create)(void *base, size_t size, int device_type,
                                          int device_index);
extern void (*shmemx_WRAPPER_heap_postinit)();
extern void (*shmem_WRAPPER_finalize)(void);
extern void (*shmem_WRAPPER_global_exit)(int);
extern int (*shmem_WRAPPER_team_translate_pe)(shmem_team_t, int, shmem_team_t);
extern int (*shmem_WRAPPER_team_n_pes)(shmem_team_t);
extern void (*shmem_WRAPPER_team_sync)(shmem_team_t);
extern int (*shmem_WRAPPER_my_pe)(void);
extern int (*shmem_WRAPPER_n_pes)(void);
extern void *(*shmem_WRAPPER_malloc)(size_t);
extern void *(*shmem_WRAPPER_calloc)(size_t, size_t);
extern void (*shmem_WRAPPER_free)(void *);
extern int (*shmem_WRAPPER_runtime_get)(int pe, char *key, void *value, size_t valuelen);

/* RMA */
extern void (*shmem_WRAPPER_uint8_put)(uint8_t *, const uint8_t *, size_t, int);
extern void (*shmem_WRAPPER_uint8_iput)(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t,
                                        int);
extern void (*shmem_WRAPPER_uint8_p)(uint8_t *, uint8_t, int);
extern void (*shmem_WRAPPER_uint16_p)(uint16_t *, uint16_t, int);
extern void (*shmem_WRAPPER_uint32_p)(uint32_t *, uint32_t, int);
extern void (*shmem_WRAPPER_uint64_p)(uint64_t *, uint64_t, int);
extern void (*shmem_WRAPPER_ulonglong_p)(unsigned long long *, unsigned long long, int);
extern void (*shmem_WRAPPER_uint8_put_nbi)(uint8_t *, const uint8_t *, size_t, int);

extern void (*shmem_WRAPPER_uint8_get)(uint8_t *, const uint8_t *, size_t, int);
extern void (*shmem_WRAPPER_uint8_iget)(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t,
                                        int);
extern uint8_t (*shmem_WRAPPER_uint8_g)(const uint8_t *, int);
extern uint16_t (*shmem_WRAPPER_uint16_g)(const uint16_t *, int);
extern uint32_t (*shmem_WRAPPER_uint32_g)(const uint32_t *, int);
extern uint64_t (*shmem_WRAPPER_uint64_g)(const uint64_t *, int);
extern unsigned long long (*shmem_WRAPPER_ulonglong_g)(const unsigned long long *, int);
extern void (*shmem_WRAPPER_uint8_get_nbi)(uint8_t *, const uint8_t *, size_t, int);

/* AMO */
extern uint32_t (*shmem_WRAPPER_uint32_atomic_fetch)(const uint32_t *, int);
extern void (*shmem_WRAPPER_uint32_atomic_set)(uint32_t *, uint32_t, int);
extern uint32_t (*shmem_WRAPPER_uint32_atomic_compare_swap)(uint32_t *, uint32_t, uint32_t, int);
extern uint32_t (*shmem_WRAPPER_uint32_atomic_swap)(uint32_t *, uint32_t, int);
extern uint32_t (*shmem_WRAPPER_uint32_atomic_fetch_inc)(uint32_t *, int);
extern void (*shmem_WRAPPER_uint32_atomic_inc)(uint32_t *, int);
extern uint32_t (*shmem_WRAPPER_uint32_atomic_fetch_add)(uint32_t *, uint32_t, int);
extern void (*shmem_WRAPPER_uint32_atomic_add)(uint32_t *, uint32_t, int);
extern uint32_t (*shmem_WRAPPER_uint32_atomic_fetch_and)(uint32_t *, uint32_t, int);
extern void (*shmem_WRAPPER_uint32_atomic_and)(uint32_t *, uint32_t, int);
extern uint32_t (*shmem_WRAPPER_uint32_atomic_fetch_or)(uint32_t *, uint32_t, int);
extern void (*shmem_WRAPPER_uint32_atomic_or)(uint32_t *, uint32_t, int);
extern uint32_t (*shmem_WRAPPER_uint32_atomic_fetch_xor)(uint32_t *, uint32_t, int);
extern void (*shmem_WRAPPER_uint32_atomic_xor)(uint32_t *, uint32_t, int);

extern uint64_t (*shmem_WRAPPER_uint64_atomic_fetch)(const uint64_t *, int);
extern void (*shmem_WRAPPER_uint64_atomic_set)(uint64_t *, uint64_t, int);
extern uint64_t (*shmem_WRAPPER_uint64_atomic_compare_swap)(uint64_t *, uint64_t, uint64_t, int);
extern uint64_t (*shmem_WRAPPER_uint64_atomic_swap)(uint64_t *, uint64_t, int);
extern uint64_t (*shmem_WRAPPER_uint64_atomic_fetch_inc)(uint64_t *, int);
extern void (*shmem_WRAPPER_uint64_atomic_inc)(uint64_t *, int);
extern uint64_t (*shmem_WRAPPER_uint64_atomic_fetch_add)(uint64_t *, uint64_t, int);
extern void (*shmem_WRAPPER_uint64_atomic_add)(uint64_t *, uint64_t, int);
extern uint64_t (*shmem_WRAPPER_uint64_atomic_fetch_and)(uint64_t *, uint64_t, int);
extern void (*shmem_WRAPPER_uint64_atomic_and)(uint64_t *, uint64_t, int);
extern uint64_t (*shmem_WRAPPER_uint64_atomic_fetch_or)(uint64_t *, uint64_t, int);
extern void (*shmem_WRAPPER_uint64_atomic_or)(uint64_t *, uint64_t, int);
extern uint64_t (*shmem_WRAPPER_uint64_atomic_fetch_xor)(uint64_t *, uint64_t, int);
extern void (*shmem_WRAPPER_uint64_atomic_xor)(uint64_t *, uint64_t, int);

extern unsigned long long (*shmem_WRAPPER_ulonglong_atomic_fetch)(const unsigned long long *, int);
extern void (*shmem_WRAPPER_ulonglong_atomic_set)(unsigned long long *, unsigned long long, int);
extern unsigned long long (*shmem_WRAPPER_ulonglong_atomic_compare_swap)(unsigned long long *,
                                                                         unsigned long long,
                                                                         unsigned long long, int);
extern unsigned long long (*shmem_WRAPPER_ulonglong_atomic_swap)(unsigned long long *,
                                                                 unsigned long long, int);
extern unsigned long long (*shmem_WRAPPER_ulonglong_atomic_fetch_inc)(unsigned long long *, int);
extern void (*shmem_WRAPPER_ulonglong_atomic_inc)(unsigned long long *, int);
extern unsigned long long (*shmem_WRAPPER_ulonglong_atomic_fetch_add)(unsigned long long *,
                                                                      unsigned long long, int);
extern void (*shmem_WRAPPER_ulonglong_atomic_add)(unsigned long long *, unsigned long long, int);
extern unsigned long long (*shmem_WRAPPER_ulonglong_atomic_fetch_and)(unsigned long long *,
                                                                      unsigned long long, int);
extern void (*shmem_WRAPPER_ulonglong_atomic_and)(unsigned long long *, unsigned long long, int);
extern unsigned long long (*shmem_WRAPPER_ulonglong_atomic_fetch_or)(unsigned long long *,
                                                                     unsigned long long, int);
extern void (*shmem_WRAPPER_ulonglong_atomic_or)(unsigned long long *, unsigned long long, int);
extern unsigned long long (*shmem_WRAPPER_ulonglong_atomic_fetch_xor)(unsigned long long *,
                                                                      unsigned long long, int);
extern void (*shmem_WRAPPER_ulonglong_atomic_xor)(unsigned long long *, unsigned long long, int);

extern int32_t (*shmem_WRAPPER_int32_atomic_fetch)(const int32_t *, int);
extern void (*shmem_WRAPPER_int32_atomic_set)(int32_t *, int32_t, int);
extern int32_t (*shmem_WRAPPER_int32_atomic_compare_swap)(int32_t *, int32_t, int32_t, int);
extern int32_t (*shmem_WRAPPER_int32_atomic_swap)(int32_t *, int32_t, int);
extern int32_t (*shmem_WRAPPER_int32_atomic_fetch_inc)(int32_t *, int);
extern void (*shmem_WRAPPER_int32_atomic_inc)(int32_t *, int);
extern int32_t (*shmem_WRAPPER_int32_atomic_fetch_add)(int32_t *, int32_t, int);
extern void (*shmem_WRAPPER_int32_atomic_add)(int32_t *, int32_t, int);
extern int32_t (*shmem_WRAPPER_int32_atomic_fetch_and)(int32_t *, int32_t, int);
extern void (*shmem_WRAPPER_int32_atomic_and)(int32_t *, int32_t, int);
extern int32_t (*shmem_WRAPPER_int32_atomic_fetch_or)(int32_t *, int32_t, int);
extern void (*shmem_WRAPPER_int32_atomic_or)(int32_t *, int32_t, int);
extern int32_t (*shmem_WRAPPER_int32_atomic_fetch_xor)(int32_t *, int32_t, int);
extern void (*shmem_WRAPPER_int32_atomic_xor)(int32_t *, int32_t, int);

extern int64_t (*shmem_WRAPPER_int64_atomic_fetch)(const int64_t *, int);
extern void (*shmem_WRAPPER_int64_atomic_set)(int64_t *, int64_t, int);
extern int64_t (*shmem_WRAPPER_int64_atomic_compare_swap)(int64_t *, int64_t, int64_t, int);
extern int64_t (*shmem_WRAPPER_int64_atomic_swap)(int64_t *, int64_t, int);
extern int64_t (*shmem_WRAPPER_int64_atomic_fetch_inc)(int64_t *, int);
extern void (*shmem_WRAPPER_int64_atomic_inc)(int64_t *, int);
extern int64_t (*shmem_WRAPPER_int64_atomic_fetch_add)(int64_t *, int64_t, int);
extern void (*shmem_WRAPPER_int64_atomic_add)(int64_t *, int64_t, int);
extern int64_t (*shmem_WRAPPER_int64_atomic_fetch_and)(int64_t *, int64_t, int);
extern void (*shmem_WRAPPER_int64_atomic_and)(int64_t *, int64_t, int);
extern int64_t (*shmem_WRAPPER_int64_atomic_fetch_or)(int64_t *, int64_t, int);
extern void (*shmem_WRAPPER_int64_atomic_or)(int64_t *, int64_t, int);
extern int64_t (*shmem_WRAPPER_int64_atomic_fetch_xor)(int64_t *, int64_t, int);
extern void (*shmem_WRAPPER_int64_atomic_xor)(int64_t *, int64_t, int);

extern long long (*shmem_WRAPPER_longlong_atomic_fetch)(const long long *, int);
extern void (*shmem_WRAPPER_longlong_atomic_set)(long long *, long long, int);
extern long long (*shmem_WRAPPER_longlong_atomic_compare_swap)(long long *, long long, long long,
                                                               int);
extern long long (*shmem_WRAPPER_longlong_atomic_swap)(long long *, long long, int);
extern long long (*shmem_WRAPPER_longlong_atomic_fetch_inc)(long long *, int);
extern void (*shmem_WRAPPER_longlong_atomic_inc)(long long *, int);
extern long long (*shmem_WRAPPER_longlong_atomic_fetch_add)(long long *, long long, int);
extern void (*shmem_WRAPPER_longlong_atomic_add)(long long *, long long, int);

extern float (*shmem_WRAPPER_float_atomic_fetch)(const float *, int);
extern void (*shmem_WRAPPER_float_atomic_set)(float *, float, int);
extern float (*shmem_WRAPPER_float_atomic_swap)(float *, float, int);

extern double (*shmem_WRAPPER_double_atomic_fetch)(const double *, int);
extern void (*shmem_WRAPPER_double_atomic_set)(double *, double, int);
extern double (*shmem_WRAPPER_double_atomic_swap)(double *, double, int);

/* Signaling */
extern void (*shmem_WRAPPER_uint8_put_signal)(uint8_t *, const uint8_t *, size_t, uint64_t *,
                                              uint64_t, int, int);
extern void (*shmem_WRAPPER_uint8_put_signal_nbi)(uint8_t *, const uint8_t *, size_t, uint64_t *,
                                                  uint64_t, int, int);
extern uint64_t (*shmem_WRAPPER_signal_fetch)(const uint64_t *);

/* Collectives */
extern void (*shmem_WRAPPER_barrier_all)(void);
extern void (*shmem_WRAPPER_sync_all)(void);
extern int (*shmem_WRAPPER_uint8_alltoall)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
extern int (*shmem_WRAPPER_uint8_broadcast)(shmem_team_t, uint8_t *, const uint8_t *, size_t, int);
extern int (*shmem_WRAPPER_uint8_collect)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
extern int (*shmem_WRAPPER_uint8_fcollect)(shmem_team_t, uint8_t *, const uint8_t *, size_t);

/* Reductions */
extern int (*shmem_WRAPPER_uint8_and_reduce)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
extern int (*shmem_WRAPPER_uint8_or_reduce)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
extern int (*shmem_WRAPPER_uint8_xor_reduce)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
extern int (*shmem_WRAPPER_uint8_max_reduce)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
extern int (*shmem_WRAPPER_uint8_min_reduce)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
extern int (*shmem_WRAPPER_uint8_sum_reduce)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
extern int (*shmem_WRAPPER_uint8_prod_reduce)(shmem_team_t, uint8_t *, const uint8_t *, size_t);

extern int (*shmem_WRAPPER_uint16_and_reduce)(shmem_team_t, uint16_t *, const uint16_t *, size_t);
extern int (*shmem_WRAPPER_uint16_or_reduce)(shmem_team_t, uint16_t *, const uint16_t *, size_t);
extern int (*shmem_WRAPPER_uint16_xor_reduce)(shmem_team_t, uint16_t *, const uint16_t *, size_t);
extern int (*shmem_WRAPPER_uint16_max_reduce)(shmem_team_t, uint16_t *, const uint16_t *, size_t);
extern int (*shmem_WRAPPER_uint16_min_reduce)(shmem_team_t, uint16_t *, const uint16_t *, size_t);
extern int (*shmem_WRAPPER_uint16_sum_reduce)(shmem_team_t, uint16_t *, const uint16_t *, size_t);
extern int (*shmem_WRAPPER_uint16_prod_reduce)(shmem_team_t, uint16_t *, const uint16_t *, size_t);

extern int (*shmem_WRAPPER_uint32_and_reduce)(shmem_team_t, uint32_t *, const uint32_t *, size_t);
extern int (*shmem_WRAPPER_uint32_or_reduce)(shmem_team_t, uint32_t *, const uint32_t *, size_t);
extern int (*shmem_WRAPPER_uint32_xor_reduce)(shmem_team_t, uint32_t *, const uint32_t *, size_t);
extern int (*shmem_WRAPPER_uint32_max_reduce)(shmem_team_t, uint32_t *, const uint32_t *, size_t);
extern int (*shmem_WRAPPER_uint32_min_reduce)(shmem_team_t, uint32_t *, const uint32_t *, size_t);
extern int (*shmem_WRAPPER_uint32_sum_reduce)(shmem_team_t, uint32_t *, const uint32_t *, size_t);
extern int (*shmem_WRAPPER_uint32_prod_reduce)(shmem_team_t, uint32_t *, const uint32_t *, size_t);

extern int (*shmem_WRAPPER_uint64_and_reduce)(shmem_team_t, uint64_t *, const uint64_t *, size_t);
extern int (*shmem_WRAPPER_uint64_or_reduce)(shmem_team_t, uint64_t *, const uint64_t *, size_t);
extern int (*shmem_WRAPPER_uint64_xor_reduce)(shmem_team_t, uint64_t *, const uint64_t *, size_t);
extern int (*shmem_WRAPPER_uint64_max_reduce)(shmem_team_t, uint64_t *, const uint64_t *, size_t);
extern int (*shmem_WRAPPER_uint64_min_reduce)(shmem_team_t, uint64_t *, const uint64_t *, size_t);
extern int (*shmem_WRAPPER_uint64_sum_reduce)(shmem_team_t, uint64_t *, const uint64_t *, size_t);
extern int (*shmem_WRAPPER_uint64_prod_reduce)(shmem_team_t, uint64_t *, const uint64_t *, size_t);

extern int (*shmem_WRAPPER_ulonglong_and_reduce)(shmem_team_t, unsigned long long *,
                                                 const unsigned long long *, size_t);
extern int (*shmem_WRAPPER_ulonglong_or_reduce)(shmem_team_t, unsigned long long *,
                                                const unsigned long long *, size_t);
extern int (*shmem_WRAPPER_ulonglong_xor_reduce)(shmem_team_t, unsigned long long *,
                                                 const unsigned long long *, size_t);
extern int (*shmem_WRAPPER_ulonglong_max_reduce)(shmem_team_t, unsigned long long *,
                                                 const unsigned long long *, size_t);
extern int (*shmem_WRAPPER_ulonglong_min_reduce)(shmem_team_t, unsigned long long *,
                                                 const unsigned long long *, size_t);
extern int (*shmem_WRAPPER_ulonglong_sum_reduce)(shmem_team_t, unsigned long long *,
                                                 const unsigned long long *, size_t);
extern int (*shmem_WRAPPER_ulonglong_prod_reduce)(shmem_team_t, unsigned long long *,
                                                  const unsigned long long *, size_t);

extern int (*shmem_WRAPPER_int8_and_reduce)(shmem_team_t, int8_t *, const int8_t *, size_t);
extern int (*shmem_WRAPPER_int8_or_reduce)(shmem_team_t, int8_t *, const int8_t *, size_t);
extern int (*shmem_WRAPPER_int8_xor_reduce)(shmem_team_t, int8_t *, const int8_t *, size_t);
extern int (*shmem_WRAPPER_int8_max_reduce)(shmem_team_t, int8_t *, const int8_t *, size_t);
extern int (*shmem_WRAPPER_int8_min_reduce)(shmem_team_t, int8_t *, const int8_t *, size_t);
extern int (*shmem_WRAPPER_int8_sum_reduce)(shmem_team_t, int8_t *, const int8_t *, size_t);
extern int (*shmem_WRAPPER_int8_prod_reduce)(shmem_team_t, int8_t *, const int8_t *, size_t);

extern int (*shmem_WRAPPER_int16_and_reduce)(shmem_team_t, int16_t *, const int16_t *, size_t);
extern int (*shmem_WRAPPER_int16_or_reduce)(shmem_team_t, int16_t *, const int16_t *, size_t);
extern int (*shmem_WRAPPER_int16_xor_reduce)(shmem_team_t, int16_t *, const int16_t *, size_t);
extern int (*shmem_WRAPPER_int16_max_reduce)(shmem_team_t, int16_t *, const int16_t *, size_t);
extern int (*shmem_WRAPPER_int16_min_reduce)(shmem_team_t, int16_t *, const int16_t *, size_t);
extern int (*shmem_WRAPPER_int16_sum_reduce)(shmem_team_t, int16_t *, const int16_t *, size_t);
extern int (*shmem_WRAPPER_int16_prod_reduce)(shmem_team_t, int16_t *, const int16_t *, size_t);

extern int (*shmem_WRAPPER_int32_and_reduce)(shmem_team_t, int32_t *, const int32_t *, size_t);
extern int (*shmem_WRAPPER_int32_or_reduce)(shmem_team_t, int32_t *, const int32_t *, size_t);
extern int (*shmem_WRAPPER_int32_xor_reduce)(shmem_team_t, int32_t *, const int32_t *, size_t);
extern int (*shmem_WRAPPER_int32_max_reduce)(shmem_team_t, int32_t *, const int32_t *, size_t);
extern int (*shmem_WRAPPER_int32_min_reduce)(shmem_team_t, int32_t *, const int32_t *, size_t);
extern int (*shmem_WRAPPER_int32_sum_reduce)(shmem_team_t, int32_t *, const int32_t *, size_t);
extern int (*shmem_WRAPPER_int32_prod_reduce)(shmem_team_t, int32_t *, const int32_t *, size_t);

extern int (*shmem_WRAPPER_int64_and_reduce)(shmem_team_t, int64_t *, const int64_t *, size_t);
extern int (*shmem_WRAPPER_int64_or_reduce)(shmem_team_t, int64_t *, const int64_t *, size_t);
extern int (*shmem_WRAPPER_int64_xor_reduce)(shmem_team_t, int64_t *, const int64_t *, size_t);
extern int (*shmem_WRAPPER_int64_max_reduce)(shmem_team_t, int64_t *, const int64_t *, size_t);
extern int (*shmem_WRAPPER_int64_min_reduce)(shmem_team_t, int64_t *, const int64_t *, size_t);
extern int (*shmem_WRAPPER_int64_sum_reduce)(shmem_team_t, int64_t *, const int64_t *, size_t);
extern int (*shmem_WRAPPER_int64_prod_reduce)(shmem_team_t, int64_t *, const int64_t *, size_t);

extern int (*shmem_WRAPPER_longlong_and_reduce)(shmem_team_t, long long *, const long long *,
                                                size_t);
extern int (*shmem_WRAPPER_longlong_or_reduce)(shmem_team_t, long long *, const long long *,
                                               size_t);
extern int (*shmem_WRAPPER_longlong_xor_reduce)(shmem_team_t, long long *, const long long *,
                                                size_t);
extern int (*shmem_WRAPPER_longlong_max_reduce)(shmem_team_t, long long *, const long long *,
                                                size_t);
extern int (*shmem_WRAPPER_longlong_min_reduce)(shmem_team_t, long long *, const long long *,
                                                size_t);
extern int (*shmem_WRAPPER_longlong_sum_reduce)(shmem_team_t, long long *, const long long *,
                                                size_t);
extern int (*shmem_WRAPPER_longlong_prod_reduce)(shmem_team_t, long long *, const long long *,
                                                 size_t);

extern int (*shmem_WRAPPER_float_max_reduce)(shmem_team_t, float *, const float *, size_t);
extern int (*shmem_WRAPPER_float_min_reduce)(shmem_team_t, float *, const float *, size_t);
extern int (*shmem_WRAPPER_float_sum_reduce)(shmem_team_t, float *, const float *, size_t);
extern int (*shmem_WRAPPER_float_prod_reduce)(shmem_team_t, float *, const float *, size_t);

extern int (*shmem_WRAPPER_double_max_reduce)(shmem_team_t, double *, const double *, size_t);
extern int (*shmem_WRAPPER_double_min_reduce)(shmem_team_t, double *, const double *, size_t);
extern int (*shmem_WRAPPER_double_sum_reduce)(shmem_team_t, double *, const double *, size_t);
extern int (*shmem_WRAPPER_double_prod_reduce)(shmem_team_t, double *, const double *, size_t);

/* Point-to-Point Synchronization */
extern int (*shmem_WRAPPER_uint32_test)(uint32_t *, int, uint32_t);
extern void (*shmem_WRAPPER_uint32_wait_until)(uint32_t *, int, uint32_t);

extern int (*shmem_WRAPPER_uint64_test)(uint64_t *, int, uint64_t);
extern void (*shmem_WRAPPER_uint64_wait_until)(uint64_t *, int, uint64_t);

extern int (*shmem_WRAPPER_ulonglong_test)(unsigned long long *, int, unsigned long long);
extern void (*shmem_WRAPPER_ulonglong_wait_until)(unsigned long long *, int, unsigned long long);

/* Memory Ordering */
extern void (*shmem_WRAPPER_fence)(void);
extern void (*shmem_WRAPPER_quiet)(void);

int ishmemi_openshmem_wrapper_init();
int ishmemi_openshmem_wrapper_fini();

#endif
