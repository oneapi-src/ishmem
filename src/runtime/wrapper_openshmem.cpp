/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/err.h"
#include "wrapper.h"
#include "wrapper_openshmem.h"
#include "env_utils.h"
#include <shmem.h>
#include <dlfcn.h>

static bool openshmem_wrapper_initialized = false;

/* Runtime */
void (*shmem_WRAPPER_init)(void);
void (*shmemx_WRAPPER_heap_preinit)(void);
int (*shmemx_WRAPPER_heap_preinit_thread)(int, int *);
void (*shmemx_WRAPPER_heap_create)(void *heap, size_t size, int device_type, int device_index);
void (*shmemx_WRAPPER_heap_postinit)(void);
void (*shmem_WRAPPER_finalize)(void);
void (*shmem_WRAPPER_global_exit)(int);
int (*shmem_WRAPPER_team_translate_pe)(shmem_team_t, int, shmem_team_t);
int (*shmem_WRAPPER_team_n_pes)(shmem_team_t);
int (*shmem_WRAPPER_team_my_pe)(shmem_team_t);
int (*shmem_WRAPPER_team_split_strided)(shmem_team_t, int, int, int, const shmem_team_config_t *,
                                        long, shmem_team_t *);
int (*shmem_WRAPPER_team_split_2d)(shmem_team_t, int, const shmem_team_config_t *, long,
                                   shmem_team_t *, const shmem_team_config_t *, long,
                                   shmem_team_t *);
void (*shmem_WRAPPER_team_destroy)(shmem_team_t);
int (*shmem_WRAPPER_team_sync)(shmem_team_t);
int (*shmem_WRAPPER_my_pe)(void);
int (*shmem_WRAPPER_n_pes)(void);
void *(*shmem_WRAPPER_malloc)(size_t);
void *(*shmem_WRAPPER_calloc)(size_t, size_t);
void (*shmem_WRAPPER_free)(void *);
int (*shmem_WRAPPER_runtime_get)(int pe, char *key, void *value, size_t valuelen);

/* RMA */
void (*shmem_WRAPPER_uint8_put)(uint8_t *, const uint8_t *, size_t, int);
void (*shmem_WRAPPER_uint8_iput)(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, int);
void (*shmem_WRAPPER_uint16_iput)(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t, int);
void (*shmem_WRAPPER_uint32_iput)(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t, int);
void (*shmem_WRAPPER_uint64_iput)(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t, int);
void (*shmem_WRAPPER_ulonglong_iput)(unsigned long long *, const unsigned long long *, ptrdiff_t,
                                     ptrdiff_t, size_t, int);
void (*shmemx_WRAPPER_uint8_ibput)(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, size_t,
                                   int);
void (*shmemx_WRAPPER_uint16_ibput)(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t,
                                    size_t, int);
void (*shmemx_WRAPPER_uint32_ibput)(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t,
                                    size_t, int);
void (*shmemx_WRAPPER_uint64_ibput)(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t,
                                    size_t, int);
void (*shmem_WRAPPER_uint8_p)(uint8_t *, uint8_t, int);
void (*shmem_WRAPPER_uint16_p)(uint16_t *, uint16_t, int);
void (*shmem_WRAPPER_uint32_p)(uint32_t *, uint32_t, int);
void (*shmem_WRAPPER_uint64_p)(uint64_t *, uint64_t, int);
void (*shmem_WRAPPER_ulonglong_p)(unsigned long long *, unsigned long long, int);
void (*shmem_WRAPPER_uint8_put_nbi)(uint8_t *, const uint8_t *, size_t, int);

void (*shmem_WRAPPER_uint8_get)(uint8_t *, const uint8_t *, size_t, int);
void (*shmem_WRAPPER_uint8_iget)(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, int);
void (*shmem_WRAPPER_uint16_iget)(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t, int);
void (*shmem_WRAPPER_uint32_iget)(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t, int);
void (*shmem_WRAPPER_uint64_iget)(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t, int);
void (*shmem_WRAPPER_ulonglong_iget)(unsigned long long *, const unsigned long long *, ptrdiff_t,
                                     ptrdiff_t, size_t, int);
void (*shmemx_WRAPPER_uint8_ibget)(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, size_t,
                                   int);
void (*shmemx_WRAPPER_uint16_ibget)(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t,
                                    size_t, int);
void (*shmemx_WRAPPER_uint32_ibget)(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t,
                                    size_t, int);
void (*shmemx_WRAPPER_uint64_ibget)(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t,
                                    size_t, int);
uint8_t (*shmem_WRAPPER_uint8_g)(const uint8_t *, int);
uint16_t (*shmem_WRAPPER_uint16_g)(const uint16_t *, int);
uint32_t (*shmem_WRAPPER_uint32_g)(const uint32_t *, int);
uint64_t (*shmem_WRAPPER_uint64_g)(const uint64_t *, int);
unsigned long long (*shmem_WRAPPER_ulonglong_g)(const unsigned long long *, int);
void (*shmem_WRAPPER_uint8_get_nbi)(uint8_t *, const uint8_t *, size_t, int);

/* AMO */
uint32_t (*shmem_WRAPPER_uint32_atomic_fetch)(const uint32_t *, int);
void (*shmem_WRAPPER_uint32_atomic_set)(uint32_t *, uint32_t, int);
uint32_t (*shmem_WRAPPER_uint32_atomic_compare_swap)(uint32_t *, uint32_t, uint32_t, int);
uint32_t (*shmem_WRAPPER_uint32_atomic_swap)(uint32_t *, uint32_t, int);
uint32_t (*shmem_WRAPPER_uint32_atomic_fetch_inc)(uint32_t *, int);
void (*shmem_WRAPPER_uint32_atomic_inc)(uint32_t *, int);
uint32_t (*shmem_WRAPPER_uint32_atomic_fetch_add)(uint32_t *, uint32_t, int);
void (*shmem_WRAPPER_uint32_atomic_add)(uint32_t *, uint32_t, int);
uint32_t (*shmem_WRAPPER_uint32_atomic_fetch_and)(uint32_t *, uint32_t, int);
void (*shmem_WRAPPER_uint32_atomic_and)(uint32_t *, uint32_t, int);
uint32_t (*shmem_WRAPPER_uint32_atomic_fetch_or)(uint32_t *, uint32_t, int);
void (*shmem_WRAPPER_uint32_atomic_or)(uint32_t *, uint32_t, int);
uint32_t (*shmem_WRAPPER_uint32_atomic_fetch_xor)(uint32_t *, uint32_t, int);
void (*shmem_WRAPPER_uint32_atomic_xor)(uint32_t *, uint32_t, int);

uint64_t (*shmem_WRAPPER_uint64_atomic_fetch)(const uint64_t *, int);
void (*shmem_WRAPPER_uint64_atomic_set)(uint64_t *, uint64_t, int);
uint64_t (*shmem_WRAPPER_uint64_atomic_compare_swap)(uint64_t *, uint64_t, uint64_t, int);
uint64_t (*shmem_WRAPPER_uint64_atomic_swap)(uint64_t *, uint64_t, int);
uint64_t (*shmem_WRAPPER_uint64_atomic_fetch_inc)(uint64_t *, int);
void (*shmem_WRAPPER_uint64_atomic_inc)(uint64_t *, int);
uint64_t (*shmem_WRAPPER_uint64_atomic_fetch_add)(uint64_t *, uint64_t, int);
void (*shmem_WRAPPER_uint64_atomic_add)(uint64_t *, uint64_t, int);
uint64_t (*shmem_WRAPPER_uint64_atomic_fetch_and)(uint64_t *, uint64_t, int);
void (*shmem_WRAPPER_uint64_atomic_and)(uint64_t *, uint64_t, int);
uint64_t (*shmem_WRAPPER_uint64_atomic_fetch_or)(uint64_t *, uint64_t, int);
void (*shmem_WRAPPER_uint64_atomic_or)(uint64_t *, uint64_t, int);
uint64_t (*shmem_WRAPPER_uint64_atomic_fetch_xor)(uint64_t *, uint64_t, int);
void (*shmem_WRAPPER_uint64_atomic_xor)(uint64_t *, uint64_t, int);

unsigned long long (*shmem_WRAPPER_ulonglong_atomic_fetch)(const unsigned long long *, int);
void (*shmem_WRAPPER_ulonglong_atomic_set)(unsigned long long *, unsigned long long, int);
unsigned long long (*shmem_WRAPPER_ulonglong_atomic_compare_swap)(unsigned long long *,
                                                                  unsigned long long,
                                                                  unsigned long long, int);
unsigned long long (*shmem_WRAPPER_ulonglong_atomic_swap)(unsigned long long *, unsigned long long,
                                                          int);
unsigned long long (*shmem_WRAPPER_ulonglong_atomic_fetch_inc)(unsigned long long *, int);
void (*shmem_WRAPPER_ulonglong_atomic_inc)(unsigned long long *, int);
unsigned long long (*shmem_WRAPPER_ulonglong_atomic_fetch_add)(unsigned long long *,
                                                               unsigned long long, int);
void (*shmem_WRAPPER_ulonglong_atomic_add)(unsigned long long *, unsigned long long, int);
unsigned long long (*shmem_WRAPPER_ulonglong_atomic_fetch_and)(unsigned long long *,
                                                               unsigned long long, int);
void (*shmem_WRAPPER_ulonglong_atomic_and)(unsigned long long *, unsigned long long, int);
unsigned long long (*shmem_WRAPPER_ulonglong_atomic_fetch_or)(unsigned long long *,
                                                              unsigned long long, int);
void (*shmem_WRAPPER_ulonglong_atomic_or)(unsigned long long *, unsigned long long, int);
unsigned long long (*shmem_WRAPPER_ulonglong_atomic_fetch_xor)(unsigned long long *,
                                                               unsigned long long, int);
void (*shmem_WRAPPER_ulonglong_atomic_xor)(unsigned long long *, unsigned long long, int);

int32_t (*shmem_WRAPPER_int32_atomic_fetch)(const int32_t *, int);
void (*shmem_WRAPPER_int32_atomic_set)(int32_t *, int32_t, int);
int32_t (*shmem_WRAPPER_int32_atomic_compare_swap)(int32_t *, int32_t, int32_t, int);
int32_t (*shmem_WRAPPER_int32_atomic_swap)(int32_t *, int32_t, int);
int32_t (*shmem_WRAPPER_int32_atomic_fetch_inc)(int32_t *, int);
void (*shmem_WRAPPER_int32_atomic_inc)(int32_t *, int);
int32_t (*shmem_WRAPPER_int32_atomic_fetch_add)(int32_t *, int32_t, int);
void (*shmem_WRAPPER_int32_atomic_add)(int32_t *, int32_t, int);
int32_t (*shmem_WRAPPER_int32_atomic_fetch_and)(int32_t *, int32_t, int);
void (*shmem_WRAPPER_int32_atomic_and)(int32_t *, int32_t, int);
int32_t (*shmem_WRAPPER_int32_atomic_fetch_or)(int32_t *, int32_t, int);
void (*shmem_WRAPPER_int32_atomic_or)(int32_t *, int32_t, int);
int32_t (*shmem_WRAPPER_int32_atomic_fetch_xor)(int32_t *, int32_t, int);
void (*shmem_WRAPPER_int32_atomic_xor)(int32_t *, int32_t, int);

int64_t (*shmem_WRAPPER_int64_atomic_fetch)(const int64_t *, int);
void (*shmem_WRAPPER_int64_atomic_set)(int64_t *, int64_t, int);
int64_t (*shmem_WRAPPER_int64_atomic_compare_swap)(int64_t *, int64_t, int64_t, int);
int64_t (*shmem_WRAPPER_int64_atomic_swap)(int64_t *, int64_t, int);
int64_t (*shmem_WRAPPER_int64_atomic_fetch_inc)(int64_t *, int);
void (*shmem_WRAPPER_int64_atomic_inc)(int64_t *, int);
int64_t (*shmem_WRAPPER_int64_atomic_fetch_add)(int64_t *, int64_t, int);
void (*shmem_WRAPPER_int64_atomic_add)(int64_t *, int64_t, int);
int64_t (*shmem_WRAPPER_int64_atomic_fetch_and)(int64_t *, int64_t, int);
void (*shmem_WRAPPER_int64_atomic_and)(int64_t *, int64_t, int);
int64_t (*shmem_WRAPPER_int64_atomic_fetch_or)(int64_t *, int64_t, int);
void (*shmem_WRAPPER_int64_atomic_or)(int64_t *, int64_t, int);
int64_t (*shmem_WRAPPER_int64_atomic_fetch_xor)(int64_t *, int64_t, int);
void (*shmem_WRAPPER_int64_atomic_xor)(int64_t *, int64_t, int);

long long (*shmem_WRAPPER_longlong_atomic_fetch)(const long long *, int);
void (*shmem_WRAPPER_longlong_atomic_set)(long long *, long long, int);
long long (*shmem_WRAPPER_longlong_atomic_compare_swap)(long long *, long long, long long, int);
long long (*shmem_WRAPPER_longlong_atomic_swap)(long long *, long long, int);
long long (*shmem_WRAPPER_longlong_atomic_fetch_inc)(long long *, int);
void (*shmem_WRAPPER_longlong_atomic_inc)(long long *, int);
long long (*shmem_WRAPPER_longlong_atomic_fetch_add)(long long *, long long, int);
void (*shmem_WRAPPER_longlong_atomic_add)(long long *, long long, int);
long long (*shmem_WRAPPER_longlong_atomic_fetch_and)(long long *, long long, int);
void (*shmem_WRAPPER_longlong_atomic_and)(long long *, long long, int);
long long (*shmem_WRAPPER_longlong_atomic_fetch_or)(long long *, long long, int);
void (*shmem_WRAPPER_longlong_atomic_or)(long long *, long long, int);
long long (*shmem_WRAPPER_longlong_atomic_fetch_xor)(long long *, long long, int);
void (*shmem_WRAPPER_longlong_atomic_xor)(long long *, long long, int);

float (*shmem_WRAPPER_float_atomic_fetch)(const float *, int);
void (*shmem_WRAPPER_float_atomic_set)(float *, float, int);
float (*shmem_WRAPPER_float_atomic_swap)(float *, float, int);

double (*shmem_WRAPPER_double_atomic_fetch)(const double *, int);
void (*shmem_WRAPPER_double_atomic_set)(double *, double, int);
double (*shmem_WRAPPER_double_atomic_swap)(double *, double, int);

/* Non-blocking AMOs */
void (*shmem_WRAPPER_uint32_atomic_fetch_nbi)(uint32_t *, const uint32_t *, int);
void (*shmem_WRAPPER_int32_atomic_fetch_nbi)(int32_t *, const int32_t *, int);
void (*shmem_WRAPPER_uint64_atomic_fetch_nbi)(uint64_t *, const uint64_t *, int);
void (*shmem_WRAPPER_int64_atomic_fetch_nbi)(int64_t *, const int64_t *, int);
void (*shmem_WRAPPER_ulonglong_atomic_fetch_nbi)(unsigned long long *, const unsigned long long *,
                                                 int);
void (*shmem_WRAPPER_longlong_atomic_fetch_nbi)(long long *, const long long *, int);
void (*shmem_WRAPPER_float_atomic_fetch_nbi)(float *, const float *, int);
void (*shmem_WRAPPER_double_atomic_fetch_nbi)(double *, const double *, int);

void (*shmem_WRAPPER_uint32_atomic_compare_swap_nbi)(uint32_t *, uint32_t *, uint32_t, uint32_t,
                                                     int);
void (*shmem_WRAPPER_int32_atomic_compare_swap_nbi)(int32_t *, int32_t *, int32_t, int32_t, int);
void (*shmem_WRAPPER_uint64_atomic_compare_swap_nbi)(uint64_t *, uint64_t *, uint64_t, uint64_t,
                                                     int);
void (*shmem_WRAPPER_int64_atomic_compare_swap_nbi)(int64_t *, int64_t *, int64_t, int64_t, int);
void (*shmem_WRAPPER_ulonglong_atomic_compare_swap_nbi)(unsigned long long *, unsigned long long *,
                                                        unsigned long long, unsigned long long,
                                                        int);
void (*shmem_WRAPPER_longlong_atomic_compare_swap_nbi)(long long *, long long *, long long,
                                                       long long, int);

void (*shmem_WRAPPER_uint32_atomic_swap_nbi)(uint32_t *, uint32_t *, uint32_t, int);
void (*shmem_WRAPPER_int32_atomic_swap_nbi)(int32_t *, int32_t *, int32_t, int);
void (*shmem_WRAPPER_uint64_atomic_swap_nbi)(uint64_t *, uint64_t *, uint64_t, int);
void (*shmem_WRAPPER_int64_atomic_swap_nbi)(int64_t *, int64_t *, int64_t, int);
void (*shmem_WRAPPER_ulonglong_atomic_swap_nbi)(unsigned long long *, unsigned long long *,
                                                unsigned long long, int);
void (*shmem_WRAPPER_longlong_atomic_swap_nbi)(long long *, long long *, long long, int);
void (*shmem_WRAPPER_float_atomic_swap_nbi)(float *, float *, float, int);
void (*shmem_WRAPPER_double_atomic_swap_nbi)(double *, double *, double, int);

void (*shmem_WRAPPER_uint32_atomic_fetch_inc_nbi)(uint32_t *, uint32_t *, int);
void (*shmem_WRAPPER_int32_atomic_fetch_inc_nbi)(int32_t *, int32_t *, int);
void (*shmem_WRAPPER_uint64_atomic_fetch_inc_nbi)(uint64_t *, uint64_t *, int);
void (*shmem_WRAPPER_int64_atomic_fetch_inc_nbi)(int64_t *, int64_t *, int);
void (*shmem_WRAPPER_ulonglong_atomic_fetch_inc_nbi)(unsigned long long *, unsigned long long *,
                                                     int);
void (*shmem_WRAPPER_longlong_atomic_fetch_inc_nbi)(long long *, long long *, int);

void (*shmem_WRAPPER_uint32_atomic_fetch_add_nbi)(uint32_t *, uint32_t *, uint32_t, int);
void (*shmem_WRAPPER_int32_atomic_fetch_add_nbi)(int32_t *, int32_t *, int32_t, int);
void (*shmem_WRAPPER_uint64_atomic_fetch_add_nbi)(uint64_t *, uint64_t *, uint64_t, int);
void (*shmem_WRAPPER_int64_atomic_fetch_add_nbi)(int64_t *, int64_t *, int64_t, int);
void (*shmem_WRAPPER_ulonglong_atomic_fetch_add_nbi)(unsigned long long *, unsigned long long *,
                                                     unsigned long long, int);
void (*shmem_WRAPPER_longlong_atomic_fetch_add_nbi)(long long *, long long *, long long, int);

void (*shmem_WRAPPER_uint32_atomic_fetch_and_nbi)(uint32_t *, uint32_t *, uint32_t, int);
void (*shmem_WRAPPER_int32_atomic_fetch_and_nbi)(int32_t *, int32_t *, int32_t, int);
void (*shmem_WRAPPER_uint64_atomic_fetch_and_nbi)(uint64_t *, uint64_t *, uint64_t, int);
void (*shmem_WRAPPER_int64_atomic_fetch_and_nbi)(int64_t *, int64_t *, int64_t, int);
void (*shmem_WRAPPER_ulonglong_atomic_fetch_and_nbi)(unsigned long long *, unsigned long long *,
                                                     unsigned long long, int);

void (*shmem_WRAPPER_uint32_atomic_fetch_or_nbi)(uint32_t *, uint32_t *, uint32_t, int);
void (*shmem_WRAPPER_int32_atomic_fetch_or_nbi)(int32_t *, int32_t *, int32_t, int);
void (*shmem_WRAPPER_uint64_atomic_fetch_or_nbi)(uint64_t *, uint64_t *, uint64_t, int);
void (*shmem_WRAPPER_int64_atomic_fetch_or_nbi)(int64_t *, int64_t *, int64_t, int);
void (*shmem_WRAPPER_ulonglong_atomic_fetch_or_nbi)(unsigned long long *, unsigned long long *,
                                                    unsigned long long, int);

void (*shmem_WRAPPER_uint32_atomic_fetch_xor_nbi)(uint32_t *, uint32_t *, uint32_t, int);
void (*shmem_WRAPPER_int32_atomic_fetch_xor_nbi)(int32_t *, int32_t *, int32_t, int);
void (*shmem_WRAPPER_uint64_atomic_fetch_xor_nbi)(uint64_t *, uint64_t *, uint64_t, int);
void (*shmem_WRAPPER_int64_atomic_fetch_xor_nbi)(int64_t *, int64_t *, int64_t, int);
void (*shmem_WRAPPER_ulonglong_atomic_fetch_xor_nbi)(unsigned long long *, unsigned long long *,
                                                     unsigned long long, int);

/* Signaling */
void (*shmem_WRAPPER_uint8_put_signal)(uint8_t *, const uint8_t *, size_t, uint64_t *, uint64_t,
                                       int, int);
void (*shmem_WRAPPER_uint8_put_signal_nbi)(uint8_t *, const uint8_t *, size_t, uint64_t *, uint64_t,
                                           int, int);
uint64_t (*shmem_WRAPPER_signal_fetch)(const uint64_t *);

/* Collectives */
void (*shmem_WRAPPER_barrier_all)(void);
void (*shmem_WRAPPER_sync_all)(void);
int (*shmem_WRAPPER_uint8_alltoall)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
int (*shmem_WRAPPER_uint8_broadcast)(shmem_team_t, uint8_t *, const uint8_t *, size_t, int);
int (*shmem_WRAPPER_uint8_collect)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
int (*shmem_WRAPPER_uint8_fcollect)(shmem_team_t, uint8_t *, const uint8_t *, size_t);

/* Reductions */
int (*shmem_WRAPPER_uchar_and_reduce)(shmem_team_t, unsigned char *, const unsigned char *, size_t);
int (*shmem_WRAPPER_int_max_reduce)(shmem_team_t, int *, const int *, size_t);
int (*shmem_WRAPPER_uint8_and_reduce)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
int (*shmem_WRAPPER_uint8_or_reduce)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
int (*shmem_WRAPPER_uint8_xor_reduce)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
int (*shmem_WRAPPER_uint8_max_reduce)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
int (*shmem_WRAPPER_uint8_min_reduce)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
int (*shmem_WRAPPER_uint8_sum_reduce)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
int (*shmem_WRAPPER_uint8_prod_reduce)(shmem_team_t, uint8_t *, const uint8_t *, size_t);

int (*shmem_WRAPPER_uint16_and_reduce)(shmem_team_t, uint16_t *, const uint16_t *, size_t);
int (*shmem_WRAPPER_uint16_or_reduce)(shmem_team_t, uint16_t *, const uint16_t *, size_t);
int (*shmem_WRAPPER_uint16_xor_reduce)(shmem_team_t, uint16_t *, const uint16_t *, size_t);
int (*shmem_WRAPPER_uint16_max_reduce)(shmem_team_t, uint16_t *, const uint16_t *, size_t);
int (*shmem_WRAPPER_uint16_min_reduce)(shmem_team_t, uint16_t *, const uint16_t *, size_t);
int (*shmem_WRAPPER_uint16_sum_reduce)(shmem_team_t, uint16_t *, const uint16_t *, size_t);
int (*shmem_WRAPPER_uint16_prod_reduce)(shmem_team_t, uint16_t *, const uint16_t *, size_t);

int (*shmem_WRAPPER_uint32_and_reduce)(shmem_team_t, uint32_t *, const uint32_t *, size_t);
int (*shmem_WRAPPER_uint32_or_reduce)(shmem_team_t, uint32_t *, const uint32_t *, size_t);
int (*shmem_WRAPPER_uint32_xor_reduce)(shmem_team_t, uint32_t *, const uint32_t *, size_t);
int (*shmem_WRAPPER_uint32_max_reduce)(shmem_team_t, uint32_t *, const uint32_t *, size_t);
int (*shmem_WRAPPER_uint32_min_reduce)(shmem_team_t, uint32_t *, const uint32_t *, size_t);
int (*shmem_WRAPPER_uint32_sum_reduce)(shmem_team_t, uint32_t *, const uint32_t *, size_t);
int (*shmem_WRAPPER_uint32_prod_reduce)(shmem_team_t, uint32_t *, const uint32_t *, size_t);

int (*shmem_WRAPPER_uint64_and_reduce)(shmem_team_t, uint64_t *, const uint64_t *, size_t);
int (*shmem_WRAPPER_uint64_or_reduce)(shmem_team_t, uint64_t *, const uint64_t *, size_t);
int (*shmem_WRAPPER_uint64_xor_reduce)(shmem_team_t, uint64_t *, const uint64_t *, size_t);
int (*shmem_WRAPPER_uint64_max_reduce)(shmem_team_t, uint64_t *, const uint64_t *, size_t);
int (*shmem_WRAPPER_uint64_min_reduce)(shmem_team_t, uint64_t *, const uint64_t *, size_t);
int (*shmem_WRAPPER_uint64_sum_reduce)(shmem_team_t, uint64_t *, const uint64_t *, size_t);
int (*shmem_WRAPPER_uint64_prod_reduce)(shmem_team_t, uint64_t *, const uint64_t *, size_t);

int (*shmem_WRAPPER_ulonglong_and_reduce)(shmem_team_t, unsigned long long *,
                                          const unsigned long long *, size_t);
int (*shmem_WRAPPER_ulonglong_or_reduce)(shmem_team_t, unsigned long long *,
                                         const unsigned long long *, size_t);
int (*shmem_WRAPPER_ulonglong_xor_reduce)(shmem_team_t, unsigned long long *,
                                          const unsigned long long *, size_t);
int (*shmem_WRAPPER_ulonglong_max_reduce)(shmem_team_t, unsigned long long *,
                                          const unsigned long long *, size_t);
int (*shmem_WRAPPER_ulonglong_min_reduce)(shmem_team_t, unsigned long long *,
                                          const unsigned long long *, size_t);
int (*shmem_WRAPPER_ulonglong_sum_reduce)(shmem_team_t, unsigned long long *,
                                          const unsigned long long *, size_t);
int (*shmem_WRAPPER_ulonglong_prod_reduce)(shmem_team_t, unsigned long long *,
                                           const unsigned long long *, size_t);

int (*shmem_WRAPPER_int8_and_reduce)(shmem_team_t, int8_t *, const int8_t *, size_t);
int (*shmem_WRAPPER_int8_or_reduce)(shmem_team_t, int8_t *, const int8_t *, size_t);
int (*shmem_WRAPPER_int8_xor_reduce)(shmem_team_t, int8_t *, const int8_t *, size_t);
int (*shmem_WRAPPER_int8_max_reduce)(shmem_team_t, int8_t *, const int8_t *, size_t);
int (*shmem_WRAPPER_int8_min_reduce)(shmem_team_t, int8_t *, const int8_t *, size_t);
int (*shmem_WRAPPER_int8_sum_reduce)(shmem_team_t, int8_t *, const int8_t *, size_t);
int (*shmem_WRAPPER_int8_prod_reduce)(shmem_team_t, int8_t *, const int8_t *, size_t);

int (*shmem_WRAPPER_int16_and_reduce)(shmem_team_t, int16_t *, const int16_t *, size_t);
int (*shmem_WRAPPER_int16_or_reduce)(shmem_team_t, int16_t *, const int16_t *, size_t);
int (*shmem_WRAPPER_int16_xor_reduce)(shmem_team_t, int16_t *, const int16_t *, size_t);
int (*shmem_WRAPPER_int16_max_reduce)(shmem_team_t, int16_t *, const int16_t *, size_t);
int (*shmem_WRAPPER_int16_min_reduce)(shmem_team_t, int16_t *, const int16_t *, size_t);
int (*shmem_WRAPPER_int16_sum_reduce)(shmem_team_t, int16_t *, const int16_t *, size_t);
int (*shmem_WRAPPER_int16_prod_reduce)(shmem_team_t, int16_t *, const int16_t *, size_t);

int (*shmem_WRAPPER_int32_and_reduce)(shmem_team_t, int32_t *, const int32_t *, size_t);
int (*shmem_WRAPPER_int32_or_reduce)(shmem_team_t, int32_t *, const int32_t *, size_t);
int (*shmem_WRAPPER_int32_xor_reduce)(shmem_team_t, int32_t *, const int32_t *, size_t);
int (*shmem_WRAPPER_int32_max_reduce)(shmem_team_t, int32_t *, const int32_t *, size_t);
int (*shmem_WRAPPER_int32_min_reduce)(shmem_team_t, int32_t *, const int32_t *, size_t);
int (*shmem_WRAPPER_int32_sum_reduce)(shmem_team_t, int32_t *, const int32_t *, size_t);
int (*shmem_WRAPPER_int32_prod_reduce)(shmem_team_t, int32_t *, const int32_t *, size_t);

int (*shmem_WRAPPER_int64_and_reduce)(shmem_team_t, int64_t *, const int64_t *, size_t);
int (*shmem_WRAPPER_int64_or_reduce)(shmem_team_t, int64_t *, const int64_t *, size_t);
int (*shmem_WRAPPER_int64_xor_reduce)(shmem_team_t, int64_t *, const int64_t *, size_t);
int (*shmem_WRAPPER_int64_max_reduce)(shmem_team_t, int64_t *, const int64_t *, size_t);
int (*shmem_WRAPPER_int64_min_reduce)(shmem_team_t, int64_t *, const int64_t *, size_t);
int (*shmem_WRAPPER_int64_sum_reduce)(shmem_team_t, int64_t *, const int64_t *, size_t);
int (*shmem_WRAPPER_int64_prod_reduce)(shmem_team_t, int64_t *, const int64_t *, size_t);

int (*shmem_WRAPPER_longlong_max_reduce)(shmem_team_t, long long *, const long long *, size_t);
int (*shmem_WRAPPER_longlong_min_reduce)(shmem_team_t, long long *, const long long *, size_t);
int (*shmem_WRAPPER_longlong_sum_reduce)(shmem_team_t, long long *, const long long *, size_t);
int (*shmem_WRAPPER_longlong_prod_reduce)(shmem_team_t, long long *, const long long *, size_t);

int (*shmem_WRAPPER_float_max_reduce)(shmem_team_t, float *, const float *, size_t);
int (*shmem_WRAPPER_float_min_reduce)(shmem_team_t, float *, const float *, size_t);
int (*shmem_WRAPPER_float_sum_reduce)(shmem_team_t, float *, const float *, size_t);
int (*shmem_WRAPPER_float_prod_reduce)(shmem_team_t, float *, const float *, size_t);

int (*shmem_WRAPPER_double_max_reduce)(shmem_team_t, double *, const double *, size_t);
int (*shmem_WRAPPER_double_min_reduce)(shmem_team_t, double *, const double *, size_t);
int (*shmem_WRAPPER_double_sum_reduce)(shmem_team_t, double *, const double *, size_t);
int (*shmem_WRAPPER_double_prod_reduce)(shmem_team_t, double *, const double *, size_t);

/* Point-to-Point Synchronization */
int (*shmem_WRAPPER_int32_test)(int32_t *, int, int32_t);
int (*shmem_WRAPPER_int32_test_all)(int32_t *, size_t, const int *, int, int32_t);
size_t (*shmem_WRAPPER_int32_test_any)(int32_t *, size_t, const int *, int, int32_t);
size_t (*shmem_WRAPPER_int32_test_some)(int32_t *, size_t, size_t *, const int *, int, int32_t);
void (*shmem_WRAPPER_int32_wait_until)(int32_t *, int, int32_t);
void (*shmem_WRAPPER_int32_wait_until_all)(int32_t *, size_t, const int *, int, int32_t);
size_t (*shmem_WRAPPER_int32_wait_until_any)(int32_t *, size_t, const int *, int, int32_t);
size_t (*shmem_WRAPPER_int32_wait_until_some)(int32_t *, size_t, size_t *, const int *, int,
                                              int32_t);

int (*shmem_WRAPPER_int64_test)(int64_t *, int, int64_t);
int (*shmem_WRAPPER_int64_test_all)(int64_t *, size_t, const int *, int, int64_t);
size_t (*shmem_WRAPPER_int64_test_any)(int64_t *, size_t, const int *, int, int64_t);
size_t (*shmem_WRAPPER_int64_test_some)(int64_t *, size_t, size_t *, const int *, int, int64_t);
void (*shmem_WRAPPER_int64_wait_until)(int64_t *, int, int64_t);
void (*shmem_WRAPPER_int64_wait_until_all)(int64_t *, size_t, const int *, int, int64_t);
size_t (*shmem_WRAPPER_int64_wait_until_any)(int64_t *, size_t, const int *, int, int64_t);
size_t (*shmem_WRAPPER_int64_wait_until_some)(int64_t *, size_t, size_t *, const int *, int,
                                              int64_t);

int (*shmem_WRAPPER_longlong_test)(long long *, int, long long);
int (*shmem_WRAPPER_longlong_test_all)(long long *, size_t, const int *, int, long long);
size_t (*shmem_WRAPPER_longlong_test_any)(long long *, size_t, const int *, int, long long);
size_t (*shmem_WRAPPER_longlong_test_some)(long long *, size_t, size_t *, const int *, int,
                                           long long);
void (*shmem_WRAPPER_longlong_wait_until)(long long *, int, long long);
void (*shmem_WRAPPER_longlong_wait_until_all)(long long *, size_t, const int *, int, long long);
size_t (*shmem_WRAPPER_longlong_wait_until_any)(long long *, size_t, const int *, int, long long);
size_t (*shmem_WRAPPER_longlong_wait_until_some)(long long *, size_t, size_t *, const int *, int,
                                                 long long);

int (*shmem_WRAPPER_uint32_test)(uint32_t *, int, uint32_t);
int (*shmem_WRAPPER_uint32_test_all)(uint32_t *, size_t, const int *, int, uint32_t);
size_t (*shmem_WRAPPER_uint32_test_any)(uint32_t *, size_t, const int *, int, uint32_t);
size_t (*shmem_WRAPPER_uint32_test_some)(uint32_t *, size_t, size_t *, const int *, int, uint32_t);
void (*shmem_WRAPPER_uint32_wait_until)(uint32_t *, int, uint32_t);
void (*shmem_WRAPPER_uint32_wait_until_all)(uint32_t *, size_t, const int *, int, uint32_t);
size_t (*shmem_WRAPPER_uint32_wait_until_any)(uint32_t *, size_t, const int *, int, uint32_t);
size_t (*shmem_WRAPPER_uint32_wait_until_some)(uint32_t *, size_t, size_t *, const int *, int,
                                               uint32_t);

int (*shmem_WRAPPER_uint64_test)(uint64_t *, int, uint64_t);
int (*shmem_WRAPPER_uint64_test_all)(uint64_t *, size_t, const int *, int, uint64_t);
size_t (*shmem_WRAPPER_uint64_test_any)(uint64_t *, size_t, const int *, int, uint64_t);
size_t (*shmem_WRAPPER_uint64_test_some)(uint64_t *, size_t, size_t *, const int *, int, uint64_t);
void (*shmem_WRAPPER_uint64_wait_until)(uint64_t *, int, uint64_t);
void (*shmem_WRAPPER_uint64_wait_until_all)(uint64_t *, size_t, const int *, int, uint64_t);
size_t (*shmem_WRAPPER_uint64_wait_until_any)(uint64_t *, size_t, const int *, int, uint64_t);
size_t (*shmem_WRAPPER_uint64_wait_until_some)(uint64_t *, size_t, size_t *, const int *, int,
                                               uint64_t);

int (*shmem_WRAPPER_ulonglong_test)(unsigned long long *, int, unsigned long long);
int (*shmem_WRAPPER_ulonglong_test_all)(unsigned long long *, size_t, const int *, int,
                                        unsigned long long);
size_t (*shmem_WRAPPER_ulonglong_test_any)(unsigned long long *, size_t, const int *, int,
                                           unsigned long long);
size_t (*shmem_WRAPPER_ulonglong_test_some)(unsigned long long *, size_t, size_t *, const int *,
                                            int, unsigned long long);
void (*shmem_WRAPPER_ulonglong_wait_until)(unsigned long long *, int, unsigned long long);
void (*shmem_WRAPPER_ulonglong_wait_until_all)(unsigned long long *, size_t, const int *, int,
                                               unsigned long long);
size_t (*shmem_WRAPPER_ulonglong_wait_until_any)(unsigned long long *, size_t, const int *, int,
                                                 unsigned long long);
size_t (*shmem_WRAPPER_ulonglong_wait_until_some)(unsigned long long *, size_t, size_t *,
                                                  const int *, int, unsigned long long);

uint64_t (*shmem_WRAPPER_signal_wait_until)(uint64_t *, int, uint64_t);

/* Memory Ordering */
void (*shmem_WRAPPER_fence)(void);
void (*shmem_WRAPPER_quiet)(void);

/* dl handle */
void *shmem_handle = nullptr;
std::vector<void **> ishmemi_shmem_handle_wrapper_list;

int ishmemi_openshmem_wrapper_fini()
{
    int ret = 0;
    for (auto p : ishmemi_shmem_handle_wrapper_list)
        *p = nullptr;
    if (shmem_handle != nullptr) {
        ret = dlclose(shmem_handle);
        shmem_handle = nullptr;
        ISHMEM_CHECK_GOTO_MSG(ret, fn_exit, "dlclose failed %s\n", dlerror());
    }
    return (0);
fn_exit:
    return (1);
}

int ishmemi_openshmem_wrapper_init()
{
    int ret = 0;

    const char *shmem_libname = ishmemi_params.SHMEM_LIB_NAME.c_str();

    /* don't initialize twice */
    if (openshmem_wrapper_initialized) goto fn_exit;
    openshmem_wrapper_initialized = true;

    shmem_handle = dlopen(shmem_libname, RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);

    if (shmem_handle == nullptr) {
        RAISE_ERROR_MSG("could not find shmem library '%s' in environment, error %s\n",
                        shmem_libname, dlerror());
    }

    /* Runtime */
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, init);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmemx, heap_preinit);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmemx, heap_preinit_thread);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmemx, heap_create);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmemx, heap_postinit);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, finalize);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, global_exit);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, team_translate_pe);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, team_n_pes);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, team_my_pe);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, team_sync);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, team_split_strided);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, team_split_2d);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, team_destroy);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, my_pe);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, n_pes);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, malloc);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, calloc);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, free);

    /* RMA */
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_put);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_iput);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint16_iput);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_iput);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_iput);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_iput);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmemx, uint8_ibput);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmemx, uint16_ibput);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmemx, uint32_ibput);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmemx, uint64_ibput);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_p);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint16_p);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_p);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_p);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_p);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_put_nbi);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_get);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_iget);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint16_iget);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_iget);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_iget);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_iget);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmemx, uint8_ibget);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmemx, uint16_ibget);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmemx, uint32_ibget);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmemx, uint64_ibget);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_g);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint16_g);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_g);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_g);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_g);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_get_nbi);

    /* AMO NBI */
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_fetch_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_fetch_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_fetch_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_fetch_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_fetch_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_atomic_fetch_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, float_atomic_fetch_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, double_atomic_fetch_nbi);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_compare_swap_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_compare_swap_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_compare_swap_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_compare_swap_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_compare_swap_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_atomic_compare_swap_nbi);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_swap_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_swap_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_swap_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_swap_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_swap_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_atomic_swap_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, float_atomic_swap_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, double_atomic_swap_nbi);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_fetch_inc_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_fetch_inc_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_fetch_inc_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_fetch_inc_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_fetch_inc_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_atomic_fetch_inc_nbi);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_fetch_add_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_fetch_add_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_fetch_add_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_fetch_add_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_fetch_add_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_atomic_fetch_add_nbi);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_fetch_and_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_fetch_and_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_fetch_and_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_fetch_and_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_fetch_and_nbi);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_fetch_or_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_fetch_or_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_fetch_or_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_fetch_or_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_fetch_or_nbi);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_fetch_xor_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_fetch_xor_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_fetch_xor_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_fetch_xor_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_fetch_xor_nbi);

    /* AMO */
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_fetch);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_set);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_compare_swap);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_swap);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_fetch_inc);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_inc);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_fetch_add);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_add);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_fetch_and);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_and);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_fetch_or);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_or);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_fetch_xor);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_atomic_xor);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_fetch);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_set);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_compare_swap);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_swap);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_fetch_inc);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_inc);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_fetch_add);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_add);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_fetch_and);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_and);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_fetch_or);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_or);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_fetch_xor);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_atomic_xor);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_fetch);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_set);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_compare_swap);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_swap);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_fetch_inc);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_inc);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_fetch_add);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_add);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_fetch_and);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_and);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_fetch_or);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_or);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_fetch_xor);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_atomic_xor);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_fetch);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_set);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_compare_swap);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_swap);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_fetch_inc);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_inc);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_fetch_add);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_add);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_fetch_and);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_and);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_fetch_or);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_or);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_fetch_xor);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_atomic_xor);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_fetch);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_set);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_compare_swap);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_swap);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_fetch_inc);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_inc);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_fetch_add);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_add);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_fetch_and);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_and);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_fetch_or);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_or);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_fetch_xor);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_atomic_xor);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_atomic_fetch);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_atomic_set);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_atomic_compare_swap);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_atomic_swap);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_atomic_fetch_inc);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_atomic_inc);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_atomic_fetch_add);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_atomic_add);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, float_atomic_fetch);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, float_atomic_set);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, float_atomic_swap);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, double_atomic_fetch);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, double_atomic_set);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, double_atomic_swap);

    /* Signaling */
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_put_signal);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_put_signal_nbi);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, signal_fetch);

    /* Collectives */
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, barrier_all);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, sync_all);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_alltoall);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_broadcast);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_collect);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_fcollect);

    /* Reductions */
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uchar_and_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int_max_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_and_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_or_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_xor_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_max_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_min_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_sum_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint8_prod_reduce);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint16_and_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint16_or_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint16_xor_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint16_max_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint16_min_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint16_sum_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint16_prod_reduce);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_and_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_or_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_xor_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_max_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_min_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_sum_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_prod_reduce);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_and_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_or_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_xor_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_max_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_min_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_sum_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_prod_reduce);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_and_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_or_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_xor_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_max_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_min_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_sum_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_prod_reduce);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int8_and_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int8_or_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int8_xor_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int8_max_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int8_min_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int8_sum_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int8_prod_reduce);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int16_and_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int16_or_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int16_xor_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int16_max_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int16_min_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int16_sum_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int16_prod_reduce);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_and_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_or_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_xor_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_max_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_min_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_sum_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_prod_reduce);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_and_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_or_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_xor_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_max_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_min_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_sum_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_prod_reduce);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_max_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_min_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_sum_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_prod_reduce);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, float_max_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, float_min_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, float_sum_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, float_prod_reduce);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, double_max_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, double_min_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, double_sum_reduce);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, double_prod_reduce);

    /* Point-to-Point Synchronization */
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_test);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_test_all);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_test_any);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_test_some);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_wait_until);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_wait_until_all);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_wait_until_any);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_wait_until_some);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_test);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_test_all);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_test_any);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_test_some);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_wait_until);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_wait_until_all);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_wait_until_any);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_wait_until_some);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_test);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_test_all);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_test_any);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_test_some);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_wait_until);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_wait_until_all);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_wait_until_any);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_wait_until_some);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_test);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_test_all);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_test_any);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_test_some);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_wait_until);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_wait_until_all);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_wait_until_any);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_wait_until_some);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_test);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_test_all);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_test_any);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_test_some);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_wait_until);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_wait_until_all);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_wait_until_any);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_wait_until_some);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_test);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_test_all);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_test_any);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_test_some);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_wait_until);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_wait_until_all);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_wait_until_any);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_wait_until_some);

    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, signal_wait_until);

    /* Memory Ordering */
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, fence);
    ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, quiet);

fn_exit:
    return ret;
}
