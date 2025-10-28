/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_WRAPPER_SHMEM_H
#define ISHMEM_RUNTIME_WRAPPER_SHMEM_H

#include <shmem.h>
/* shmemx header needed for SHMEMX_TEAM_NODE and heap preinit */
#include <shmemx.h>

/* Ignore unused function warnings for the template wrappers below */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

/* The definitions in this namespace are template definitions for the wrapper functions so that
 * the implementation in runtime_openshmem.cpp can be generalized
 */
namespace ishmemi_openshmem_wrappers {
    int init_wrappers(void);
    int fini_wrappers(void);

    extern shmem_team_t SHMEM_TEAM_WORLD;
    extern shmem_team_t SHMEM_TEAM_SHARED;
    extern shmem_team_t SHMEMX_TEAM_NODE;

    /* clang-format off */
    /* Define the requirement for specialized template */
    template<typename T> struct assert_dependency : public std::false_type {};
    template<typename T, ishmemi_op_t OP> struct assert_dependency_op : public std::false_type {};

    /* Define the generic function pointer types */
    /* RMA */
    template <typename T> using rma_type = void (*)(T *, const T *, size_t, int);
    template <typename T> using irma_type = void (*)(T *, const T *, ptrdiff_t, ptrdiff_t, size_t, int);
    template <typename T> using ibrma_type = void (*)(T *, const T *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
    template <typename T> using p_type = void (*)(T *, T, int);
    template <typename T> using g_type = T (*)(const T *, int);

    /* Non-blocking AMO */
    template <typename T> using atomic_fetch_nbi_type = void (*)(T *, const T *, int);
    template <typename T> using atomic_fetch_inc_nbi_type = void (*)(T *, T *, int);
    template <typename T> using atomic_fetch_op_nbi_type = void (*)(T *, T *, T, int);
    template <typename T> using atomic_compare_swap_nbi_type = void (*)(T *, T *, T, T, int);
    template <typename T> using atomic_swap_nbi_type = void (*)(T *, T *, T, int);

    /* AMO */
    template <typename T> using atomic_fetch_type = T (*)(const T *, int);
    template <typename T> using atomic_compare_swap_type = T (*)(T *, T, T, int);
    template <typename T> using atomic_swap_type = T (*)(T *, T, int);
    template <typename T> using atomic_set_type = void (*)(T *, T, int);
    template <typename T> using atomic_fetch_inc_type = T (*)(T *, int);
    template <typename T> using atomic_fetch_op_type = T (*)(T *, T, int);
    template <typename T> using atomic_inc_type = void (*)(T *, int);
    template <typename T> using atomic_op_type = void (*)(T *, T, int);

    /* Reductions */
    template <typename T> using reduce_type = int (*)(shmem_team_t, T *, const T *, size_t);

    /* Scan */
    template <typename T> using scan_type = int (*)(shmem_team_t, T *, const T *, size_t);

    /* Point-to-Point Synchronization */
    template <typename T> using test_type = int (*)(T *, int, T);
    template <typename T> using test_all_type = int (*)(T *, size_t, const int *, int, T);
    template <typename T> using test_any_type = size_t (*)(T *, size_t, const int *, int, T);
    template <typename T> using test_some_type = size_t (*)(T *, size_t, size_t *, const int *, int, T);
    template <typename T> using wait_until_type = void (*)(T *, int, T);
    template <typename T> using wait_until_all_type = void (*)(T *, size_t, const int *, int, T);
    template <typename T> using wait_until_any_type = size_t (*)(T *, size_t, const int *, int, T);
    template <typename T> using wait_until_some_type = size_t (*)(T *, size_t, size_t *, const int *, int, T);
    template <typename T> using test_all_vector_type = int (*)(T *, size_t, const int *, int, T *);
    template <typename T> using test_any_vector_type = size_t (*)(T *, size_t, const int *, int, T *);
    template <typename T> using test_some_vector_type = size_t (*)(T *, size_t, size_t *, const int *, int, T *);
    template <typename T> using wait_until_all_vector_type = void (*)(T *, size_t, const int *, int, T *);
    template <typename T> using wait_until_any_vector_type = size_t (*)(T *, size_t, const int *, int, T *);
    template <typename T> using wait_until_some_vector_type = size_t (*)(T *, size_t, size_t *, const int *, int, T *);
    /* clang-format on */

    extern void (*init)(void);
    extern void (*heap_preinit)(void);
    extern int (*heap_preinit_thread)(int, int *);
    extern void (*heap_create)(void *, size_t, int, int);
    extern void (*heap_postinit)();
    extern void (*finalize)(void);
    extern void (*global_exit)(int);
    extern int (*team_translate_pe)(shmem_team_t, int, shmem_team_t);
    extern int (*team_n_pes)(shmem_team_t);
    extern int (*team_my_pe)(shmem_team_t);
    extern int (*team_sync)(shmem_team_t);
    extern int (*team_split_strided)(shmem_team_t, int, int, int, const shmem_team_config_t *, long,
                                     shmem_team_t *);
    extern int (*team_split_2d)(shmem_team_t, int, const shmem_team_config_t *, long,
                                shmem_team_t *, const shmem_team_config_t *, long, shmem_team_t *);
    extern void (*team_destroy)(shmem_team_t);
    extern int (*my_pe)(void);
    extern int (*n_pes)(void);
    extern void *(*malloc)(size_t);
    extern void *(*calloc)(size_t, size_t);
    extern void (*free)(void *);
    extern int (*runtime_get)(int, char *, void *, size_t);
    extern bool (*addr_accessible)(const void *, int);

    /* RMA */
    extern rma_type<uint8_t> uint8_put;
    extern irma_type<uint8_t> uint8_iput;
    extern irma_type<uint16_t> uint16_iput;
    extern irma_type<uint32_t> uint32_iput;
    extern irma_type<uint64_t> uint64_iput;
    extern irma_type<unsigned long long> ulonglong_iput;

    extern ibrma_type<uint8_t> uint8_ibput;
    extern ibrma_type<uint16_t> uint16_ibput;
    extern ibrma_type<uint32_t> uint32_ibput;
    extern ibrma_type<uint64_t> uint64_ibput;
    extern p_type<uint8_t> uint8_p;
    extern p_type<uint16_t> uint16_p;
    extern p_type<uint32_t> uint32_p;
    extern p_type<uint64_t> uint64_p;
    extern p_type<unsigned long long> ulonglong_p;
    extern p_type<float> float_p;
    extern p_type<double> double_p;
    extern rma_type<uint8_t> uint8_put_nbi;

    extern rma_type<uint8_t> uint8_get;
    extern irma_type<uint8_t> uint8_iget;
    extern irma_type<uint16_t> uint16_iget;
    extern irma_type<uint32_t> uint32_iget;
    extern irma_type<uint64_t> uint64_iget;
    extern irma_type<unsigned long long> ulonglong_iget;
    extern ibrma_type<uint8_t> uint8_ibget;
    extern ibrma_type<uint16_t> uint16_ibget;
    extern ibrma_type<uint32_t> uint32_ibget;
    extern ibrma_type<uint64_t> uint64_ibget;
    extern g_type<uint8_t> uint8_g;
    extern g_type<uint16_t> uint16_g;
    extern g_type<uint32_t> uint32_g;
    extern g_type<uint64_t> uint64_g;
    extern g_type<unsigned long long> ulonglong_g;
    extern g_type<float> float_g;
    extern g_type<double> double_g;
    extern rma_type<uint8_t> uint8_get_nbi;

    /* Non-blocking AMOs */
    extern atomic_fetch_nbi_type<uint32_t> uint32_atomic_fetch_nbi;
    extern atomic_fetch_nbi_type<int32_t> int32_atomic_fetch_nbi;
    extern atomic_fetch_nbi_type<uint64_t> uint64_atomic_fetch_nbi;
    extern atomic_fetch_nbi_type<int64_t> int64_atomic_fetch_nbi;
    extern atomic_fetch_nbi_type<unsigned long long> ulonglong_atomic_fetch_nbi;
    extern atomic_fetch_nbi_type<long long> longlong_atomic_fetch_nbi;
    extern atomic_fetch_nbi_type<float> float_atomic_fetch_nbi;
    extern atomic_fetch_nbi_type<double> double_atomic_fetch_nbi;

    extern atomic_compare_swap_nbi_type<uint32_t> uint32_atomic_compare_swap_nbi;
    extern atomic_compare_swap_nbi_type<int32_t> int32_atomic_compare_swap_nbi;
    extern atomic_compare_swap_nbi_type<uint64_t> uint64_atomic_compare_swap_nbi;
    extern atomic_compare_swap_nbi_type<int64_t> int64_atomic_compare_swap_nbi;
    extern atomic_compare_swap_nbi_type<unsigned long long> ulonglong_atomic_compare_swap_nbi;
    extern atomic_compare_swap_nbi_type<long long> longlong_atomic_compare_swap_nbi;

    extern atomic_swap_nbi_type<uint32_t> uint32_atomic_swap_nbi;
    extern atomic_swap_nbi_type<int32_t> int32_atomic_swap_nbi;
    extern atomic_swap_nbi_type<uint64_t> uint64_atomic_swap_nbi;
    extern atomic_swap_nbi_type<int64_t> int64_atomic_swap_nbi;
    extern atomic_swap_nbi_type<unsigned long long> ulonglong_atomic_swap_nbi;
    extern atomic_swap_nbi_type<long long> longlong_atomic_swap_nbi;
    extern atomic_swap_nbi_type<float> float_atomic_swap_nbi;
    extern atomic_swap_nbi_type<double> double_atomic_swap_nbi;

    extern atomic_fetch_inc_nbi_type<uint32_t> uint32_atomic_fetch_inc_nbi;
    extern atomic_fetch_inc_nbi_type<int32_t> int32_atomic_fetch_inc_nbi;
    extern atomic_fetch_inc_nbi_type<uint64_t> uint64_atomic_fetch_inc_nbi;
    extern atomic_fetch_inc_nbi_type<int64_t> int64_atomic_fetch_inc_nbi;
    extern atomic_fetch_inc_nbi_type<unsigned long long> ulonglong_atomic_fetch_inc_nbi;
    extern atomic_fetch_inc_nbi_type<long long> longlong_atomic_fetch_inc_nbi;

    extern atomic_fetch_op_nbi_type<uint32_t> uint32_atomic_fetch_add_nbi;
    extern atomic_fetch_op_nbi_type<int32_t> int32_atomic_fetch_add_nbi;
    extern atomic_fetch_op_nbi_type<uint64_t> uint64_atomic_fetch_add_nbi;
    extern atomic_fetch_op_nbi_type<int64_t> int64_atomic_fetch_add_nbi;
    extern atomic_fetch_op_nbi_type<unsigned long long> ulonglong_atomic_fetch_add_nbi;
    extern atomic_fetch_op_nbi_type<long long> longlong_atomic_fetch_add_nbi;

    extern atomic_fetch_op_nbi_type<uint32_t> uint32_atomic_fetch_and_nbi;
    extern atomic_fetch_op_nbi_type<int32_t> int32_atomic_fetch_and_nbi;
    extern atomic_fetch_op_nbi_type<uint64_t> uint64_atomic_fetch_and_nbi;
    extern atomic_fetch_op_nbi_type<int64_t> int64_atomic_fetch_and_nbi;
    extern atomic_fetch_op_nbi_type<unsigned long long> ulonglong_atomic_fetch_and_nbi;

    extern atomic_fetch_op_nbi_type<uint32_t> uint32_atomic_fetch_or_nbi;
    extern atomic_fetch_op_nbi_type<int32_t> int32_atomic_fetch_or_nbi;
    extern atomic_fetch_op_nbi_type<uint64_t> uint64_atomic_fetch_or_nbi;
    extern atomic_fetch_op_nbi_type<int64_t> int64_atomic_fetch_or_nbi;
    extern atomic_fetch_op_nbi_type<unsigned long long> ulonglong_atomic_fetch_or_nbi;

    extern atomic_fetch_op_nbi_type<uint32_t> uint32_atomic_fetch_xor_nbi;
    extern atomic_fetch_op_nbi_type<int32_t> int32_atomic_fetch_xor_nbi;
    extern atomic_fetch_op_nbi_type<uint64_t> uint64_atomic_fetch_xor_nbi;
    extern atomic_fetch_op_nbi_type<int64_t> int64_atomic_fetch_xor_nbi;
    extern atomic_fetch_op_nbi_type<unsigned long long> ulonglong_atomic_fetch_xor_nbi;

    /* AMO */
    extern atomic_fetch_type<uint32_t> uint32_atomic_fetch;
    extern atomic_set_type<uint32_t> uint32_atomic_set;
    extern atomic_compare_swap_type<uint32_t> uint32_atomic_compare_swap;
    extern atomic_swap_type<uint32_t> uint32_atomic_swap;
    extern atomic_fetch_inc_type<uint32_t> uint32_atomic_fetch_inc;
    extern atomic_inc_type<uint32_t> uint32_atomic_inc;
    extern atomic_fetch_op_type<uint32_t> uint32_atomic_fetch_add;
    extern atomic_op_type<uint32_t> uint32_atomic_add;
    extern atomic_fetch_op_type<uint32_t> uint32_atomic_fetch_and;
    extern atomic_op_type<uint32_t> uint32_atomic_and;
    extern atomic_fetch_op_type<uint32_t> uint32_atomic_fetch_or;
    extern atomic_op_type<uint32_t> uint32_atomic_or;
    extern atomic_fetch_op_type<uint32_t> uint32_atomic_fetch_xor;
    extern atomic_op_type<uint32_t> uint32_atomic_xor;

    extern atomic_fetch_type<uint64_t> uint64_atomic_fetch;
    extern atomic_set_type<uint64_t> uint64_atomic_set;
    extern atomic_compare_swap_type<uint64_t> uint64_atomic_compare_swap;
    extern atomic_swap_type<uint64_t> uint64_atomic_swap;
    extern atomic_fetch_inc_type<uint64_t> uint64_atomic_fetch_inc;
    extern atomic_inc_type<uint64_t> uint64_atomic_inc;
    extern atomic_fetch_op_type<uint64_t> uint64_atomic_fetch_add;
    extern atomic_op_type<uint64_t> uint64_atomic_add;
    extern atomic_fetch_op_type<uint64_t> uint64_atomic_fetch_and;
    extern atomic_op_type<uint64_t> uint64_atomic_and;
    extern atomic_fetch_op_type<uint64_t> uint64_atomic_fetch_or;
    extern atomic_op_type<uint64_t> uint64_atomic_or;
    extern atomic_fetch_op_type<uint64_t> uint64_atomic_fetch_xor;
    extern atomic_op_type<uint64_t> uint64_atomic_xor;

    extern atomic_fetch_type<unsigned long long> ulonglong_atomic_fetch;
    extern atomic_set_type<unsigned long long> ulonglong_atomic_set;
    extern atomic_compare_swap_type<unsigned long long> ulonglong_atomic_compare_swap;
    extern atomic_swap_type<unsigned long long> ulonglong_atomic_swap;
    extern atomic_fetch_inc_type<unsigned long long> ulonglong_atomic_fetch_inc;
    extern atomic_inc_type<unsigned long long> ulonglong_atomic_inc;
    extern atomic_fetch_op_type<unsigned long long> ulonglong_atomic_fetch_add;
    extern atomic_op_type<unsigned long long> ulonglong_atomic_add;
    extern atomic_fetch_op_type<unsigned long long> ulonglong_atomic_fetch_and;
    extern atomic_op_type<unsigned long long> ulonglong_atomic_and;
    extern atomic_fetch_op_type<unsigned long long> ulonglong_atomic_fetch_or;
    extern atomic_op_type<unsigned long long> ulonglong_atomic_or;
    extern atomic_fetch_op_type<unsigned long long> ulonglong_atomic_fetch_xor;
    extern atomic_op_type<unsigned long long> ulonglong_atomic_xor;

    extern atomic_fetch_type<int32_t> int32_atomic_fetch;
    extern atomic_set_type<int32_t> int32_atomic_set;
    extern atomic_compare_swap_type<int32_t> int32_atomic_compare_swap;
    extern atomic_swap_type<int32_t> int32_atomic_swap;
    extern atomic_fetch_inc_type<int32_t> int32_atomic_fetch_inc;
    extern atomic_inc_type<int32_t> int32_atomic_inc;
    extern atomic_fetch_op_type<int32_t> int32_atomic_fetch_add;
    extern atomic_op_type<int32_t> int32_atomic_add;
    extern atomic_fetch_op_type<int32_t> int32_atomic_fetch_and;
    extern atomic_op_type<int32_t> int32_atomic_and;
    extern atomic_fetch_op_type<int32_t> int32_atomic_fetch_or;
    extern atomic_op_type<int32_t> int32_atomic_or;
    extern atomic_fetch_op_type<int32_t> int32_atomic_fetch_xor;
    extern atomic_op_type<int32_t> int32_atomic_xor;

    extern atomic_fetch_type<int64_t> int64_atomic_fetch;
    extern atomic_set_type<int64_t> int64_atomic_set;
    extern atomic_compare_swap_type<int64_t> int64_atomic_compare_swap;
    extern atomic_swap_type<int64_t> int64_atomic_swap;
    extern atomic_fetch_inc_type<int64_t> int64_atomic_fetch_inc;
    extern atomic_inc_type<int64_t> int64_atomic_inc;
    extern atomic_fetch_op_type<int64_t> int64_atomic_fetch_add;
    extern atomic_op_type<int64_t> int64_atomic_add;
    extern atomic_fetch_op_type<int64_t> int64_atomic_fetch_and;
    extern atomic_op_type<int64_t> int64_atomic_and;
    extern atomic_fetch_op_type<int64_t> int64_atomic_fetch_or;
    extern atomic_op_type<int64_t> int64_atomic_or;
    extern atomic_fetch_op_type<int64_t> int64_atomic_fetch_xor;
    extern atomic_op_type<int64_t> int64_atomic_xor;

    extern atomic_fetch_type<long long> longlong_atomic_fetch;
    extern atomic_set_type<long long> longlong_atomic_set;
    extern atomic_compare_swap_type<long long> longlong_atomic_compare_swap;
    extern atomic_swap_type<long long> longlong_atomic_swap;
    extern atomic_fetch_inc_type<long long> longlong_atomic_fetch_inc;
    extern atomic_inc_type<long long> longlong_atomic_inc;
    extern atomic_fetch_op_type<long long> longlong_atomic_fetch_add;
    extern atomic_op_type<long long> longlong_atomic_add;

    extern atomic_fetch_type<float> float_atomic_fetch;
    extern atomic_set_type<float> float_atomic_set;
    extern atomic_swap_type<float> float_atomic_swap;

    extern atomic_fetch_type<double> double_atomic_fetch;
    extern atomic_set_type<double> double_atomic_set;
    extern atomic_swap_type<double> double_atomic_swap;

    /* Signaling */
    extern void (*uint8_put_signal)(uint8_t *, const uint8_t *, size_t, uint64_t *, uint64_t, int,
                                    int);
    extern void (*uint8_put_signal_nbi)(uint8_t *, const uint8_t *, size_t, uint64_t *, uint64_t,
                                        int, int);
    extern uint64_t (*signal_fetch)(const uint64_t *);

    /* Collectives */
    extern void (*barrier_all)(void);
    extern void (*sync_all)(void);
    extern int (*uint8_alltoall)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
    extern int (*uint8_broadcast)(shmem_team_t, uint8_t *, const uint8_t *, size_t, int);
    extern int (*uint8_collect)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
    extern int (*uint8_fcollect)(shmem_team_t, uint8_t *, const uint8_t *, size_t);

    /* Reductions */
    extern reduce_type<unsigned char> uchar_and_reduce;
    extern reduce_type<int> int_max_reduce;

    extern reduce_type<uint8_t> uint8_and_reduce;
    extern reduce_type<uint8_t> uint8_or_reduce;
    extern reduce_type<uint8_t> uint8_xor_reduce;
    extern reduce_type<uint8_t> uint8_max_reduce;
    extern reduce_type<uint8_t> uint8_min_reduce;
    extern reduce_type<uint8_t> uint8_sum_reduce;
    extern reduce_type<uint8_t> uint8_prod_reduce;

    extern reduce_type<uint16_t> uint16_and_reduce;
    extern reduce_type<uint16_t> uint16_or_reduce;
    extern reduce_type<uint16_t> uint16_xor_reduce;
    extern reduce_type<uint16_t> uint16_max_reduce;
    extern reduce_type<uint16_t> uint16_min_reduce;
    extern reduce_type<uint16_t> uint16_sum_reduce;
    extern reduce_type<uint16_t> uint16_prod_reduce;

    extern reduce_type<uint32_t> uint32_and_reduce;
    extern reduce_type<uint32_t> uint32_or_reduce;
    extern reduce_type<uint32_t> uint32_xor_reduce;
    extern reduce_type<uint32_t> uint32_max_reduce;
    extern reduce_type<uint32_t> uint32_min_reduce;
    extern reduce_type<uint32_t> uint32_sum_reduce;
    extern reduce_type<uint32_t> uint32_prod_reduce;

    extern reduce_type<uint64_t> uint64_and_reduce;
    extern reduce_type<uint64_t> uint64_or_reduce;
    extern reduce_type<uint64_t> uint64_xor_reduce;
    extern reduce_type<uint64_t> uint64_max_reduce;
    extern reduce_type<uint64_t> uint64_min_reduce;
    extern reduce_type<uint64_t> uint64_sum_reduce;
    extern reduce_type<uint64_t> uint64_prod_reduce;

    extern reduce_type<unsigned long long> ulonglong_and_reduce;
    extern reduce_type<unsigned long long> ulonglong_or_reduce;
    extern reduce_type<unsigned long long> ulonglong_xor_reduce;
    extern reduce_type<unsigned long long> ulonglong_max_reduce;
    extern reduce_type<unsigned long long> ulonglong_min_reduce;
    extern reduce_type<unsigned long long> ulonglong_sum_reduce;
    extern reduce_type<unsigned long long> ulonglong_prod_reduce;

    extern reduce_type<int8_t> int8_and_reduce;
    extern reduce_type<int8_t> int8_or_reduce;
    extern reduce_type<int8_t> int8_xor_reduce;
    extern reduce_type<int8_t> int8_max_reduce;
    extern reduce_type<int8_t> int8_min_reduce;
    extern reduce_type<int8_t> int8_sum_reduce;
    extern reduce_type<int8_t> int8_prod_reduce;

    extern reduce_type<int16_t> int16_and_reduce;
    extern reduce_type<int16_t> int16_or_reduce;
    extern reduce_type<int16_t> int16_xor_reduce;
    extern reduce_type<int16_t> int16_max_reduce;
    extern reduce_type<int16_t> int16_min_reduce;
    extern reduce_type<int16_t> int16_sum_reduce;
    extern reduce_type<int16_t> int16_prod_reduce;

    extern reduce_type<int32_t> int32_and_reduce;
    extern reduce_type<int32_t> int32_or_reduce;
    extern reduce_type<int32_t> int32_xor_reduce;
    extern reduce_type<int32_t> int32_max_reduce;
    extern reduce_type<int32_t> int32_min_reduce;
    extern reduce_type<int32_t> int32_sum_reduce;
    extern reduce_type<int32_t> int32_prod_reduce;

    extern reduce_type<int64_t> int64_and_reduce;
    extern reduce_type<int64_t> int64_or_reduce;
    extern reduce_type<int64_t> int64_xor_reduce;
    extern reduce_type<int64_t> int64_max_reduce;
    extern reduce_type<int64_t> int64_min_reduce;
    extern reduce_type<int64_t> int64_sum_reduce;
    extern reduce_type<int64_t> int64_prod_reduce;

    extern reduce_type<long long> longlong_max_reduce;
    extern reduce_type<long long> longlong_min_reduce;
    extern reduce_type<long long> longlong_sum_reduce;
    extern reduce_type<long long> longlong_prod_reduce;

    extern reduce_type<float> float_max_reduce;
    extern reduce_type<float> float_min_reduce;
    extern reduce_type<float> float_sum_reduce;
    extern reduce_type<float> float_prod_reduce;

    extern reduce_type<double> double_max_reduce;
    extern reduce_type<double> double_min_reduce;
    extern reduce_type<double> double_sum_reduce;
    extern reduce_type<double> double_prod_reduce;

    /* Scan */
    extern bool inscan_exists;
    extern scan_type<uint8_t> uint8_sum_inscan;
    extern scan_type<uint16_t> uint16_sum_inscan;
    extern scan_type<uint32_t> uint32_sum_inscan;
    extern scan_type<uint64_t> uint64_sum_inscan;
    extern scan_type<unsigned long long> ulonglong_sum_inscan;
    extern scan_type<int8_t> int8_sum_inscan;
    extern scan_type<int16_t> int16_sum_inscan;
    extern scan_type<int32_t> int32_sum_inscan;
    extern scan_type<int64_t> int64_sum_inscan;
    extern scan_type<long long> longlong_sum_inscan;
    extern scan_type<float> float_sum_inscan;
    extern scan_type<double> double_sum_inscan;

    extern bool exscan_exists;
    extern scan_type<uint8_t> uint8_sum_exscan;
    extern scan_type<uint16_t> uint16_sum_exscan;
    extern scan_type<uint32_t> uint32_sum_exscan;
    extern scan_type<uint64_t> uint64_sum_exscan;
    extern scan_type<unsigned long long> ulonglong_sum_exscan;
    extern scan_type<int8_t> int8_sum_exscan;
    extern scan_type<int16_t> int16_sum_exscan;
    extern scan_type<int32_t> int32_sum_exscan;
    extern scan_type<int64_t> int64_sum_exscan;
    extern scan_type<long long> longlong_sum_exscan;
    extern scan_type<float> float_sum_exscan;
    extern scan_type<double> double_sum_exscan;

    /* Point-to-Point Synchronization */
    extern test_type<int32_t> int32_test;
    extern test_all_type<int32_t> int32_test_all;
    extern test_any_type<int32_t> int32_test_any;
    extern test_some_type<int32_t> int32_test_some;
    extern wait_until_type<int32_t> int32_wait_until;
    extern wait_until_all_type<int32_t> int32_wait_until_all;
    extern wait_until_any_type<int32_t> int32_wait_until_any;
    extern wait_until_some_type<int32_t> int32_wait_until_some;
    extern test_all_vector_type<int32_t> int32_test_all_vector;
    extern test_any_vector_type<int32_t> int32_test_any_vector;
    extern test_some_vector_type<int32_t> int32_test_some_vector;
    extern wait_until_all_vector_type<int32_t> int32_wait_until_all_vector;
    extern wait_until_any_vector_type<int32_t> int32_wait_until_any_vector;
    extern wait_until_some_vector_type<int32_t> int32_wait_until_some_vector;

    extern test_type<int64_t> int64_test;
    extern test_all_type<int64_t> int64_test_all;
    extern test_any_type<int64_t> int64_test_any;
    extern test_some_type<int64_t> int64_test_some;
    extern wait_until_type<int64_t> int64_wait_until;
    extern wait_until_all_type<int64_t> int64_wait_until_all;
    extern wait_until_any_type<int64_t> int64_wait_until_any;
    extern wait_until_some_type<int64_t> int64_wait_until_some;
    extern test_all_vector_type<int64_t> int64_test_all_vector;
    extern test_any_vector_type<int64_t> int64_test_any_vector;
    extern test_some_vector_type<int64_t> int64_test_some_vector;
    extern wait_until_all_vector_type<int64_t> int64_wait_until_all_vector;
    extern wait_until_any_vector_type<int64_t> int64_wait_until_any_vector;
    extern wait_until_some_vector_type<int64_t> int64_wait_until_some_vector;

    extern test_type<long long> longlong_test;
    extern test_all_type<long long> longlong_test_all;
    extern test_any_type<long long> longlong_test_any;
    extern test_some_type<long long> longlong_test_some;
    extern wait_until_type<long long> longlong_wait_until;
    extern wait_until_all_type<long long> longlong_wait_until_all;
    extern wait_until_any_type<long long> longlong_wait_until_any;
    extern wait_until_some_type<long long> longlong_wait_until_some;
    extern test_all_vector_type<long long> longlong_test_all_vector;
    extern test_any_vector_type<long long> longlong_test_any_vector;
    extern test_some_vector_type<long long> longlong_test_some_vector;
    extern wait_until_all_vector_type<long long> longlong_wait_until_all_vector;
    extern wait_until_any_vector_type<long long> longlong_wait_until_any_vector;
    extern wait_until_some_vector_type<long long> longlong_wait_until_some_vector;

    extern test_type<uint32_t> uint32_test;
    extern test_all_type<uint32_t> uint32_test_all;
    extern test_any_type<uint32_t> uint32_test_any;
    extern test_some_type<uint32_t> uint32_test_some;
    extern wait_until_type<uint32_t> uint32_wait_until;
    extern wait_until_all_type<uint32_t> uint32_wait_until_all;
    extern wait_until_any_type<uint32_t> uint32_wait_until_any;
    extern wait_until_some_type<uint32_t> uint32_wait_until_some;
    extern test_all_vector_type<uint32_t> uint32_test_all_vector;
    extern test_any_vector_type<uint32_t> uint32_test_any_vector;
    extern test_some_vector_type<uint32_t> uint32_test_some_vector;

    extern wait_until_all_vector_type<uint32_t> uint32_wait_until_all_vector;
    extern wait_until_any_vector_type<uint32_t> uint32_wait_until_any_vector;
    extern wait_until_some_vector_type<uint32_t> uint32_wait_until_some_vector;

    extern test_type<uint64_t> uint64_test;
    extern test_all_type<uint64_t> uint64_test_all;
    extern test_any_type<uint64_t> uint64_test_any;
    extern test_some_type<uint64_t> uint64_test_some;
    extern wait_until_type<uint64_t> uint64_wait_until;
    extern wait_until_all_type<uint64_t> uint64_wait_until_all;
    extern wait_until_any_type<uint64_t> uint64_wait_until_any;
    extern wait_until_some_type<uint64_t> uint64_wait_until_some;
    extern test_all_vector_type<uint64_t> uint64_test_all_vector;
    extern test_any_vector_type<uint64_t> uint64_test_any_vector;
    extern test_some_vector_type<uint64_t> uint64_test_some_vector;
    extern wait_until_all_vector_type<uint64_t> uint64_wait_until_all_vector;
    extern wait_until_any_vector_type<uint64_t> uint64_wait_until_any_vector;
    extern wait_until_some_vector_type<uint64_t> uint64_wait_until_some_vector;

    extern test_type<unsigned long long> ulonglong_test;
    extern test_all_type<unsigned long long> ulonglong_test_all;
    extern test_any_type<unsigned long long> ulonglong_test_any;
    extern test_some_type<unsigned long long> ulonglong_test_some;
    extern wait_until_type<unsigned long long> ulonglong_wait_until;
    extern wait_until_all_type<unsigned long long> ulonglong_wait_until_all;
    extern wait_until_any_type<unsigned long long> ulonglong_wait_until_any;
    extern wait_until_some_type<unsigned long long> ulonglong_wait_until_some;
    extern test_all_vector_type<unsigned long long> ulonglong_test_all_vector;
    extern test_any_vector_type<unsigned long long> ulonglong_test_any_vector;
    extern test_some_vector_type<unsigned long long> ulonglong_test_some_vector;
    extern wait_until_all_vector_type<unsigned long long> ulonglong_wait_until_all_vector;
    extern wait_until_any_vector_type<unsigned long long> ulonglong_wait_until_any_vector;
    extern wait_until_some_vector_type<unsigned long long> ulonglong_wait_until_some_vector;

    extern uint64_t (*signal_wait_until)(uint64_t *, int, uint64_t);

    /* Memory Ordering */
    extern void (*fence)(void);
    extern void (*quiet)(void);

    /* Define the template specializations for each shmem wrapper function which has more than one
     * supported type */
    /* clang-format off */
    template <typename T> static constexpr irma_type<T> iput() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto iput<uint8_t>() -> irma_type<uint8_t> { return uint8_iput; }
    template <> inline auto iput<uint16_t>() -> irma_type<uint16_t> { return uint16_iput; }
    template <> inline auto iput<uint32_t>() -> irma_type<uint32_t> { return uint32_iput; }
    template <> inline auto iput<uint64_t>() -> irma_type<uint64_t> { return uint64_iput; }
    template <> inline auto iput<unsigned long long>() -> irma_type<unsigned long long> { return ulonglong_iput; }

    template <typename T> static constexpr ibrma_type<T> ibput() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto ibput<uint8_t>() -> ibrma_type<uint8_t> { return uint8_ibput; }
    template <> inline auto ibput<uint16_t>() -> ibrma_type<uint16_t> { return uint16_ibput; }
    template <> inline auto ibput<uint32_t>() -> ibrma_type<uint32_t> { return uint32_ibput; }
    template <> inline auto ibput<uint64_t>() -> ibrma_type<uint64_t> { return uint64_ibput; }

    template <typename T> static constexpr p_type<T> p() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto p<uint8_t>() -> p_type<uint8_t> { return uint8_p; }
    template <> inline auto p<uint16_t>() -> p_type<uint16_t> { return uint16_p; }
    template <> inline auto p<uint32_t>() -> p_type<uint32_t> { return uint32_p; }
    template <> inline auto p<uint64_t>() -> p_type<uint64_t> { return uint64_p; }
    template <> inline auto p<unsigned long long>() -> p_type<unsigned long long> { return ulonglong_p; }
    template <> inline auto p<float>() -> p_type<float> { return float_p; }
    template <> inline auto p<double>() -> p_type<double> { return double_p; }

    template <typename T> static constexpr irma_type<T> iget() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto iget<uint8_t>() -> irma_type<uint8_t> { return uint8_iget; }
    template <> inline auto iget<uint16_t>() -> irma_type<uint16_t> { return uint16_iget; }
    template <> inline auto iget<uint32_t>() -> irma_type<uint32_t> { return uint32_iget; }
    template <> inline auto iget<uint64_t>() -> irma_type<uint64_t> { return uint64_iget; }
    template <> inline auto iget<unsigned long long>() -> irma_type<unsigned long long> { return ulonglong_iget; }

    template <typename T> static constexpr ibrma_type<T> ibget() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto ibget<uint8_t>() -> ibrma_type<uint8_t> { return uint8_ibget; }
    template <> inline auto ibget<uint16_t>() -> ibrma_type<uint16_t> { return uint16_ibget; }
    template <> inline auto ibget<uint32_t>() -> ibrma_type<uint32_t> { return uint32_ibget; }
    template <> inline auto ibget<uint64_t>() -> ibrma_type<uint64_t> { return uint64_ibget; }

    template <typename T> static constexpr g_type<T> g() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto g<uint8_t>() -> g_type<uint8_t> { return uint8_g; }
    template <> inline auto g<uint16_t>() -> g_type<uint16_t> { return uint16_g; }
    template <> inline auto g<uint32_t>() -> g_type<uint32_t> { return uint32_g; }
    template <> inline auto g<uint64_t>() -> g_type<uint64_t> { return uint64_g; }
    template <> inline auto g<unsigned long long>() -> g_type<unsigned long long> { return ulonglong_g; }
    template <> inline auto g<float>() -> g_type<float> { return float_g; }
    template <> inline auto g<double>() -> g_type<double> { return double_g; }

/* Define the template specializations for each shmem wrapper function which has more than one
* supported type */
    template <typename T> static constexpr atomic_fetch_nbi_type<T> atomic_fetch_nbi() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_fetch_nbi<uint32_t>() -> atomic_fetch_nbi_type<uint32_t> { return uint32_atomic_fetch_nbi; }
    template <> inline auto atomic_fetch_nbi<uint64_t>() -> atomic_fetch_nbi_type<uint64_t> { return uint64_atomic_fetch_nbi; }
    template <> inline auto atomic_fetch_nbi<unsigned long long>() -> atomic_fetch_nbi_type<unsigned long long> { return ulonglong_atomic_fetch_nbi; }
    template <> inline auto atomic_fetch_nbi<int32_t>() -> atomic_fetch_nbi_type<int32_t> { return int32_atomic_fetch_nbi; }
    template <> inline auto atomic_fetch_nbi<int64_t>() -> atomic_fetch_nbi_type<int64_t> { return int64_atomic_fetch_nbi; }
    template <> inline auto atomic_fetch_nbi<long long>() -> atomic_fetch_nbi_type<long long> { return longlong_atomic_fetch_nbi; }
    template <> inline auto atomic_fetch_nbi<float>() -> atomic_fetch_nbi_type<float> { return float_atomic_fetch_nbi; }
    template <> inline auto atomic_fetch_nbi<double>() -> atomic_fetch_nbi_type<double> { return double_atomic_fetch_nbi; }

    template <typename T> static constexpr atomic_compare_swap_nbi_type<T> atomic_compare_swap_nbi() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_compare_swap_nbi<uint32_t>() -> atomic_compare_swap_nbi_type<uint32_t> { return uint32_atomic_compare_swap_nbi; }
    template <> inline auto atomic_compare_swap_nbi<uint64_t>() -> atomic_compare_swap_nbi_type<uint64_t> { return uint64_atomic_compare_swap_nbi; }
 template <>
auto atomic_compare_swap_nbi<unsigned long long>()
-> atomic_compare_swap_nbi_type<unsigned long long>
{
return ulonglong_atomic_compare_swap_nbi;
}
    template <> inline auto atomic_compare_swap_nbi<int32_t>() -> atomic_compare_swap_nbi_type<int32_t> { return int32_atomic_compare_swap_nbi; }
    template <> inline auto atomic_compare_swap_nbi<int64_t>() -> atomic_compare_swap_nbi_type<int64_t> { return int64_atomic_compare_swap_nbi; }
    template <> inline auto atomic_compare_swap_nbi<long long>() -> atomic_compare_swap_nbi_type<long long> { return longlong_atomic_compare_swap_nbi; }

    template <typename T> static constexpr atomic_swap_nbi_type<T> atomic_swap_nbi() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_swap_nbi<uint32_t>() -> atomic_swap_nbi_type<uint32_t> { return uint32_atomic_swap_nbi; }
    template <> inline auto atomic_swap_nbi<uint64_t>() -> atomic_swap_nbi_type<uint64_t> { return uint64_atomic_swap_nbi; }
    template <> inline auto atomic_swap_nbi<unsigned long long>() -> atomic_swap_nbi_type<unsigned long long> { return ulonglong_atomic_swap_nbi; }
    template <> inline auto atomic_swap_nbi<int32_t>() -> atomic_swap_nbi_type<int32_t> { return int32_atomic_swap_nbi; }
    template <> inline auto atomic_swap_nbi<int64_t>() -> atomic_swap_nbi_type<int64_t> { return int64_atomic_swap_nbi; }
    template <> inline auto atomic_swap_nbi<long long>() -> atomic_swap_nbi_type<long long> { return longlong_atomic_swap_nbi; }
    template <> inline auto atomic_swap_nbi<float>() -> atomic_swap_nbi_type<float> { return float_atomic_swap_nbi; }
    template <> inline auto atomic_swap_nbi<double>() -> atomic_swap_nbi_type<double> { return double_atomic_swap_nbi; }

    template <typename T> static constexpr atomic_fetch_inc_nbi_type<T> atomic_fetch_inc_nbi() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_fetch_inc_nbi<uint32_t>() -> atomic_fetch_inc_nbi_type<uint32_t> { return uint32_atomic_fetch_inc_nbi; }
    template <> inline auto atomic_fetch_inc_nbi<uint64_t>() -> atomic_fetch_inc_nbi_type<uint64_t> { return uint64_atomic_fetch_inc_nbi; }
    template <> inline auto atomic_fetch_inc_nbi<unsigned long long>() -> atomic_fetch_inc_nbi_type<unsigned long long> { return ulonglong_atomic_fetch_inc_nbi; }
    template <> inline auto atomic_fetch_inc_nbi<int32_t>() -> atomic_fetch_inc_nbi_type<int32_t> { return int32_atomic_fetch_inc_nbi; }
    template <> inline auto atomic_fetch_inc_nbi<int64_t>() -> atomic_fetch_inc_nbi_type<int64_t> { return int64_atomic_fetch_inc_nbi; }
    template <> inline auto atomic_fetch_inc_nbi<long long>() -> atomic_fetch_inc_nbi_type<long long> { return longlong_atomic_fetch_inc_nbi; }

    template <typename T> static constexpr atomic_fetch_op_nbi_type<T> atomic_fetch_add_nbi() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_fetch_add_nbi<uint32_t>() -> atomic_fetch_op_nbi_type<uint32_t> { return uint32_atomic_fetch_add_nbi; }
    template <> inline auto atomic_fetch_add_nbi<uint64_t>() -> atomic_fetch_op_nbi_type<uint64_t> { return uint64_atomic_fetch_add_nbi; }
    template <> inline auto atomic_fetch_add_nbi<unsigned long long>() -> atomic_fetch_op_nbi_type<unsigned long long> { return ulonglong_atomic_fetch_add_nbi; }
    template <> inline auto atomic_fetch_add_nbi<int32_t>() -> atomic_fetch_op_nbi_type<int32_t> { return int32_atomic_fetch_add_nbi; }
    template <> inline auto atomic_fetch_add_nbi<int64_t>() -> atomic_fetch_op_nbi_type<int64_t> { return int64_atomic_fetch_add_nbi; }
    template <> inline auto atomic_fetch_add_nbi<long long>() -> atomic_fetch_op_nbi_type<long long> { return longlong_atomic_fetch_add_nbi; }

    template <typename T> static constexpr atomic_fetch_op_nbi_type<T> atomic_fetch_and_nbi() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_fetch_and_nbi<uint32_t>() -> atomic_fetch_op_nbi_type<uint32_t> { return uint32_atomic_fetch_and_nbi; }
    template <> inline auto atomic_fetch_and_nbi<uint64_t>() -> atomic_fetch_op_nbi_type<uint64_t> { return uint64_atomic_fetch_and_nbi; }
    template <> inline auto atomic_fetch_and_nbi<unsigned long long>() -> atomic_fetch_op_nbi_type<unsigned long long> { return ulonglong_atomic_fetch_and_nbi; }
    template <> inline auto atomic_fetch_and_nbi<int32_t>() -> atomic_fetch_op_nbi_type<int32_t> { return int32_atomic_fetch_and_nbi; }
    template <> inline auto atomic_fetch_and_nbi<int64_t>() -> atomic_fetch_op_nbi_type<int64_t> { return int64_atomic_fetch_and_nbi; }

    template <typename T> static constexpr atomic_fetch_op_nbi_type<T> atomic_fetch_or_nbi() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_fetch_or_nbi<uint32_t>() -> atomic_fetch_op_nbi_type<uint32_t> { return uint32_atomic_fetch_or_nbi; }
    template <> inline auto atomic_fetch_or_nbi<uint64_t>() -> atomic_fetch_op_nbi_type<uint64_t> { return uint64_atomic_fetch_or_nbi; }
    template <> inline auto atomic_fetch_or_nbi<unsigned long long>() -> atomic_fetch_op_nbi_type<unsigned long long> { return ulonglong_atomic_fetch_or_nbi; }
    template <> inline auto atomic_fetch_or_nbi<int32_t>() -> atomic_fetch_op_nbi_type<int32_t> { return int32_atomic_fetch_or_nbi; }
    template <> inline auto atomic_fetch_or_nbi<int64_t>() -> atomic_fetch_op_nbi_type<int64_t> { return int64_atomic_fetch_or_nbi; }

    template <typename T> static constexpr atomic_fetch_op_nbi_type<T> atomic_fetch_xor_nbi() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_fetch_xor_nbi<uint32_t>() -> atomic_fetch_op_nbi_type<uint32_t> { return uint32_atomic_fetch_xor_nbi; }
    template <> inline auto atomic_fetch_xor_nbi<uint64_t>() -> atomic_fetch_op_nbi_type<uint64_t> { return uint64_atomic_fetch_xor_nbi; }
    template <> inline auto atomic_fetch_xor_nbi<unsigned long long>() -> atomic_fetch_op_nbi_type<unsigned long long> { return ulonglong_atomic_fetch_xor_nbi; }
    template <> inline auto atomic_fetch_xor_nbi<int32_t>() -> atomic_fetch_op_nbi_type<int32_t> { return int32_atomic_fetch_xor_nbi; }
    template <> inline auto atomic_fetch_xor_nbi<int64_t>() -> atomic_fetch_op_nbi_type<int64_t> { return int64_atomic_fetch_xor_nbi; }

/* Define the template specializations for each shmem wrapper function which has more than one
* supported type */
    template <typename T> static constexpr atomic_fetch_type<T> atomic_fetch() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_fetch<uint32_t>() -> atomic_fetch_type<uint32_t> { return uint32_atomic_fetch; }
    template <> inline auto atomic_fetch<uint64_t>() -> atomic_fetch_type<uint64_t> { return uint64_atomic_fetch; }
    template <> inline auto atomic_fetch<unsigned long long>() -> atomic_fetch_type<unsigned long long> { return ulonglong_atomic_fetch; }
    template <> inline auto atomic_fetch<int32_t>() -> atomic_fetch_type<int32_t> { return int32_atomic_fetch; }
    template <> inline auto atomic_fetch<int64_t>() -> atomic_fetch_type<int64_t> { return int64_atomic_fetch; }
    template <> inline auto atomic_fetch<long long>() -> atomic_fetch_type<long long> { return longlong_atomic_fetch; }
    template <> inline auto atomic_fetch<float>() -> atomic_fetch_type<float> { return float_atomic_fetch; }
    template <> inline auto atomic_fetch<double>() -> atomic_fetch_type<double> { return double_atomic_fetch; }

    template <typename T> static constexpr atomic_compare_swap_type<T> atomic_compare_swap() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_compare_swap<uint32_t>() -> atomic_compare_swap_type<uint32_t> { return uint32_atomic_compare_swap; }
    template <> inline auto atomic_compare_swap<uint64_t>() -> atomic_compare_swap_type<uint64_t> { return uint64_atomic_compare_swap; }
    template <> inline auto atomic_compare_swap<unsigned long long>() -> atomic_compare_swap_type<unsigned long long> { return ulonglong_atomic_compare_swap; }
    template <> inline auto atomic_compare_swap<int32_t>() -> atomic_compare_swap_type<int32_t> { return int32_atomic_compare_swap; }
    template <> inline auto atomic_compare_swap<int64_t>() -> atomic_compare_swap_type<int64_t> { return int64_atomic_compare_swap; }
    template <> inline auto atomic_compare_swap<long long>() -> atomic_compare_swap_type<long long> { return longlong_atomic_compare_swap; }

    template <typename T> static constexpr atomic_swap_type<T> atomic_swap() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_swap<uint32_t>() -> atomic_swap_type<uint32_t> { return uint32_atomic_swap; }
    template <> inline auto atomic_swap<uint64_t>() -> atomic_swap_type<uint64_t> { return uint64_atomic_swap; }
    template <> inline auto atomic_swap<unsigned long long>() -> atomic_swap_type<unsigned long long> { return ulonglong_atomic_swap; }
    template <> inline auto atomic_swap<int32_t>() -> atomic_swap_type<int32_t> { return int32_atomic_swap; }
    template <> inline auto atomic_swap<int64_t>() -> atomic_swap_type<int64_t> { return int64_atomic_swap; }
    template <> inline auto atomic_swap<long long>() -> atomic_swap_type<long long> { return longlong_atomic_swap; }
    template <> inline auto atomic_swap<float>() -> atomic_swap_type<float> { return float_atomic_swap; }
    template <> inline auto atomic_swap<double>() -> atomic_swap_type<double> { return double_atomic_swap; }

    template <typename T> static constexpr atomic_set_type<T> atomic_set() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_set<uint32_t>() -> atomic_set_type<uint32_t> { return uint32_atomic_set; }
    template <> inline auto atomic_set<uint64_t>() -> atomic_set_type<uint64_t> { return uint64_atomic_set; }
    template <> inline auto atomic_set<unsigned long long>() -> atomic_set_type<unsigned long long> { return ulonglong_atomic_set; }
    template <> inline auto atomic_set<int32_t>() -> atomic_set_type<int32_t> { return int32_atomic_set; }
    template <> inline auto atomic_set<int64_t>() -> atomic_set_type<int64_t> { return int64_atomic_set; }
    template <> inline auto atomic_set<long long>() -> atomic_set_type<long long> { return longlong_atomic_set; }
    template <> inline auto atomic_set<float>() -> atomic_set_type<float> { return float_atomic_set; }
    template <> inline auto atomic_set<double>() -> atomic_set_type<double> { return double_atomic_set; }

    template <typename T> static constexpr atomic_fetch_inc_type<T> atomic_fetch_inc() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_fetch_inc<uint32_t>() -> atomic_fetch_inc_type<uint32_t> { return uint32_atomic_fetch_inc; }
    template <> inline auto atomic_fetch_inc<uint64_t>() -> atomic_fetch_inc_type<uint64_t> { return uint64_atomic_fetch_inc; }
    template <> inline auto atomic_fetch_inc<unsigned long long>() -> atomic_fetch_inc_type<unsigned long long> { return ulonglong_atomic_fetch_inc; }
    template <> inline auto atomic_fetch_inc<int32_t>() -> atomic_fetch_inc_type<int32_t> { return int32_atomic_fetch_inc; }
    template <> inline auto atomic_fetch_inc<int64_t>() -> atomic_fetch_inc_type<int64_t> { return int64_atomic_fetch_inc; }
    template <> inline auto atomic_fetch_inc<long long>() -> atomic_fetch_inc_type<long long> { return longlong_atomic_fetch_inc; }

    template <typename T> static constexpr atomic_fetch_op_type<T> atomic_fetch_add() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_fetch_add<uint32_t>() -> atomic_fetch_op_type<uint32_t> { return uint32_atomic_fetch_add; }
    template <> inline auto atomic_fetch_add<uint64_t>() -> atomic_fetch_op_type<uint64_t> { return uint64_atomic_fetch_add; }
    template <> inline auto atomic_fetch_add<unsigned long long>() -> atomic_fetch_op_type<unsigned long long> { return ulonglong_atomic_fetch_add; }
    template <> inline auto atomic_fetch_add<int32_t>() -> atomic_fetch_op_type<int32_t> { return int32_atomic_fetch_add; }
    template <> inline auto atomic_fetch_add<int64_t>() -> atomic_fetch_op_type<int64_t> { return int64_atomic_fetch_add; }
    template <> inline auto atomic_fetch_add<long long>() -> atomic_fetch_op_type<long long> { return longlong_atomic_fetch_add; }

    template <typename T> static constexpr atomic_fetch_op_type<T> atomic_fetch_and() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_fetch_and<uint32_t>() -> atomic_fetch_op_type<uint32_t> { return uint32_atomic_fetch_and; }
    template <> inline auto atomic_fetch_and<uint64_t>() -> atomic_fetch_op_type<uint64_t> { return uint64_atomic_fetch_and; }
    template <> inline auto atomic_fetch_and<unsigned long long>() -> atomic_fetch_op_type<unsigned long long> { return ulonglong_atomic_fetch_and; }
    template <> inline auto atomic_fetch_and<int32_t>() -> atomic_fetch_op_type<int32_t> { return int32_atomic_fetch_and; }
    template <> inline auto atomic_fetch_and<int64_t>() -> atomic_fetch_op_type<int64_t> { return int64_atomic_fetch_and; }

    template <typename T> static constexpr atomic_fetch_op_type<T> atomic_fetch_or() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_fetch_or<uint32_t>() -> atomic_fetch_op_type<uint32_t> { return uint32_atomic_fetch_or; }
    template <> inline auto atomic_fetch_or<uint64_t>() -> atomic_fetch_op_type<uint64_t> { return uint64_atomic_fetch_or; }
    template <> inline auto atomic_fetch_or<unsigned long long>() -> atomic_fetch_op_type<unsigned long long> { return ulonglong_atomic_fetch_or; }
    template <> inline auto atomic_fetch_or<int32_t>() -> atomic_fetch_op_type<int32_t> { return int32_atomic_fetch_or; }
    template <> inline auto atomic_fetch_or<int64_t>() -> atomic_fetch_op_type<int64_t> { return int64_atomic_fetch_or; }

    template <typename T> static constexpr atomic_fetch_op_type<T> atomic_fetch_xor() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_fetch_xor<uint32_t>() -> atomic_fetch_op_type<uint32_t> { return uint32_atomic_fetch_xor; }
    template <> inline auto atomic_fetch_xor<uint64_t>() -> atomic_fetch_op_type<uint64_t> { return uint64_atomic_fetch_xor; }
    template <> inline auto atomic_fetch_xor<unsigned long long>() -> atomic_fetch_op_type<unsigned long long> { return ulonglong_atomic_fetch_xor; }
    template <> inline auto atomic_fetch_xor<int32_t>() -> atomic_fetch_op_type<int32_t> { return int32_atomic_fetch_xor; }
    template <> inline auto atomic_fetch_xor<int64_t>() -> atomic_fetch_op_type<int64_t> { return int64_atomic_fetch_xor; }

    template <typename T> static constexpr atomic_inc_type<T> atomic_inc() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_inc<uint32_t>() -> atomic_inc_type<uint32_t> { return uint32_atomic_inc; }
    template <> inline auto atomic_inc<uint64_t>() -> atomic_inc_type<uint64_t> { return uint64_atomic_inc; }
    template <> inline auto atomic_inc<unsigned long long>() -> atomic_inc_type<unsigned long long> { return ulonglong_atomic_inc; }
    template <> inline auto atomic_inc<int32_t>() -> atomic_inc_type<int32_t> { return int32_atomic_inc; }
    template <> inline auto atomic_inc<int64_t>() -> atomic_inc_type<int64_t> { return int64_atomic_inc; }
    template <> inline auto atomic_inc<long long>() -> atomic_inc_type<long long> { return longlong_atomic_inc; }

    template <typename T> static constexpr atomic_op_type<T> atomic_add() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_add<uint32_t>() -> atomic_op_type<uint32_t> { return uint32_atomic_add; }
    template <> inline auto atomic_add<uint64_t>() -> atomic_op_type<uint64_t> { return uint64_atomic_add; }
    template <> inline auto atomic_add<unsigned long long>() -> atomic_op_type<unsigned long long> { return ulonglong_atomic_add; }
    template <> inline auto atomic_add<int32_t>() -> atomic_op_type<int32_t> { return int32_atomic_add; }
    template <> inline auto atomic_add<int64_t>() -> atomic_op_type<int64_t> { return int64_atomic_add; }
    template <> inline auto atomic_add<long long>() -> atomic_op_type<long long> { return longlong_atomic_add; }

    template <typename T> static constexpr atomic_op_type<T> atomic_and() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_and<uint32_t>() -> atomic_op_type<uint32_t> { return uint32_atomic_and; }
    template <> inline auto atomic_and<uint64_t>() -> atomic_op_type<uint64_t> { return uint64_atomic_and; }
    template <> inline auto atomic_and<unsigned long long>() -> atomic_op_type<unsigned long long> { return ulonglong_atomic_and; }
    template <> inline auto atomic_and<int32_t>() -> atomic_op_type<int32_t> { return int32_atomic_and; }
    template <> inline auto atomic_and<int64_t>() -> atomic_op_type<int64_t> { return int64_atomic_and; }

    template <typename T> static constexpr atomic_op_type<T> atomic_or() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_or<uint32_t>() -> atomic_op_type<uint32_t> { return uint32_atomic_or; }
    template <> inline auto atomic_or<uint64_t>() -> atomic_op_type<uint64_t> { return uint64_atomic_or; }
    template <> inline auto atomic_or<unsigned long long>() -> atomic_op_type<unsigned long long> { return ulonglong_atomic_or; }
    template <> inline auto atomic_or<int32_t>() -> atomic_op_type<int32_t> { return int32_atomic_or; }
    template <> inline auto atomic_or<int64_t>() -> atomic_op_type<int64_t> { return int64_atomic_or; }

    template <typename T> static constexpr atomic_op_type<T> atomic_xor() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto atomic_xor<uint32_t>() -> atomic_op_type<uint32_t> { return uint32_atomic_xor; }
    template <> inline auto atomic_xor<uint64_t>() -> atomic_op_type<uint64_t> { return uint64_atomic_xor; }
    template <> inline auto atomic_xor<unsigned long long>() -> atomic_op_type<unsigned long long> { return ulonglong_atomic_xor; }
    template <> inline auto atomic_xor<int32_t>() -> atomic_op_type<int32_t> { return int32_atomic_xor; }
    template <> inline auto atomic_xor<int64_t>() -> atomic_op_type<int64_t> { return int64_atomic_xor; }

    template <typename T, ishmemi_op_t OP> static constexpr reduce_type<T> reduce() { static_assert(assert_dependency_op<T, OP>::value, "Undefined wrapper function"); }
    template <> inline auto reduce<uint8_t, AND_REDUCE>() -> reduce_type<uint8_t> { return uint8_and_reduce; }
    template <> inline auto reduce<uint8_t, OR_REDUCE>() -> reduce_type<uint8_t> { return uint8_or_reduce; }
    template <> inline auto reduce<uint8_t, XOR_REDUCE>() -> reduce_type<uint8_t> { return uint8_xor_reduce; }
    template <> inline auto reduce<uint8_t, MAX_REDUCE>() -> reduce_type<uint8_t> { return uint8_max_reduce; }
    template <> inline auto reduce<uint8_t, MIN_REDUCE>() -> reduce_type<uint8_t> { return uint8_min_reduce; }
    template <> inline auto reduce<uint8_t, SUM_REDUCE>() -> reduce_type<uint8_t> { return uint8_sum_reduce; }
    template <> inline auto reduce<uint8_t, PROD_REDUCE>() -> reduce_type<uint8_t> { return uint8_prod_reduce; }

    template <> inline auto reduce<uint16_t, AND_REDUCE>() -> reduce_type<uint16_t> { return uint16_and_reduce; }
    template <> inline auto reduce<uint16_t, OR_REDUCE>() -> reduce_type<uint16_t> { return uint16_or_reduce; }
    template <> inline auto reduce<uint16_t, XOR_REDUCE>() -> reduce_type<uint16_t> { return uint16_xor_reduce; }
    template <> inline auto reduce<uint16_t, MAX_REDUCE>() -> reduce_type<uint16_t> { return uint16_max_reduce; }
    template <> inline auto reduce<uint16_t, MIN_REDUCE>() -> reduce_type<uint16_t> { return uint16_min_reduce; }
    template <> inline auto reduce<uint16_t, SUM_REDUCE>() -> reduce_type<uint16_t> { return uint16_sum_reduce; }
    template <> inline auto reduce<uint16_t, PROD_REDUCE>() -> reduce_type<uint16_t> { return uint16_prod_reduce; }

    template <> inline auto reduce<uint32_t, AND_REDUCE>() -> reduce_type<uint32_t> { return uint32_and_reduce; }
    template <> inline auto reduce<uint32_t, OR_REDUCE>() -> reduce_type<uint32_t> { return uint32_or_reduce; }
    template <> inline auto reduce<uint32_t, XOR_REDUCE>() -> reduce_type<uint32_t> { return uint32_xor_reduce; }
    template <> inline auto reduce<uint32_t, MAX_REDUCE>() -> reduce_type<uint32_t> { return uint32_max_reduce; }
    template <> inline auto reduce<uint32_t, MIN_REDUCE>() -> reduce_type<uint32_t> { return uint32_min_reduce; }
    template <> inline auto reduce<uint32_t, SUM_REDUCE>() -> reduce_type<uint32_t> { return uint32_sum_reduce; }
    template <> inline auto reduce<uint32_t, PROD_REDUCE>() -> reduce_type<uint32_t> { return uint32_prod_reduce; }

    template <> inline auto reduce<uint64_t, AND_REDUCE>() -> reduce_type<uint64_t> { return uint64_and_reduce; }
    template <> inline auto reduce<uint64_t, OR_REDUCE>() -> reduce_type<uint64_t> { return uint64_or_reduce; }
    template <> inline auto reduce<uint64_t, XOR_REDUCE>() -> reduce_type<uint64_t> { return uint64_xor_reduce; }
    template <> inline auto reduce<uint64_t, MAX_REDUCE>() -> reduce_type<uint64_t> { return uint64_max_reduce; }
    template <> inline auto reduce<uint64_t, MIN_REDUCE>() -> reduce_type<uint64_t> { return uint64_min_reduce; }
    template <> inline auto reduce<uint64_t, SUM_REDUCE>() -> reduce_type<uint64_t> { return uint64_sum_reduce; }
    template <> inline auto reduce<uint64_t, PROD_REDUCE>() -> reduce_type<uint64_t> { return uint64_prod_reduce; }

    template <> inline auto reduce<unsigned long long, AND_REDUCE>() -> reduce_type<unsigned long long> { return ulonglong_and_reduce; }
    template <> inline auto reduce<unsigned long long, OR_REDUCE>() -> reduce_type<unsigned long long> { return ulonglong_or_reduce; }
    template <> inline auto reduce<unsigned long long, XOR_REDUCE>() -> reduce_type<unsigned long long> { return ulonglong_xor_reduce; }
    template <> inline auto reduce<unsigned long long, MAX_REDUCE>() -> reduce_type<unsigned long long> { return ulonglong_max_reduce; }
    template <> inline auto reduce<unsigned long long, MIN_REDUCE>() -> reduce_type<unsigned long long> { return ulonglong_min_reduce; }
    template <> inline auto reduce<unsigned long long, SUM_REDUCE>() -> reduce_type<unsigned long long> { return ulonglong_sum_reduce; }
    template <> inline auto reduce<unsigned long long, PROD_REDUCE>() -> reduce_type<unsigned long long> { return ulonglong_prod_reduce; }

    template <> inline auto reduce<int8_t, AND_REDUCE>() -> reduce_type<int8_t> { return int8_and_reduce; }
    template <> inline auto reduce<int8_t, OR_REDUCE>() -> reduce_type<int8_t> { return int8_or_reduce; }
    template <> inline auto reduce<int8_t, XOR_REDUCE>() -> reduce_type<int8_t> { return int8_xor_reduce; }
    template <> inline auto reduce<int8_t, MAX_REDUCE>() -> reduce_type<int8_t> { return int8_max_reduce; }
    template <> inline auto reduce<int8_t, MIN_REDUCE>() -> reduce_type<int8_t> { return int8_min_reduce; }
    template <> inline auto reduce<int8_t, SUM_REDUCE>() -> reduce_type<int8_t> { return int8_sum_reduce; }
    template <> inline auto reduce<int8_t, PROD_REDUCE>() -> reduce_type<int8_t> { return int8_prod_reduce; }

    template <> inline auto reduce<int16_t, AND_REDUCE>() -> reduce_type<int16_t> { return int16_and_reduce; }
    template <> inline auto reduce<int16_t, OR_REDUCE>() -> reduce_type<int16_t> { return int16_or_reduce; }
    template <> inline auto reduce<int16_t, XOR_REDUCE>() -> reduce_type<int16_t> { return int16_xor_reduce; }
    template <> inline auto reduce<int16_t, MAX_REDUCE>() -> reduce_type<int16_t> { return int16_max_reduce; }
    template <> inline auto reduce<int16_t, MIN_REDUCE>() -> reduce_type<int16_t> { return int16_min_reduce; }
    template <> inline auto reduce<int16_t, SUM_REDUCE>() -> reduce_type<int16_t> { return int16_sum_reduce; }
    template <> inline auto reduce<int16_t, PROD_REDUCE>() -> reduce_type<int16_t> { return int16_prod_reduce; }

    template <> inline auto reduce<int32_t, AND_REDUCE>() -> reduce_type<int32_t> { return int32_and_reduce; }
    template <> inline auto reduce<int32_t, OR_REDUCE>() -> reduce_type<int32_t> { return int32_or_reduce; }
    template <> inline auto reduce<int32_t, XOR_REDUCE>() -> reduce_type<int32_t> { return int32_xor_reduce; }
    template <> inline auto reduce<int32_t, MAX_REDUCE>() -> reduce_type<int32_t> { return int32_max_reduce; }
    template <> inline auto reduce<int32_t, MIN_REDUCE>() -> reduce_type<int32_t> { return int32_min_reduce; }
    template <> inline auto reduce<int32_t, SUM_REDUCE>() -> reduce_type<int32_t> { return int32_sum_reduce; }
    template <> inline auto reduce<int32_t, PROD_REDUCE>() -> reduce_type<int32_t> { return int32_prod_reduce; }

    template <> inline auto reduce<int64_t, AND_REDUCE>() -> reduce_type<int64_t> { return int64_and_reduce; }
    template <> inline auto reduce<int64_t, OR_REDUCE>() -> reduce_type<int64_t> { return int64_or_reduce; }
    template <> inline auto reduce<int64_t, XOR_REDUCE>() -> reduce_type<int64_t> { return int64_xor_reduce; }
    template <> inline auto reduce<int64_t, MAX_REDUCE>() -> reduce_type<int64_t> { return int64_max_reduce; }
    template <> inline auto reduce<int64_t, MIN_REDUCE>() -> reduce_type<int64_t> { return int64_min_reduce; }
    template <> inline auto reduce<int64_t, SUM_REDUCE>() -> reduce_type<int64_t> { return int64_sum_reduce; }
    template <> inline auto reduce<int64_t, PROD_REDUCE>() -> reduce_type<int64_t> { return int64_prod_reduce; }

    template <> inline auto reduce<long long, MAX_REDUCE>() -> reduce_type<long long> { return longlong_max_reduce; }
    template <> inline auto reduce<long long, MIN_REDUCE>() -> reduce_type<long long> { return longlong_min_reduce; }
    template <> inline auto reduce<long long, SUM_REDUCE>() -> reduce_type<long long> { return longlong_sum_reduce; }
    template <> inline auto reduce<long long, PROD_REDUCE>() -> reduce_type<long long> { return longlong_prod_reduce; }

    template <> inline auto reduce<float, MAX_REDUCE>() -> reduce_type<float> { return float_max_reduce; }
    template <> inline auto reduce<float, MIN_REDUCE>() -> reduce_type<float> { return float_min_reduce; }
    template <> inline auto reduce<float, SUM_REDUCE>() -> reduce_type<float> { return float_sum_reduce; }
    template <> inline auto reduce<float, PROD_REDUCE>() -> reduce_type<float> { return float_prod_reduce; }

    template <> inline auto reduce<double, MAX_REDUCE>() -> reduce_type<double> { return double_max_reduce; }
    template <> inline auto reduce<double, MIN_REDUCE>() -> reduce_type<double> { return double_min_reduce; }
    template <> inline auto reduce<double, SUM_REDUCE>() -> reduce_type<double> { return double_sum_reduce; }
    template <> inline auto reduce<double, PROD_REDUCE>() -> reduce_type<double> { return double_prod_reduce; }

    template <typename T> static constexpr scan_type<T> inscan() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto inscan<uint8_t>() -> scan_type<uint8_t> { return uint8_sum_inscan; }
    template <> inline auto inscan<uint16_t>() -> scan_type<uint16_t> { return uint16_sum_inscan; }
    template <> inline auto inscan<uint32_t>() -> scan_type<uint32_t> { return uint32_sum_inscan; }
    template <> inline auto inscan<uint64_t>() -> scan_type<uint64_t> { return uint64_sum_inscan; }
    template <> inline auto inscan<unsigned long long>() -> scan_type<unsigned long long> { return ulonglong_sum_inscan; }
    template <> inline auto inscan<int8_t>() -> scan_type<int8_t> { return int8_sum_inscan; }
    template <> inline auto inscan<int16_t>() -> scan_type<int16_t> { return int16_sum_inscan; }
    template <> inline auto inscan<int32_t>() -> scan_type<int32_t> { return int32_sum_inscan; }
    template <> inline auto inscan<int64_t>() -> scan_type<int64_t> { return int64_sum_inscan; }
    template <> inline auto inscan<long long>() -> scan_type<long long> { return longlong_sum_inscan; }
    template <> inline auto inscan<float>() -> scan_type<float> { return float_sum_inscan; }
    template <> inline auto inscan<double>() -> scan_type<double> { return double_sum_inscan; }

    template <typename T> static constexpr scan_type<T> exscan() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto exscan<uint8_t>() -> scan_type<uint8_t> { return uint8_sum_exscan; }
    template <> inline auto exscan<uint16_t>() -> scan_type<uint16_t> { return uint16_sum_exscan; }
    template <> inline auto exscan<uint32_t>() -> scan_type<uint32_t> { return uint32_sum_exscan; }
    template <> inline auto exscan<uint64_t>() -> scan_type<uint64_t> { return uint64_sum_exscan; }
    template <> inline auto exscan<unsigned long long>() -> scan_type<unsigned long long> { return ulonglong_sum_exscan; }
    template <> inline auto exscan<int8_t>() -> scan_type<int8_t> { return int8_sum_exscan; }
    template <> inline auto exscan<int16_t>() -> scan_type<int16_t> { return int16_sum_exscan; }
    template <> inline auto exscan<int32_t>() -> scan_type<int32_t> { return int32_sum_exscan; }
    template <> inline auto exscan<int64_t>() -> scan_type<int64_t> { return int64_sum_exscan; }
    template <> inline auto exscan<long long>() -> scan_type<long long> { return longlong_sum_exscan; }
    template <> inline auto exscan<float>() -> scan_type<float> { return float_sum_exscan; }
    template <> inline auto exscan<double>() -> scan_type<double> { return double_sum_exscan; }
    

    template <typename T> static constexpr test_type<T> test() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto test<uint32_t>() -> test_type<uint32_t> { return uint32_test; }
    template <> inline auto test<uint64_t>() -> test_type<uint64_t> { return uint64_test; }
    template <> inline auto test<unsigned long long>() -> test_type<unsigned long long> { return ulonglong_test; }
    template <> inline auto test<int32_t>() -> test_type<int32_t> { return int32_test; }
    template <> inline auto test<int64_t>() -> test_type<int64_t> { return int64_test; }
    template <> inline auto test<long long>() -> test_type<long long> { return longlong_test; }

    template <typename T> static constexpr test_all_type<T> test_all() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto test_all<uint32_t>() -> test_all_type<uint32_t> { return uint32_test_all; }
    template <> inline auto test_all<uint64_t>() -> test_all_type<uint64_t> { return uint64_test_all; }
    template <> inline auto test_all<unsigned long long>() -> test_all_type<unsigned long long> { return ulonglong_test_all; }
    template <> inline auto test_all<int32_t>() -> test_all_type<int32_t> { return int32_test_all; }
    template <> inline auto test_all<int64_t>() -> test_all_type<int64_t> { return int64_test_all; }
    template <> inline auto test_all<long long>() -> test_all_type<long long> { return longlong_test_all; }

    template <typename T> static constexpr test_any_type<T> test_any() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto test_any<uint32_t>() -> test_any_type<uint32_t> { return uint32_test_any; }
    template <> inline auto test_any<uint64_t>() -> test_any_type<uint64_t> { return uint64_test_any; }
    template <> inline auto test_any<unsigned long long>() -> test_any_type<unsigned long long> { return ulonglong_test_any; }
    template <> inline auto test_any<int32_t>() -> test_any_type<int32_t> { return int32_test_any; }
    template <> inline auto test_any<int64_t>() -> test_any_type<int64_t> { return int64_test_any; }
    template <> inline auto test_any<long long>() -> test_any_type<long long> { return longlong_test_any; }

    template <typename T> static constexpr test_some_type<T> test_some() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto test_some<uint32_t>() -> test_some_type<uint32_t> { return uint32_test_some; }
    template <> inline auto test_some<uint64_t>() -> test_some_type<uint64_t> { return uint64_test_some; }
    template <> inline auto test_some<unsigned long long>() -> test_some_type<unsigned long long> { return ulonglong_test_some; }
    template <> inline auto test_some<int32_t>() -> test_some_type<int32_t> { return int32_test_some; }
    template <> inline auto test_some<int64_t>() -> test_some_type<int64_t> { return int64_test_some; }
    template <> inline auto test_some<long long>() -> test_some_type<long long> { return longlong_test_some; }

    template <typename T> static constexpr wait_until_type<T> wait_until() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto wait_until<uint32_t>() -> wait_until_type<uint32_t> { return uint32_wait_until; }
    template <> inline auto wait_until<uint64_t>() -> wait_until_type<uint64_t> { return uint64_wait_until; }
    template <> inline auto wait_until<unsigned long long>() -> wait_until_type<unsigned long long> { return ulonglong_wait_until; }
    template <> inline auto wait_until<int32_t>() -> wait_until_type<int32_t> { return int32_wait_until; }
    template <> inline auto wait_until<int64_t>() -> wait_until_type<int64_t> { return int64_wait_until; }
    template <> inline auto wait_until<long long>() -> wait_until_type<long long> { return longlong_wait_until; }

    template <typename T> static constexpr wait_until_all_type<T> wait_until_all() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto wait_until_all<uint32_t>() -> wait_until_all_type<uint32_t> { return uint32_wait_until_all; }
    template <> inline auto wait_until_all<uint64_t>() -> wait_until_all_type<uint64_t> { return uint64_wait_until_all; }
    template <> inline auto wait_until_all<unsigned long long>() -> wait_until_all_type<unsigned long long> { return ulonglong_wait_until_all; }
    template <> inline auto wait_until_all<int32_t>() -> wait_until_all_type<int32_t> { return int32_wait_until_all; }
    template <> inline auto wait_until_all<int64_t>() -> wait_until_all_type<int64_t> { return int64_wait_until_all; }
    template <> inline auto wait_until_all<long long>() -> wait_until_all_type<long long> { return longlong_wait_until_all; }

    template <typename T> static constexpr wait_until_any_type<T> wait_until_any() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto wait_until_any<uint32_t>() -> wait_until_any_type<uint32_t> { return uint32_wait_until_any; }
    template <> inline auto wait_until_any<uint64_t>() -> wait_until_any_type<uint64_t> { return uint64_wait_until_any; }
    template <> inline auto wait_until_any<unsigned long long>() -> wait_until_any_type<unsigned long long> { return ulonglong_wait_until_any; }
    template <> inline auto wait_until_any<int32_t>() -> wait_until_any_type<int32_t> { return int32_wait_until_any; }
    template <> inline auto wait_until_any<int64_t>() -> wait_until_any_type<int64_t> { return int64_wait_until_any; }
    template <> inline auto wait_until_any<long long>() -> wait_until_any_type<long long> { return longlong_wait_until_any; }

    template <typename T> static constexpr wait_until_some_type<T> wait_until_some() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto wait_until_some<uint32_t>() -> wait_until_some_type<uint32_t> { return uint32_wait_until_some; }
    template <> inline auto wait_until_some<uint64_t>() -> wait_until_some_type<uint64_t> { return uint64_wait_until_some; }
    template <> inline auto wait_until_some<unsigned long long>() -> wait_until_some_type<unsigned long long> { return ulonglong_wait_until_some; }
    template <> inline auto wait_until_some<int32_t>() -> wait_until_some_type<int32_t> { return int32_wait_until_some; }
    template <> inline auto wait_until_some<int64_t>() -> wait_until_some_type<int64_t> { return int64_wait_until_some; }
    template <> inline auto wait_until_some<long long>() -> wait_until_some_type<long long> { return longlong_wait_until_some; }

    template <typename T> static constexpr test_all_vector_type<T> test_all_vector() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto test_all_vector<uint32_t>() -> test_all_vector_type<uint32_t> { return uint32_test_all_vector; }
    template <> inline auto test_all_vector<uint64_t>() -> test_all_vector_type<uint64_t> { return uint64_test_all_vector; }
    template <> inline auto test_all_vector<unsigned long long>() -> test_all_vector_type<unsigned long long> { return ulonglong_test_all_vector; }
    template <> inline auto test_all_vector<int32_t>() -> test_all_vector_type<int32_t> { return int32_test_all_vector; }
    template <> inline auto test_all_vector<int64_t>() -> test_all_vector_type<int64_t> { return int64_test_all_vector; }
    template <> inline auto test_all_vector<long long>() -> test_all_vector_type<long long> { return longlong_test_all_vector; }

    template <typename T> static constexpr test_any_vector_type<T> test_any_vector() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto test_any_vector<uint32_t>() -> test_any_vector_type<uint32_t> { return uint32_test_any_vector; }
    template <> inline auto test_any_vector<uint64_t>() -> test_any_vector_type<uint64_t> { return uint64_test_any_vector; }
    template <> inline auto test_any_vector<unsigned long long>() -> test_any_vector_type<unsigned long long> { return ulonglong_test_any_vector; }
    template <> inline auto test_any_vector<int32_t>() -> test_any_vector_type<int32_t> { return int32_test_any_vector; }
    template <> inline auto test_any_vector<int64_t>() -> test_any_vector_type<int64_t> { return int64_test_any_vector; }
    template <> inline auto test_any_vector<long long>() -> test_any_vector_type<long long> { return longlong_test_any_vector; }

    template <typename T> static constexpr test_some_vector_type<T> test_some_vector() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto test_some_vector<uint32_t>() -> test_some_vector_type<uint32_t> { return uint32_test_some_vector; }
    template <> inline auto test_some_vector<uint64_t>() -> test_some_vector_type<uint64_t> { return uint64_test_some_vector; }
    template <> inline auto test_some_vector<unsigned long long>() -> test_some_vector_type<unsigned long long> { return ulonglong_test_some_vector; }
    template <> inline auto test_some_vector<int32_t>() -> test_some_vector_type<int32_t> { return int32_test_some_vector; }
    template <> inline auto test_some_vector<int64_t>() -> test_some_vector_type<int64_t> { return int64_test_some_vector; }
    template <> inline auto test_some_vector<long long>() -> test_some_vector_type<long long> { return longlong_test_some_vector; }

    template <typename T> static constexpr wait_until_all_vector_type<T> wait_until_all_vector() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto wait_until_all_vector<uint32_t>() -> wait_until_all_vector_type<uint32_t> { return uint32_wait_until_all_vector; }
    template <> inline auto wait_until_all_vector<uint64_t>() -> wait_until_all_vector_type<uint64_t> { return uint64_wait_until_all_vector; }
    template <> inline auto wait_until_all_vector<unsigned long long>() -> wait_until_all_vector_type<unsigned long long> { return ulonglong_wait_until_all_vector; }
    template <> inline auto wait_until_all_vector<int32_t>() -> wait_until_all_vector_type<int32_t> { return int32_wait_until_all_vector; }
    template <> inline auto wait_until_all_vector<int64_t>() -> wait_until_all_vector_type<int64_t> { return int64_wait_until_all_vector; }
    template <> inline auto wait_until_all_vector<long long>() -> wait_until_all_vector_type<long long> { return longlong_wait_until_all_vector; }

    template <typename T> static constexpr wait_until_any_vector_type<T> wait_until_any_vector() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto wait_until_any_vector<uint32_t>() -> wait_until_any_vector_type<uint32_t> { return uint32_wait_until_any_vector; }
    template <> inline auto wait_until_any_vector<uint64_t>() -> wait_until_any_vector_type<uint64_t> { return uint64_wait_until_any_vector; }
    template <> inline auto wait_until_any_vector<unsigned long long>() -> wait_until_any_vector_type<unsigned long long> { return ulonglong_wait_until_any_vector; }
    template <> inline auto wait_until_any_vector<int32_t>() -> wait_until_any_vector_type<int32_t> { return int32_wait_until_any_vector; }
    template <> inline auto wait_until_any_vector<int64_t>() -> wait_until_any_vector_type<int64_t> { return int64_wait_until_any_vector; }
    template <> inline auto wait_until_any_vector<long long>() -> wait_until_any_vector_type<long long> { return longlong_wait_until_any_vector; }

    template <typename T> static constexpr wait_until_some_vector_type<T> wait_until_some_vector() { static_assert(assert_dependency<T>::value, "Undefined wrapper function"); }
    template <> inline auto wait_until_some_vector<uint32_t>() -> wait_until_some_vector_type<uint32_t> { return uint32_wait_until_some_vector; }
    template <> inline auto wait_until_some_vector<uint64_t>() -> wait_until_some_vector_type<uint64_t> { return uint64_wait_until_some_vector; }
    template <> inline auto wait_until_some_vector<unsigned long long>() -> wait_until_some_vector_type<unsigned long long> { return ulonglong_wait_until_some_vector; }
    template <> inline auto wait_until_some_vector<int32_t>() -> wait_until_some_vector_type<int32_t> { return int32_wait_until_some_vector; }
    template <> inline auto wait_until_some_vector<int64_t>() -> wait_until_some_vector_type<int64_t> { return int64_wait_until_some_vector; }
    template <> inline auto wait_until_some_vector<long long>() -> wait_until_some_vector_type<long long> { return longlong_wait_until_some_vector; }
    /* clang-format on */
}  // namespace ishmemi_openshmem_wrappers

#pragma GCC diagnostic pop

#endif
