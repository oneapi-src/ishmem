/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/err.h"
#include "wrapper.h"
#include "wrapper_openshmem.h"
#include "ishmem/env_utils.h"
#include <shmem.h>
#include <dlfcn.h>

namespace ishmemi_openshmem_wrappers {
    static bool initialized = false;

    shmem_team_t *TEAM_WORLD;
    shmem_team_t *TEAM_SHARED;
    shmem_team_t *TEAM_NODE;

    shmem_team_t SHMEM_TEAM_WORLD;
    shmem_team_t SHMEM_TEAM_SHARED;
    shmem_team_t SHMEMX_TEAM_NODE;

    void (*init)(void);
    void (*heap_preinit)(void);
    int (*heap_preinit_thread)(int, int *);
    void (*heap_create)(void *, size_t, int, int);
    void (*heap_postinit)(void);
    void (*finalize)(void);
    void (*global_exit)(int);
    int (*team_translate_pe)(shmem_team_t, int, shmem_team_t);
    int (*team_n_pes)(shmem_team_t);
    int (*team_my_pe)(shmem_team_t);
    int (*team_sync)(shmem_team_t);
    int (*team_split_strided)(shmem_team_t, int, int, int, const shmem_team_config_t *, long,
                              shmem_team_t *);
    int (*team_split_2d)(shmem_team_t, int, const shmem_team_config_t *, long, shmem_team_t *,
                         const shmem_team_config_t *, long, shmem_team_t *);
    void (*team_destroy)(shmem_team_t);
    int (*my_pe)(void);
    int (*n_pes)(void);
    void *(*malloc)(size_t);
    void *(*calloc)(size_t, size_t);
    void (*free)(void *);
    bool (*addr_accessible)(const void *, int);
    int (*runtime_get)(int, char *, void *, size_t);

    /* RMA */
    rma_type<uint8_t> uint8_put;
    irma_type<uint8_t> uint8_iput;
    irma_type<uint16_t> uint16_iput;
    irma_type<uint32_t> uint32_iput;
    irma_type<uint64_t> uint64_iput;
    irma_type<unsigned long long> ulonglong_iput;

    ibrma_type<uint8_t> uint8_ibput;
    ibrma_type<uint16_t> uint16_ibput;
    ibrma_type<uint32_t> uint32_ibput;
    ibrma_type<uint64_t> uint64_ibput;
    p_type<uint8_t> uint8_p;
    p_type<uint16_t> uint16_p;
    p_type<uint32_t> uint32_p;
    p_type<uint64_t> uint64_p;
    p_type<unsigned long long> ulonglong_p;
    p_type<float> float_p;
    p_type<double> double_p;
    rma_type<uint8_t> uint8_put_nbi;

    rma_type<uint8_t> uint8_get;
    irma_type<uint8_t> uint8_iget;
    irma_type<uint16_t> uint16_iget;
    irma_type<uint32_t> uint32_iget;
    irma_type<uint64_t> uint64_iget;
    irma_type<unsigned long long> ulonglong_iget;
    ibrma_type<uint8_t> uint8_ibget;
    ibrma_type<uint16_t> uint16_ibget;
    ibrma_type<uint32_t> uint32_ibget;
    ibrma_type<uint64_t> uint64_ibget;
    g_type<uint8_t> uint8_g;
    g_type<uint16_t> uint16_g;
    g_type<uint32_t> uint32_g;
    g_type<uint64_t> uint64_g;
    g_type<unsigned long long> ulonglong_g;
    g_type<float> float_g;
    g_type<double> double_g;
    rma_type<uint8_t> uint8_get_nbi;

    /* Non-blocking AMOs */
    atomic_fetch_nbi_type<uint32_t> uint32_atomic_fetch_nbi;
    atomic_fetch_nbi_type<int32_t> int32_atomic_fetch_nbi;
    atomic_fetch_nbi_type<uint64_t> uint64_atomic_fetch_nbi;
    atomic_fetch_nbi_type<int64_t> int64_atomic_fetch_nbi;
    atomic_fetch_nbi_type<unsigned long long> ulonglong_atomic_fetch_nbi;
    atomic_fetch_nbi_type<long long> longlong_atomic_fetch_nbi;
    atomic_fetch_nbi_type<float> float_atomic_fetch_nbi;
    atomic_fetch_nbi_type<double> double_atomic_fetch_nbi;

    atomic_compare_swap_nbi_type<uint32_t> uint32_atomic_compare_swap_nbi;
    atomic_compare_swap_nbi_type<int32_t> int32_atomic_compare_swap_nbi;
    atomic_compare_swap_nbi_type<uint64_t> uint64_atomic_compare_swap_nbi;
    atomic_compare_swap_nbi_type<int64_t> int64_atomic_compare_swap_nbi;
    atomic_compare_swap_nbi_type<unsigned long long> ulonglong_atomic_compare_swap_nbi;
    atomic_compare_swap_nbi_type<long long> longlong_atomic_compare_swap_nbi;

    atomic_swap_nbi_type<uint32_t> uint32_atomic_swap_nbi;
    atomic_swap_nbi_type<int32_t> int32_atomic_swap_nbi;
    atomic_swap_nbi_type<uint64_t> uint64_atomic_swap_nbi;
    atomic_swap_nbi_type<int64_t> int64_atomic_swap_nbi;
    atomic_swap_nbi_type<unsigned long long> ulonglong_atomic_swap_nbi;
    atomic_swap_nbi_type<long long> longlong_atomic_swap_nbi;
    atomic_swap_nbi_type<float> float_atomic_swap_nbi;
    atomic_swap_nbi_type<double> double_atomic_swap_nbi;

    atomic_fetch_inc_nbi_type<uint32_t> uint32_atomic_fetch_inc_nbi;
    atomic_fetch_inc_nbi_type<int32_t> int32_atomic_fetch_inc_nbi;
    atomic_fetch_inc_nbi_type<uint64_t> uint64_atomic_fetch_inc_nbi;
    atomic_fetch_inc_nbi_type<int64_t> int64_atomic_fetch_inc_nbi;
    atomic_fetch_inc_nbi_type<unsigned long long> ulonglong_atomic_fetch_inc_nbi;
    atomic_fetch_inc_nbi_type<long long> longlong_atomic_fetch_inc_nbi;

    atomic_fetch_op_nbi_type<uint32_t> uint32_atomic_fetch_add_nbi;
    atomic_fetch_op_nbi_type<int32_t> int32_atomic_fetch_add_nbi;
    atomic_fetch_op_nbi_type<uint64_t> uint64_atomic_fetch_add_nbi;
    atomic_fetch_op_nbi_type<int64_t> int64_atomic_fetch_add_nbi;
    atomic_fetch_op_nbi_type<unsigned long long> ulonglong_atomic_fetch_add_nbi;
    atomic_fetch_op_nbi_type<long long> longlong_atomic_fetch_add_nbi;

    atomic_fetch_op_nbi_type<uint32_t> uint32_atomic_fetch_and_nbi;
    atomic_fetch_op_nbi_type<int32_t> int32_atomic_fetch_and_nbi;
    atomic_fetch_op_nbi_type<uint64_t> uint64_atomic_fetch_and_nbi;
    atomic_fetch_op_nbi_type<int64_t> int64_atomic_fetch_and_nbi;
    atomic_fetch_op_nbi_type<unsigned long long> ulonglong_atomic_fetch_and_nbi;

    atomic_fetch_op_nbi_type<uint32_t> uint32_atomic_fetch_or_nbi;
    atomic_fetch_op_nbi_type<int32_t> int32_atomic_fetch_or_nbi;
    atomic_fetch_op_nbi_type<uint64_t> uint64_atomic_fetch_or_nbi;
    atomic_fetch_op_nbi_type<int64_t> int64_atomic_fetch_or_nbi;
    atomic_fetch_op_nbi_type<unsigned long long> ulonglong_atomic_fetch_or_nbi;

    atomic_fetch_op_nbi_type<uint32_t> uint32_atomic_fetch_xor_nbi;
    atomic_fetch_op_nbi_type<int32_t> int32_atomic_fetch_xor_nbi;
    atomic_fetch_op_nbi_type<uint64_t> uint64_atomic_fetch_xor_nbi;
    atomic_fetch_op_nbi_type<int64_t> int64_atomic_fetch_xor_nbi;
    atomic_fetch_op_nbi_type<unsigned long long> ulonglong_atomic_fetch_xor_nbi;

    /* AMO */
    atomic_fetch_type<uint32_t> uint32_atomic_fetch;
    atomic_set_type<uint32_t> uint32_atomic_set;
    atomic_compare_swap_type<uint32_t> uint32_atomic_compare_swap;
    atomic_swap_type<uint32_t> uint32_atomic_swap;
    atomic_fetch_inc_type<uint32_t> uint32_atomic_fetch_inc;
    atomic_inc_type<uint32_t> uint32_atomic_inc;
    atomic_fetch_op_type<uint32_t> uint32_atomic_fetch_add;
    atomic_op_type<uint32_t> uint32_atomic_add;
    atomic_fetch_op_type<uint32_t> uint32_atomic_fetch_and;
    atomic_op_type<uint32_t> uint32_atomic_and;
    atomic_fetch_op_type<uint32_t> uint32_atomic_fetch_or;
    atomic_op_type<uint32_t> uint32_atomic_or;
    atomic_fetch_op_type<uint32_t> uint32_atomic_fetch_xor;
    atomic_op_type<uint32_t> uint32_atomic_xor;

    atomic_fetch_type<uint64_t> uint64_atomic_fetch;
    atomic_set_type<uint64_t> uint64_atomic_set;
    atomic_compare_swap_type<uint64_t> uint64_atomic_compare_swap;
    atomic_swap_type<uint64_t> uint64_atomic_swap;
    atomic_fetch_inc_type<uint64_t> uint64_atomic_fetch_inc;
    atomic_inc_type<uint64_t> uint64_atomic_inc;
    atomic_fetch_op_type<uint64_t> uint64_atomic_fetch_add;
    atomic_op_type<uint64_t> uint64_atomic_add;
    atomic_fetch_op_type<uint64_t> uint64_atomic_fetch_and;
    atomic_op_type<uint64_t> uint64_atomic_and;
    atomic_fetch_op_type<uint64_t> uint64_atomic_fetch_or;
    atomic_op_type<uint64_t> uint64_atomic_or;
    atomic_fetch_op_type<uint64_t> uint64_atomic_fetch_xor;
    atomic_op_type<uint64_t> uint64_atomic_xor;

    atomic_fetch_type<unsigned long long> ulonglong_atomic_fetch;
    atomic_set_type<unsigned long long> ulonglong_atomic_set;
    atomic_compare_swap_type<unsigned long long> ulonglong_atomic_compare_swap;
    atomic_swap_type<unsigned long long> ulonglong_atomic_swap;
    atomic_fetch_inc_type<unsigned long long> ulonglong_atomic_fetch_inc;
    atomic_inc_type<unsigned long long> ulonglong_atomic_inc;
    atomic_fetch_op_type<unsigned long long> ulonglong_atomic_fetch_add;
    atomic_op_type<unsigned long long> ulonglong_atomic_add;
    atomic_fetch_op_type<unsigned long long> ulonglong_atomic_fetch_and;
    atomic_op_type<unsigned long long> ulonglong_atomic_and;
    atomic_fetch_op_type<unsigned long long> ulonglong_atomic_fetch_or;
    atomic_op_type<unsigned long long> ulonglong_atomic_or;
    atomic_fetch_op_type<unsigned long long> ulonglong_atomic_fetch_xor;
    atomic_op_type<unsigned long long> ulonglong_atomic_xor;

    atomic_fetch_type<int32_t> int32_atomic_fetch;
    atomic_set_type<int32_t> int32_atomic_set;
    atomic_compare_swap_type<int32_t> int32_atomic_compare_swap;
    atomic_swap_type<int32_t> int32_atomic_swap;
    atomic_fetch_inc_type<int32_t> int32_atomic_fetch_inc;
    atomic_inc_type<int32_t> int32_atomic_inc;
    atomic_fetch_op_type<int32_t> int32_atomic_fetch_add;
    atomic_op_type<int32_t> int32_atomic_add;
    atomic_fetch_op_type<int32_t> int32_atomic_fetch_and;
    atomic_op_type<int32_t> int32_atomic_and;
    atomic_fetch_op_type<int32_t> int32_atomic_fetch_or;
    atomic_op_type<int32_t> int32_atomic_or;
    atomic_fetch_op_type<int32_t> int32_atomic_fetch_xor;
    atomic_op_type<int32_t> int32_atomic_xor;

    atomic_fetch_type<int64_t> int64_atomic_fetch;
    atomic_set_type<int64_t> int64_atomic_set;
    atomic_compare_swap_type<int64_t> int64_atomic_compare_swap;
    atomic_swap_type<int64_t> int64_atomic_swap;
    atomic_fetch_inc_type<int64_t> int64_atomic_fetch_inc;
    atomic_inc_type<int64_t> int64_atomic_inc;
    atomic_fetch_op_type<int64_t> int64_atomic_fetch_add;
    atomic_op_type<int64_t> int64_atomic_add;
    atomic_fetch_op_type<int64_t> int64_atomic_fetch_and;
    atomic_op_type<int64_t> int64_atomic_and;
    atomic_fetch_op_type<int64_t> int64_atomic_fetch_or;
    atomic_op_type<int64_t> int64_atomic_or;
    atomic_fetch_op_type<int64_t> int64_atomic_fetch_xor;
    atomic_op_type<int64_t> int64_atomic_xor;

    atomic_fetch_type<long long> longlong_atomic_fetch;
    atomic_set_type<long long> longlong_atomic_set;
    atomic_compare_swap_type<long long> longlong_atomic_compare_swap;
    atomic_swap_type<long long> longlong_atomic_swap;
    atomic_fetch_inc_type<long long> longlong_atomic_fetch_inc;
    atomic_inc_type<long long> longlong_atomic_inc;
    atomic_fetch_op_type<long long> longlong_atomic_fetch_add;
    atomic_op_type<long long> longlong_atomic_add;

    atomic_fetch_type<float> float_atomic_fetch;
    atomic_set_type<float> float_atomic_set;
    atomic_swap_type<float> float_atomic_swap;

    atomic_fetch_type<double> double_atomic_fetch;
    atomic_set_type<double> double_atomic_set;
    atomic_swap_type<double> double_atomic_swap;

    /* Signaling */
    void (*uint8_put_signal)(uint8_t *, const uint8_t *, size_t, uint64_t *, uint64_t, int, int);
    void (*uint8_put_signal_nbi)(uint8_t *, const uint8_t *, size_t, uint64_t *, uint64_t, int,
                                 int);
    uint64_t (*signal_fetch)(const uint64_t *);

    /* Collectives */
    void (*barrier_all)(void);
    void (*sync_all)(void);
    int (*uint8_alltoall)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
    int (*uint8_broadcast)(shmem_team_t, uint8_t *, const uint8_t *, size_t, int);
    int (*uint8_collect)(shmem_team_t, uint8_t *, const uint8_t *, size_t);
    int (*uint8_fcollect)(shmem_team_t, uint8_t *, const uint8_t *, size_t);

    /* Reductions */
    reduce_type<unsigned char> uchar_and_reduce;
    reduce_type<int> int_max_reduce;

    reduce_type<uint8_t> uint8_and_reduce;
    reduce_type<uint8_t> uint8_or_reduce;
    reduce_type<uint8_t> uint8_xor_reduce;
    reduce_type<uint8_t> uint8_max_reduce;
    reduce_type<uint8_t> uint8_min_reduce;
    reduce_type<uint8_t> uint8_sum_reduce;
    reduce_type<uint8_t> uint8_prod_reduce;

    reduce_type<uint16_t> uint16_and_reduce;
    reduce_type<uint16_t> uint16_or_reduce;
    reduce_type<uint16_t> uint16_xor_reduce;
    reduce_type<uint16_t> uint16_max_reduce;
    reduce_type<uint16_t> uint16_min_reduce;
    reduce_type<uint16_t> uint16_sum_reduce;
    reduce_type<uint16_t> uint16_prod_reduce;

    reduce_type<uint32_t> uint32_and_reduce;
    reduce_type<uint32_t> uint32_or_reduce;
    reduce_type<uint32_t> uint32_xor_reduce;
    reduce_type<uint32_t> uint32_max_reduce;
    reduce_type<uint32_t> uint32_min_reduce;
    reduce_type<uint32_t> uint32_sum_reduce;
    reduce_type<uint32_t> uint32_prod_reduce;

    reduce_type<uint64_t> uint64_and_reduce;
    reduce_type<uint64_t> uint64_or_reduce;
    reduce_type<uint64_t> uint64_xor_reduce;
    reduce_type<uint64_t> uint64_max_reduce;
    reduce_type<uint64_t> uint64_min_reduce;
    reduce_type<uint64_t> uint64_sum_reduce;
    reduce_type<uint64_t> uint64_prod_reduce;

    reduce_type<unsigned long long> ulonglong_and_reduce;
    reduce_type<unsigned long long> ulonglong_or_reduce;
    reduce_type<unsigned long long> ulonglong_xor_reduce;
    reduce_type<unsigned long long> ulonglong_max_reduce;
    reduce_type<unsigned long long> ulonglong_min_reduce;
    reduce_type<unsigned long long> ulonglong_sum_reduce;
    reduce_type<unsigned long long> ulonglong_prod_reduce;

    reduce_type<int8_t> int8_and_reduce;
    reduce_type<int8_t> int8_or_reduce;
    reduce_type<int8_t> int8_xor_reduce;
    reduce_type<int8_t> int8_max_reduce;
    reduce_type<int8_t> int8_min_reduce;
    reduce_type<int8_t> int8_sum_reduce;
    reduce_type<int8_t> int8_prod_reduce;

    reduce_type<int16_t> int16_and_reduce;
    reduce_type<int16_t> int16_or_reduce;
    reduce_type<int16_t> int16_xor_reduce;
    reduce_type<int16_t> int16_max_reduce;
    reduce_type<int16_t> int16_min_reduce;
    reduce_type<int16_t> int16_sum_reduce;
    reduce_type<int16_t> int16_prod_reduce;

    reduce_type<int32_t> int32_and_reduce;
    reduce_type<int32_t> int32_or_reduce;
    reduce_type<int32_t> int32_xor_reduce;
    reduce_type<int32_t> int32_max_reduce;
    reduce_type<int32_t> int32_min_reduce;
    reduce_type<int32_t> int32_sum_reduce;
    reduce_type<int32_t> int32_prod_reduce;

    reduce_type<int64_t> int64_and_reduce;
    reduce_type<int64_t> int64_or_reduce;
    reduce_type<int64_t> int64_xor_reduce;
    reduce_type<int64_t> int64_max_reduce;
    reduce_type<int64_t> int64_min_reduce;
    reduce_type<int64_t> int64_sum_reduce;
    reduce_type<int64_t> int64_prod_reduce;

    reduce_type<long long> longlong_max_reduce;
    reduce_type<long long> longlong_min_reduce;
    reduce_type<long long> longlong_sum_reduce;
    reduce_type<long long> longlong_prod_reduce;

    reduce_type<float> float_max_reduce;
    reduce_type<float> float_min_reduce;
    reduce_type<float> float_sum_reduce;
    reduce_type<float> float_prod_reduce;

    reduce_type<double> double_max_reduce;
    reduce_type<double> double_min_reduce;
    reduce_type<double> double_sum_reduce;
    reduce_type<double> double_prod_reduce;

    /* Scan */
    bool inscan_exists;
    scan_type<uint8_t> uint8_sum_inscan;
    scan_type<uint16_t> uint16_sum_inscan;
    scan_type<uint32_t> uint32_sum_inscan;
    scan_type<uint64_t> uint64_sum_inscan;
    scan_type<unsigned long long> ulonglong_sum_inscan;
    scan_type<int8_t> int8_sum_inscan;
    scan_type<int16_t> int16_sum_inscan;
    scan_type<int32_t> int32_sum_inscan;
    scan_type<int64_t> int64_sum_inscan;
    scan_type<long long> longlong_sum_inscan;
    scan_type<float> float_sum_inscan;
    scan_type<double> double_sum_inscan;

    bool exscan_exists;
    scan_type<uint8_t> uint8_sum_exscan;
    scan_type<uint16_t> uint16_sum_exscan;
    scan_type<uint32_t> uint32_sum_exscan;
    scan_type<uint64_t> uint64_sum_exscan;
    scan_type<unsigned long long> ulonglong_sum_exscan;
    scan_type<int8_t> int8_sum_exscan;
    scan_type<int16_t> int16_sum_exscan;
    scan_type<int32_t> int32_sum_exscan;
    scan_type<int64_t> int64_sum_exscan;
    scan_type<long long> longlong_sum_exscan;
    scan_type<float> float_sum_exscan;
    scan_type<double> double_sum_exscan;

    /* Point-to-Point Synchronization */
    test_type<int32_t> int32_test;
    test_all_type<int32_t> int32_test_all;
    test_any_type<int32_t> int32_test_any;
    test_some_type<int32_t> int32_test_some;
    wait_until_type<int32_t> int32_wait_until;
    wait_until_all_type<int32_t> int32_wait_until_all;
    wait_until_any_type<int32_t> int32_wait_until_any;
    wait_until_some_type<int32_t> int32_wait_until_some;
    test_all_vector_type<int32_t> int32_test_all_vector;
    test_any_vector_type<int32_t> int32_test_any_vector;
    test_some_vector_type<int32_t> int32_test_some_vector;
    wait_until_all_vector_type<int32_t> int32_wait_until_all_vector;
    wait_until_any_vector_type<int32_t> int32_wait_until_any_vector;
    wait_until_some_vector_type<int32_t> int32_wait_until_some_vector;

    test_type<int64_t> int64_test;
    test_all_type<int64_t> int64_test_all;
    test_any_type<int64_t> int64_test_any;
    test_some_type<int64_t> int64_test_some;
    wait_until_type<int64_t> int64_wait_until;
    wait_until_all_type<int64_t> int64_wait_until_all;
    wait_until_any_type<int64_t> int64_wait_until_any;
    wait_until_some_type<int64_t> int64_wait_until_some;
    test_all_vector_type<int64_t> int64_test_all_vector;
    test_any_vector_type<int64_t> int64_test_any_vector;
    test_some_vector_type<int64_t> int64_test_some_vector;
    wait_until_all_vector_type<int64_t> int64_wait_until_all_vector;
    wait_until_any_vector_type<int64_t> int64_wait_until_any_vector;
    wait_until_some_vector_type<int64_t> int64_wait_until_some_vector;

    test_type<long long> longlong_test;
    test_all_type<long long> longlong_test_all;
    test_any_type<long long> longlong_test_any;
    test_some_type<long long> longlong_test_some;
    wait_until_type<long long> longlong_wait_until;
    wait_until_all_type<long long> longlong_wait_until_all;
    wait_until_any_type<long long> longlong_wait_until_any;
    wait_until_some_type<long long> longlong_wait_until_some;
    test_all_vector_type<long long> longlong_test_all_vector;
    test_any_vector_type<long long> longlong_test_any_vector;
    test_some_vector_type<long long> longlong_test_some_vector;
    wait_until_all_vector_type<long long> longlong_wait_until_all_vector;
    wait_until_any_vector_type<long long> longlong_wait_until_any_vector;
    wait_until_some_vector_type<long long> longlong_wait_until_some_vector;

    test_type<uint32_t> uint32_test;
    test_all_type<uint32_t> uint32_test_all;
    test_any_type<uint32_t> uint32_test_any;
    test_some_type<uint32_t> uint32_test_some;
    wait_until_type<uint32_t> uint32_wait_until;
    wait_until_all_type<uint32_t> uint32_wait_until_all;
    wait_until_any_type<uint32_t> uint32_wait_until_any;
    wait_until_some_type<uint32_t> uint32_wait_until_some;
    test_all_vector_type<uint32_t> uint32_test_all_vector;
    test_any_vector_type<uint32_t> uint32_test_any_vector;
    test_some_vector_type<uint32_t> uint32_test_some_vector;
    wait_until_all_vector_type<uint32_t> uint32_wait_until_all_vector;
    wait_until_any_vector_type<uint32_t> uint32_wait_until_any_vector;
    wait_until_some_vector_type<uint32_t> uint32_wait_until_some_vector;

    test_type<uint64_t> uint64_test;
    test_all_type<uint64_t> uint64_test_all;
    test_any_type<uint64_t> uint64_test_any;
    test_some_type<uint64_t> uint64_test_some;
    wait_until_type<uint64_t> uint64_wait_until;
    wait_until_all_type<uint64_t> uint64_wait_until_all;
    wait_until_any_type<uint64_t> uint64_wait_until_any;
    wait_until_some_type<uint64_t> uint64_wait_until_some;
    test_all_vector_type<uint64_t> uint64_test_all_vector;
    test_any_vector_type<uint64_t> uint64_test_any_vector;
    test_some_vector_type<uint64_t> uint64_test_some_vector;
    wait_until_all_vector_type<uint64_t> uint64_wait_until_all_vector;
    wait_until_any_vector_type<uint64_t> uint64_wait_until_any_vector;
    wait_until_some_vector_type<uint64_t> uint64_wait_until_some_vector;

    test_type<unsigned long long> ulonglong_test;
    test_all_type<unsigned long long> ulonglong_test_all;
    test_any_type<unsigned long long> ulonglong_test_any;
    test_some_type<unsigned long long> ulonglong_test_some;
    wait_until_type<unsigned long long> ulonglong_wait_until;
    wait_until_all_type<unsigned long long> ulonglong_wait_until_all;
    wait_until_any_type<unsigned long long> ulonglong_wait_until_any;
    wait_until_some_type<unsigned long long> ulonglong_wait_until_some;
    test_all_vector_type<unsigned long long> ulonglong_test_all_vector;
    test_any_vector_type<unsigned long long> ulonglong_test_any_vector;
    test_some_vector_type<unsigned long long> ulonglong_test_some_vector;
    wait_until_all_vector_type<unsigned long long> ulonglong_wait_until_all_vector;
    wait_until_any_vector_type<unsigned long long> ulonglong_wait_until_any_vector;
    wait_until_some_vector_type<unsigned long long> ulonglong_wait_until_some_vector;

    uint64_t (*signal_wait_until)(uint64_t *, int, uint64_t);

    /* Memory Ordering */
    void (*fence)(void);
    void (*quiet)(void);

    /* dl handle */
    void *shmem_handle = nullptr;
    std::vector<void **> wrapper_list;

    int fini_wrappers(void)
    {
        int ret = 0;
        for (auto p : wrapper_list)
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

    int init_wrappers(void)
    {
        int ret = 0;
        const char *shmem_libname = ishmemi_params.SHMEM_LIB_NAME.c_str();

        /* don't initialize twice */
        if (ishmemi_openshmem_wrappers::initialized) goto fn_exit;
        ishmemi_openshmem_wrappers::initialized = true;

        shmem_handle = dlopen(shmem_libname, RTLD_NOW | RTLD_GLOBAL);

        if (shmem_handle == nullptr) {
            RAISE_ERROR_MSG("could not find shmem library '%s' in environment, error %s\n",
                            shmem_libname, dlerror());
        }

        /* Load the SHMEM_TEAM symbols */
        ISHMEMI_LINK_SYMBOL(shmem_handle, SHMEM, TEAM_WORLD);
        ISHMEMI_LINK_SYMBOL(shmem_handle, SHMEM, TEAM_SHARED);
        ISHMEMI_LINK_SYMBOL(shmem_handle, SHMEMX, TEAM_NODE);

        SHMEM_TEAM_WORLD = *TEAM_WORLD;
        SHMEM_TEAM_SHARED = *TEAM_SHARED;
        SHMEMX_TEAM_NODE = *TEAM_NODE;

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
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, addr_accessible);

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
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, float_p);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, double_p);
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
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, float_g);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, double_g);
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

        /* Scan */
        inscan_exists = true;
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, uint8_sum_inscan, inscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, uint16_sum_inscan, inscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, uint32_sum_inscan, inscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, uint64_sum_inscan, inscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, ulonglong_sum_inscan, inscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, int8_sum_inscan, inscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, int16_sum_inscan, inscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, int32_sum_inscan, inscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, int64_sum_inscan, inscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, longlong_sum_inscan, inscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, float_sum_inscan, inscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, double_sum_inscan, inscan_exists);

        exscan_exists = true;
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, uint8_sum_exscan, exscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, uint16_sum_exscan, exscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, uint32_sum_exscan, exscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, uint64_sum_exscan, exscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, ulonglong_sum_exscan, exscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, int8_sum_exscan, exscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, int16_sum_exscan, exscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, int32_sum_exscan, exscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, int64_sum_exscan, exscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, longlong_sum_exscan, exscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, float_sum_exscan, exscan_exists);
        ISHMEMI_TRY_LINK_SYMBOL(shmem_handle, shmemx, double_sum_exscan, exscan_exists);

        /* Point-to-Point Synchronization */
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_test);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_test_all);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_test_any);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_test_some);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_wait_until);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_wait_until_all);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_wait_until_any);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_wait_until_some);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_test_all_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_test_any_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_test_some_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_wait_until_all_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_wait_until_any_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int32_wait_until_some_vector);

        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_test);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_test_all);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_test_any);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_test_some);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_wait_until);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_wait_until_all);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_wait_until_any);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_wait_until_some);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_test_all_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_test_any_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_test_some_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_wait_until_all_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_wait_until_any_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, int64_wait_until_some_vector);

        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_test);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_test_all);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_test_any);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_test_some);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_wait_until);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_wait_until_all);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_wait_until_any);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_wait_until_some);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_test_all_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_test_any_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_test_some_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_wait_until_all_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_wait_until_any_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, longlong_wait_until_some_vector);

        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_test);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_test_all);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_test_any);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_test_some);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_wait_until);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_wait_until_all);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_wait_until_any);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_wait_until_some);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_test_all_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_test_any_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_test_some_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_wait_until_all_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_wait_until_any_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint32_wait_until_some_vector);

        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_test);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_test_all);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_test_any);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_test_some);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_wait_until);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_wait_until_all);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_wait_until_any);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_wait_until_some);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_test_all_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_test_any_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_test_some_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_wait_until_all_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_wait_until_any_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, uint64_wait_until_some_vector);

        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_test);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_test_all);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_test_any);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_test_some);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_wait_until);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_wait_until_all);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_wait_until_any);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_wait_until_some);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_test_all_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_test_any_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_test_some_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_wait_until_all_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_wait_until_any_vector);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, ulonglong_wait_until_some_vector);

        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, signal_wait_until);

        /* Memory Ordering */
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, fence);
        ISHMEMI_LINK_SYMBOL(shmem_handle, shmem, quiet);

    fn_exit:
        return ret;
    }
}  // namespace ishmemi_openshmem_wrappers
