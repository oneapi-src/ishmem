/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* Wrappers to interface with OpenSHMEM runtime */
#include "ishmem_config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "runtime.h"
#include "wrapper.h"
#include "collectives.h"
#include "internal.h"
#include "impl_proxy.h"  // ISHMEMI_RUNTIME_REQUEST_HELPER

/* Operations that need multiple function pointers:
 * - p          (uint8, uint16, uint32, uint64, ulonglong)
 * - g          (uint8, uint16, uint32, uint64, ulonglong)
 * - reductions (uint8, uint16, uint32, uint64, ulonglong,
 *               int8, int16, int32, int64, longlong,
 *               float, double, longdouble)
 * - amos       (uint32, uint64, ulonglong,
 *               int32, int64, longlong,
 *               float, double, longdouble)
 * - test       (uint32, uint64, ulonglong)
 * - wait_until (uint32, uint64, ulonglong)
 */

/* Enabling SHMEMX_TEAM_NODE by default which provides a team that shares a compute node */
#define ISHMEMI_TEAM_NODE SHMEMX_TEAM_NODE

static int rank = -1;
static int size = 0;
static bool initialized_openshmem = false;

int ishmemi_runtime_openshmem_funcptr_fini();

int ishmemi_runtime_openshmem_fini(void)
{
    int ret = 0;

    if (initialized_openshmem) {
        shmem_WRAPPER_finalize();
        initialized_openshmem = false;
    }
    ishmemi_runtime_openshmem_funcptr_fini(); /* free function tables */
    ishmemi_openshmem_wrapper_fini();         /* close shared library */

    return ret;
}

void ishmemi_runtime_openshmem_abort(int exit_code, const char msg[])
{
    std::cerr << "[ABORT] " << msg << std::endl;
    shmem_WRAPPER_global_exit(exit_code);
}

int ishmemi_runtime_openshmem_get_rank(void)
{
    return rank;
}

int ishmemi_runtime_openshmem_get_size(void)
{
    return size;
}

int ishmemi_runtime_openshmem_get_node_rank(int pe)
{
    return shmem_WRAPPER_team_translate_pe(SHMEM_TEAM_WORLD, pe, ISHMEMI_TEAM_NODE);
}

int ishmemi_runtime_openshmem_get_node_size(void)
{
    return shmem_WRAPPER_team_n_pes(ISHMEMI_TEAM_NODE);
}

void ishmemi_runtime_openshmem_fence(void)
{
    // Ensure L0 operations are finished
    ishmemi_level_zero_sync();

    // Ensure all operations faciliated by OpenSHMEM backend are finished
    shmem_WRAPPER_fence();
}

void ishmemi_runtime_openshmem_quiet(void)
{
    // Ensure L0 operations are finished
    ishmemi_level_zero_sync();

    // Ensure all operations faciliated by OpenSHMEM backend are finished
    shmem_WRAPPER_quiet();
}

void ishmemi_runtime_openshmem_barrier(void)
{
    // Ensure L0 operations are finished
    ishmemi_level_zero_sync();

    // Ensure all operations faciliated by OpenSHMEM backend are finished
    shmem_WRAPPER_barrier_all();
}

void ishmemi_runtime_openshmem_sync(void)
{
    shmem_WRAPPER_sync_all();
}

void ishmemi_runtime_openshmem_node_barrier(void)
{
    // FIXME: No team barrier supported
    shmem_WRAPPER_team_sync(ISHMEMI_TEAM_NODE);
}

void ishmemi_runtime_openshmem_bcast(void *buf, size_t count, int root)
{
    shmem_WRAPPER_uint8_broadcast(SHMEM_TEAM_WORLD, (uint8_t *) buf, (uint8_t *) buf, count, root);
}

void ishmemi_runtime_openshmem_node_bcast(void *buf, size_t count, int root)
{
    shmem_WRAPPER_uint8_broadcast(ISHMEMI_TEAM_NODE, (uint8_t *) buf, (uint8_t *) buf, count, root);
}

void ishmemi_runtime_openshmem_node_fcollect(void *dst, void *src, size_t count)
{
    shmem_WRAPPER_uint8_fcollect(ISHMEMI_TEAM_NODE, (uint8_t *) dst, (uint8_t *) src, count);
}

void ishmemi_runtime_openshmem_fcollect(void *dst, void *src, size_t count)
{
    shmem_WRAPPER_uint8_fcollect(SHMEM_TEAM_WORLD, (uint8_t *) dst, (uint8_t *) src, count);
}

bool ishmemi_runtime_openshmem_is_local(int pe)
{
    return (ishmemi_runtime_openshmem_get_node_rank(pe) != -1);
}

int ishmemi_runtime_openshmem_get(int pe, char *key, void *value, size_t valuelen)
{
    return (shmem_WRAPPER_runtime_get(pe, key, value, valuelen));
}

/* RMA */
void ishmemi_openshmem_uint8_put(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    shmem_WRAPPER_uint8_put(dest, src, nelems, pe);
}

void ishmemi_openshmem_uint8_iput(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    shmem_WRAPPER_uint8_iput(dest, src, dst, sst, nelems, pe);
}

void ishmemi_openshmem_uint8_p(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    shmem_WRAPPER_uint8_p(dest, val, pe);
}

void ishmemi_openshmem_uint16_p(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    shmem_WRAPPER_uint16_p(dest, val, pe);
}

void ishmemi_openshmem_uint32_p(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    shmem_WRAPPER_uint32_p(dest, val, pe);
}

void ishmemi_openshmem_uint64_p(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    shmem_WRAPPER_uint64_p(dest, val, pe);
}

void ishmemi_openshmem_ulonglong_p(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    shmem_WRAPPER_ulonglong_p(dest, val, pe);
}

void ishmemi_openshmem_uint8_put_nbi(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    shmem_WRAPPER_uint8_put_nbi(dest, src, nelems, pe);
}

void ishmemi_openshmem_uint8_get(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    shmem_WRAPPER_uint8_get(dest, src, nelems, pe);
}

void ishmemi_openshmem_uint8_iget(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    shmem_WRAPPER_uint8_iget(dest, src, dst, sst, nelems, pe);
}

void ishmemi_openshmem_uint8_g(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.ui8 = shmem_WRAPPER_uint8_g(src, pe);
}

void ishmemi_openshmem_uint16_g(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.ui16 = shmem_WRAPPER_uint16_g(src, pe);
}

void ishmemi_openshmem_uint32_g(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.ui32 = shmem_WRAPPER_uint32_g(src, pe);
}

void ishmemi_openshmem_uint64_g(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.ui64 = shmem_WRAPPER_uint64_g(src, pe);
}

void ishmemi_openshmem_ulonglong_g(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.ull = shmem_WRAPPER_ulonglong_g(src, pe);
}

void ishmemi_openshmem_uint8_get_nbi(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    shmem_WRAPPER_uint8_get_nbi(dest, src, nelems, pe);
}

/* AMO */
void ishmemi_openshmem_uint32_atomic_fetch(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.ui32 = shmem_WRAPPER_uint32_atomic_fetch(src, pe);
}

void ishmemi_openshmem_uint32_atomic_set(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    shmem_WRAPPER_uint32_atomic_set(dest, val, pe);
}

void ishmemi_openshmem_uint32_atomic_compare_swap(ishmemi_request_t *msg,
                                                  ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.ui32 = shmem_WRAPPER_uint32_atomic_compare_swap(dest, cond, val, pe);
}

void ishmemi_openshmem_uint32_atomic_swap(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.ui32 = shmem_WRAPPER_uint32_atomic_swap(dest, val, pe);
}

void ishmemi_openshmem_uint32_atomic_fetch_inc(ishmemi_request_t *msg,
                                               ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.ui32 = shmem_WRAPPER_uint32_atomic_fetch_inc(dest, pe);
}

void ishmemi_openshmem_uint32_atomic_inc(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    shmem_WRAPPER_uint32_atomic_inc(dest, pe);
}

void ishmemi_openshmem_uint32_atomic_fetch_add(ishmemi_request_t *msg,
                                               ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.ui32 = shmem_WRAPPER_uint32_atomic_fetch_add(dest, val, pe);
}

void ishmemi_openshmem_uint32_atomic_add(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    shmem_WRAPPER_uint32_atomic_add(dest, val, pe);
}

void ishmemi_openshmem_uint32_atomic_fetch_and(ishmemi_request_t *msg,
                                               ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.ui32 = shmem_WRAPPER_uint32_atomic_fetch_and(dest, val, pe);
}

void ishmemi_openshmem_uint32_atomic_and(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    shmem_WRAPPER_uint32_atomic_and(dest, val, pe);
}

void ishmemi_openshmem_uint32_atomic_fetch_or(ishmemi_request_t *msg,
                                              ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.ui32 = shmem_WRAPPER_uint32_atomic_fetch_or(dest, val, pe);
}

void ishmemi_openshmem_uint32_atomic_or(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    shmem_WRAPPER_uint32_atomic_or(dest, val, pe);
}

void ishmemi_openshmem_uint32_atomic_fetch_xor(ishmemi_request_t *msg,
                                               ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.ui32 = shmem_WRAPPER_uint32_atomic_fetch_xor(dest, val, pe);
}

void ishmemi_openshmem_uint32_atomic_xor(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    shmem_WRAPPER_uint32_atomic_xor(dest, val, pe);
}

void ishmemi_openshmem_uint64_atomic_fetch(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.ui64 = shmem_WRAPPER_uint64_atomic_fetch(src, pe);
}

void ishmemi_openshmem_uint64_atomic_set(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    shmem_WRAPPER_uint64_atomic_set(dest, val, pe);
}

void ishmemi_openshmem_uint64_atomic_compare_swap(ishmemi_request_t *msg,
                                                  ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.ui64 = shmem_WRAPPER_uint64_atomic_compare_swap(dest, cond, val, pe);
}

void ishmemi_openshmem_uint64_atomic_swap(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.ui64 = shmem_WRAPPER_uint64_atomic_swap(dest, val, pe);
}

void ishmemi_openshmem_uint64_atomic_fetch_inc(ishmemi_request_t *msg,
                                               ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.ui64 = shmem_WRAPPER_uint64_atomic_fetch_inc(dest, pe);
}

void ishmemi_openshmem_uint64_atomic_inc(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    shmem_WRAPPER_uint64_atomic_inc(dest, pe);
}

void ishmemi_openshmem_uint64_atomic_fetch_add(ishmemi_request_t *msg,
                                               ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.ui64 = shmem_WRAPPER_uint64_atomic_fetch_add(dest, val, pe);
}

void ishmemi_openshmem_uint64_atomic_add(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    shmem_WRAPPER_uint64_atomic_add(dest, val, pe);
}

void ishmemi_openshmem_uint64_atomic_fetch_and(ishmemi_request_t *msg,
                                               ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.ui64 = shmem_WRAPPER_uint64_atomic_fetch_and(dest, val, pe);
}

void ishmemi_openshmem_uint64_atomic_and(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    shmem_WRAPPER_uint64_atomic_and(dest, val, pe);
}

void ishmemi_openshmem_uint64_atomic_fetch_or(ishmemi_request_t *msg,
                                              ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.ui64 = shmem_WRAPPER_uint64_atomic_fetch_or(dest, val, pe);
}

void ishmemi_openshmem_uint64_atomic_or(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    shmem_WRAPPER_uint64_atomic_or(dest, val, pe);
}

void ishmemi_openshmem_uint64_atomic_fetch_xor(ishmemi_request_t *msg,
                                               ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.ui64 = shmem_WRAPPER_uint64_atomic_fetch_xor(dest, val, pe);
}

void ishmemi_openshmem_uint64_atomic_xor(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    shmem_WRAPPER_uint64_atomic_xor(dest, val, pe);
}

void ishmemi_openshmem_ulonglong_atomic_fetch(ishmemi_request_t *msg,
                                              ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.ull = shmem_WRAPPER_ulonglong_atomic_fetch(src, pe);
}

void ishmemi_openshmem_ulonglong_atomic_set(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    shmem_WRAPPER_ulonglong_atomic_set(dest, val, pe);
}

void ishmemi_openshmem_ulonglong_atomic_compare_swap(ishmemi_request_t *msg,
                                                     ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.ull = shmem_WRAPPER_ulonglong_atomic_compare_swap(dest, cond, val, pe);
}

void ishmemi_openshmem_ulonglong_atomic_swap(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.ull = shmem_WRAPPER_ulonglong_atomic_swap(dest, val, pe);
}

void ishmemi_openshmem_ulonglong_atomic_fetch_inc(ishmemi_request_t *msg,
                                                  ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.ull = shmem_WRAPPER_ulonglong_atomic_fetch_inc(dest, pe);
}

void ishmemi_openshmem_ulonglong_atomic_inc(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    shmem_WRAPPER_ulonglong_atomic_inc(dest, pe);
}

void ishmemi_openshmem_ulonglong_atomic_fetch_add(ishmemi_request_t *msg,
                                                  ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.ull = shmem_WRAPPER_ulonglong_atomic_fetch_add(dest, val, pe);
}

void ishmemi_openshmem_ulonglong_atomic_add(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    shmem_WRAPPER_ulonglong_atomic_add(dest, val, pe);
}

void ishmemi_openshmem_ulonglong_atomic_fetch_and(ishmemi_request_t *msg,
                                                  ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.ull = shmem_WRAPPER_ulonglong_atomic_fetch_and(dest, val, pe);
}

void ishmemi_openshmem_ulonglong_atomic_and(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    shmem_WRAPPER_ulonglong_atomic_and(dest, val, pe);
}

void ishmemi_openshmem_ulonglong_atomic_fetch_or(ishmemi_request_t *msg,
                                                 ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.ull = shmem_WRAPPER_ulonglong_atomic_fetch_or(dest, val, pe);
}

void ishmemi_openshmem_ulonglong_atomic_or(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    shmem_WRAPPER_ulonglong_atomic_or(dest, val, pe);
}

void ishmemi_openshmem_ulonglong_atomic_fetch_xor(ishmemi_request_t *msg,
                                                  ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.ull = shmem_WRAPPER_ulonglong_atomic_fetch_xor(dest, val, pe);
}

void ishmemi_openshmem_ulonglong_atomic_xor(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    shmem_WRAPPER_ulonglong_atomic_xor(dest, val, pe);
}

void ishmemi_openshmem_int32_atomic_fetch(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i32 = shmem_WRAPPER_int32_atomic_fetch(src, pe);
}

void ishmemi_openshmem_int32_atomic_set(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    shmem_WRAPPER_int32_atomic_set(dest, val, pe);
}

void ishmemi_openshmem_int32_atomic_compare_swap(ishmemi_request_t *msg,
                                                 ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i32 = shmem_WRAPPER_int32_atomic_compare_swap(dest, cond, val, pe);
}

void ishmemi_openshmem_int32_atomic_swap(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i32 = shmem_WRAPPER_int32_atomic_swap(dest, val, pe);
}

void ishmemi_openshmem_int32_atomic_fetch_inc(ishmemi_request_t *msg,
                                              ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i32 = shmem_WRAPPER_int32_atomic_fetch_inc(dest, pe);
}

void ishmemi_openshmem_int32_atomic_inc(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    shmem_WRAPPER_int32_atomic_inc(dest, pe);
}

void ishmemi_openshmem_int32_atomic_fetch_add(ishmemi_request_t *msg,
                                              ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i32 = shmem_WRAPPER_int32_atomic_fetch_add(dest, val, pe);
}

void ishmemi_openshmem_int32_atomic_add(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    shmem_WRAPPER_int32_atomic_add(dest, val, pe);
}

void ishmemi_openshmem_int32_atomic_fetch_and(ishmemi_request_t *msg,
                                              ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i32 = shmem_WRAPPER_int32_atomic_fetch_and(dest, val, pe);
}

void ishmemi_openshmem_int32_atomic_and(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    shmem_WRAPPER_int32_atomic_and(dest, val, pe);
}

void ishmemi_openshmem_int32_atomic_fetch_or(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i32 = shmem_WRAPPER_int32_atomic_fetch_or(dest, val, pe);
}

void ishmemi_openshmem_int32_atomic_or(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    shmem_WRAPPER_int32_atomic_or(dest, val, pe);
}

void ishmemi_openshmem_int32_atomic_fetch_xor(ishmemi_request_t *msg,
                                              ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i32 = shmem_WRAPPER_int32_atomic_fetch_xor(dest, val, pe);
}

void ishmemi_openshmem_int32_atomic_xor(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    shmem_WRAPPER_int32_atomic_xor(dest, val, pe);
}

void ishmemi_openshmem_int64_atomic_fetch(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i64 = shmem_WRAPPER_int64_atomic_fetch(src, pe);
}

void ishmemi_openshmem_int64_atomic_set(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    shmem_WRAPPER_int64_atomic_set(dest, val, pe);
}

void ishmemi_openshmem_int64_atomic_compare_swap(ishmemi_request_t *msg,
                                                 ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i64 = shmem_WRAPPER_int64_atomic_compare_swap(dest, cond, val, pe);
}

void ishmemi_openshmem_int64_atomic_swap(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i64 = shmem_WRAPPER_int64_atomic_swap(dest, val, pe);
}

void ishmemi_openshmem_int64_atomic_fetch_inc(ishmemi_request_t *msg,
                                              ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i64 = shmem_WRAPPER_int64_atomic_fetch_inc(dest, pe);
}

void ishmemi_openshmem_int64_atomic_inc(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    shmem_WRAPPER_int64_atomic_inc(dest, pe);
}

void ishmemi_openshmem_int64_atomic_fetch_add(ishmemi_request_t *msg,
                                              ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i64 = shmem_WRAPPER_int64_atomic_fetch_add(dest, val, pe);
}

void ishmemi_openshmem_int64_atomic_add(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    shmem_WRAPPER_int64_atomic_add(dest, val, pe);
}

void ishmemi_openshmem_int64_atomic_fetch_and(ishmemi_request_t *msg,
                                              ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i64 = shmem_WRAPPER_int64_atomic_fetch_and(dest, val, pe);
}

void ishmemi_openshmem_int64_atomic_and(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    shmem_WRAPPER_int64_atomic_and(dest, val, pe);
}

void ishmemi_openshmem_int64_atomic_fetch_or(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i64 = shmem_WRAPPER_int64_atomic_fetch_or(dest, val, pe);
}

void ishmemi_openshmem_int64_atomic_or(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    shmem_WRAPPER_int64_atomic_or(dest, val, pe);
}

void ishmemi_openshmem_int64_atomic_fetch_xor(ishmemi_request_t *msg,
                                              ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i64 = shmem_WRAPPER_int64_atomic_fetch_xor(dest, val, pe);
}

void ishmemi_openshmem_int64_atomic_xor(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    shmem_WRAPPER_int64_atomic_xor(dest, val, pe);
}

void ishmemi_openshmem_longlong_atomic_fetch(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.ll = shmem_WRAPPER_longlong_atomic_fetch(src, pe);
}

void ishmemi_openshmem_longlong_atomic_set(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    shmem_WRAPPER_longlong_atomic_set(dest, val, pe);
}

void ishmemi_openshmem_longlong_atomic_compare_swap(ishmemi_request_t *msg,
                                                    ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.ll = shmem_WRAPPER_longlong_atomic_compare_swap(dest, cond, val, pe);
}

void ishmemi_openshmem_longlong_atomic_swap(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.ll = shmem_WRAPPER_longlong_atomic_swap(dest, val, pe);
}

void ishmemi_openshmem_longlong_atomic_fetch_inc(ishmemi_request_t *msg,
                                                 ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.ll = shmem_WRAPPER_longlong_atomic_fetch_inc(dest, pe);
}

void ishmemi_openshmem_longlong_atomic_inc(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    shmem_WRAPPER_longlong_atomic_inc(dest, pe);
}

void ishmemi_openshmem_longlong_atomic_fetch_add(ishmemi_request_t *msg,
                                                 ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.ll = shmem_WRAPPER_longlong_atomic_fetch_add(dest, val, pe);
}

void ishmemi_openshmem_longlong_atomic_add(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    shmem_WRAPPER_longlong_atomic_add(dest, val, pe);
}

void ishmemi_openshmem_float_atomic_fetch(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    comp->completion.ret.f = shmem_WRAPPER_float_atomic_fetch(src, pe);
}

void ishmemi_openshmem_float_atomic_set(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    shmem_WRAPPER_float_atomic_set(dest, val, pe);
}

void ishmemi_openshmem_float_atomic_swap(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    comp->completion.ret.f = shmem_WRAPPER_float_atomic_swap(dest, val, pe);
}

void ishmemi_openshmem_double_atomic_fetch(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    comp->completion.ret.ld = shmem_WRAPPER_double_atomic_fetch(src, pe);
}

void ishmemi_openshmem_double_atomic_set(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    shmem_WRAPPER_double_atomic_set(dest, val, pe);
}

void ishmemi_openshmem_double_atomic_swap(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    comp->completion.ret.ld = shmem_WRAPPER_double_atomic_swap(dest, val, pe);
}

/* Signaling */
void ishmemi_openshmem_uint8_put_signal(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    shmem_WRAPPER_uint8_put_signal(dest, src, nelems, sig_addr, signal, sig_op, pe);
}

void ishmemi_openshmem_uint8_put_signal_nbi(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    shmem_WRAPPER_uint8_put_signal_nbi(dest, src, nelems, sig_addr, signal, sig_op, pe);
}

void ishmemi_openshmem_signal_fetch(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.ui64 = shmem_WRAPPER_signal_fetch(sig_addr);
}

/* Collectives */
void ishmemi_openshmem_barrier_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ishmemi_runtime_openshmem_barrier();
}

void ishmemi_openshmem_sync_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ishmemi_runtime_openshmem_sync();
}

void ishmemi_openshmem_uint8_alltoall(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_alltoall(team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_broadcast(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_broadcast(team, dest, src, nelems, root);
}

void ishmemi_openshmem_uint8_collect(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_collect(team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_fcollect(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_fcollect(team, dest, src, nelems);
}

/* Reductions */
void ishmemi_openshmem_uint8_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_and_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_or_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_xor_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_max_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_min_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_sum_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_prod_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint16_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = shmem_WRAPPER_uint16_and_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint16_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = shmem_WRAPPER_uint16_or_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint16_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = shmem_WRAPPER_uint16_xor_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint16_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = shmem_WRAPPER_uint16_max_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint16_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = shmem_WRAPPER_uint16_min_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint16_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = shmem_WRAPPER_uint16_sum_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint16_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = shmem_WRAPPER_uint16_prod_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint32_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_and_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint32_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_or_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint32_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_xor_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint32_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_max_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint32_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_min_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint32_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_sum_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint32_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_prod_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint64_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_and_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint64_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_or_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint64_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_xor_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint64_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_max_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint64_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_min_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint64_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_sum_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_uint64_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_prod_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_ulonglong_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = shmem_WRAPPER_ulonglong_and_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_ulonglong_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = shmem_WRAPPER_ulonglong_or_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_ulonglong_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = shmem_WRAPPER_ulonglong_xor_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_ulonglong_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = shmem_WRAPPER_ulonglong_max_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_ulonglong_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = shmem_WRAPPER_ulonglong_min_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_ulonglong_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = shmem_WRAPPER_ulonglong_sum_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_ulonglong_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = shmem_WRAPPER_ulonglong_prod_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int8_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = shmem_WRAPPER_int8_and_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int8_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = shmem_WRAPPER_int8_or_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int8_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = shmem_WRAPPER_int8_xor_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int8_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = shmem_WRAPPER_int8_max_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int8_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = shmem_WRAPPER_int8_min_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int8_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = shmem_WRAPPER_int8_sum_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int8_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = shmem_WRAPPER_int8_prod_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int16_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = shmem_WRAPPER_int16_and_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int16_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = shmem_WRAPPER_int16_or_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int16_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = shmem_WRAPPER_int16_xor_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int16_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = shmem_WRAPPER_int16_max_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int16_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = shmem_WRAPPER_int16_min_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int16_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = shmem_WRAPPER_int16_sum_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int16_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = shmem_WRAPPER_int16_prod_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int32_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = shmem_WRAPPER_int32_and_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int32_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = shmem_WRAPPER_int32_or_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int32_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = shmem_WRAPPER_int32_xor_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int32_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = shmem_WRAPPER_int32_max_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int32_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = shmem_WRAPPER_int32_min_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int32_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = shmem_WRAPPER_int32_sum_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int32_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = shmem_WRAPPER_int32_prod_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int64_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = shmem_WRAPPER_int64_and_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int64_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = shmem_WRAPPER_int64_or_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int64_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = shmem_WRAPPER_int64_xor_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int64_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = shmem_WRAPPER_int64_max_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int64_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = shmem_WRAPPER_int64_min_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int64_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = shmem_WRAPPER_int64_sum_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_int64_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = shmem_WRAPPER_int64_prod_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_longlong_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i = shmem_WRAPPER_longlong_and_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_longlong_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i = shmem_WRAPPER_longlong_or_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_longlong_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i = shmem_WRAPPER_longlong_xor_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_longlong_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i = shmem_WRAPPER_longlong_max_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_longlong_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i = shmem_WRAPPER_longlong_min_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_longlong_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i = shmem_WRAPPER_longlong_sum_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_longlong_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i = shmem_WRAPPER_longlong_prod_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_float_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    comp->completion.ret.i = shmem_WRAPPER_float_max_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_float_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    comp->completion.ret.i = shmem_WRAPPER_float_min_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_float_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    comp->completion.ret.i = shmem_WRAPPER_float_sum_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_float_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    comp->completion.ret.i = shmem_WRAPPER_float_prod_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_double_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    comp->completion.ret.i = shmem_WRAPPER_double_max_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_double_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    comp->completion.ret.i = shmem_WRAPPER_double_min_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_double_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    comp->completion.ret.i = shmem_WRAPPER_double_sum_reduce(team, dest, src, nelems);
}

void ishmemi_openshmem_double_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    comp->completion.ret.i = shmem_WRAPPER_double_prod_reduce(team, dest, src, nelems);
}

/* Point-to-Point Synchronization */
void ishmemi_openshmem_uint32_test(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_test(dest, cmp, cmp_value);
}

void ishmemi_openshmem_uint32_wait_until(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    shmem_WRAPPER_uint32_wait_until(dest, cmp, cmp_value);
}

void ishmemi_openshmem_uint64_test(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_test(dest, cmp, cmp_value);
}

void ishmemi_openshmem_uint64_wait_until(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    shmem_WRAPPER_uint64_wait_until(dest, cmp, cmp_value);
}

void ishmemi_openshmem_ulonglong_test(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = shmem_WRAPPER_ulonglong_test(dest, cmp, cmp_value);
}

void ishmemi_openshmem_ulonglong_wait_until(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    shmem_WRAPPER_ulonglong_wait_until(dest, cmp, cmp_value);
}

/* Memory Ordering */
void ishmemi_openshmem_fence(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ishmemi_runtime_openshmem_fence();
}

void ishmemi_openshmem_quiet(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ishmemi_runtime_openshmem_quiet();
}

void ishmemi_runtime_openshmem_unsupported(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEM_ERROR_MSG("Encountered type '%s' unsupported for operation '%s'\n",
                     ishmemi_type_str[msg->type], ishmemi_op_str[msg->op]);
    ishmemi_cpu_info->proxy_state = EXIT;
}

int ishmemi_runtime_openshmem_funcptr_fini()
{
    for (int i = 0; i < ISHMEMI_OP_END; ++i) {
        for (int j = 0; j < ishmemi_runtime_proxy_func_num_types; ++j) {
            ishmemi_proxy_funcs[i][j] = ishmemi_runtime_openshmem_unsupported;
        }
        ISHMEMI_FREE(free, ishmemi_proxy_funcs[i]);
    }
    ISHMEMI_FREE(free, ishmemi_proxy_funcs);
    return (0);
}

void ishmemi_runtime_openshmem_funcptr_init()
{
    ishmemi_proxy_funcs = (ishmemi_runtime_proxy_func_t **) malloc(
        sizeof(ishmemi_runtime_proxy_func_t *) * ISHMEMI_OP_END);
    ISHMEM_CHECK_GOTO_MSG(ishmemi_proxy_funcs == nullptr, fn_exit,
                          "Allocation of ishmemi_proxy_funcs failed\n");

    /* Initialize every function with the "unsupported op" function */
    /* Note: KILL operation is covered inside the proxy directly - it is the same for all backends
     * currently */
    for (int i = 0; i < ISHMEMI_OP_END; ++i) {
        ishmemi_proxy_funcs[i] = (ishmemi_runtime_proxy_func_t *) malloc(
            sizeof(ishmemi_runtime_proxy_func_t) * ishmemi_runtime_proxy_func_num_types);
        for (int j = 0; j < ishmemi_runtime_proxy_func_num_types; ++j) {
            ishmemi_proxy_funcs[i][j] = ishmemi_runtime_openshmem_unsupported;
        }
    }

    /* Fill in the supported functions */
    /* RMA */
    ishmemi_proxy_funcs[PUT][UINT8] = ishmemi_openshmem_uint8_put;
    ishmemi_proxy_funcs[IPUT][UINT8] = ishmemi_openshmem_uint8_iput;
    ishmemi_proxy_funcs[P][UINT8] = ishmemi_openshmem_uint8_p;
    ishmemi_proxy_funcs[P][UINT16] = ishmemi_openshmem_uint16_p;
    ishmemi_proxy_funcs[P][UINT32] = ishmemi_openshmem_uint32_p;
    ishmemi_proxy_funcs[P][UINT64] = ishmemi_openshmem_uint64_p;
    ishmemi_proxy_funcs[P][ULONGLONG] = ishmemi_openshmem_ulonglong_p;
    ishmemi_proxy_funcs[PUT_NBI][UINT8] = ishmemi_openshmem_uint8_put_nbi;

    ishmemi_proxy_funcs[GET][UINT8] = ishmemi_openshmem_uint8_get;
    ishmemi_proxy_funcs[IGET][UINT8] = ishmemi_openshmem_uint8_iget;
    ishmemi_proxy_funcs[G][UINT8] = ishmemi_openshmem_uint8_g;
    ishmemi_proxy_funcs[G][UINT16] = ishmemi_openshmem_uint16_g;
    ishmemi_proxy_funcs[G][UINT32] = ishmemi_openshmem_uint32_g;
    ishmemi_proxy_funcs[G][UINT64] = ishmemi_openshmem_uint64_g;
    ishmemi_proxy_funcs[G][ULONGLONG] = ishmemi_openshmem_ulonglong_g;
    ishmemi_proxy_funcs[GET_NBI][UINT8] = ishmemi_openshmem_uint8_get_nbi;

    /* AMO */
    ishmemi_proxy_funcs[AMO_FETCH][UINT32] = ishmemi_openshmem_uint32_atomic_fetch;
    ishmemi_proxy_funcs[AMO_SET][UINT32] = ishmemi_openshmem_uint32_atomic_set;
    ishmemi_proxy_funcs[AMO_COMPARE_SWAP][UINT32] = ishmemi_openshmem_uint32_atomic_compare_swap;
    ishmemi_proxy_funcs[AMO_SWAP][UINT32] = ishmemi_openshmem_uint32_atomic_swap;
    ishmemi_proxy_funcs[AMO_FETCH_INC][UINT32] = ishmemi_openshmem_uint32_atomic_fetch_inc;
    ishmemi_proxy_funcs[AMO_INC][UINT32] = ishmemi_openshmem_uint32_atomic_inc;
    ishmemi_proxy_funcs[AMO_FETCH_ADD][UINT32] = ishmemi_openshmem_uint32_atomic_fetch_add;
    ishmemi_proxy_funcs[AMO_ADD][UINT32] = ishmemi_openshmem_uint32_atomic_add;
    ishmemi_proxy_funcs[AMO_FETCH_AND][UINT32] = ishmemi_openshmem_uint32_atomic_fetch_and;
    ishmemi_proxy_funcs[AMO_AND][UINT32] = ishmemi_openshmem_uint32_atomic_and;
    ishmemi_proxy_funcs[AMO_FETCH_OR][UINT32] = ishmemi_openshmem_uint32_atomic_fetch_or;
    ishmemi_proxy_funcs[AMO_OR][UINT32] = ishmemi_openshmem_uint32_atomic_or;
    ishmemi_proxy_funcs[AMO_FETCH_XOR][UINT32] = ishmemi_openshmem_uint32_atomic_fetch_xor;
    ishmemi_proxy_funcs[AMO_XOR][UINT32] = ishmemi_openshmem_uint32_atomic_xor;

    ishmemi_proxy_funcs[AMO_FETCH][UINT64] = ishmemi_openshmem_uint64_atomic_fetch;
    ishmemi_proxy_funcs[AMO_SET][UINT64] = ishmemi_openshmem_uint64_atomic_set;
    ishmemi_proxy_funcs[AMO_COMPARE_SWAP][UINT64] = ishmemi_openshmem_uint64_atomic_compare_swap;
    ishmemi_proxy_funcs[AMO_SWAP][UINT64] = ishmemi_openshmem_uint64_atomic_swap;
    ishmemi_proxy_funcs[AMO_FETCH_INC][UINT64] = ishmemi_openshmem_uint64_atomic_fetch_inc;
    ishmemi_proxy_funcs[AMO_INC][UINT64] = ishmemi_openshmem_uint64_atomic_inc;
    ishmemi_proxy_funcs[AMO_FETCH_ADD][UINT64] = ishmemi_openshmem_uint64_atomic_fetch_add;
    ishmemi_proxy_funcs[AMO_ADD][UINT64] = ishmemi_openshmem_uint64_atomic_add;
    ishmemi_proxy_funcs[AMO_FETCH_AND][UINT64] = ishmemi_openshmem_uint64_atomic_fetch_and;
    ishmemi_proxy_funcs[AMO_AND][UINT64] = ishmemi_openshmem_uint64_atomic_and;
    ishmemi_proxy_funcs[AMO_FETCH_OR][UINT64] = ishmemi_openshmem_uint64_atomic_fetch_or;
    ishmemi_proxy_funcs[AMO_OR][UINT64] = ishmemi_openshmem_uint64_atomic_or;
    ishmemi_proxy_funcs[AMO_FETCH_XOR][UINT64] = ishmemi_openshmem_uint64_atomic_fetch_xor;
    ishmemi_proxy_funcs[AMO_XOR][UINT64] = ishmemi_openshmem_uint64_atomic_xor;

    ishmemi_proxy_funcs[AMO_FETCH][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_fetch;
    ishmemi_proxy_funcs[AMO_SET][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_set;
    ishmemi_proxy_funcs[AMO_COMPARE_SWAP][ULONGLONG] =
        ishmemi_openshmem_ulonglong_atomic_compare_swap;
    ishmemi_proxy_funcs[AMO_SWAP][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_swap;
    ishmemi_proxy_funcs[AMO_FETCH_INC][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_fetch_inc;
    ishmemi_proxy_funcs[AMO_INC][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_inc;
    ishmemi_proxy_funcs[AMO_FETCH_ADD][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_fetch_add;
    ishmemi_proxy_funcs[AMO_ADD][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_add;
    ishmemi_proxy_funcs[AMO_FETCH_AND][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_fetch_and;
    ishmemi_proxy_funcs[AMO_AND][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_and;
    ishmemi_proxy_funcs[AMO_FETCH_OR][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_fetch_or;
    ishmemi_proxy_funcs[AMO_OR][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_or;
    ishmemi_proxy_funcs[AMO_FETCH_XOR][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_fetch_xor;
    ishmemi_proxy_funcs[AMO_XOR][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_xor;

    ishmemi_proxy_funcs[AMO_FETCH][INT32] = ishmemi_openshmem_int32_atomic_fetch;
    ishmemi_proxy_funcs[AMO_SET][INT32] = ishmemi_openshmem_int32_atomic_set;
    ishmemi_proxy_funcs[AMO_COMPARE_SWAP][INT32] = ishmemi_openshmem_int32_atomic_compare_swap;
    ishmemi_proxy_funcs[AMO_SWAP][INT32] = ishmemi_openshmem_int32_atomic_swap;
    ishmemi_proxy_funcs[AMO_FETCH_INC][INT32] = ishmemi_openshmem_int32_atomic_fetch_inc;
    ishmemi_proxy_funcs[AMO_INC][INT32] = ishmemi_openshmem_int32_atomic_inc;
    ishmemi_proxy_funcs[AMO_FETCH_ADD][INT32] = ishmemi_openshmem_int32_atomic_fetch_add;
    ishmemi_proxy_funcs[AMO_ADD][INT32] = ishmemi_openshmem_int32_atomic_add;
    ishmemi_proxy_funcs[AMO_FETCH_AND][INT32] = ishmemi_openshmem_int32_atomic_fetch_and;
    ishmemi_proxy_funcs[AMO_AND][INT32] = ishmemi_openshmem_int32_atomic_and;
    ishmemi_proxy_funcs[AMO_FETCH_OR][INT32] = ishmemi_openshmem_int32_atomic_fetch_or;
    ishmemi_proxy_funcs[AMO_OR][INT32] = ishmemi_openshmem_int32_atomic_or;
    ishmemi_proxy_funcs[AMO_FETCH_XOR][INT32] = ishmemi_openshmem_int32_atomic_fetch_xor;
    ishmemi_proxy_funcs[AMO_XOR][INT32] = ishmemi_openshmem_int32_atomic_xor;

    ishmemi_proxy_funcs[AMO_FETCH][INT64] = ishmemi_openshmem_int64_atomic_fetch;
    ishmemi_proxy_funcs[AMO_SET][INT64] = ishmemi_openshmem_int64_atomic_set;
    ishmemi_proxy_funcs[AMO_COMPARE_SWAP][INT64] = ishmemi_openshmem_int64_atomic_compare_swap;
    ishmemi_proxy_funcs[AMO_SWAP][INT64] = ishmemi_openshmem_int64_atomic_swap;
    ishmemi_proxy_funcs[AMO_FETCH_INC][INT64] = ishmemi_openshmem_int64_atomic_fetch_inc;
    ishmemi_proxy_funcs[AMO_INC][INT64] = ishmemi_openshmem_int64_atomic_inc;
    ishmemi_proxy_funcs[AMO_FETCH_ADD][INT64] = ishmemi_openshmem_int64_atomic_fetch_add;
    ishmemi_proxy_funcs[AMO_ADD][INT64] = ishmemi_openshmem_int64_atomic_add;
    ishmemi_proxy_funcs[AMO_FETCH_AND][INT64] = ishmemi_openshmem_int64_atomic_fetch_and;
    ishmemi_proxy_funcs[AMO_AND][INT64] = ishmemi_openshmem_int64_atomic_and;
    ishmemi_proxy_funcs[AMO_FETCH_OR][INT64] = ishmemi_openshmem_int64_atomic_fetch_or;
    ishmemi_proxy_funcs[AMO_OR][INT64] = ishmemi_openshmem_int64_atomic_or;
    ishmemi_proxy_funcs[AMO_FETCH_XOR][INT64] = ishmemi_openshmem_int64_atomic_fetch_xor;
    ishmemi_proxy_funcs[AMO_XOR][INT64] = ishmemi_openshmem_int64_atomic_xor;

    ishmemi_proxy_funcs[AMO_FETCH][LONGLONG] = ishmemi_openshmem_longlong_atomic_fetch;
    ishmemi_proxy_funcs[AMO_SET][LONGLONG] = ishmemi_openshmem_longlong_atomic_set;
    ishmemi_proxy_funcs[AMO_COMPARE_SWAP][LONGLONG] =
        ishmemi_openshmem_longlong_atomic_compare_swap;
    ishmemi_proxy_funcs[AMO_SWAP][LONGLONG] = ishmemi_openshmem_longlong_atomic_swap;
    ishmemi_proxy_funcs[AMO_FETCH_INC][LONGLONG] = ishmemi_openshmem_longlong_atomic_fetch_inc;
    ishmemi_proxy_funcs[AMO_INC][LONGLONG] = ishmemi_openshmem_longlong_atomic_inc;
    ishmemi_proxy_funcs[AMO_FETCH_ADD][LONGLONG] = ishmemi_openshmem_longlong_atomic_fetch_add;
    ishmemi_proxy_funcs[AMO_ADD][LONGLONG] = ishmemi_openshmem_longlong_atomic_add;

    ishmemi_proxy_funcs[AMO_FETCH][FLOAT] = ishmemi_openshmem_float_atomic_fetch;
    ishmemi_proxy_funcs[AMO_SET][FLOAT] = ishmemi_openshmem_float_atomic_set;
    ishmemi_proxy_funcs[AMO_SWAP][FLOAT] = ishmemi_openshmem_float_atomic_swap;

    ishmemi_proxy_funcs[AMO_FETCH][DOUBLE] = ishmemi_openshmem_double_atomic_fetch;
    ishmemi_proxy_funcs[AMO_SET][DOUBLE] = ishmemi_openshmem_double_atomic_set;
    ishmemi_proxy_funcs[AMO_SWAP][DOUBLE] = ishmemi_openshmem_double_atomic_swap;

    /* Signaling */
    ishmemi_proxy_funcs[PUT_SIGNAL][UINT8] = ishmemi_openshmem_uint8_put_signal;
    ishmemi_proxy_funcs[PUT_SIGNAL_NBI][UINT8] = ishmemi_openshmem_uint8_put_signal_nbi;
    ishmemi_proxy_funcs[SIGNAL_FETCH][0] = ishmemi_openshmem_signal_fetch;

    /* Collectives */
    ishmemi_proxy_funcs[BARRIER][0] = ishmemi_openshmem_barrier_all;
    ishmemi_proxy_funcs[SYNC][0] = ishmemi_openshmem_sync_all;
    ishmemi_proxy_funcs[ALLTOALL][UINT8] = ishmemi_openshmem_uint8_alltoall;
    ishmemi_proxy_funcs[BCAST][UINT8] = ishmemi_openshmem_uint8_broadcast;
    ishmemi_proxy_funcs[COLLECT][UINT8] = ishmemi_openshmem_uint8_collect;
    ishmemi_proxy_funcs[FCOLLECT][UINT8] = ishmemi_openshmem_uint8_fcollect;

    /* Reductions */
    ishmemi_proxy_funcs[AND_REDUCE][UINT8] = ishmemi_openshmem_uint8_and_reduce;
    ishmemi_proxy_funcs[OR_REDUCE][UINT8] = ishmemi_openshmem_uint8_or_reduce;
    ishmemi_proxy_funcs[XOR_REDUCE][UINT8] = ishmemi_openshmem_uint8_xor_reduce;
    ishmemi_proxy_funcs[MAX_REDUCE][UINT8] = ishmemi_openshmem_uint8_max_reduce;
    ishmemi_proxy_funcs[MIN_REDUCE][UINT8] = ishmemi_openshmem_uint8_min_reduce;
    ishmemi_proxy_funcs[SUM_REDUCE][UINT8] = ishmemi_openshmem_uint8_sum_reduce;
    ishmemi_proxy_funcs[PROD_REDUCE][UINT8] = ishmemi_openshmem_uint8_prod_reduce;

    ishmemi_proxy_funcs[AND_REDUCE][UINT16] = ishmemi_openshmem_uint16_and_reduce;
    ishmemi_proxy_funcs[OR_REDUCE][UINT16] = ishmemi_openshmem_uint16_or_reduce;
    ishmemi_proxy_funcs[XOR_REDUCE][UINT16] = ishmemi_openshmem_uint16_xor_reduce;
    ishmemi_proxy_funcs[MAX_REDUCE][UINT16] = ishmemi_openshmem_uint16_max_reduce;
    ishmemi_proxy_funcs[MIN_REDUCE][UINT16] = ishmemi_openshmem_uint16_min_reduce;
    ishmemi_proxy_funcs[SUM_REDUCE][UINT16] = ishmemi_openshmem_uint16_sum_reduce;
    ishmemi_proxy_funcs[PROD_REDUCE][UINT16] = ishmemi_openshmem_uint16_prod_reduce;

    ishmemi_proxy_funcs[AND_REDUCE][UINT32] = ishmemi_openshmem_uint32_and_reduce;
    ishmemi_proxy_funcs[OR_REDUCE][UINT32] = ishmemi_openshmem_uint32_or_reduce;
    ishmemi_proxy_funcs[XOR_REDUCE][UINT32] = ishmemi_openshmem_uint32_xor_reduce;
    ishmemi_proxy_funcs[MAX_REDUCE][UINT32] = ishmemi_openshmem_uint32_max_reduce;
    ishmemi_proxy_funcs[MIN_REDUCE][UINT32] = ishmemi_openshmem_uint32_min_reduce;
    ishmemi_proxy_funcs[SUM_REDUCE][UINT32] = ishmemi_openshmem_uint32_sum_reduce;
    ishmemi_proxy_funcs[PROD_REDUCE][UINT32] = ishmemi_openshmem_uint32_prod_reduce;

    ishmemi_proxy_funcs[AND_REDUCE][UINT64] = ishmemi_openshmem_uint64_and_reduce;
    ishmemi_proxy_funcs[OR_REDUCE][UINT64] = ishmemi_openshmem_uint64_or_reduce;
    ishmemi_proxy_funcs[XOR_REDUCE][UINT64] = ishmemi_openshmem_uint64_xor_reduce;
    ishmemi_proxy_funcs[MAX_REDUCE][UINT64] = ishmemi_openshmem_uint64_max_reduce;
    ishmemi_proxy_funcs[MIN_REDUCE][UINT64] = ishmemi_openshmem_uint64_min_reduce;
    ishmemi_proxy_funcs[SUM_REDUCE][UINT64] = ishmemi_openshmem_uint64_sum_reduce;
    ishmemi_proxy_funcs[PROD_REDUCE][UINT64] = ishmemi_openshmem_uint64_prod_reduce;

    ishmemi_proxy_funcs[AND_REDUCE][ULONGLONG] = ishmemi_openshmem_ulonglong_and_reduce;
    ishmemi_proxy_funcs[OR_REDUCE][ULONGLONG] = ishmemi_openshmem_ulonglong_or_reduce;
    ishmemi_proxy_funcs[XOR_REDUCE][ULONGLONG] = ishmemi_openshmem_ulonglong_xor_reduce;
    ishmemi_proxy_funcs[MAX_REDUCE][ULONGLONG] = ishmemi_openshmem_ulonglong_max_reduce;
    ishmemi_proxy_funcs[MIN_REDUCE][ULONGLONG] = ishmemi_openshmem_ulonglong_min_reduce;
    ishmemi_proxy_funcs[SUM_REDUCE][ULONGLONG] = ishmemi_openshmem_ulonglong_sum_reduce;
    ishmemi_proxy_funcs[PROD_REDUCE][ULONGLONG] = ishmemi_openshmem_ulonglong_prod_reduce;

    ishmemi_proxy_funcs[AND_REDUCE][INT8] = ishmemi_openshmem_int8_and_reduce;
    ishmemi_proxy_funcs[OR_REDUCE][INT8] = ishmemi_openshmem_int8_or_reduce;
    ishmemi_proxy_funcs[XOR_REDUCE][INT8] = ishmemi_openshmem_int8_xor_reduce;
    ishmemi_proxy_funcs[MAX_REDUCE][INT8] = ishmemi_openshmem_int8_max_reduce;
    ishmemi_proxy_funcs[MIN_REDUCE][INT8] = ishmemi_openshmem_int8_min_reduce;
    ishmemi_proxy_funcs[SUM_REDUCE][INT8] = ishmemi_openshmem_int8_sum_reduce;
    ishmemi_proxy_funcs[PROD_REDUCE][INT8] = ishmemi_openshmem_int8_prod_reduce;

    ishmemi_proxy_funcs[AND_REDUCE][INT16] = ishmemi_openshmem_int16_and_reduce;
    ishmemi_proxy_funcs[OR_REDUCE][INT16] = ishmemi_openshmem_int16_or_reduce;
    ishmemi_proxy_funcs[XOR_REDUCE][INT16] = ishmemi_openshmem_int16_xor_reduce;
    ishmemi_proxy_funcs[MAX_REDUCE][INT16] = ishmemi_openshmem_int16_max_reduce;
    ishmemi_proxy_funcs[MIN_REDUCE][INT16] = ishmemi_openshmem_int16_min_reduce;
    ishmemi_proxy_funcs[SUM_REDUCE][INT16] = ishmemi_openshmem_int16_sum_reduce;
    ishmemi_proxy_funcs[PROD_REDUCE][INT16] = ishmemi_openshmem_int16_prod_reduce;

    ishmemi_proxy_funcs[AND_REDUCE][INT32] = ishmemi_openshmem_int32_and_reduce;
    ishmemi_proxy_funcs[OR_REDUCE][INT32] = ishmemi_openshmem_int32_or_reduce;
    ishmemi_proxy_funcs[XOR_REDUCE][INT32] = ishmemi_openshmem_int32_xor_reduce;
    ishmemi_proxy_funcs[MAX_REDUCE][INT32] = ishmemi_openshmem_int32_max_reduce;
    ishmemi_proxy_funcs[MIN_REDUCE][INT32] = ishmemi_openshmem_int32_min_reduce;
    ishmemi_proxy_funcs[SUM_REDUCE][INT32] = ishmemi_openshmem_int32_sum_reduce;
    ishmemi_proxy_funcs[PROD_REDUCE][INT32] = ishmemi_openshmem_int32_prod_reduce;

    ishmemi_proxy_funcs[AND_REDUCE][INT64] = ishmemi_openshmem_int64_and_reduce;
    ishmemi_proxy_funcs[OR_REDUCE][INT64] = ishmemi_openshmem_int64_or_reduce;
    ishmemi_proxy_funcs[XOR_REDUCE][INT64] = ishmemi_openshmem_int64_xor_reduce;
    ishmemi_proxy_funcs[MAX_REDUCE][INT64] = ishmemi_openshmem_int64_max_reduce;
    ishmemi_proxy_funcs[MIN_REDUCE][INT64] = ishmemi_openshmem_int64_min_reduce;
    ishmemi_proxy_funcs[SUM_REDUCE][INT64] = ishmemi_openshmem_int64_sum_reduce;
    ishmemi_proxy_funcs[PROD_REDUCE][INT64] = ishmemi_openshmem_int64_prod_reduce;

    ishmemi_proxy_funcs[MAX_REDUCE][LONGLONG] = ishmemi_openshmem_longlong_max_reduce;
    ishmemi_proxy_funcs[MIN_REDUCE][LONGLONG] = ishmemi_openshmem_longlong_min_reduce;
    ishmemi_proxy_funcs[SUM_REDUCE][LONGLONG] = ishmemi_openshmem_longlong_sum_reduce;
    ishmemi_proxy_funcs[PROD_REDUCE][LONGLONG] = ishmemi_openshmem_longlong_prod_reduce;

    ishmemi_proxy_funcs[MAX_REDUCE][FLOAT] = ishmemi_openshmem_float_max_reduce;
    ishmemi_proxy_funcs[MIN_REDUCE][FLOAT] = ishmemi_openshmem_float_min_reduce;
    ishmemi_proxy_funcs[SUM_REDUCE][FLOAT] = ishmemi_openshmem_float_sum_reduce;
    ishmemi_proxy_funcs[PROD_REDUCE][FLOAT] = ishmemi_openshmem_float_prod_reduce;

    ishmemi_proxy_funcs[MAX_REDUCE][DOUBLE] = ishmemi_openshmem_double_max_reduce;
    ishmemi_proxy_funcs[MIN_REDUCE][DOUBLE] = ishmemi_openshmem_double_min_reduce;
    ishmemi_proxy_funcs[SUM_REDUCE][DOUBLE] = ishmemi_openshmem_double_sum_reduce;
    ishmemi_proxy_funcs[PROD_REDUCE][DOUBLE] = ishmemi_openshmem_double_prod_reduce;

    /* Point-to-Point Synchronization */
    ishmemi_proxy_funcs[TEST][UINT32] = ishmemi_openshmem_uint32_test;
    ishmemi_proxy_funcs[WAIT][UINT32] = ishmemi_openshmem_uint32_wait_until;

    ishmemi_proxy_funcs[TEST][UINT64] = ishmemi_openshmem_uint64_test;
    ishmemi_proxy_funcs[WAIT][UINT64] = ishmemi_openshmem_uint64_wait_until;

    ishmemi_proxy_funcs[TEST][ULONGLONG] = ishmemi_openshmem_ulonglong_test;
    ishmemi_proxy_funcs[WAIT][ULONGLONG] = ishmemi_openshmem_ulonglong_wait_until;

    /* Memory Ordering */
    ishmemi_proxy_funcs[FENCE][0] = ishmemi_openshmem_fence;
    ishmemi_proxy_funcs[QUIET][0] = ishmemi_openshmem_quiet;

fn_exit:
    return;
}

int ishmemi_runtime_openshmem_init(bool initialize_runtime)
{
    int ret = 0, tl_provided;

    /* Setup OpenSHMEM dlsym links */
    ret = ishmemi_openshmem_wrapper_init();
    if (ret != 0) return ret;

    if (initialize_runtime && !initialized_openshmem) {
        ret = shmemx_WRAPPER_heap_preinit_thread(SHMEM_THREAD_MULTIPLE, &tl_provided);
        ISHMEM_CHECK_RETURN_MSG(
            ret | (tl_provided != SHMEM_THREAD_MULTIPLE),
            "SHMEM initialization failed with ret = %d and thread level provided = %d\n", ret,
            tl_provided);
        initialized_openshmem = true;
    }

    /* Setup internal runtime info */
    rank = shmem_WRAPPER_my_pe();
    size = shmem_WRAPPER_n_pes();

    /* Setup runtime function pointers */
    ishmemi_runtime_fini = ishmemi_runtime_openshmem_fini;
    ishmemi_runtime_abort = ishmemi_runtime_openshmem_abort;
    ishmemi_runtime_get_rank = ishmemi_runtime_openshmem_get_rank;
    ishmemi_runtime_get_size = ishmemi_runtime_openshmem_get_size;
    ishmemi_runtime_get_node_rank = ishmemi_runtime_openshmem_get_node_rank;
    ishmemi_runtime_get_node_size = ishmemi_runtime_openshmem_get_node_size;
    ishmemi_runtime_sync = ishmemi_runtime_openshmem_sync;
    ishmemi_runtime_fence = ishmemi_runtime_openshmem_fence;
    ishmemi_runtime_quiet = ishmemi_runtime_openshmem_quiet;
    ishmemi_runtime_barrier = ishmemi_runtime_openshmem_barrier;
    ishmemi_runtime_node_barrier = ishmemi_runtime_openshmem_node_barrier;
    ishmemi_runtime_bcast = ishmemi_runtime_openshmem_bcast;
    ishmemi_runtime_node_bcast = ishmemi_runtime_openshmem_node_bcast;
    ishmemi_runtime_node_fcollect = ishmemi_runtime_openshmem_node_fcollect;
    ishmemi_runtime_fcollect = ishmemi_runtime_openshmem_fcollect;
    ishmemi_runtime_is_local = ishmemi_runtime_openshmem_is_local;
    ishmemi_runtime_get = ishmemi_runtime_openshmem_get;
    ishmemi_runtime_malloc = shmem_WRAPPER_malloc;
    ishmemi_runtime_calloc = shmem_WRAPPER_calloc;
    ishmemi_runtime_free = shmem_WRAPPER_free;

    ishmemi_runtime_openshmem_funcptr_init();

    return ret;
}

void ishmemi_runtime_openshmem_heap_create(void *base, size_t size)
{
    if (initialized_openshmem) {
        /* only ZE device support for now */
        shmemx_WRAPPER_heap_create(base, size, SHMEMX_EXTERNAL_HEAP_ZE, 0);
        shmemx_WRAPPER_heap_postinit();
    }
}
