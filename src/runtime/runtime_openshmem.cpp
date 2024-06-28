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
#include "ishmem/err.h"
#include "proxy_impl.h"

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

void ishmemi_runtime_openshmem_barrier_all(void)
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

int ishmemi_runtime_openshmem_team_sync(ishmemi_runtime_team_t team)
{
    return shmem_WRAPPER_team_sync(team);
}

int ishmemi_runtime_openshmem_team_sanity_checks(ishmemi_runtime_team_predefined_t team,
                                                 int team_size, int expected_team_size, int team_pe,
                                                 int expected_team_pe, int world_pe,
                                                 int expected_world_pe)
{
    int ret = 0;
    if (team_size != expected_team_size) {
        ret = -1;
        ISHMEM_DEBUG_MSG("Expected %s team size: %d, got %d\n",
                         ishmemi_runtime_team_predefined_string(team), expected_team_size,
                         team_size);
    }
    if (team_pe != expected_team_pe) {
        ret = -1;
        ISHMEM_DEBUG_MSG("Expected %s team PE: %d, got %d\n",
                         ishmemi_runtime_team_predefined_string(team), expected_team_pe, team_pe);
    }
    if (world_pe != expected_world_pe) {
        ret = -1;
        ISHMEM_DEBUG_MSG("Expected %s world PE: %d, got %d\n",
                         ishmemi_runtime_team_predefined_string(team), expected_world_pe, world_pe);
    }
    return ret;
}

int ishmemi_runtime_openshmem_team_predefined_set(
    ishmemi_runtime_team_t *team, ishmemi_runtime_team_predefined_t predefined_team_name,
    int expected_team_size, int expected_world_pe, int expected_team_pe)
{
    int ret = 0;
    int team_size = -1, team_pe = -1, world_pe = -1;

    switch (predefined_team_name) {
        case WORLD:
            *team = SHMEM_TEAM_WORLD;
            break;
        case SHARED:
            /* The runtime's SHMEM_TEAM_SHARED may be incompatible with ISHMEM_TEAM_SHARED, so
             * check that the PE identifiers and team sizes match: */
            team_size = shmem_WRAPPER_team_n_pes(SHMEM_TEAM_SHARED);
            team_pe =
                shmem_WRAPPER_team_translate_pe(SHMEM_TEAM_WORLD, ishmemi_my_pe, SHMEM_TEAM_SHARED);
            world_pe = shmem_WRAPPER_team_translate_pe(SHMEM_TEAM_SHARED, expected_team_pe,
                                                       SHMEM_TEAM_WORLD);
            ret = ishmemi_runtime_openshmem_team_sanity_checks(
                SHARED, team_size, expected_team_size, team_pe, expected_team_pe, world_pe,
                expected_world_pe);
            /* If the shared teams do not match, try SHMEMX_TEAM_NODE: */
            if (ret != 0) {
                ISHMEM_DEBUG_MSG("Incompatible SHMEM_TEAM_SHARED: instead use SHMEMX_TEAM_NODE\n");
                team_size = shmem_WRAPPER_team_n_pes(SHMEMX_TEAM_NODE);
                team_pe = shmem_WRAPPER_team_translate_pe(SHMEM_TEAM_WORLD, ishmemi_my_pe,
                                                          SHMEMX_TEAM_NODE);
                world_pe = shmem_WRAPPER_team_translate_pe(SHMEMX_TEAM_NODE, expected_team_pe,
                                                           SHMEM_TEAM_WORLD);
                ret = ishmemi_runtime_openshmem_team_sanity_checks(
                    NODE, team_size, expected_team_size, team_pe, expected_team_pe, world_pe,
                    expected_world_pe);
                ISHMEM_CHECK_RETURN_MSG(
                    ret, "Runtime ISHMEM_TEAM_SHARED unable to use SHMEMX_TEAM_NODE\n");
                *team = SHMEMX_TEAM_NODE;
            } else {
                *team = SHMEM_TEAM_SHARED;
            }
            break;
        case NODE:
            *team = SHMEMX_TEAM_NODE;
            break;
        default:
            return -1;
    }
    return ret;
}

void ishmemi_runtime_openshmem_node_barrier(void)
{
    // TODO: quiet ctx's on ISHMEMI_TEAM_NODE
    shmem_WRAPPER_quiet();
    shmem_WRAPPER_team_sync(ISHMEMI_TEAM_NODE);
}

void ishmemi_runtime_openshmem_heap_create(void *base, size_t size)
{
    if (initialized_openshmem) {
        /* only ZE device support for now */
        shmemx_WRAPPER_heap_create(base, size, SHMEMX_EXTERNAL_HEAP_ZE, 0);
        shmemx_WRAPPER_heap_postinit();
    }
}

int ishmemi_runtime_openshmem_team_split_strided(ishmemi_runtime_team_t parent_team, int PE_start,
                                                 int PE_stride, int PE_size,
                                                 const ishmemi_runtime_team_config_t *config,
                                                 long config_mask, ishmemi_runtime_team_t *new_team)
{
    int ret = shmem_WRAPPER_team_split_strided(parent_team, PE_start, PE_stride, PE_size, config,
                                               config_mask, new_team);
    return ret;
}

int ishmemi_runtime_openshmem_uchar_and_reduce(ishmemi_runtime_team_t team, unsigned char *dest,
                                               const unsigned char *source, size_t nreduce)
{
    int ret = shmem_WRAPPER_uchar_and_reduce(team, dest, source, nreduce);
    return ret;
}

int ishmemi_runtime_openshmem_int_max_reduce(ishmemi_runtime_team_t team, int *dest,
                                             const int *source, size_t nreduce)
{
    int ret = shmem_WRAPPER_int_max_reduce(team, dest, source, nreduce);
    return ret;
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

void ishmemi_openshmem_uint16_iput(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    shmem_WRAPPER_uint16_iput(dest, src, dst, sst, nelems, pe);
}

void ishmemi_openshmem_uint32_iput(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    shmem_WRAPPER_uint32_iput(dest, src, dst, sst, nelems, pe);
}

void ishmemi_openshmem_uint64_iput(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    shmem_WRAPPER_uint64_iput(dest, src, dst, sst, nelems, pe);
}

void ishmemi_openshmem_ulonglong_iput(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    shmem_WRAPPER_ulonglong_iput(dest, src, dst, sst, nelems, pe);
}

void ishmemi_openshmem_uint8_ibput(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    shmemx_WRAPPER_uint8_ibput(dest, src, dst, sst, bsize, nelems, pe);
}

void ishmemi_openshmem_uint16_ibput(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    shmemx_WRAPPER_uint16_ibput(dest, src, dst, sst, bsize, nelems, pe);
}

void ishmemi_openshmem_uint32_ibput(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    shmemx_WRAPPER_uint32_ibput(dest, src, dst, sst, bsize, nelems, pe);
}

void ishmemi_openshmem_uint64_ibput(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    shmemx_WRAPPER_uint64_ibput(dest, src, dst, sst, bsize, nelems, pe);
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

void ishmemi_openshmem_uint16_iget(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    shmem_WRAPPER_uint16_iget(dest, src, dst, sst, nelems, pe);
}

void ishmemi_openshmem_uint32_iget(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    shmem_WRAPPER_uint32_iget(dest, src, dst, sst, nelems, pe);
}

void ishmemi_openshmem_uint64_iget(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    shmem_WRAPPER_uint64_iget(dest, src, dst, sst, nelems, pe);
}

void ishmemi_openshmem_uint8_ibget(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    shmemx_WRAPPER_uint8_ibget(dest, src, dst, sst, bsize, nelems, pe);
}

void ishmemi_openshmem_uint16_ibget(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    shmemx_WRAPPER_uint16_ibget(dest, src, dst, sst, bsize, nelems, pe);
}

void ishmemi_openshmem_uint32_ibget(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    shmemx_WRAPPER_uint32_ibget(dest, src, dst, sst, bsize, nelems, pe);
}

void ishmemi_openshmem_uint64_ibget(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    shmemx_WRAPPER_uint64_ibget(dest, src, dst, sst, bsize, nelems, pe);
}

void ishmemi_openshmem_ulonglong_iget(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    shmem_WRAPPER_ulonglong_iget(dest, src, dst, sst, nelems, pe);
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

/* AMO NBI */
#define ISHMEMI_GENERATE_AMO_FETCH_NBI_WRAPPER(TYPENAME, TYPEFIELD, TYPE)                          \
    void ishmemi_openshmem_##TYPENAME##_atomic_fetch_nbi(ishmemi_request_t *msg,                   \
                                                         ishmemi_ringcompletion_t *comp)           \
    {                                                                                              \
        ISHMEMI_RUNTIME_REQUEST_HELPER(TYPE, TYPEFIELD);                                           \
        shmem_WRAPPER_##TYPENAME##_atomic_fetch_nbi(fetch, dest, pe);                              \
    }

ISHMEMI_GENERATE_AMO_FETCH_NBI_WRAPPER(uint32, ui32, uint32_t)
ISHMEMI_GENERATE_AMO_FETCH_NBI_WRAPPER(uint64, ui64, uint64_t)
ISHMEMI_GENERATE_AMO_FETCH_NBI_WRAPPER(int32, i32, int32_t)
ISHMEMI_GENERATE_AMO_FETCH_NBI_WRAPPER(int64, i64, int64_t)
ISHMEMI_GENERATE_AMO_FETCH_NBI_WRAPPER(ulonglong, ul, unsigned long long)
ISHMEMI_GENERATE_AMO_FETCH_NBI_WRAPPER(longlong, ll, long long)
ISHMEMI_GENERATE_AMO_FETCH_NBI_WRAPPER(float, f, float)
ISHMEMI_GENERATE_AMO_FETCH_NBI_WRAPPER(double, ld, double)

#define ISHMEMI_GENERATE_AMO_COMPARE_SWAP_NBI_WRAPPER(TYPENAME, TYPEFIELD, TYPE)                   \
    void ishmemi_openshmem_##TYPENAME##_atomic_compare_swap_nbi(ishmemi_request_t *msg,            \
                                                                ishmemi_ringcompletion_t *comp)    \
    {                                                                                              \
        ISHMEMI_RUNTIME_REQUEST_HELPER(TYPE, TYPEFIELD);                                           \
        shmem_WRAPPER_##TYPENAME##_atomic_compare_swap_nbi(fetch, dest, cond, val, pe);            \
    }

ISHMEMI_GENERATE_AMO_COMPARE_SWAP_NBI_WRAPPER(uint32, ui32, uint32_t)
ISHMEMI_GENERATE_AMO_COMPARE_SWAP_NBI_WRAPPER(uint64, ui64, uint64_t)
ISHMEMI_GENERATE_AMO_COMPARE_SWAP_NBI_WRAPPER(int32, i32, int32_t)
ISHMEMI_GENERATE_AMO_COMPARE_SWAP_NBI_WRAPPER(int64, i64, int64_t)
ISHMEMI_GENERATE_AMO_COMPARE_SWAP_NBI_WRAPPER(ulonglong, ul, unsigned long long)
ISHMEMI_GENERATE_AMO_COMPARE_SWAP_NBI_WRAPPER(longlong, ll, long long)

#define ISHMEMI_GENERATE_AMO_SWAP_NBI_WRAPPER(TYPENAME, TYPEFIELD, TYPE)                           \
    void ishmemi_openshmem_##TYPENAME##_atomic_swap_nbi(ishmemi_request_t *msg,                    \
                                                        ishmemi_ringcompletion_t *comp)            \
    {                                                                                              \
        ISHMEMI_RUNTIME_REQUEST_HELPER(TYPE, TYPEFIELD);                                           \
        shmem_WRAPPER_##TYPENAME##_atomic_swap_nbi(fetch, dest, val, pe);                          \
    }

ISHMEMI_GENERATE_AMO_SWAP_NBI_WRAPPER(uint32, ui32, uint32_t)
ISHMEMI_GENERATE_AMO_SWAP_NBI_WRAPPER(uint64, ui64, uint64_t)
ISHMEMI_GENERATE_AMO_SWAP_NBI_WRAPPER(int32, i32, int32_t)
ISHMEMI_GENERATE_AMO_SWAP_NBI_WRAPPER(int64, i64, int64_t)
ISHMEMI_GENERATE_AMO_SWAP_NBI_WRAPPER(ulonglong, ul, unsigned long long)
ISHMEMI_GENERATE_AMO_SWAP_NBI_WRAPPER(longlong, ll, long long)
ISHMEMI_GENERATE_AMO_SWAP_NBI_WRAPPER(float, f, float)
ISHMEMI_GENERATE_AMO_SWAP_NBI_WRAPPER(double, ld, double)

#define ISHMEMI_GENERATE_AMO_FETCH_INC_NBI_WRAPPER(TYPENAME, TYPEFIELD, TYPE)                      \
    void ishmemi_openshmem_##TYPENAME##_atomic_fetch_inc_nbi(ishmemi_request_t *msg,               \
                                                             ishmemi_ringcompletion_t *comp)       \
    {                                                                                              \
        ISHMEMI_RUNTIME_REQUEST_HELPER(TYPE, TYPEFIELD);                                           \
        shmem_WRAPPER_##TYPENAME##_atomic_fetch_inc_nbi(fetch, dest, pe);                          \
    }

ISHMEMI_GENERATE_AMO_FETCH_INC_NBI_WRAPPER(uint32, ui32, uint32_t)
ISHMEMI_GENERATE_AMO_FETCH_INC_NBI_WRAPPER(uint64, ui64, uint64_t)
ISHMEMI_GENERATE_AMO_FETCH_INC_NBI_WRAPPER(int32, i32, int32_t)
ISHMEMI_GENERATE_AMO_FETCH_INC_NBI_WRAPPER(int64, i64, int64_t)
ISHMEMI_GENERATE_AMO_FETCH_INC_NBI_WRAPPER(ulonglong, ul, unsigned long long)
ISHMEMI_GENERATE_AMO_FETCH_INC_NBI_WRAPPER(longlong, ll, long long)

#define ISHMEMI_GENERATE_AMO_FETCH_ADD_NBI_WRAPPER(TYPENAME, TYPEFIELD, TYPE)                      \
    void ishmemi_openshmem_##TYPENAME##_atomic_fetch_add_nbi(ishmemi_request_t *msg,               \
                                                             ishmemi_ringcompletion_t *comp)       \
    {                                                                                              \
        ISHMEMI_RUNTIME_REQUEST_HELPER(TYPE, TYPEFIELD);                                           \
        shmem_WRAPPER_##TYPENAME##_atomic_fetch_add_nbi(fetch, dest, val, pe);                     \
    }

ISHMEMI_GENERATE_AMO_FETCH_ADD_NBI_WRAPPER(uint32, ui32, uint32_t)
ISHMEMI_GENERATE_AMO_FETCH_ADD_NBI_WRAPPER(uint64, ui64, uint64_t)
ISHMEMI_GENERATE_AMO_FETCH_ADD_NBI_WRAPPER(int32, i32, int32_t)
ISHMEMI_GENERATE_AMO_FETCH_ADD_NBI_WRAPPER(int64, i64, int64_t)
ISHMEMI_GENERATE_AMO_FETCH_ADD_NBI_WRAPPER(ulonglong, ul, unsigned long long)
ISHMEMI_GENERATE_AMO_FETCH_ADD_NBI_WRAPPER(longlong, ll, long long)

#define ISHMEMI_GENERATE_AMO_FETCH_AND_NBI_WRAPPER(TYPENAME, TYPEFIELD, TYPE)                      \
    void ishmemi_openshmem_##TYPENAME##_atomic_fetch_and_nbi(ishmemi_request_t *msg,               \
                                                             ishmemi_ringcompletion_t *comp)       \
    {                                                                                              \
        ISHMEMI_RUNTIME_REQUEST_HELPER(TYPE, TYPEFIELD);                                           \
        shmem_WRAPPER_##TYPENAME##_atomic_fetch_and_nbi(fetch, dest, val, pe);                     \
    }

ISHMEMI_GENERATE_AMO_FETCH_AND_NBI_WRAPPER(uint32, ui32, uint32_t)
ISHMEMI_GENERATE_AMO_FETCH_AND_NBI_WRAPPER(uint64, ui64, uint64_t)
ISHMEMI_GENERATE_AMO_FETCH_AND_NBI_WRAPPER(int32, i32, int32_t)
ISHMEMI_GENERATE_AMO_FETCH_AND_NBI_WRAPPER(int64, i64, int64_t)
ISHMEMI_GENERATE_AMO_FETCH_AND_NBI_WRAPPER(ulonglong, ul, unsigned long long)

#define ISHMEMI_GENERATE_AMO_FETCH_OR_NBI_WRAPPER(TYPENAME, TYPEFIELD, TYPE)                       \
    void ishmemi_openshmem_##TYPENAME##_atomic_fetch_or_nbi(ishmemi_request_t *msg,                \
                                                            ishmemi_ringcompletion_t *comp)        \
    {                                                                                              \
        ISHMEMI_RUNTIME_REQUEST_HELPER(TYPE, TYPEFIELD);                                           \
        shmem_WRAPPER_##TYPENAME##_atomic_fetch_or_nbi(fetch, dest, val, pe);                      \
    }

ISHMEMI_GENERATE_AMO_FETCH_OR_NBI_WRAPPER(uint32, ui32, uint32_t)
ISHMEMI_GENERATE_AMO_FETCH_OR_NBI_WRAPPER(uint64, ui64, uint64_t)
ISHMEMI_GENERATE_AMO_FETCH_OR_NBI_WRAPPER(int32, i32, int32_t)
ISHMEMI_GENERATE_AMO_FETCH_OR_NBI_WRAPPER(int64, i64, int64_t)
ISHMEMI_GENERATE_AMO_FETCH_OR_NBI_WRAPPER(ulonglong, ul, unsigned long long)

#define ISHMEMI_GENERATE_AMO_FETCH_XOR_NBI_WRAPPER(TYPENAME, TYPEFIELD, TYPE)                      \
    void ishmemi_openshmem_##TYPENAME##_atomic_fetch_xor_nbi(ishmemi_request_t *msg,               \
                                                             ishmemi_ringcompletion_t *comp)       \
    {                                                                                              \
        ISHMEMI_RUNTIME_REQUEST_HELPER(TYPE, TYPEFIELD);                                           \
        shmem_WRAPPER_##TYPENAME##_atomic_fetch_xor_nbi(fetch, dest, val, pe);                     \
    }

ISHMEMI_GENERATE_AMO_FETCH_XOR_NBI_WRAPPER(uint32, ui32, uint32_t)
ISHMEMI_GENERATE_AMO_FETCH_XOR_NBI_WRAPPER(uint64, ui64, uint64_t)
ISHMEMI_GENERATE_AMO_FETCH_XOR_NBI_WRAPPER(int32, i32, int32_t)
ISHMEMI_GENERATE_AMO_FETCH_XOR_NBI_WRAPPER(int64, i64, int64_t)
ISHMEMI_GENERATE_AMO_FETCH_XOR_NBI_WRAPPER(ulonglong, ul, unsigned long long)

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
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.ui64 = shmem_WRAPPER_signal_fetch(sig_addr);
}

void ishmemi_openshmem_signal_add(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    /* TODO: Use signal add */
    shmem_WRAPPER_uint64_atomic_add(dest, val, pe);
}

void ishmemi_openshmem_signal_set(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    /* TODO: Use signal set */
    shmem_WRAPPER_uint64_atomic_set(dest, val, pe);
}

/* Teams */
void ishmemi_openshmem_team_my_pe(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_team_my_pe(team->shmem_team);
}

void ishmemi_openshmem_team_n_pes(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_team_n_pes(team->shmem_team);
}

void ishmemi_openshmem_team_sync(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_team_sync(team->shmem_team);
}

/* Collectives */
void ishmemi_openshmem_barrier_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ishmemi_runtime_openshmem_barrier_all();
}

void ishmemi_openshmem_sync_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ishmemi_runtime_openshmem_sync();
}

void ishmemi_openshmem_uint8_alltoall(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_alltoall(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_broadcast(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i =
        shmem_WRAPPER_uint8_broadcast(team->shmem_team, dest, src, nelems, root);
}

void ishmemi_openshmem_uint8_collect(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_collect(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_fcollect(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_fcollect(team->shmem_team, dest, src, nelems);
}

/* Reductions */
void ishmemi_openshmem_uint8_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_and_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_or_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_xor_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_max_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_min_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_sum_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint8_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = shmem_WRAPPER_uint8_prod_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint16_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = shmem_WRAPPER_uint16_and_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint16_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = shmem_WRAPPER_uint16_or_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint16_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = shmem_WRAPPER_uint16_xor_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint16_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = shmem_WRAPPER_uint16_max_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint16_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = shmem_WRAPPER_uint16_min_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint16_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = shmem_WRAPPER_uint16_sum_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint16_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = shmem_WRAPPER_uint16_prod_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint32_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_and_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint32_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_or_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint32_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_xor_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint32_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_max_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint32_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_min_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint32_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_sum_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint32_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = shmem_WRAPPER_uint32_prod_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint64_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_and_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint64_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_or_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint64_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_xor_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint64_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_max_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint64_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_min_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint64_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_sum_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_uint64_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = shmem_WRAPPER_uint64_prod_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_ulonglong_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i =
        shmem_WRAPPER_ulonglong_and_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_ulonglong_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = shmem_WRAPPER_ulonglong_or_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_ulonglong_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i =
        shmem_WRAPPER_ulonglong_xor_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_ulonglong_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i =
        shmem_WRAPPER_ulonglong_max_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_ulonglong_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i =
        shmem_WRAPPER_ulonglong_min_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_ulonglong_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i =
        shmem_WRAPPER_ulonglong_sum_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_ulonglong_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i =
        shmem_WRAPPER_ulonglong_prod_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int8_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = shmem_WRAPPER_int8_and_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int8_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = shmem_WRAPPER_int8_or_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int8_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = shmem_WRAPPER_int8_xor_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int8_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = shmem_WRAPPER_int8_max_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int8_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = shmem_WRAPPER_int8_min_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int8_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = shmem_WRAPPER_int8_sum_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int8_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = shmem_WRAPPER_int8_prod_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int16_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = shmem_WRAPPER_int16_and_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int16_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = shmem_WRAPPER_int16_or_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int16_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = shmem_WRAPPER_int16_xor_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int16_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = shmem_WRAPPER_int16_max_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int16_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = shmem_WRAPPER_int16_min_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int16_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = shmem_WRAPPER_int16_sum_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int16_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = shmem_WRAPPER_int16_prod_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int32_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = shmem_WRAPPER_int32_and_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int32_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = shmem_WRAPPER_int32_or_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int32_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = shmem_WRAPPER_int32_xor_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int32_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = shmem_WRAPPER_int32_max_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int32_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = shmem_WRAPPER_int32_min_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int32_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = shmem_WRAPPER_int32_sum_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int32_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = shmem_WRAPPER_int32_prod_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int64_and_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = shmem_WRAPPER_int64_and_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int64_or_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = shmem_WRAPPER_int64_or_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int64_xor_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = shmem_WRAPPER_int64_xor_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int64_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = shmem_WRAPPER_int64_max_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int64_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = shmem_WRAPPER_int64_min_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int64_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = shmem_WRAPPER_int64_sum_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_int64_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = shmem_WRAPPER_int64_prod_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_longlong_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i = shmem_WRAPPER_longlong_max_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_longlong_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i = shmem_WRAPPER_longlong_min_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_longlong_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i = shmem_WRAPPER_longlong_sum_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_longlong_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i =
        shmem_WRAPPER_longlong_prod_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_float_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    comp->completion.ret.i = shmem_WRAPPER_float_max_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_float_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    comp->completion.ret.i = shmem_WRAPPER_float_min_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_float_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    comp->completion.ret.i = shmem_WRAPPER_float_sum_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_float_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    comp->completion.ret.i = shmem_WRAPPER_float_prod_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_double_max_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    comp->completion.ret.i = shmem_WRAPPER_double_max_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_double_min_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    comp->completion.ret.i = shmem_WRAPPER_double_min_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_double_sum_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    comp->completion.ret.i = shmem_WRAPPER_double_sum_reduce(team->shmem_team, dest, src, nelems);
}

void ishmemi_openshmem_double_prod_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    comp->completion.ret.i = shmem_WRAPPER_double_prod_reduce(team->shmem_team, dest, src, nelems);
}

/* Point-to-Point Synchronization */
void ishmemi_openshmem_int32_test(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int32_t, dest);
    }
    comp->completion.ret.i = shmem_WRAPPER_int32_test(dest, cmp, cmp_value);
}

void ishmemi_openshmem_int32_test_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int32_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.i = shmem_WRAPPER_int32_test_all(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_int32_test_any(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int32_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt = shmem_WRAPPER_int32_test_any(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_int32_test_some(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int32_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(indices)) {
        indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_int32_test_some(dest, nelems, indices, status, cmp, cmp_value);
}

void ishmemi_openshmem_int32_wait_until(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int32_t, dest);
    }
    shmem_WRAPPER_int32_wait_until(dest, cmp, cmp_value);
}

void ishmemi_openshmem_int32_wait_until_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int32_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    shmem_WRAPPER_int32_wait_until_all(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_int32_wait_until_any(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int32_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_int32_wait_until_any(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_int32_wait_until_some(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int32_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(indices)) {
        indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_int32_wait_until_some(dest, nelems, indices, status, cmp, cmp_value);
}

void ishmemi_openshmem_int64_test(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int64_t, dest);
    }
    comp->completion.ret.i = shmem_WRAPPER_int64_test(dest, cmp, cmp_value);
}

void ishmemi_openshmem_int64_test_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int64_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.i = shmem_WRAPPER_int64_test_all(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_int64_test_any(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int64_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt = shmem_WRAPPER_int64_test_any(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_int64_test_some(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int64_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(indices)) {
        indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_int64_test_some(dest, nelems, indices, status, cmp, cmp_value);
}

void ishmemi_openshmem_int64_wait_until(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int64_t, dest);
    }
    shmem_WRAPPER_int64_wait_until(dest, cmp, cmp_value);
}

void ishmemi_openshmem_int64_wait_until_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int64_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    shmem_WRAPPER_int64_wait_until_all(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_int64_wait_until_any(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int64_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_int64_wait_until_any(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_int64_wait_until_some(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(int64_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(indices)) {
        indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_int64_wait_until_some(dest, nelems, indices, status, cmp, cmp_value);
}

void ishmemi_openshmem_longlong_test(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(long long, dest);
    }
    comp->completion.ret.i = shmem_WRAPPER_longlong_test(dest, cmp, cmp_value);
}

void ishmemi_openshmem_longlong_test_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(long long, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.i = shmem_WRAPPER_longlong_test_all(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_longlong_test_any(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(long long, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_longlong_test_any(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_longlong_test_some(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(long long, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(indices)) {
        indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_longlong_test_some(dest, nelems, indices, status, cmp, cmp_value);
}

void ishmemi_openshmem_longlong_wait_until(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(long long, dest);
    }
    shmem_WRAPPER_longlong_wait_until(dest, cmp, cmp_value);
}

void ishmemi_openshmem_longlong_wait_until_all(ishmemi_request_t *msg,
                                               ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(long long, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    shmem_WRAPPER_longlong_wait_until_all(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_longlong_wait_until_any(ishmemi_request_t *msg,
                                               ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(long long, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_longlong_wait_until_any(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_longlong_wait_until_some(ishmemi_request_t *msg,
                                                ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(long long, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(indices)) {
        indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_longlong_wait_until_some(dest, nelems, indices, status, cmp, cmp_value);
}

void ishmemi_openshmem_uint32_test(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint32_t, dest);
    }
    comp->completion.ret.i = shmem_WRAPPER_uint32_test(dest, cmp, cmp_value);
}

void ishmemi_openshmem_uint32_test_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint32_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.i = shmem_WRAPPER_uint32_test_all(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_uint32_test_any(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint32_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt = shmem_WRAPPER_uint32_test_any(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_uint32_test_some(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint32_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(indices)) {
        indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_uint32_test_some(dest, nelems, indices, status, cmp, cmp_value);
}

void ishmemi_openshmem_uint32_wait_until(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint32_t, dest);
    }
    shmem_WRAPPER_uint32_wait_until(dest, cmp, cmp_value);
}

void ishmemi_openshmem_uint32_wait_until_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint32_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    shmem_WRAPPER_uint32_wait_until_all(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_uint32_wait_until_any(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint32_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_uint32_wait_until_any(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_uint32_wait_until_some(ishmemi_request_t *msg,
                                              ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint32_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(indices)) {
        indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_uint32_wait_until_some(dest, nelems, indices, status, cmp, cmp_value);
}

void ishmemi_openshmem_uint64_test(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint64_t, dest);
    }
    comp->completion.ret.i = shmem_WRAPPER_uint64_test(dest, cmp, cmp_value);
}

void ishmemi_openshmem_uint64_test_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint64_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.i = shmem_WRAPPER_uint64_test_all(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_uint64_test_any(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint64_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt = shmem_WRAPPER_uint64_test_any(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_uint64_test_some(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint64_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(indices)) {
        indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_uint64_test_some(dest, nelems, indices, status, cmp, cmp_value);
}

void ishmemi_openshmem_uint64_wait_until(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint64_t, dest);
    }
    shmem_WRAPPER_uint64_wait_until(dest, cmp, cmp_value);
}

void ishmemi_openshmem_uint64_wait_until_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint64_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    shmem_WRAPPER_uint64_wait_until_all(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_uint64_wait_until_any(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint64_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_uint64_wait_until_any(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_uint64_wait_until_some(ishmemi_request_t *msg,
                                              ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint64_t, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(indices)) {
        indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_uint64_wait_until_some(dest, nelems, indices, status, cmp, cmp_value);
}

void ishmemi_openshmem_signal_wait_until(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    if (ISHMEMI_HOST_IN_HEAP(sig_addr)) {
        sig_addr = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint64_t, sig_addr);
    }
    comp->completion.ret.ui64 = shmem_WRAPPER_signal_wait_until(sig_addr, cmp, cmp_value);
}

void ishmemi_openshmem_ulonglong_test(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(unsigned long long, dest);
    }
    comp->completion.ret.i = shmem_WRAPPER_ulonglong_test(dest, cmp, cmp_value);
}

void ishmemi_openshmem_ulonglong_test_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(unsigned long long, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.i = shmem_WRAPPER_ulonglong_test_all(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_ulonglong_test_any(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(unsigned long long, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_ulonglong_test_any(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_ulonglong_test_some(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(unsigned long long, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(indices)) {
        indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_ulonglong_test_some(dest, nelems, indices, status, cmp, cmp_value);
}

void ishmemi_openshmem_ulonglong_wait_until(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(unsigned long long, dest);
    }
    shmem_WRAPPER_ulonglong_wait_until(dest, cmp, cmp_value);
}

void ishmemi_openshmem_ulonglong_wait_until_all(ishmemi_request_t *msg,
                                                ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(unsigned long long, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    shmem_WRAPPER_ulonglong_wait_until_all(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_ulonglong_wait_until_any(ishmemi_request_t *msg,
                                                ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(unsigned long long, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_ulonglong_wait_until_any(dest, nelems, status, cmp, cmp_value);
}

void ishmemi_openshmem_ulonglong_wait_until_some(ishmemi_request_t *msg,
                                                 ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    if (ISHMEMI_HOST_IN_HEAP(dest)) {
        dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(unsigned long long, dest);
    }
    if (ISHMEMI_HOST_IN_HEAP(indices)) {
        indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
    }
    if (ISHMEMI_HOST_IN_HEAP(status)) {
        status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
    }
    comp->completion.ret.szt =
        shmem_WRAPPER_ulonglong_wait_until_some(dest, nelems, indices, status, cmp, cmp_value);
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
    ishmemi_proxy_funcs[IPUT][UINT16] = ishmemi_openshmem_uint16_iput;
    ishmemi_proxy_funcs[IPUT][UINT32] = ishmemi_openshmem_uint32_iput;
    ishmemi_proxy_funcs[IPUT][UINT64] = ishmemi_openshmem_uint64_iput;
    ishmemi_proxy_funcs[IPUT][ULONGLONG] = ishmemi_openshmem_uint64_iput;
    ishmemi_proxy_funcs[IBPUT][UINT8] = ishmemi_openshmem_uint8_ibput;
    ishmemi_proxy_funcs[IBPUT][UINT16] = ishmemi_openshmem_uint16_ibput;
    ishmemi_proxy_funcs[IBPUT][UINT32] = ishmemi_openshmem_uint32_ibput;
    ishmemi_proxy_funcs[IBPUT][UINT64] = ishmemi_openshmem_uint64_ibput;
    ishmemi_proxy_funcs[P][UINT8] = ishmemi_openshmem_uint8_p;
    ishmemi_proxy_funcs[P][UINT16] = ishmemi_openshmem_uint16_p;
    ishmemi_proxy_funcs[P][UINT32] = ishmemi_openshmem_uint32_p;
    ishmemi_proxy_funcs[P][UINT64] = ishmemi_openshmem_uint64_p;
    ishmemi_proxy_funcs[P][ULONGLONG] = ishmemi_openshmem_ulonglong_p;
    ishmemi_proxy_funcs[PUT_NBI][UINT8] = ishmemi_openshmem_uint8_put_nbi;

    ishmemi_proxy_funcs[GET][UINT8] = ishmemi_openshmem_uint8_get;
    ishmemi_proxy_funcs[IGET][UINT8] = ishmemi_openshmem_uint8_iget;
    ishmemi_proxy_funcs[IGET][UINT16] = ishmemi_openshmem_uint16_iget;
    ishmemi_proxy_funcs[IGET][UINT32] = ishmemi_openshmem_uint32_iget;
    ishmemi_proxy_funcs[IGET][UINT64] = ishmemi_openshmem_uint64_iget;
    ishmemi_proxy_funcs[IGET][ULONGLONG] = ishmemi_openshmem_uint64_iget;
    ishmemi_proxy_funcs[IBGET][UINT8] = ishmemi_openshmem_uint8_ibget;
    ishmemi_proxy_funcs[IBGET][UINT16] = ishmemi_openshmem_uint16_ibget;
    ishmemi_proxy_funcs[IBGET][UINT32] = ishmemi_openshmem_uint32_ibget;
    ishmemi_proxy_funcs[IBGET][UINT64] = ishmemi_openshmem_uint64_ibget;
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

    /* AMO NBI */
    ishmemi_proxy_funcs[AMO_FETCH_NBI][UINT32] = ishmemi_openshmem_uint32_atomic_fetch_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_NBI][INT32] = ishmemi_openshmem_int32_atomic_fetch_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_NBI][UINT64] = ishmemi_openshmem_uint64_atomic_fetch_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_NBI][INT64] = ishmemi_openshmem_int64_atomic_fetch_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_NBI][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_fetch_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_NBI][LONGLONG] = ishmemi_openshmem_longlong_atomic_fetch_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_NBI][FLOAT] = ishmemi_openshmem_float_atomic_fetch_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_NBI][DOUBLE] = ishmemi_openshmem_double_atomic_fetch_nbi;

    ishmemi_proxy_funcs[AMO_COMPARE_SWAP_NBI][UINT32] =
        ishmemi_openshmem_uint32_atomic_compare_swap_nbi;
    ishmemi_proxy_funcs[AMO_COMPARE_SWAP_NBI][INT32] =
        ishmemi_openshmem_int32_atomic_compare_swap_nbi;
    ishmemi_proxy_funcs[AMO_COMPARE_SWAP_NBI][UINT64] =
        ishmemi_openshmem_uint64_atomic_compare_swap_nbi;
    ishmemi_proxy_funcs[AMO_COMPARE_SWAP_NBI][INT64] =
        ishmemi_openshmem_int64_atomic_compare_swap_nbi;
    ishmemi_proxy_funcs[AMO_COMPARE_SWAP_NBI][ULONGLONG] =
        ishmemi_openshmem_ulonglong_atomic_compare_swap_nbi;
    ishmemi_proxy_funcs[AMO_COMPARE_SWAP_NBI][LONGLONG] =
        ishmemi_openshmem_longlong_atomic_compare_swap_nbi;

    ishmemi_proxy_funcs[AMO_SWAP_NBI][UINT32] = ishmemi_openshmem_uint32_atomic_swap_nbi;
    ishmemi_proxy_funcs[AMO_SWAP_NBI][INT32] = ishmemi_openshmem_int32_atomic_swap_nbi;
    ishmemi_proxy_funcs[AMO_SWAP_NBI][UINT64] = ishmemi_openshmem_uint64_atomic_swap_nbi;
    ishmemi_proxy_funcs[AMO_SWAP_NBI][INT64] = ishmemi_openshmem_int64_atomic_swap_nbi;
    ishmemi_proxy_funcs[AMO_SWAP_NBI][ULONGLONG] = ishmemi_openshmem_ulonglong_atomic_swap_nbi;
    ishmemi_proxy_funcs[AMO_SWAP_NBI][LONGLONG] = ishmemi_openshmem_longlong_atomic_swap_nbi;
    ishmemi_proxy_funcs[AMO_SWAP_NBI][FLOAT] = ishmemi_openshmem_float_atomic_swap_nbi;
    ishmemi_proxy_funcs[AMO_SWAP_NBI][DOUBLE] = ishmemi_openshmem_double_atomic_swap_nbi;

    ishmemi_proxy_funcs[AMO_FETCH_INC_NBI][UINT32] = ishmemi_openshmem_uint32_atomic_fetch_inc_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_INC_NBI][INT32] = ishmemi_openshmem_int32_atomic_fetch_inc_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_INC_NBI][UINT64] = ishmemi_openshmem_uint64_atomic_fetch_inc_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_INC_NBI][INT64] = ishmemi_openshmem_int64_atomic_fetch_inc_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_INC_NBI][ULONGLONG] =
        ishmemi_openshmem_ulonglong_atomic_fetch_inc_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_INC_NBI][LONGLONG] =
        ishmemi_openshmem_longlong_atomic_fetch_inc_nbi;

    ishmemi_proxy_funcs[AMO_FETCH_ADD_NBI][UINT32] = ishmemi_openshmem_uint32_atomic_fetch_add_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_ADD_NBI][INT32] = ishmemi_openshmem_int32_atomic_fetch_add_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_ADD_NBI][UINT64] = ishmemi_openshmem_uint64_atomic_fetch_add_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_ADD_NBI][INT64] = ishmemi_openshmem_int64_atomic_fetch_add_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_ADD_NBI][ULONGLONG] =
        ishmemi_openshmem_ulonglong_atomic_fetch_add_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_ADD_NBI][LONGLONG] =
        ishmemi_openshmem_longlong_atomic_fetch_add_nbi;

    ishmemi_proxy_funcs[AMO_FETCH_AND_NBI][UINT32] = ishmemi_openshmem_uint32_atomic_fetch_and_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_AND_NBI][INT32] = ishmemi_openshmem_int32_atomic_fetch_and_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_AND_NBI][UINT64] = ishmemi_openshmem_uint64_atomic_fetch_and_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_AND_NBI][INT64] = ishmemi_openshmem_int64_atomic_fetch_and_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_AND_NBI][ULONGLONG] =
        ishmemi_openshmem_ulonglong_atomic_fetch_and_nbi;

    ishmemi_proxy_funcs[AMO_FETCH_OR_NBI][UINT32] = ishmemi_openshmem_uint32_atomic_fetch_or_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_OR_NBI][INT32] = ishmemi_openshmem_int32_atomic_fetch_or_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_OR_NBI][UINT64] = ishmemi_openshmem_uint64_atomic_fetch_or_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_OR_NBI][INT64] = ishmemi_openshmem_int64_atomic_fetch_or_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_OR_NBI][ULONGLONG] =
        ishmemi_openshmem_ulonglong_atomic_fetch_or_nbi;

    ishmemi_proxy_funcs[AMO_FETCH_XOR_NBI][UINT32] = ishmemi_openshmem_uint32_atomic_fetch_xor_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_XOR_NBI][INT32] = ishmemi_openshmem_int32_atomic_fetch_xor_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_XOR_NBI][UINT64] = ishmemi_openshmem_uint64_atomic_fetch_xor_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_XOR_NBI][INT64] = ishmemi_openshmem_int64_atomic_fetch_xor_nbi;
    ishmemi_proxy_funcs[AMO_FETCH_XOR_NBI][ULONGLONG] =
        ishmemi_openshmem_ulonglong_atomic_fetch_xor_nbi;

    /* Signaling */
    ishmemi_proxy_funcs[PUT_SIGNAL][UINT8] = ishmemi_openshmem_uint8_put_signal;
    ishmemi_proxy_funcs[PUT_SIGNAL_NBI][UINT8] = ishmemi_openshmem_uint8_put_signal_nbi;
    ishmemi_proxy_funcs[SIGNAL_FETCH][0] = ishmemi_openshmem_signal_fetch;
    ishmemi_proxy_funcs[SIGNAL_ADD][UINT64] = ishmemi_openshmem_signal_add;
    ishmemi_proxy_funcs[SIGNAL_SET][UINT64] = ishmemi_openshmem_signal_set;

    /* Teams */
    ishmemi_proxy_funcs[TEAM_MY_PE][0] = ishmemi_openshmem_team_my_pe;
    ishmemi_proxy_funcs[TEAM_N_PES][0] = ishmemi_openshmem_team_n_pes;
    ishmemi_proxy_funcs[TEAM_SYNC][0] = ishmemi_openshmem_team_sync;

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
    ishmemi_proxy_funcs[TEST][INT32] = ishmemi_openshmem_int32_test;
    ishmemi_proxy_funcs[TEST_ALL][INT32] = ishmemi_openshmem_int32_test_all;
    ishmemi_proxy_funcs[TEST_ANY][INT32] = ishmemi_openshmem_int32_test_any;
    ishmemi_proxy_funcs[TEST_SOME][INT32] = ishmemi_openshmem_int32_test_some;
    ishmemi_proxy_funcs[WAIT][INT32] = ishmemi_openshmem_int32_wait_until;
    ishmemi_proxy_funcs[WAIT_ALL][INT32] = ishmemi_openshmem_int32_wait_until_all;
    ishmemi_proxy_funcs[WAIT_ANY][INT32] = ishmemi_openshmem_int32_wait_until_any;
    ishmemi_proxy_funcs[WAIT_SOME][INT32] = ishmemi_openshmem_int32_wait_until_some;

    ishmemi_proxy_funcs[TEST][INT64] = ishmemi_openshmem_int64_test;
    ishmemi_proxy_funcs[TEST_ALL][INT64] = ishmemi_openshmem_int64_test_all;
    ishmemi_proxy_funcs[TEST_ANY][INT64] = ishmemi_openshmem_int64_test_any;
    ishmemi_proxy_funcs[TEST_SOME][INT64] = ishmemi_openshmem_int64_test_some;
    ishmemi_proxy_funcs[WAIT][INT64] = ishmemi_openshmem_int64_wait_until;
    ishmemi_proxy_funcs[WAIT_ALL][INT64] = ishmemi_openshmem_int64_wait_until_all;
    ishmemi_proxy_funcs[WAIT_ANY][INT64] = ishmemi_openshmem_int64_wait_until_any;
    ishmemi_proxy_funcs[WAIT_SOME][INT64] = ishmemi_openshmem_int64_wait_until_some;

    ishmemi_proxy_funcs[TEST][LONGLONG] = ishmemi_openshmem_longlong_test;
    ishmemi_proxy_funcs[TEST_ALL][LONGLONG] = ishmemi_openshmem_longlong_test_all;
    ishmemi_proxy_funcs[TEST_ANY][LONGLONG] = ishmemi_openshmem_longlong_test_any;
    ishmemi_proxy_funcs[TEST_SOME][LONGLONG] = ishmemi_openshmem_longlong_test_some;
    ishmemi_proxy_funcs[WAIT][LONGLONG] = ishmemi_openshmem_longlong_wait_until;
    ishmemi_proxy_funcs[WAIT_ALL][LONGLONG] = ishmemi_openshmem_longlong_wait_until_all;
    ishmemi_proxy_funcs[WAIT_ANY][LONGLONG] = ishmemi_openshmem_longlong_wait_until_any;
    ishmemi_proxy_funcs[WAIT_SOME][LONGLONG] = ishmemi_openshmem_longlong_wait_until_some;

    ishmemi_proxy_funcs[TEST][UINT32] = ishmemi_openshmem_uint32_test;
    ishmemi_proxy_funcs[TEST_ALL][UINT32] = ishmemi_openshmem_uint32_test_all;
    ishmemi_proxy_funcs[TEST_ANY][UINT32] = ishmemi_openshmem_uint32_test_any;
    ishmemi_proxy_funcs[TEST_SOME][UINT32] = ishmemi_openshmem_uint32_test_some;
    ishmemi_proxy_funcs[WAIT][UINT32] = ishmemi_openshmem_uint32_wait_until;
    ishmemi_proxy_funcs[WAIT_ALL][UINT32] = ishmemi_openshmem_uint32_wait_until_all;
    ishmemi_proxy_funcs[WAIT_ANY][UINT32] = ishmemi_openshmem_uint32_wait_until_any;
    ishmemi_proxy_funcs[WAIT_SOME][UINT32] = ishmemi_openshmem_uint32_wait_until_some;

    ishmemi_proxy_funcs[TEST][UINT64] = ishmemi_openshmem_uint64_test;
    ishmemi_proxy_funcs[TEST_ALL][UINT64] = ishmemi_openshmem_uint64_test_all;
    ishmemi_proxy_funcs[TEST_ANY][UINT64] = ishmemi_openshmem_uint64_test_any;
    ishmemi_proxy_funcs[TEST_SOME][UINT64] = ishmemi_openshmem_uint64_test_some;
    ishmemi_proxy_funcs[WAIT][UINT64] = ishmemi_openshmem_uint64_wait_until;
    ishmemi_proxy_funcs[WAIT_ALL][UINT64] = ishmemi_openshmem_uint64_wait_until_all;
    ishmemi_proxy_funcs[WAIT_ANY][UINT64] = ishmemi_openshmem_uint64_wait_until_any;
    ishmemi_proxy_funcs[WAIT_SOME][UINT64] = ishmemi_openshmem_uint64_wait_until_some;

    ishmemi_proxy_funcs[TEST][ULONGLONG] = ishmemi_openshmem_ulonglong_test;
    ishmemi_proxy_funcs[TEST_ALL][ULONGLONG] = ishmemi_openshmem_ulonglong_test_all;
    ishmemi_proxy_funcs[TEST_ANY][ULONGLONG] = ishmemi_openshmem_ulonglong_test_any;
    ishmemi_proxy_funcs[TEST_SOME][ULONGLONG] = ishmemi_openshmem_ulonglong_test_some;
    ishmemi_proxy_funcs[WAIT][ULONGLONG] = ishmemi_openshmem_ulonglong_wait_until;
    ishmemi_proxy_funcs[WAIT_ALL][ULONGLONG] = ishmemi_openshmem_ulonglong_wait_until_all;
    ishmemi_proxy_funcs[WAIT_ANY][ULONGLONG] = ishmemi_openshmem_ulonglong_wait_until_any;
    ishmemi_proxy_funcs[WAIT_SOME][ULONGLONG] = ishmemi_openshmem_ulonglong_wait_until_some;

    ishmemi_proxy_funcs[SIGNAL_WAIT_UNTIL][UINT64] = ishmemi_openshmem_signal_wait_until;

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
    ishmemi_runtime_fence = ishmemi_runtime_openshmem_fence;
    ishmemi_runtime_quiet = ishmemi_runtime_openshmem_quiet;
    ishmemi_runtime_barrier_all = ishmemi_runtime_openshmem_barrier_all;
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
    ishmemi_runtime_team_split_strided = ishmemi_runtime_openshmem_team_split_strided;
    ishmemi_runtime_team_sync = ishmemi_runtime_openshmem_team_sync;
    ishmemi_runtime_team_predefined_set = ishmemi_runtime_openshmem_team_predefined_set;
    ishmemi_runtime_uchar_and_reduce = ishmemi_runtime_openshmem_uchar_and_reduce;
    ishmemi_runtime_int_max_reduce = ishmemi_runtime_openshmem_int_max_reduce;

    ishmemi_runtime_openshmem_funcptr_init();

    return ret;
}
