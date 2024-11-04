/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* Wrappers to interface with OpenSHMEM runtime */
#include "ishmem/config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "runtime.h"
#include "runtime_openshmem.h"
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

/* Enabling ishmemi_openshmem_wrappers::SHMEMX_TEAM_NODE by default which provides a team that
 * shares a compute node */
#define ISHMEMI_TEAM_NODE ishmemi_openshmem_wrappers::SHMEMX_TEAM_NODE

ishmemi_runtime_openshmem::ishmemi_runtime_openshmem(bool initialize_runtime, bool _oshmpi)
{
    int ret = 0, tl_provided;

    /* Setup OpenSHMEM dlsym links */
    ret = ishmemi_openshmem_wrappers::init_wrappers();
    ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "Failed to load SHMEM library\n");

    /* Initialize the runtime if requested */
    if (initialize_runtime && !this->initialized) {
        ret = ishmemi_openshmem_wrappers::heap_preinit_thread(SHMEM_THREAD_MULTIPLE, &tl_provided);
        ISHMEM_CHECK_GOTO_MSG(
            ret | (tl_provided != SHMEM_THREAD_MULTIPLE), fn_fail,
            "SHMEM initialization failed with ret = %d and thread level provided = %d\n", ret,
            tl_provided);
        this->initialized = true;
    }

    /* Setup internal runtime info */
    this->oshmpi = _oshmpi;
    this->rank = ishmemi_openshmem_wrappers::my_pe();
    this->size = ishmemi_openshmem_wrappers::n_pes();

    /* Initialize the function pointer table */
    this->funcptr_init();

fn_fail:
    return;
}

ishmemi_runtime_openshmem::~ishmemi_runtime_openshmem(void)
{
    /* Finalize the runtime if necessary */
    if (this->initialized) {
        ishmemi_openshmem_wrappers::finalize();
        this->initialized = false;
    }

    /* Cleanup the function pointer table */
    this->funcptr_fini();

    /* Close the shared library */
    ishmemi_openshmem_wrappers::fini_wrappers();
}

void ishmemi_runtime_openshmem::heap_create(void *base, size_t size)
{
    if (this->initialized) {
        /* only ZE device support for now */
        ishmemi_openshmem_wrappers::heap_create(base, size, SHMEMX_EXTERNAL_HEAP_ZE, 0);
        ishmemi_openshmem_wrappers::heap_postinit();
    }
}

/* Query APIs */
int ishmemi_runtime_openshmem::get_rank(void)
{
    return this->rank;
}

int ishmemi_runtime_openshmem::get_size(void)
{
    return this->size;
}

int ishmemi_runtime_openshmem::get_node_rank(int pe)
{
    return ishmemi_openshmem_wrappers::team_translate_pe(
        ishmemi_openshmem_wrappers::SHMEM_TEAM_WORLD, pe, ISHMEMI_TEAM_NODE);
}

int ishmemi_runtime_openshmem::get_node_size(void)
{
    return ishmemi_openshmem_wrappers::team_n_pes(ISHMEMI_TEAM_NODE);
}

bool ishmemi_runtime_openshmem::is_local(int pe)
{
    return (this->get_node_rank(pe) != -1);
}

bool ishmemi_runtime_openshmem::is_symmetric_address(const void *addr)
{
    return ishmemi_openshmem_wrappers::addr_accessible(addr, rank);
}

/* Memory APIs */
void *ishmemi_runtime_openshmem::malloc(size_t size)
{
    return ishmemi_openshmem_wrappers::malloc(size);
}

void *ishmemi_runtime_openshmem::calloc(size_t num, size_t size)
{
    return ishmemi_openshmem_wrappers::calloc(num, size);
}

void ishmemi_runtime_openshmem::free(void *ptr)
{
    ishmemi_openshmem_wrappers::free(ptr);
}

/* Team APIs */
int ishmemi_runtime_openshmem::team_sync(ishmemi_runtime_team_t team)
{
    return ishmemi_openshmem_wrappers::team_sync(team.shmem);
}

int ishmemi_runtime_openshmem::team_sanity_checks(ishmemi_runtime_team_predefined_t team,
                                                  int team_size, int expected_team_size,
                                                  int team_pe, int expected_team_pe, int world_pe,
                                                  int expected_world_pe)
{
    int ret = 0;
    if (team_size != expected_team_size) {
        ret = -1;
        ISHMEM_DEBUG_MSG("Expected %s team size: %d, got %d\n", this->team_predefined_string(team),
                         expected_team_size, team_size);
    }
    if (team_pe != expected_team_pe) {
        ret = -1;
        ISHMEM_DEBUG_MSG("Expected %s team PE: %d, got %d\n", this->team_predefined_string(team),
                         expected_team_pe, team_pe);
    }
    if (world_pe != expected_world_pe) {
        ret = -1;
        ISHMEM_DEBUG_MSG("Expected %s world PE: %d, got %d\n", this->team_predefined_string(team),
                         expected_world_pe, world_pe);
    }
    return ret;
}

int ishmemi_runtime_openshmem::team_predefined_set(
    ishmemi_runtime_team_t *team, ishmemi_runtime_team_predefined_t predefined_team_name,
    int expected_team_size, int expected_world_pe, int expected_team_pe)
{
    int ret = 0;
    int team_size = -1, team_pe = -1, world_pe = -1;

    switch (predefined_team_name) {
        case WORLD:
            team->shmem = ishmemi_openshmem_wrappers::SHMEM_TEAM_WORLD;
            break;
        case SHARED:
            /* The runtime's ishmemi_openshmem_wrappers::SHMEM_TEAM_SHARED may be incompatible with
             * ISHMEM_TEAM_SHARED, so check that the PE identifiers and team sizes match: */
            team_size = ishmemi_openshmem_wrappers::team_n_pes(
                ishmemi_openshmem_wrappers::SHMEM_TEAM_SHARED);
            team_pe = ishmemi_openshmem_wrappers::team_translate_pe(
                ishmemi_openshmem_wrappers::SHMEM_TEAM_WORLD, ishmemi_my_pe,
                ishmemi_openshmem_wrappers::SHMEM_TEAM_SHARED);
            world_pe = ishmemi_openshmem_wrappers::team_translate_pe(
                ishmemi_openshmem_wrappers::SHMEM_TEAM_SHARED, expected_team_pe,
                ishmemi_openshmem_wrappers::SHMEM_TEAM_WORLD);
            ret = this->team_sanity_checks(SHARED, team_size, expected_team_size, team_pe,
                                           expected_team_pe, world_pe, expected_world_pe);
            /* If the shared teams do not match, try ishmemi_openshmem_wrappers::SHMEMX_TEAM_NODE:
             */
            if (ret != 0) {
                ISHMEM_DEBUG_MSG(
                    "Incompatible ishmemi_openshmem_wrappers::SHMEM_TEAM_SHARED: instead use "
                    "ishmemi_openshmem_wrappers::SHMEMX_TEAM_NODE\n");
                team_size = ishmemi_openshmem_wrappers::team_n_pes(
                    ishmemi_openshmem_wrappers::SHMEMX_TEAM_NODE);
                team_pe = ishmemi_openshmem_wrappers::team_translate_pe(
                    ishmemi_openshmem_wrappers::SHMEM_TEAM_WORLD, ishmemi_my_pe,
                    ishmemi_openshmem_wrappers::SHMEMX_TEAM_NODE);
                world_pe = ishmemi_openshmem_wrappers::team_translate_pe(
                    ishmemi_openshmem_wrappers::SHMEMX_TEAM_NODE, expected_team_pe,
                    ishmemi_openshmem_wrappers::SHMEM_TEAM_WORLD);
                ret = this->team_sanity_checks(NODE, team_size, expected_team_size, team_pe,
                                               expected_team_pe, world_pe, expected_world_pe);
                ISHMEM_CHECK_RETURN_MSG(ret,
                                        "Runtime ISHMEM_TEAM_SHARED unable to use "
                                        "ishmemi_openshmem_wrappers::SHMEMX_TEAM_NODE\n");
                team->shmem = ishmemi_openshmem_wrappers::SHMEMX_TEAM_NODE;
            } else {
                team->shmem = ishmemi_openshmem_wrappers::SHMEM_TEAM_SHARED;
            }
            break;
        case NODE:
            team->shmem = ishmemi_openshmem_wrappers::SHMEMX_TEAM_NODE;
            break;
        default:
            return -1;
    }
    return ret;
}

int ishmemi_runtime_openshmem::team_split_strided(ishmemi_runtime_team_t parent_team, int PE_start,
                                                  int PE_stride, int PE_size,
                                                  const ishmemi_runtime_team_config_t *config,
                                                  long config_mask,
                                                  ishmemi_runtime_team_t *new_team)
{
    int ret = ishmemi_openshmem_wrappers::team_split_strided(parent_team.shmem, PE_start, PE_stride,
                                                             PE_size, &config->shmem, config_mask,
                                                             &new_team->shmem);
    return ret;
}

/* Operation APIs */
void ishmemi_runtime_openshmem::abort(int exit_code, const char msg[])
{
    std::cerr << "[ABORT] " << msg << std::endl;
    ishmemi_openshmem_wrappers::global_exit(exit_code);
}

int ishmemi_runtime_openshmem::get_kvs(int pe, char *key, void *value, size_t valuelen)
{
    return ishmemi_openshmem_wrappers::runtime_get(pe, key, value, valuelen);
}

void ishmemi_runtime_openshmem::team_destroy(ishmemi_runtime_team_t team)
{
    ishmemi_openshmem_wrappers::team_destroy(team.shmem);
}

int ishmemi_runtime_openshmem::uchar_and_reduce(ishmemi_runtime_team_t team, unsigned char *dest,
                                                const unsigned char *source, size_t nreduce)
{
    int ret = ishmemi_openshmem_wrappers::uchar_and_reduce(team.shmem, dest, source, nreduce);
    return ret;
}

int ishmemi_runtime_openshmem::int_max_reduce(ishmemi_runtime_team_t team, int *dest,
                                              const int *source, size_t nreduce)
{
    int ret = ishmemi_openshmem_wrappers::int_max_reduce(team.shmem, dest, source, nreduce);
    return ret;
}

void ishmemi_runtime_openshmem::bcast(void *buf, size_t count, int root)
{
    ishmemi_openshmem_wrappers::uint8_broadcast(ishmemi_openshmem_wrappers::SHMEM_TEAM_WORLD,
                                                (uint8_t *) buf, (uint8_t *) buf, count, root);
}

void ishmemi_runtime_openshmem::node_bcast(void *buf, size_t count, int root)
{
    ishmemi_openshmem_wrappers::uint8_broadcast(ISHMEMI_TEAM_NODE, (uint8_t *) buf, (uint8_t *) buf,
                                                count, root);
}

void ishmemi_runtime_openshmem::fcollect(void *dst, void *src, size_t count)
{
    ishmemi_openshmem_wrappers::uint8_fcollect(ishmemi_openshmem_wrappers::SHMEM_TEAM_WORLD,
                                               (uint8_t *) dst, (uint8_t *) src, count);
}

void ishmemi_runtime_openshmem::node_fcollect(void *dst, void *src, size_t count)
{
    ishmemi_openshmem_wrappers::uint8_fcollect(ISHMEMI_TEAM_NODE, (uint8_t *) dst, (uint8_t *) src,
                                               count);
}

void ishmemi_runtime_openshmem::barrier_all(void)
{
    /* Ensure L0 operations are finished */
    ishmemi_level_zero_sync();

    /* Ensure all operations faciliated by OpenSHMEM backend are finished */
    ishmemi_openshmem_wrappers::barrier_all();
}

void ishmemi_runtime_openshmem::node_barrier(void)
{
    /* TODO: quiet ctx's on ISHMEMI_TEAM_NODE */
    ishmemi_openshmem_wrappers::quiet();
    ishmemi_openshmem_wrappers::team_sync(ISHMEMI_TEAM_NODE);
}

void ishmemi_runtime_openshmem::fence(void)
{
    /* Ensure L0 operations are finished */
    ishmemi_level_zero_sync();

    /* Ensure all operations faciliated by OpenSHMEM backend are finished */
    ishmemi_openshmem_wrappers::fence();
}

void ishmemi_runtime_openshmem::quiet(void)
{
    /* Ensure L0 operations are finished */
    ishmemi_level_zero_sync();

    /* Ensure all operations faciliated by OpenSHMEM backend are finished */
    ishmemi_openshmem_wrappers::quiet();
}

void ishmemi_runtime_openshmem::sync(void)
{
    ishmemi_openshmem_wrappers::sync_all();
}

void ishmemi_runtime_openshmem::progress(void) {}

/* Private functions */
/* RMA */
int ishmemi_openshmem_uint8_put(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, PUT);
    ishmemi_openshmem_wrappers::uint8_put(dest, src, nelems, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_iput(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::iput<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, IPUT);
    wrapper(dest, src, dst, sst, nelems, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_ibput(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::ibput<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, IBPUT);
    wrapper(dest, src, dst, sst, bsize, nelems, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_p(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::p<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, P);
    wrapper(dest, val, pe);
    return 0;
}

int ishmemi_openshmem_uint8_put_nbi(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, PUT_NBI);
    ishmemi_openshmem_wrappers::uint8_put_nbi(dest, src, nelems, pe);
    return 0;
}

int ishmemi_openshmem_uint8_get(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, GET);
    ishmemi_openshmem_wrappers::uint8_get(dest, src, nelems, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_iget(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::iget<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, IGET);
    wrapper(dest, src, dst, sst, nelems, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_ibget(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::ibget<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, IBGET);
    wrapper(dest, src, dst, sst, bsize, nelems, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_g(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::g<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, G);
    ishmemi_union_set_field_value<T, G>(comp->completion.ret, wrapper(src, pe));
    return 0;
}

int ishmemi_openshmem_uint8_get_nbi(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, GET_NBI);
    ishmemi_openshmem_wrappers::uint8_get_nbi(dest, src, nelems, pe);
    return 0;
}

/* Non-blocking AMO */
template <typename T>
int ishmemi_openshmem_atomic_fetch_nbi(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_fetch_nbi<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_FETCH_NBI);
    wrapper(fetch, dest, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_compare_swap_nbi(ishmemi_request_t *msg,
                                              ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_compare_swap_nbi<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_COMPARE_SWAP_NBI);
    wrapper(fetch, dest, cond, val, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_swap_nbi(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_swap_nbi<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_SWAP_NBI);
    wrapper(fetch, dest, val, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_fetch_inc_nbi(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_fetch_inc_nbi<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_FETCH_INC_NBI);
    wrapper(fetch, dest, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_fetch_add_nbi(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_fetch_add_nbi<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_FETCH_ADD_NBI);
    wrapper(fetch, dest, val, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_fetch_and_nbi(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_fetch_and_nbi<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_FETCH_AND_NBI);
    wrapper(fetch, dest, val, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_fetch_or_nbi(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_fetch_or_nbi<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_FETCH_OR_NBI);
    wrapper(fetch, dest, val, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_fetch_xor_nbi(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_fetch_xor_nbi<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_FETCH_XOR_NBI);
    wrapper(fetch, dest, val, pe);
    return 0;
}

/* AMO */
template <typename T>
int ishmemi_openshmem_atomic_fetch(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_fetch<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_FETCH);
    ishmemi_union_set_field_value<T, AMO_FETCH>(comp->completion.ret, wrapper(src, pe));
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_set(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_set<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_SET);
    wrapper(dest, val, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_compare_swap(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_compare_swap<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_COMPARE_SWAP);
    ishmemi_union_set_field_value<T, AMO_COMPARE_SWAP>(comp->completion.ret,
                                                       wrapper(dest, cond, val, pe));
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_swap(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_swap<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_SWAP);
    ishmemi_union_set_field_value<T, AMO_SWAP>(comp->completion.ret, wrapper(dest, val, pe));
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_fetch_inc(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_fetch_inc<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_FETCH_INC);
    ishmemi_union_set_field_value<T, AMO_FETCH_INC>(comp->completion.ret, wrapper(dest, pe));
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_fetch_add(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_fetch_add<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_FETCH_ADD);
    ishmemi_union_set_field_value<T, AMO_FETCH_ADD>(comp->completion.ret, wrapper(dest, val, pe));
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_fetch_and(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_fetch_and<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_FETCH_AND);
    ishmemi_union_set_field_value<T, AMO_FETCH_AND>(comp->completion.ret, wrapper(dest, val, pe));
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_fetch_or(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_fetch_or<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_FETCH_OR);
    ishmemi_union_set_field_value<T, AMO_FETCH_OR>(comp->completion.ret, wrapper(dest, val, pe));
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_fetch_xor(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_fetch_xor<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_FETCH_XOR);
    ishmemi_union_set_field_value<T, AMO_FETCH_XOR>(comp->completion.ret, wrapper(dest, val, pe));
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_inc(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_inc<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_INC);
    wrapper(dest, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_add(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_add<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_ADD);
    wrapper(dest, val, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_and(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_and<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_AND);
    wrapper(dest, val, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_or(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_or<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_OR);
    wrapper(dest, val, pe);
    return 0;
}

template <typename T>
int ishmemi_openshmem_atomic_xor(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::atomic_xor<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, AMO_XOR);
    wrapper(dest, val, pe);
    return 0;
}

/* Signaling */
int ishmemi_openshmem_uint8_put_signal(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, PUT_SIGNAL);
    ishmemi_openshmem_wrappers::uint8_put_signal(dest, src, nelems, sig_addr, signal, sig_op, pe);
    return 0;
}

int ishmemi_openshmem_uint8_put_signal_nbi(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, PUT_SIGNAL_NBI);
    ishmemi_openshmem_wrappers::uint8_put_signal_nbi(dest, src, nelems, sig_addr, signal, sig_op,
                                                     pe);
    return 0;
}

int ishmemi_openshmem_signal_fetch(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, SIGNAL_FETCH);
    comp->completion.ret.ui64 = ishmemi_openshmem_wrappers::signal_fetch(sig_addr);
    return 0;
}

int ishmemi_openshmem_signal_add(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, AMO_ADD);
    /* TODO: Use signal add */
    ishmemi_openshmem_wrappers::uint64_atomic_add(dest, val, pe);
    return 0;
}

int ishmemi_openshmem_signal_set(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, AMO_SET);
    /* TODO: Use signal set */
    ishmemi_openshmem_wrappers::uint64_atomic_set(dest, val, pe);
    return 0;
}

/* Teams */
int ishmemi_openshmem_team_my_pe(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, TEAM_MY_PE);
    comp->completion.ret.i = ishmemi_openshmem_wrappers::team_my_pe(team_ptr->runtime_team.shmem);
    return comp->completion.ret.i;
}

int ishmemi_openshmem_team_n_pes(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, TEAM_N_PES);
    comp->completion.ret.i = ishmemi_openshmem_wrappers::team_n_pes(team_ptr->runtime_team.shmem);
    return comp->completion.ret.i;
}

int ishmemi_openshmem_team_sync(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, TEAM_SYNC);
    comp->completion.ret.i = ishmemi_openshmem_wrappers::team_sync(team_ptr->runtime_team.shmem);
    return comp->completion.ret.i;
}

/* Collectives */
int ishmemi_openshmem_barrier_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    /* Ensure L0 operations are finished */
    ishmemi_level_zero_sync();

    /* Ensure all operations faciliated by OpenSHMEM backend are finished */
    ishmemi_openshmem_wrappers::barrier_all();
    return 0;
}

int ishmemi_openshmem_sync_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ishmemi_openshmem_wrappers::sync_all();
    return 0;
}

int ishmemi_openshmem_uint8_alltoall(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ALLTOALL);
    comp->completion.ret.i =
        ishmemi_openshmem_wrappers::uint8_alltoall(team_ptr->runtime_team.shmem, dest, src, nelems);
    return comp->completion.ret.i;
}

int ishmemi_openshmem_uint8_broadcast(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, BCAST);
    comp->completion.ret.i = ishmemi_openshmem_wrappers::uint8_broadcast(
        team_ptr->runtime_team.shmem, dest, src, nelems, root);
    return comp->completion.ret.i;
}

int ishmemi_openshmem_uint8_collect(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, COLLECT);
    comp->completion.ret.i =
        ishmemi_openshmem_wrappers::uint8_collect(team_ptr->runtime_team.shmem, dest, src, nelems);
    return comp->completion.ret.i;
}

int ishmemi_openshmem_uint8_fcollect(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, FCOLLECT);
    comp->completion.ret.i =
        ishmemi_openshmem_wrappers::uint8_fcollect(team_ptr->runtime_team.shmem, dest, src, nelems);
    return comp->completion.ret.i;
}

/* Reductions */
template <typename T, ishmemi_op_t OP>
int ishmemi_openshmem_reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, OP);
    assert(OP == msg->op);

    if constexpr (ishmemi_op_is_reduction<OP>()) {
        auto wrapper = ishmemi_openshmem_wrappers::reduce<T, OP>();
        comp->completion.ret.i = wrapper(team_ptr->runtime_team.shmem, dest, src, nelems);
    } else {
        /* This branch will only occur if the template wrappers or the proxy_funcs are incorrectly
         * defined */
        ISHMEM_ERROR_MSG("Encountered unknown reduction operation '%s'\n", ishmemi_op_str[OP]);
        comp->completion.ret.i = -1;
        ishmemi_cpu_info->proxy_state = EXIT;
    }
    return comp->completion.ret.i;
}

/* Point-to-Point Synchronization */
template <typename T, bool OSHMPI>
int ishmemi_openshmem_test(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::test<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, TEST);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(dest)) {
            dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, dest);
        }
    }

    comp->completion.ret.i = wrapper(dest, cmp, cmp_value);
    return comp->completion.ret.i;
}

template <typename T, bool OSHMPI>
int ishmemi_openshmem_test_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::test_all<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, TEST_ALL);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(dest)) {
            dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, dest);
        }
        if (ISHMEMI_HOST_IN_HEAP(status)) {
            status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
        }
    }

    comp->completion.ret.i = wrapper(dest, nelems, status, cmp, cmp_value);
    return comp->completion.ret.i;
}

template <typename T, bool OSHMPI>
int ishmemi_openshmem_test_any(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::test_any<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, TEST_ANY);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(dest)) {
            dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, dest);
        }
        if (ISHMEMI_HOST_IN_HEAP(status)) {
            status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
        }
    }

    comp->completion.ret.szt = wrapper(dest, nelems, status, cmp, cmp_value);
    return 0;
}

template <typename T, bool OSHMPI>
int ishmemi_openshmem_test_some(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::test_some<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, TEST_SOME);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(dest)) {
            dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, dest);
        }
        if (ISHMEMI_HOST_IN_HEAP(indices)) {
            indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
        }
        if (ISHMEMI_HOST_IN_HEAP(status)) {
            status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
        }
    }

    comp->completion.ret.szt = wrapper(dest, nelems, indices, status, cmp, cmp_value);
    return 0;
}

template <typename T, bool OSHMPI>
int ishmemi_openshmem_wait_until(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::wait_until<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, WAIT);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(dest)) {
            dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, dest);
        }
    }

    wrapper(dest, cmp, cmp_value);
    return 0;
}

template <typename T, bool OSHMPI>
int ishmemi_openshmem_wait_until_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::wait_until_all<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, WAIT_ALL);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(dest)) {
            dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, dest);
        }
        if (ISHMEMI_HOST_IN_HEAP(status)) {
            status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
        }
    }

    wrapper(dest, nelems, status, cmp, cmp_value);
    return 0;
}

template <typename T, bool OSHMPI>
int ishmemi_openshmem_wait_until_any(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::wait_until_any<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, WAIT_ANY);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(dest)) {
            dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, dest);
        }
        if (ISHMEMI_HOST_IN_HEAP(status)) {
            status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
        }
    }

    comp->completion.ret.szt = wrapper(dest, nelems, status, cmp, cmp_value);
    return 0;
}

template <typename T, bool OSHMPI>
int ishmemi_openshmem_wait_until_some(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::wait_until_some<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, WAIT_SOME);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(dest)) {
            dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, dest);
        }
        if (ISHMEMI_HOST_IN_HEAP(indices)) {
            indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
        }
        if (ISHMEMI_HOST_IN_HEAP(status)) {
            status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
        }
    }

    comp->completion.ret.szt = wrapper(dest, nelems, indices, status, cmp, cmp_value);
    return 0;
}

template <typename T, bool OSHMPI>
int ishmemi_openshmem_test_all_vector(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::test_all_vector<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, TEST_ALL_VECTOR);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(dest)) {
            dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, dest);
        }
        if (ISHMEMI_HOST_IN_HEAP(status)) {
            status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
        }
        if (ISHMEMI_HOST_IN_HEAP(cmp_values)) {
            cmp_values = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, cmp_values);
        }
    }

    comp->completion.ret.i = wrapper(dest, nelems, status, cmp, const_cast<T *>(cmp_values));
    return comp->completion.ret.i;
}

template <typename T, bool OSHMPI>
int ishmemi_openshmem_test_any_vector(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::test_any_vector<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, TEST_ANY_VECTOR);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(dest)) {
            dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, dest);
        }
        if (ISHMEMI_HOST_IN_HEAP(status)) {
            status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
        }
        if (ISHMEMI_HOST_IN_HEAP(cmp_values)) {
            cmp_values = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, cmp_values);
        }
    }

    comp->completion.ret.szt = wrapper(dest, nelems, status, cmp, const_cast<T *>(cmp_values));
    return 0;
}

template <typename T, bool OSHMPI>
int ishmemi_openshmem_test_some_vector(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::test_some_vector<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, TEST_SOME_VECTOR);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(dest)) {
            dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, dest);
        }
        if (ISHMEMI_HOST_IN_HEAP(indices)) {
            indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
        }
        if (ISHMEMI_HOST_IN_HEAP(status)) {
            status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
        }
        if (ISHMEMI_HOST_IN_HEAP(cmp_values)) {
            cmp_values = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, cmp_values);
        }
    }

    comp->completion.ret.szt =
        wrapper(dest, nelems, indices, status, cmp, const_cast<T *>(cmp_values));
    return 0;
}

template <typename T, bool OSHMPI>
int ishmemi_openshmem_wait_until_all_vector(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::wait_until_all_vector<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, WAIT_ALL_VECTOR);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(dest)) {
            dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, dest);
        }
        if (ISHMEMI_HOST_IN_HEAP(status)) {
            status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
        }
        if (ISHMEMI_HOST_IN_HEAP(cmp_values)) {
            cmp_values = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, cmp_values);
        }
    }

    wrapper(dest, nelems, status, cmp, const_cast<T *>(cmp_values));
    return 0;
}

template <typename T, bool OSHMPI>
int ishmemi_openshmem_wait_until_any_vector(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::wait_until_any_vector<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, WAIT_ANY_VECTOR);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(dest)) {
            dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, dest);
        }
        if (ISHMEMI_HOST_IN_HEAP(status)) {
            status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
        }
        if (ISHMEMI_HOST_IN_HEAP(cmp_values)) {
            cmp_values = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, cmp_values);
        }
    }

    comp->completion.ret.szt = wrapper(dest, nelems, status, cmp, const_cast<T *>(cmp_values));
    return 0;
}

template <typename T, bool OSHMPI>
int ishmemi_openshmem_wait_until_some_vector(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    auto wrapper = ishmemi_openshmem_wrappers::wait_until_some_vector<T>();
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, WAIT_SOME_VECTOR);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(dest)) {
            dest = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, dest);
        }
        if (ISHMEMI_HOST_IN_HEAP(indices)) {
            indices = ISHMEMI_DEVICE_TO_MMAP_ADDR(size_t, indices);
        }
        if (ISHMEMI_HOST_IN_HEAP(status)) {
            status = ISHMEMI_DEVICE_TO_MMAP_ADDR(int, status);
        }
        if (ISHMEMI_HOST_IN_HEAP(cmp_values)) {
            cmp_values = ISHMEMI_DEVICE_TO_MMAP_ADDR(T, cmp_values);
        }
    }

    comp->completion.ret.szt =
        wrapper(dest, nelems, indices, status, cmp, const_cast<T *>(cmp_values));
    return 0;
}

template <bool OSHMPI>
int ishmemi_openshmem_signal_wait_until(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, SIGNAL_WAIT_UNTIL);

    if constexpr (!OSHMPI) {
        if (ISHMEMI_HOST_IN_HEAP(sig_addr)) {
            sig_addr = ISHMEMI_DEVICE_TO_MMAP_ADDR(uint64_t, sig_addr);
        }
    }

    comp->completion.ret.ui64 =
        ishmemi_openshmem_wrappers::signal_wait_until(sig_addr, cmp, cmp_value);
    return 0;
}

/* Memory Ordering */
int ishmemi_openshmem_fence(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    /* Ensure L0 operations are finished */
    ishmemi_level_zero_sync();

    /* Ensure all operations faciliated by OpenSHMEM backend are finished */
    ishmemi_openshmem_wrappers::fence();
    return 0;
}

int ishmemi_openshmem_quiet(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    /* Ensure L0 operations are finished */
    ishmemi_level_zero_sync();

    /* Ensure all operations faciliated by OpenSHMEM backend are finished */
    ishmemi_openshmem_wrappers::quiet();
    return 0;
}

/* Function table */
void ishmemi_runtime_openshmem::funcptr_init()
{
    proxy_funcs = (ishmemi_runtime_proxy_func_t **) ::malloc(
        sizeof(ishmemi_runtime_proxy_func_t *) * ISHMEMI_OP_END);
    ISHMEM_CHECK_GOTO_MSG(proxy_funcs == nullptr, fn_exit, "Allocation of proxy_funcs failed\n");

    /* Initialize every function with the "unsupported op" function */
    /* Note: KILL operation is covered inside the proxy directly - it is the same for all backends
     * currently */
    for (int i = 0; i < ISHMEMI_OP_END; ++i) {
        proxy_funcs[i] = (ishmemi_runtime_proxy_func_t *) ::malloc(
            sizeof(ishmemi_runtime_proxy_func_t) * ishmemi_runtime_type::proxy_func_num_types);
        for (int j = 0; j < ishmemi_runtime_type::proxy_func_num_types; ++j) {
            proxy_funcs[i][j] = ishmemi_runtime_type::unsupported;
        }
    }

    /* Fill in the supported functions */
    /* RMA */
    proxy_funcs[PUT][UINT8] = ishmemi_openshmem_uint8_put;
    proxy_funcs[IPUT][UINT8] = ishmemi_openshmem_iput<uint8_t>;
    proxy_funcs[IPUT][UINT16] = ishmemi_openshmem_iput<uint16_t>;
    proxy_funcs[IPUT][UINT32] = ishmemi_openshmem_iput<uint32_t>;
    proxy_funcs[IPUT][UINT64] = ishmemi_openshmem_iput<uint64_t>;
    proxy_funcs[IPUT][ULONGLONG] = ishmemi_openshmem_iput<unsigned long long>;
    proxy_funcs[IBPUT][UINT8] = ishmemi_openshmem_ibput<uint8_t>;
    proxy_funcs[IBPUT][UINT16] = ishmemi_openshmem_ibput<uint16_t>;
    proxy_funcs[IBPUT][UINT32] = ishmemi_openshmem_ibput<uint32_t>;
    proxy_funcs[IBPUT][UINT64] = ishmemi_openshmem_ibput<uint64_t>;
    proxy_funcs[P][UINT8] = ishmemi_openshmem_p<uint8_t>;
    proxy_funcs[P][UINT16] = ishmemi_openshmem_p<uint16_t>;
    proxy_funcs[P][UINT32] = ishmemi_openshmem_p<uint32_t>;
    proxy_funcs[P][UINT64] = ishmemi_openshmem_p<uint64_t>;
    proxy_funcs[P][ULONGLONG] = ishmemi_openshmem_p<unsigned long long>;
    proxy_funcs[P][FLOAT] = ishmemi_openshmem_p<float>;
    proxy_funcs[P][DOUBLE] = ishmemi_openshmem_p<double>;
    proxy_funcs[PUT_NBI][UINT8] = ishmemi_openshmem_uint8_put_nbi;

    proxy_funcs[GET][UINT8] = ishmemi_openshmem_uint8_get;
    proxy_funcs[IGET][UINT8] = ishmemi_openshmem_iget<uint8_t>;
    proxy_funcs[IGET][UINT16] = ishmemi_openshmem_iget<uint16_t>;
    proxy_funcs[IGET][UINT32] = ishmemi_openshmem_iget<uint32_t>;
    proxy_funcs[IGET][UINT64] = ishmemi_openshmem_iget<uint64_t>;
    proxy_funcs[IGET][ULONGLONG] = ishmemi_openshmem_iget<unsigned long long>;
    proxy_funcs[IBGET][UINT8] = ishmemi_openshmem_ibget<uint8_t>;
    proxy_funcs[IBGET][UINT16] = ishmemi_openshmem_ibget<uint16_t>;
    proxy_funcs[IBGET][UINT32] = ishmemi_openshmem_ibget<uint32_t>;
    proxy_funcs[IBGET][UINT64] = ishmemi_openshmem_ibget<uint64_t>;
    proxy_funcs[G][UINT8] = ishmemi_openshmem_g<uint8_t>;
    proxy_funcs[G][UINT16] = ishmemi_openshmem_g<uint16_t>;
    proxy_funcs[G][UINT32] = ishmemi_openshmem_g<uint32_t>;
    proxy_funcs[G][UINT64] = ishmemi_openshmem_g<uint64_t>;
    proxy_funcs[G][ULONGLONG] = ishmemi_openshmem_g<unsigned long long>;
    proxy_funcs[G][FLOAT] = ishmemi_openshmem_g<float>;
    proxy_funcs[G][DOUBLE] = ishmemi_openshmem_g<double>;
    proxy_funcs[GET_NBI][UINT8] = ishmemi_openshmem_uint8_get_nbi;

    /* AMO */
    proxy_funcs[AMO_FETCH][UINT32] = ishmemi_openshmem_atomic_fetch<uint32_t>;
    proxy_funcs[AMO_SET][UINT32] = ishmemi_openshmem_atomic_set<uint32_t>;
    proxy_funcs[AMO_COMPARE_SWAP][UINT32] = ishmemi_openshmem_atomic_compare_swap<uint32_t>;
    proxy_funcs[AMO_SWAP][UINT32] = ishmemi_openshmem_atomic_swap<uint32_t>;
    proxy_funcs[AMO_FETCH_INC][UINT32] = ishmemi_openshmem_atomic_fetch_inc<uint32_t>;
    proxy_funcs[AMO_INC][UINT32] = ishmemi_openshmem_atomic_inc<uint32_t>;
    proxy_funcs[AMO_FETCH_ADD][UINT32] = ishmemi_openshmem_atomic_fetch_add<uint32_t>;
    proxy_funcs[AMO_ADD][UINT32] = ishmemi_openshmem_atomic_add<uint32_t>;
    proxy_funcs[AMO_FETCH_AND][UINT32] = ishmemi_openshmem_atomic_fetch_and<uint32_t>;
    proxy_funcs[AMO_AND][UINT32] = ishmemi_openshmem_atomic_and<uint32_t>;
    proxy_funcs[AMO_FETCH_OR][UINT32] = ishmemi_openshmem_atomic_fetch_or<uint32_t>;
    proxy_funcs[AMO_OR][UINT32] = ishmemi_openshmem_atomic_or<uint32_t>;
    proxy_funcs[AMO_FETCH_XOR][UINT32] = ishmemi_openshmem_atomic_fetch_xor<uint32_t>;
    proxy_funcs[AMO_XOR][UINT32] = ishmemi_openshmem_atomic_xor<uint32_t>;

    proxy_funcs[AMO_FETCH][UINT64] = ishmemi_openshmem_atomic_fetch<uint64_t>;
    proxy_funcs[AMO_SET][UINT64] = ishmemi_openshmem_atomic_set<uint64_t>;
    proxy_funcs[AMO_COMPARE_SWAP][UINT64] = ishmemi_openshmem_atomic_compare_swap<uint64_t>;
    proxy_funcs[AMO_SWAP][UINT64] = ishmemi_openshmem_atomic_swap<uint64_t>;
    proxy_funcs[AMO_FETCH_INC][UINT64] = ishmemi_openshmem_atomic_fetch_inc<uint64_t>;
    proxy_funcs[AMO_INC][UINT64] = ishmemi_openshmem_atomic_inc<uint64_t>;
    proxy_funcs[AMO_FETCH_ADD][UINT64] = ishmemi_openshmem_atomic_fetch_add<uint64_t>;
    proxy_funcs[AMO_ADD][UINT64] = ishmemi_openshmem_atomic_add<uint64_t>;
    proxy_funcs[AMO_FETCH_AND][UINT64] = ishmemi_openshmem_atomic_fetch_and<uint64_t>;
    proxy_funcs[AMO_AND][UINT64] = ishmemi_openshmem_atomic_and<uint64_t>;
    proxy_funcs[AMO_FETCH_OR][UINT64] = ishmemi_openshmem_atomic_fetch_or<uint64_t>;
    proxy_funcs[AMO_OR][UINT64] = ishmemi_openshmem_atomic_or<uint64_t>;
    proxy_funcs[AMO_FETCH_XOR][UINT64] = ishmemi_openshmem_atomic_fetch_xor<uint64_t>;
    proxy_funcs[AMO_XOR][UINT64] = ishmemi_openshmem_atomic_xor<uint64_t>;

    proxy_funcs[AMO_FETCH][ULONGLONG] = ishmemi_openshmem_atomic_fetch<unsigned long long>;
    proxy_funcs[AMO_SET][ULONGLONG] = ishmemi_openshmem_atomic_set<unsigned long long>;
    proxy_funcs[AMO_COMPARE_SWAP][ULONGLONG] =
        ishmemi_openshmem_atomic_compare_swap<unsigned long long>;
    proxy_funcs[AMO_SWAP][ULONGLONG] = ishmemi_openshmem_atomic_swap<unsigned long long>;
    proxy_funcs[AMO_FETCH_INC][ULONGLONG] = ishmemi_openshmem_atomic_fetch_inc<unsigned long long>;
    proxy_funcs[AMO_INC][ULONGLONG] = ishmemi_openshmem_atomic_inc<unsigned long long>;
    proxy_funcs[AMO_FETCH_ADD][ULONGLONG] = ishmemi_openshmem_atomic_fetch_add<unsigned long long>;
    proxy_funcs[AMO_ADD][ULONGLONG] = ishmemi_openshmem_atomic_add<unsigned long long>;
    proxy_funcs[AMO_FETCH_AND][ULONGLONG] = ishmemi_openshmem_atomic_fetch_and<unsigned long long>;
    proxy_funcs[AMO_AND][ULONGLONG] = ishmemi_openshmem_atomic_and<unsigned long long>;
    proxy_funcs[AMO_FETCH_OR][ULONGLONG] = ishmemi_openshmem_atomic_fetch_or<unsigned long long>;
    proxy_funcs[AMO_OR][ULONGLONG] = ishmemi_openshmem_atomic_or<unsigned long long>;
    proxy_funcs[AMO_FETCH_XOR][ULONGLONG] = ishmemi_openshmem_atomic_fetch_xor<unsigned long long>;
    proxy_funcs[AMO_XOR][ULONGLONG] = ishmemi_openshmem_atomic_xor<unsigned long long>;

    proxy_funcs[AMO_FETCH][INT32] = ishmemi_openshmem_atomic_fetch<int32_t>;
    proxy_funcs[AMO_SET][INT32] = ishmemi_openshmem_atomic_set<int32_t>;
    proxy_funcs[AMO_COMPARE_SWAP][INT32] = ishmemi_openshmem_atomic_compare_swap<int32_t>;
    proxy_funcs[AMO_SWAP][INT32] = ishmemi_openshmem_atomic_swap<int32_t>;
    proxy_funcs[AMO_FETCH_INC][INT32] = ishmemi_openshmem_atomic_fetch_inc<int32_t>;
    proxy_funcs[AMO_INC][INT32] = ishmemi_openshmem_atomic_inc<int32_t>;
    proxy_funcs[AMO_FETCH_ADD][INT32] = ishmemi_openshmem_atomic_fetch_add<int32_t>;
    proxy_funcs[AMO_ADD][INT32] = ishmemi_openshmem_atomic_add<int32_t>;
    proxy_funcs[AMO_FETCH_AND][INT32] = ishmemi_openshmem_atomic_fetch_and<int32_t>;
    proxy_funcs[AMO_AND][INT32] = ishmemi_openshmem_atomic_and<int32_t>;
    proxy_funcs[AMO_FETCH_OR][INT32] = ishmemi_openshmem_atomic_fetch_or<int32_t>;
    proxy_funcs[AMO_OR][INT32] = ishmemi_openshmem_atomic_or<int32_t>;
    proxy_funcs[AMO_FETCH_XOR][INT32] = ishmemi_openshmem_atomic_fetch_xor<int32_t>;
    proxy_funcs[AMO_XOR][INT32] = ishmemi_openshmem_atomic_xor<int32_t>;

    proxy_funcs[AMO_FETCH][INT64] = ishmemi_openshmem_atomic_fetch<int64_t>;
    proxy_funcs[AMO_SET][INT64] = ishmemi_openshmem_atomic_set<int64_t>;
    proxy_funcs[AMO_COMPARE_SWAP][INT64] = ishmemi_openshmem_atomic_compare_swap<int64_t>;
    proxy_funcs[AMO_SWAP][INT64] = ishmemi_openshmem_atomic_swap<int64_t>;
    proxy_funcs[AMO_FETCH_INC][INT64] = ishmemi_openshmem_atomic_fetch_inc<int64_t>;
    proxy_funcs[AMO_INC][INT64] = ishmemi_openshmem_atomic_inc<int64_t>;
    proxy_funcs[AMO_FETCH_ADD][INT64] = ishmemi_openshmem_atomic_fetch_add<int64_t>;
    proxy_funcs[AMO_ADD][INT64] = ishmemi_openshmem_atomic_add<int64_t>;
    proxy_funcs[AMO_FETCH_AND][INT64] = ishmemi_openshmem_atomic_fetch_and<int64_t>;
    proxy_funcs[AMO_AND][INT64] = ishmemi_openshmem_atomic_and<int64_t>;
    proxy_funcs[AMO_FETCH_OR][INT64] = ishmemi_openshmem_atomic_fetch_or<int64_t>;
    proxy_funcs[AMO_OR][INT64] = ishmemi_openshmem_atomic_or<int64_t>;
    proxy_funcs[AMO_FETCH_XOR][INT64] = ishmemi_openshmem_atomic_fetch_xor<int64_t>;
    proxy_funcs[AMO_XOR][INT64] = ishmemi_openshmem_atomic_xor<int64_t>;

    proxy_funcs[AMO_FETCH][LONGLONG] = ishmemi_openshmem_atomic_fetch<long long>;
    proxy_funcs[AMO_SET][LONGLONG] = ishmemi_openshmem_atomic_set<long long>;
    proxy_funcs[AMO_COMPARE_SWAP][LONGLONG] = ishmemi_openshmem_atomic_compare_swap<long long>;
    proxy_funcs[AMO_SWAP][LONGLONG] = ishmemi_openshmem_atomic_swap<long long>;
    proxy_funcs[AMO_FETCH_INC][LONGLONG] = ishmemi_openshmem_atomic_fetch_inc<long long>;
    proxy_funcs[AMO_INC][LONGLONG] = ishmemi_openshmem_atomic_inc<long long>;
    proxy_funcs[AMO_FETCH_ADD][LONGLONG] = ishmemi_openshmem_atomic_fetch_add<long long>;
    proxy_funcs[AMO_ADD][LONGLONG] = ishmemi_openshmem_atomic_add<long long>;

    proxy_funcs[AMO_FETCH][FLOAT] = ishmemi_openshmem_atomic_fetch<float>;
    proxy_funcs[AMO_SET][FLOAT] = ishmemi_openshmem_atomic_set<float>;
    proxy_funcs[AMO_SWAP][FLOAT] = ishmemi_openshmem_atomic_swap<float>;

    proxy_funcs[AMO_FETCH][DOUBLE] = ishmemi_openshmem_atomic_fetch<double>;
    proxy_funcs[AMO_SET][DOUBLE] = ishmemi_openshmem_atomic_set<double>;
    proxy_funcs[AMO_SWAP][DOUBLE] = ishmemi_openshmem_atomic_swap<double>;

    /* AMO NBI */
    proxy_funcs[AMO_FETCH_NBI][UINT32] = ishmemi_openshmem_atomic_fetch_nbi<uint32_t>;
    proxy_funcs[AMO_FETCH_NBI][INT32] = ishmemi_openshmem_atomic_fetch_nbi<int32_t>;
    proxy_funcs[AMO_FETCH_NBI][UINT64] = ishmemi_openshmem_atomic_fetch_nbi<uint64_t>;
    proxy_funcs[AMO_FETCH_NBI][INT64] = ishmemi_openshmem_atomic_fetch_nbi<int64_t>;
    proxy_funcs[AMO_FETCH_NBI][ULONGLONG] = ishmemi_openshmem_atomic_fetch_nbi<unsigned long long>;
    proxy_funcs[AMO_FETCH_NBI][LONGLONG] = ishmemi_openshmem_atomic_fetch_nbi<long long>;
    proxy_funcs[AMO_FETCH_NBI][FLOAT] = ishmemi_openshmem_atomic_fetch_nbi<float>;
    proxy_funcs[AMO_FETCH_NBI][DOUBLE] = ishmemi_openshmem_atomic_fetch_nbi<double>;

    proxy_funcs[AMO_COMPARE_SWAP_NBI][UINT32] = ishmemi_openshmem_atomic_compare_swap_nbi<uint32_t>;
    proxy_funcs[AMO_COMPARE_SWAP_NBI][INT32] = ishmemi_openshmem_atomic_compare_swap_nbi<int32_t>;
    proxy_funcs[AMO_COMPARE_SWAP_NBI][UINT64] = ishmemi_openshmem_atomic_compare_swap_nbi<uint64_t>;
    proxy_funcs[AMO_COMPARE_SWAP_NBI][INT64] = ishmemi_openshmem_atomic_compare_swap_nbi<int64_t>;
    proxy_funcs[AMO_COMPARE_SWAP_NBI][ULONGLONG] =
        ishmemi_openshmem_atomic_compare_swap_nbi<unsigned long long>;
    proxy_funcs[AMO_COMPARE_SWAP_NBI][LONGLONG] =
        ishmemi_openshmem_atomic_compare_swap_nbi<long long>;

    proxy_funcs[AMO_SWAP_NBI][UINT32] = ishmemi_openshmem_atomic_swap_nbi<uint32_t>;
    proxy_funcs[AMO_SWAP_NBI][INT32] = ishmemi_openshmem_atomic_swap_nbi<int32_t>;
    proxy_funcs[AMO_SWAP_NBI][UINT64] = ishmemi_openshmem_atomic_swap_nbi<uint64_t>;
    proxy_funcs[AMO_SWAP_NBI][INT64] = ishmemi_openshmem_atomic_swap_nbi<int64_t>;
    proxy_funcs[AMO_SWAP_NBI][ULONGLONG] = ishmemi_openshmem_atomic_swap_nbi<unsigned long long>;
    proxy_funcs[AMO_SWAP_NBI][LONGLONG] = ishmemi_openshmem_atomic_swap_nbi<long long>;
    proxy_funcs[AMO_SWAP_NBI][FLOAT] = ishmemi_openshmem_atomic_swap_nbi<float>;
    proxy_funcs[AMO_SWAP_NBI][DOUBLE] = ishmemi_openshmem_atomic_swap_nbi<double>;

    proxy_funcs[AMO_FETCH_INC_NBI][UINT32] = ishmemi_openshmem_atomic_fetch_inc_nbi<uint32_t>;
    proxy_funcs[AMO_FETCH_INC_NBI][INT32] = ishmemi_openshmem_atomic_fetch_inc_nbi<int32_t>;
    proxy_funcs[AMO_FETCH_INC_NBI][UINT64] = ishmemi_openshmem_atomic_fetch_inc_nbi<uint64_t>;
    proxy_funcs[AMO_FETCH_INC_NBI][INT64] = ishmemi_openshmem_atomic_fetch_inc_nbi<int64_t>;
    proxy_funcs[AMO_FETCH_INC_NBI][ULONGLONG] =
        ishmemi_openshmem_atomic_fetch_inc_nbi<unsigned long long>;
    proxy_funcs[AMO_FETCH_INC_NBI][LONGLONG] = ishmemi_openshmem_atomic_fetch_inc_nbi<long long>;

    proxy_funcs[AMO_FETCH_ADD_NBI][UINT32] = ishmemi_openshmem_atomic_fetch_add_nbi<uint32_t>;
    proxy_funcs[AMO_FETCH_ADD_NBI][INT32] = ishmemi_openshmem_atomic_fetch_add_nbi<int32_t>;
    proxy_funcs[AMO_FETCH_ADD_NBI][UINT64] = ishmemi_openshmem_atomic_fetch_add_nbi<uint64_t>;
    proxy_funcs[AMO_FETCH_ADD_NBI][INT64] = ishmemi_openshmem_atomic_fetch_add_nbi<int64_t>;
    proxy_funcs[AMO_FETCH_ADD_NBI][ULONGLONG] =
        ishmemi_openshmem_atomic_fetch_add_nbi<unsigned long long>;
    proxy_funcs[AMO_FETCH_ADD_NBI][LONGLONG] = ishmemi_openshmem_atomic_fetch_add_nbi<long long>;

    proxy_funcs[AMO_FETCH_AND_NBI][UINT32] = ishmemi_openshmem_atomic_fetch_and_nbi<uint32_t>;
    proxy_funcs[AMO_FETCH_AND_NBI][INT32] = ishmemi_openshmem_atomic_fetch_and_nbi<int32_t>;
    proxy_funcs[AMO_FETCH_AND_NBI][UINT64] = ishmemi_openshmem_atomic_fetch_and_nbi<uint64_t>;
    proxy_funcs[AMO_FETCH_AND_NBI][INT64] = ishmemi_openshmem_atomic_fetch_and_nbi<int64_t>;
    proxy_funcs[AMO_FETCH_AND_NBI][ULONGLONG] =
        ishmemi_openshmem_atomic_fetch_and_nbi<unsigned long long>;

    proxy_funcs[AMO_FETCH_OR_NBI][UINT32] = ishmemi_openshmem_atomic_fetch_or_nbi<uint32_t>;
    proxy_funcs[AMO_FETCH_OR_NBI][INT32] = ishmemi_openshmem_atomic_fetch_or_nbi<int32_t>;
    proxy_funcs[AMO_FETCH_OR_NBI][UINT64] = ishmemi_openshmem_atomic_fetch_or_nbi<uint64_t>;
    proxy_funcs[AMO_FETCH_OR_NBI][INT64] = ishmemi_openshmem_atomic_fetch_or_nbi<int64_t>;
    proxy_funcs[AMO_FETCH_OR_NBI][ULONGLONG] =
        ishmemi_openshmem_atomic_fetch_or_nbi<unsigned long long>;

    proxy_funcs[AMO_FETCH_XOR_NBI][UINT32] = ishmemi_openshmem_atomic_fetch_xor_nbi<uint32_t>;
    proxy_funcs[AMO_FETCH_XOR_NBI][INT32] = ishmemi_openshmem_atomic_fetch_xor_nbi<int32_t>;
    proxy_funcs[AMO_FETCH_XOR_NBI][UINT64] = ishmemi_openshmem_atomic_fetch_xor_nbi<uint64_t>;
    proxy_funcs[AMO_FETCH_XOR_NBI][INT64] = ishmemi_openshmem_atomic_fetch_xor_nbi<int64_t>;
    proxy_funcs[AMO_FETCH_XOR_NBI][ULONGLONG] =
        ishmemi_openshmem_atomic_fetch_xor_nbi<unsigned long long>;

    /* Signaling */
    proxy_funcs[PUT_SIGNAL][UINT8] = ishmemi_openshmem_uint8_put_signal;
    proxy_funcs[PUT_SIGNAL_NBI][UINT8] = ishmemi_openshmem_uint8_put_signal_nbi;
    proxy_funcs[SIGNAL_FETCH][0] = ishmemi_openshmem_signal_fetch;
    proxy_funcs[SIGNAL_ADD][UINT64] = ishmemi_openshmem_signal_add;
    proxy_funcs[SIGNAL_SET][UINT64] = ishmemi_openshmem_signal_set;

    /* Teams */
    proxy_funcs[TEAM_MY_PE][NONE] = ishmemi_openshmem_team_my_pe;
    proxy_funcs[TEAM_N_PES][NONE] = ishmemi_openshmem_team_n_pes;
    proxy_funcs[TEAM_SYNC][NONE] = ishmemi_openshmem_team_sync;

    /* Collectives */
    proxy_funcs[BARRIER][NONE] = ishmemi_openshmem_barrier_all;
    proxy_funcs[SYNC][NONE] = ishmemi_openshmem_sync_all;
    proxy_funcs[ALLTOALL][UINT8] = ishmemi_openshmem_uint8_alltoall;
    proxy_funcs[BCAST][UINT8] = ishmemi_openshmem_uint8_broadcast;
    proxy_funcs[COLLECT][UINT8] = ishmemi_openshmem_uint8_collect;
    proxy_funcs[FCOLLECT][UINT8] = ishmemi_openshmem_uint8_fcollect;

    /* Reductions */
    proxy_funcs[AND_REDUCE][UINT8] = ishmemi_openshmem_reduce<uint8_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][UINT8] = ishmemi_openshmem_reduce<uint8_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][UINT8] = ishmemi_openshmem_reduce<uint8_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][UINT8] = ishmemi_openshmem_reduce<uint8_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][UINT8] = ishmemi_openshmem_reduce<uint8_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][UINT8] = ishmemi_openshmem_reduce<uint8_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][UINT8] = ishmemi_openshmem_reduce<uint8_t, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][UINT16] = ishmemi_openshmem_reduce<uint16_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][UINT16] = ishmemi_openshmem_reduce<uint16_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][UINT16] = ishmemi_openshmem_reduce<uint16_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][UINT16] = ishmemi_openshmem_reduce<uint16_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][UINT16] = ishmemi_openshmem_reduce<uint16_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][UINT16] = ishmemi_openshmem_reduce<uint16_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][UINT16] = ishmemi_openshmem_reduce<uint16_t, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][UINT32] = ishmemi_openshmem_reduce<uint32_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][UINT32] = ishmemi_openshmem_reduce<uint32_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][UINT32] = ishmemi_openshmem_reduce<uint32_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][UINT32] = ishmemi_openshmem_reduce<uint32_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][UINT32] = ishmemi_openshmem_reduce<uint32_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][UINT32] = ishmemi_openshmem_reduce<uint32_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][UINT32] = ishmemi_openshmem_reduce<uint32_t, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][UINT64] = ishmemi_openshmem_reduce<uint64_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][UINT64] = ishmemi_openshmem_reduce<uint64_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][UINT64] = ishmemi_openshmem_reduce<uint64_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][UINT64] = ishmemi_openshmem_reduce<uint64_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][UINT64] = ishmemi_openshmem_reduce<uint64_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][UINT64] = ishmemi_openshmem_reduce<uint64_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][UINT64] = ishmemi_openshmem_reduce<uint64_t, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][ULONGLONG] = ishmemi_openshmem_reduce<unsigned long long, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][ULONGLONG] = ishmemi_openshmem_reduce<unsigned long long, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][ULONGLONG] = ishmemi_openshmem_reduce<unsigned long long, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][ULONGLONG] = ishmemi_openshmem_reduce<unsigned long long, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][ULONGLONG] = ishmemi_openshmem_reduce<unsigned long long, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][ULONGLONG] = ishmemi_openshmem_reduce<unsigned long long, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][ULONGLONG] = ishmemi_openshmem_reduce<unsigned long long, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][INT8] = ishmemi_openshmem_reduce<int8_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][INT8] = ishmemi_openshmem_reduce<int8_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][INT8] = ishmemi_openshmem_reduce<int8_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][INT8] = ishmemi_openshmem_reduce<int8_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][INT8] = ishmemi_openshmem_reduce<int8_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][INT8] = ishmemi_openshmem_reduce<int8_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][INT8] = ishmemi_openshmem_reduce<int8_t, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][INT16] = ishmemi_openshmem_reduce<int16_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][INT16] = ishmemi_openshmem_reduce<int16_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][INT16] = ishmemi_openshmem_reduce<int16_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][INT16] = ishmemi_openshmem_reduce<int16_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][INT16] = ishmemi_openshmem_reduce<int16_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][INT16] = ishmemi_openshmem_reduce<int16_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][INT16] = ishmemi_openshmem_reduce<int16_t, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][INT32] = ishmemi_openshmem_reduce<int32_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][INT32] = ishmemi_openshmem_reduce<int32_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][INT32] = ishmemi_openshmem_reduce<int32_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][INT32] = ishmemi_openshmem_reduce<int32_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][INT32] = ishmemi_openshmem_reduce<int32_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][INT32] = ishmemi_openshmem_reduce<int32_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][INT32] = ishmemi_openshmem_reduce<int32_t, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][INT64] = ishmemi_openshmem_reduce<int64_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][INT64] = ishmemi_openshmem_reduce<int64_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][INT64] = ishmemi_openshmem_reduce<int64_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][INT64] = ishmemi_openshmem_reduce<int64_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][INT64] = ishmemi_openshmem_reduce<int64_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][INT64] = ishmemi_openshmem_reduce<int64_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][INT64] = ishmemi_openshmem_reduce<int64_t, PROD_REDUCE>;

    proxy_funcs[MAX_REDUCE][LONGLONG] = ishmemi_openshmem_reduce<long long, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][LONGLONG] = ishmemi_openshmem_reduce<long long, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][LONGLONG] = ishmemi_openshmem_reduce<long long, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][LONGLONG] = ishmemi_openshmem_reduce<long long, PROD_REDUCE>;

    proxy_funcs[MAX_REDUCE][FLOAT] = ishmemi_openshmem_reduce<float, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][FLOAT] = ishmemi_openshmem_reduce<float, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][FLOAT] = ishmemi_openshmem_reduce<float, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][FLOAT] = ishmemi_openshmem_reduce<float, PROD_REDUCE>;

    proxy_funcs[MAX_REDUCE][DOUBLE] = ishmemi_openshmem_reduce<double, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][DOUBLE] = ishmemi_openshmem_reduce<double, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][DOUBLE] = ishmemi_openshmem_reduce<double, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][DOUBLE] = ishmemi_openshmem_reduce<double, PROD_REDUCE>;

    /* Point-to-Point Synchronization */
    if (oshmpi) {
        proxy_funcs[TEST][INT32] = ishmemi_openshmem_test<int32_t, true>;
        proxy_funcs[TEST_ALL][INT32] = ishmemi_openshmem_test_all<int32_t, true>;
        proxy_funcs[TEST_ANY][INT32] = ishmemi_openshmem_test_any<int32_t, true>;
        proxy_funcs[TEST_SOME][INT32] = ishmemi_openshmem_test_some<int32_t, true>;
        proxy_funcs[WAIT][INT32] = ishmemi_openshmem_wait_until<int32_t, true>;
        proxy_funcs[WAIT_ALL][INT32] = ishmemi_openshmem_wait_until_all<int32_t, true>;
        proxy_funcs[WAIT_ANY][INT32] = ishmemi_openshmem_wait_until_any<int32_t, true>;
        proxy_funcs[WAIT_SOME][INT32] = ishmemi_openshmem_wait_until_some<int32_t, true>;
        proxy_funcs[TEST_ALL_VECTOR][INT32] = ishmemi_openshmem_test_all_vector<int32_t, true>;
        proxy_funcs[TEST_ANY_VECTOR][INT32] = ishmemi_openshmem_test_any_vector<int32_t, true>;
        proxy_funcs[TEST_SOME_VECTOR][INT32] = ishmemi_openshmem_test_some_vector<int32_t, true>;
        proxy_funcs[WAIT_ALL_VECTOR][INT32] =
            ishmemi_openshmem_wait_until_all_vector<int32_t, true>;
        proxy_funcs[WAIT_ANY_VECTOR][INT32] =
            ishmemi_openshmem_wait_until_any_vector<int32_t, true>;
        proxy_funcs[WAIT_SOME_VECTOR][INT32] =
            ishmemi_openshmem_wait_until_some_vector<int32_t, true>;

        proxy_funcs[TEST][INT64] = ishmemi_openshmem_test<int64_t, true>;
        proxy_funcs[TEST_ALL][INT64] = ishmemi_openshmem_test_all<int64_t, true>;
        proxy_funcs[TEST_ANY][INT64] = ishmemi_openshmem_test_any<int64_t, true>;
        proxy_funcs[TEST_SOME][INT64] = ishmemi_openshmem_test_some<int64_t, true>;
        proxy_funcs[WAIT][INT64] = ishmemi_openshmem_wait_until<int64_t, true>;
        proxy_funcs[WAIT_ALL][INT64] = ishmemi_openshmem_wait_until_all<int64_t, true>;
        proxy_funcs[WAIT_ANY][INT64] = ishmemi_openshmem_wait_until_any<int64_t, true>;
        proxy_funcs[WAIT_SOME][INT64] = ishmemi_openshmem_wait_until_some<int64_t, true>;
        proxy_funcs[TEST_ALL_VECTOR][INT64] = ishmemi_openshmem_test_all_vector<int64_t, true>;
        proxy_funcs[TEST_ANY_VECTOR][INT64] = ishmemi_openshmem_test_any_vector<int64_t, true>;
        proxy_funcs[TEST_SOME_VECTOR][INT64] = ishmemi_openshmem_test_some_vector<int64_t, true>;
        proxy_funcs[WAIT_ALL_VECTOR][INT64] =
            ishmemi_openshmem_wait_until_all_vector<int64_t, true>;
        proxy_funcs[WAIT_ANY_VECTOR][INT64] =
            ishmemi_openshmem_wait_until_any_vector<int64_t, true>;
        proxy_funcs[WAIT_SOME_VECTOR][INT64] =
            ishmemi_openshmem_wait_until_some_vector<int64_t, true>;

        proxy_funcs[TEST][LONGLONG] = ishmemi_openshmem_test<long long, true>;
        proxy_funcs[TEST_ALL][LONGLONG] = ishmemi_openshmem_test_all<long long, true>;
        proxy_funcs[TEST_ANY][LONGLONG] = ishmemi_openshmem_test_any<long long, true>;
        proxy_funcs[TEST_SOME][LONGLONG] = ishmemi_openshmem_test_some<long long, true>;
        proxy_funcs[WAIT][LONGLONG] = ishmemi_openshmem_wait_until<long long, true>;
        proxy_funcs[WAIT_ALL][LONGLONG] = ishmemi_openshmem_wait_until_all<long long, true>;
        proxy_funcs[WAIT_ANY][LONGLONG] = ishmemi_openshmem_wait_until_any<long long, true>;
        proxy_funcs[WAIT_SOME][LONGLONG] = ishmemi_openshmem_wait_until_some<long long, true>;
        proxy_funcs[TEST_ALL_VECTOR][LONGLONG] = ishmemi_openshmem_test_all_vector<long long, true>;
        proxy_funcs[TEST_ANY_VECTOR][LONGLONG] = ishmemi_openshmem_test_any_vector<long long, true>;
        proxy_funcs[TEST_SOME_VECTOR][LONGLONG] =
            ishmemi_openshmem_test_some_vector<long long, true>;
        proxy_funcs[WAIT_ALL_VECTOR][LONGLONG] =
            ishmemi_openshmem_wait_until_all_vector<long long, true>;
        proxy_funcs[WAIT_ANY_VECTOR][LONGLONG] =
            ishmemi_openshmem_wait_until_any_vector<long long, true>;
        proxy_funcs[WAIT_SOME_VECTOR][LONGLONG] =
            ishmemi_openshmem_wait_until_some_vector<long long, true>;

        proxy_funcs[TEST][UINT32] = ishmemi_openshmem_test<uint32_t, true>;
        proxy_funcs[TEST_ALL][UINT32] = ishmemi_openshmem_test_all<uint32_t, true>;
        proxy_funcs[TEST_ANY][UINT32] = ishmemi_openshmem_test_any<uint32_t, true>;
        proxy_funcs[TEST_SOME][UINT32] = ishmemi_openshmem_test_some<uint32_t, true>;
        proxy_funcs[WAIT][UINT32] = ishmemi_openshmem_wait_until<uint32_t, true>;
        proxy_funcs[WAIT_ALL][UINT32] = ishmemi_openshmem_wait_until_all<uint32_t, true>;
        proxy_funcs[WAIT_ANY][UINT32] = ishmemi_openshmem_wait_until_any<uint32_t, true>;
        proxy_funcs[WAIT_SOME][UINT32] = ishmemi_openshmem_wait_until_some<uint32_t, true>;
        proxy_funcs[TEST_ALL_VECTOR][UINT32] = ishmemi_openshmem_test_all_vector<uint32_t, true>;
        proxy_funcs[TEST_ANY_VECTOR][UINT32] = ishmemi_openshmem_test_any_vector<uint32_t, true>;
        proxy_funcs[TEST_SOME_VECTOR][UINT32] = ishmemi_openshmem_test_some_vector<uint32_t, true>;
        proxy_funcs[WAIT_ALL_VECTOR][UINT32] =
            ishmemi_openshmem_wait_until_all_vector<uint32_t, true>;
        proxy_funcs[WAIT_ANY_VECTOR][UINT32] =
            ishmemi_openshmem_wait_until_any_vector<uint32_t, true>;
        proxy_funcs[WAIT_SOME_VECTOR][UINT32] =
            ishmemi_openshmem_wait_until_some_vector<uint32_t, true>;

        proxy_funcs[TEST][UINT64] = ishmemi_openshmem_test<uint64_t, true>;
        proxy_funcs[TEST_ALL][UINT64] = ishmemi_openshmem_test_all<uint64_t, true>;
        proxy_funcs[TEST_ANY][UINT64] = ishmemi_openshmem_test_any<uint64_t, true>;
        proxy_funcs[TEST_SOME][UINT64] = ishmemi_openshmem_test_some<uint64_t, true>;
        proxy_funcs[WAIT][UINT64] = ishmemi_openshmem_wait_until<uint64_t, true>;
        proxy_funcs[WAIT_ALL][UINT64] = ishmemi_openshmem_wait_until_all<uint64_t, true>;
        proxy_funcs[WAIT_ANY][UINT64] = ishmemi_openshmem_wait_until_any<uint64_t, true>;
        proxy_funcs[WAIT_SOME][UINT64] = ishmemi_openshmem_wait_until_some<uint64_t, true>;
        proxy_funcs[TEST_ALL_VECTOR][UINT64] = ishmemi_openshmem_test_all_vector<uint64_t, true>;
        proxy_funcs[TEST_ANY_VECTOR][UINT64] = ishmemi_openshmem_test_any_vector<uint64_t, true>;
        proxy_funcs[TEST_SOME_VECTOR][UINT64] = ishmemi_openshmem_test_some_vector<uint64_t, true>;
        proxy_funcs[WAIT_ALL_VECTOR][UINT64] =
            ishmemi_openshmem_wait_until_all_vector<uint64_t, true>;
        proxy_funcs[WAIT_ANY_VECTOR][UINT64] =
            ishmemi_openshmem_wait_until_any_vector<uint64_t, true>;
        proxy_funcs[WAIT_SOME_VECTOR][UINT64] =
            ishmemi_openshmem_wait_until_some_vector<uint64_t, true>;

        proxy_funcs[TEST][ULONGLONG] = ishmemi_openshmem_test<unsigned long long, true>;
        proxy_funcs[TEST_ALL][ULONGLONG] = ishmemi_openshmem_test_all<unsigned long long, true>;
        proxy_funcs[TEST_ANY][ULONGLONG] = ishmemi_openshmem_test_any<unsigned long long, true>;
        proxy_funcs[TEST_SOME][ULONGLONG] = ishmemi_openshmem_test_some<unsigned long long, true>;
        proxy_funcs[WAIT][ULONGLONG] = ishmemi_openshmem_wait_until<unsigned long long, true>;
        proxy_funcs[WAIT_ALL][ULONGLONG] =
            ishmemi_openshmem_wait_until_all<unsigned long long, true>;
        proxy_funcs[WAIT_ANY][ULONGLONG] =
            ishmemi_openshmem_wait_until_any<unsigned long long, true>;
        proxy_funcs[WAIT_SOME][ULONGLONG] =
            ishmemi_openshmem_wait_until_some<unsigned long long, true>;
        proxy_funcs[TEST_ALL_VECTOR][ULONGLONG] =
            ishmemi_openshmem_test_all_vector<unsigned long long, true>;
        proxy_funcs[TEST_ANY_VECTOR][ULONGLONG] =
            ishmemi_openshmem_test_any_vector<unsigned long long, true>;
        proxy_funcs[TEST_SOME_VECTOR][ULONGLONG] =
            ishmemi_openshmem_test_some_vector<unsigned long long, true>;
        proxy_funcs[WAIT_ALL_VECTOR][ULONGLONG] =
            ishmemi_openshmem_wait_until_all_vector<unsigned long long, true>;
        proxy_funcs[WAIT_ANY_VECTOR][ULONGLONG] =
            ishmemi_openshmem_wait_until_any_vector<unsigned long long, true>;
        proxy_funcs[WAIT_SOME_VECTOR][ULONGLONG] =
            ishmemi_openshmem_wait_until_some_vector<unsigned long long, true>;

        proxy_funcs[SIGNAL_WAIT_UNTIL][UINT64] = ishmemi_openshmem_signal_wait_until<true>;
    } else {
        proxy_funcs[TEST][INT32] = ishmemi_openshmem_test<int32_t, false>;
        proxy_funcs[TEST_ALL][INT32] = ishmemi_openshmem_test_all<int32_t, false>;
        proxy_funcs[TEST_ANY][INT32] = ishmemi_openshmem_test_any<int32_t, false>;
        proxy_funcs[TEST_SOME][INT32] = ishmemi_openshmem_test_some<int32_t, false>;
        proxy_funcs[WAIT][INT32] = ishmemi_openshmem_wait_until<int32_t, false>;
        proxy_funcs[WAIT_ALL][INT32] = ishmemi_openshmem_wait_until_all<int32_t, false>;
        proxy_funcs[WAIT_ANY][INT32] = ishmemi_openshmem_wait_until_any<int32_t, false>;
        proxy_funcs[WAIT_SOME][INT32] = ishmemi_openshmem_wait_until_some<int32_t, false>;
        proxy_funcs[TEST_ALL_VECTOR][INT32] = ishmemi_openshmem_test_all_vector<int32_t, false>;
        proxy_funcs[TEST_ANY_VECTOR][INT32] = ishmemi_openshmem_test_any_vector<int32_t, false>;
        proxy_funcs[TEST_SOME_VECTOR][INT32] = ishmemi_openshmem_test_some_vector<int32_t, false>;
        proxy_funcs[WAIT_ALL_VECTOR][INT32] =
            ishmemi_openshmem_wait_until_all_vector<int32_t, false>;
        proxy_funcs[WAIT_ANY_VECTOR][INT32] =
            ishmemi_openshmem_wait_until_any_vector<int32_t, false>;
        proxy_funcs[WAIT_SOME_VECTOR][INT32] =
            ishmemi_openshmem_wait_until_some_vector<int32_t, false>;

        proxy_funcs[TEST][INT64] = ishmemi_openshmem_test<int64_t, false>;
        proxy_funcs[TEST_ALL][INT64] = ishmemi_openshmem_test_all<int64_t, false>;
        proxy_funcs[TEST_ANY][INT64] = ishmemi_openshmem_test_any<int64_t, false>;
        proxy_funcs[TEST_SOME][INT64] = ishmemi_openshmem_test_some<int64_t, false>;
        proxy_funcs[WAIT][INT64] = ishmemi_openshmem_wait_until<int64_t, false>;
        proxy_funcs[WAIT_ALL][INT64] = ishmemi_openshmem_wait_until_all<int64_t, false>;
        proxy_funcs[WAIT_ANY][INT64] = ishmemi_openshmem_wait_until_any<int64_t, false>;
        proxy_funcs[WAIT_SOME][INT64] = ishmemi_openshmem_wait_until_some<int64_t, false>;
        proxy_funcs[TEST_ALL_VECTOR][INT64] = ishmemi_openshmem_test_all_vector<int64_t, false>;
        proxy_funcs[TEST_ANY_VECTOR][INT64] = ishmemi_openshmem_test_any_vector<int64_t, false>;
        proxy_funcs[TEST_SOME_VECTOR][INT64] = ishmemi_openshmem_test_some_vector<int64_t, false>;
        proxy_funcs[WAIT_ALL_VECTOR][INT64] =
            ishmemi_openshmem_wait_until_all_vector<int64_t, false>;
        proxy_funcs[WAIT_ANY_VECTOR][INT64] =
            ishmemi_openshmem_wait_until_any_vector<int64_t, false>;
        proxy_funcs[WAIT_SOME_VECTOR][INT64] =
            ishmemi_openshmem_wait_until_some_vector<int64_t, false>;

        proxy_funcs[TEST][LONGLONG] = ishmemi_openshmem_test<long long, false>;
        proxy_funcs[TEST_ALL][LONGLONG] = ishmemi_openshmem_test_all<long long, false>;
        proxy_funcs[TEST_ANY][LONGLONG] = ishmemi_openshmem_test_any<long long, false>;
        proxy_funcs[TEST_SOME][LONGLONG] = ishmemi_openshmem_test_some<long long, false>;
        proxy_funcs[WAIT][LONGLONG] = ishmemi_openshmem_wait_until<long long, false>;
        proxy_funcs[WAIT_ALL][LONGLONG] = ishmemi_openshmem_wait_until_all<long long, false>;
        proxy_funcs[WAIT_ANY][LONGLONG] = ishmemi_openshmem_wait_until_any<long long, false>;
        proxy_funcs[WAIT_SOME][LONGLONG] = ishmemi_openshmem_wait_until_some<long long, false>;
        proxy_funcs[TEST_ALL_VECTOR][LONGLONG] =
            ishmemi_openshmem_test_all_vector<long long, false>;
        proxy_funcs[TEST_ANY_VECTOR][LONGLONG] =
            ishmemi_openshmem_test_any_vector<long long, false>;
        proxy_funcs[TEST_SOME_VECTOR][LONGLONG] =
            ishmemi_openshmem_test_some_vector<long long, false>;
        proxy_funcs[WAIT_ALL_VECTOR][LONGLONG] =
            ishmemi_openshmem_wait_until_all_vector<long long, false>;
        proxy_funcs[WAIT_ANY_VECTOR][LONGLONG] =
            ishmemi_openshmem_wait_until_any_vector<long long, false>;
        proxy_funcs[WAIT_SOME_VECTOR][LONGLONG] =
            ishmemi_openshmem_wait_until_some_vector<long long, false>;

        proxy_funcs[TEST][UINT32] = ishmemi_openshmem_test<uint32_t, false>;
        proxy_funcs[TEST_ALL][UINT32] = ishmemi_openshmem_test_all<uint32_t, false>;
        proxy_funcs[TEST_ANY][UINT32] = ishmemi_openshmem_test_any<uint32_t, false>;
        proxy_funcs[TEST_SOME][UINT32] = ishmemi_openshmem_test_some<uint32_t, false>;
        proxy_funcs[WAIT][UINT32] = ishmemi_openshmem_wait_until<uint32_t, false>;
        proxy_funcs[WAIT_ALL][UINT32] = ishmemi_openshmem_wait_until_all<uint32_t, false>;
        proxy_funcs[WAIT_ANY][UINT32] = ishmemi_openshmem_wait_until_any<uint32_t, false>;
        proxy_funcs[WAIT_SOME][UINT32] = ishmemi_openshmem_wait_until_some<uint32_t, false>;
        proxy_funcs[TEST_ALL_VECTOR][UINT32] = ishmemi_openshmem_test_all_vector<uint32_t, false>;
        proxy_funcs[TEST_ANY_VECTOR][UINT32] = ishmemi_openshmem_test_any_vector<uint32_t, false>;
        proxy_funcs[TEST_SOME_VECTOR][UINT32] = ishmemi_openshmem_test_some_vector<uint32_t, false>;
        proxy_funcs[WAIT_ALL_VECTOR][UINT32] =
            ishmemi_openshmem_wait_until_all_vector<uint32_t, false>;
        proxy_funcs[WAIT_ANY_VECTOR][UINT32] =
            ishmemi_openshmem_wait_until_any_vector<uint32_t, false>;
        proxy_funcs[WAIT_SOME_VECTOR][UINT32] =
            ishmemi_openshmem_wait_until_some_vector<uint32_t, false>;

        proxy_funcs[TEST][UINT64] = ishmemi_openshmem_test<uint64_t, false>;
        proxy_funcs[TEST_ALL][UINT64] = ishmemi_openshmem_test_all<uint64_t, false>;
        proxy_funcs[TEST_ANY][UINT64] = ishmemi_openshmem_test_any<uint64_t, false>;
        proxy_funcs[TEST_SOME][UINT64] = ishmemi_openshmem_test_some<uint64_t, false>;
        proxy_funcs[WAIT][UINT64] = ishmemi_openshmem_wait_until<uint64_t, false>;
        proxy_funcs[WAIT_ALL][UINT64] = ishmemi_openshmem_wait_until_all<uint64_t, false>;
        proxy_funcs[WAIT_ANY][UINT64] = ishmemi_openshmem_wait_until_any<uint64_t, false>;
        proxy_funcs[WAIT_SOME][UINT64] = ishmemi_openshmem_wait_until_some<uint64_t, false>;
        proxy_funcs[TEST_ALL_VECTOR][UINT64] = ishmemi_openshmem_test_all_vector<uint64_t, false>;
        proxy_funcs[TEST_ANY_VECTOR][UINT64] = ishmemi_openshmem_test_any_vector<uint64_t, false>;
        proxy_funcs[TEST_SOME_VECTOR][UINT64] = ishmemi_openshmem_test_some_vector<uint64_t, false>;
        proxy_funcs[WAIT_ALL_VECTOR][UINT64] =
            ishmemi_openshmem_wait_until_all_vector<uint64_t, false>;
        proxy_funcs[WAIT_ANY_VECTOR][UINT64] =
            ishmemi_openshmem_wait_until_any_vector<uint64_t, false>;
        proxy_funcs[WAIT_SOME_VECTOR][UINT64] =
            ishmemi_openshmem_wait_until_some_vector<uint64_t, false>;

        proxy_funcs[TEST][ULONGLONG] = ishmemi_openshmem_test<unsigned long long, false>;
        proxy_funcs[TEST_ALL][ULONGLONG] = ishmemi_openshmem_test_all<unsigned long long, false>;
        proxy_funcs[TEST_ANY][ULONGLONG] = ishmemi_openshmem_test_any<unsigned long long, false>;
        proxy_funcs[TEST_SOME][ULONGLONG] = ishmemi_openshmem_test_some<unsigned long long, false>;
        proxy_funcs[WAIT][ULONGLONG] = ishmemi_openshmem_wait_until<unsigned long long, false>;
        proxy_funcs[WAIT_ALL][ULONGLONG] =
            ishmemi_openshmem_wait_until_all<unsigned long long, false>;
        proxy_funcs[WAIT_ANY][ULONGLONG] =
            ishmemi_openshmem_wait_until_any<unsigned long long, false>;
        proxy_funcs[WAIT_SOME][ULONGLONG] =
            ishmemi_openshmem_wait_until_some<unsigned long long, false>;
        proxy_funcs[TEST_ALL_VECTOR][ULONGLONG] =
            ishmemi_openshmem_test_all_vector<unsigned long long, false>;
        proxy_funcs[TEST_ANY_VECTOR][ULONGLONG] =
            ishmemi_openshmem_test_any_vector<unsigned long long, false>;
        proxy_funcs[TEST_SOME_VECTOR][ULONGLONG] =
            ishmemi_openshmem_test_some_vector<unsigned long long, false>;
        proxy_funcs[WAIT_ALL_VECTOR][ULONGLONG] =
            ishmemi_openshmem_wait_until_all_vector<unsigned long long, false>;
        proxy_funcs[WAIT_ANY_VECTOR][ULONGLONG] =
            ishmemi_openshmem_wait_until_any_vector<unsigned long long, false>;
        proxy_funcs[WAIT_SOME_VECTOR][ULONGLONG] =
            ishmemi_openshmem_wait_until_some_vector<unsigned long long, false>;

        proxy_funcs[SIGNAL_WAIT_UNTIL][UINT64] = ishmemi_openshmem_signal_wait_until<false>;
    }

    /* Memory Ordering */
    proxy_funcs[FENCE][NONE] = ishmemi_openshmem_fence;
    proxy_funcs[QUIET][NONE] = ishmemi_openshmem_quiet;

fn_exit:
    return;
}

void ishmemi_runtime_openshmem::funcptr_fini()
{
    for (int i = 0; i < ISHMEMI_OP_END; ++i) {
        for (int j = 0; j < ishmemi_runtime_type::proxy_func_num_types; ++j) {
            proxy_funcs[i][j] = ishmemi_runtime_type::unsupported;
        }
        ISHMEMI_FREE(::free, proxy_funcs[i]);
    }

    ISHMEMI_FREE(::free, proxy_funcs);
}
