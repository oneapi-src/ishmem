/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/err.h"
#include "accelerator.h"
#include "collectives.h"
#include "proxy_impl.h"
#include "ipc.h"
#include "memory.h"
#include "proxy.h"
#include "runtime.h"
#include "collectives/sync_impl.h"
#include "collectives/reduce_impl.h"
#include "collectives/broadcast_impl.h"
#include "collectives/collect_impl.h"
#include "collectives/alltoall_impl.h"
#include <cstdlib>  // abort

int ishmemi_my_pe;
int ishmemi_n_pes;

/* Note: these string array is in the same order as ishmemi_op_t in types.h */
const char *ishmemi_op_str[ISHMEMI_OP_END + 1];
/* Note: these string array is in the same order as ishmemi_type_t in types.h */
const char *ishmemi_type_str[ISHMEMI_TYPE_END + 1];

/* Macros for generating ishmem external API implementation */
/* Name convention is based on shmem function arguments
   - P: pointer (i.e. src/dst)
   - S: size_t (i.e. buffer size)
   - I: int (i.e. pe)
   - T: type (i.e. value for AMO)
   - G: group (for work group operations

   For example:
   - R_PTTI: pointer, type, type, int; returns type data -- used for
   atomic_compare_swap
   - I_PPS: pointer, pointer, size; returns int -- used for reductions
*/

/* Each of these macros has two implementations
 * The __SYCL_DEVICE_ONLY__ version is active for device code
 * The !__SYCL_DEVICE_ONLY__ version is active for host code
 *
 * Each of the above branches has code for the intra-node case
 * and separate code for the inter-node case
 *
 * pes for which local_pes[pe] is non-zero are local
 * for these, the value of ipc_buffers[local_pes[pe]] is the base of the
 * associated symmetric heap
 */
/* TODO
 * in these functions, we pass a *ishmemi_ringcompletion_t around, but the
 * value passed to the proxy is the <index> of the completion structure.  This
 * should be simplified perhaps by passing the index to the helper functions
 * rather than the pointer
 */

/* device global pointer to global_info structure */
sycl::ext::oneapi::experimental::device_global<ishmemi_info_t *> global_info;

#define ISHMEMI_API_IMPL(SUFFIX)                                                                   \
    void ishmemx_##SUFFIX()                                                                        \
    {                                                                                              \
        if constexpr (ishmemi_is_device) {                                                         \
            ishmemi_request_t req;                                                                 \
            req.op = ISHMEMI_OP_##SUFFIX;                                                          \
            req.type = MEM;                                                                        \
            ishmemi_proxy_blocking_request(req);                                                   \
        }                                                                                          \
    }

#define ISHMEMI_API_IMPL_NO_R(SUFFIX)                                                              \
    void ishmemx_##SUFFIX()                                                                        \
    {                                                                                              \
        if constexpr (ishmemi_is_device) {                                                         \
            ishmemi_request_t req;                                                                 \
            req.op = ISHMEMI_OP_##SUFFIX;                                                          \
            req.type = MEM;                                                                        \
            ishmemi_proxy_nonblocking_request(req);                                                \
        }                                                                                          \
    }

/* This API is only for test purposes */
ISHMEMI_API_IMPL(debug_test)
ISHMEMI_API_IMPL(nop)
ISHMEMI_API_IMPL_NO_R(nop_no_r)

void ishmemi_init_op_str()
{
    ishmemi_op_str[PUT] = "put";
    ishmemi_op_str[IPUT] = "iput";
    ishmemi_op_str[P] = "p";
    ishmemi_op_str[GET] = "get";
    ishmemi_op_str[IGET] = "iget";
    ishmemi_op_str[G] = "g";
    ishmemi_op_str[PUT_NBI] = "put_nbi";
    ishmemi_op_str[GET_NBI] = "get_nbi";
    ishmemi_op_str[AMO_FETCH] = "atomic_fetch";
    ishmemi_op_str[AMO_SET] = "atomic_set";
    ishmemi_op_str[AMO_COMPARE_SWAP] = "atomic_compare_swap";
    ishmemi_op_str[AMO_SWAP] = "atomic_swap";
    ishmemi_op_str[AMO_FETCH_INC] = "atomic_fetch_inc";
    ishmemi_op_str[AMO_INC] = "atomic_inc";
    ishmemi_op_str[AMO_FETCH_ADD] = "atomic_fetch_add";
    ishmemi_op_str[AMO_ADD] = "atomic_add";
    ishmemi_op_str[AMO_FETCH_AND] = "atomic_fetch_and";
    ishmemi_op_str[AMO_AND] = "atomic_and";
    ishmemi_op_str[AMO_FETCH_OR] = "atomic_fetch_or";
    ishmemi_op_str[AMO_OR] = "atomic_or";
    ishmemi_op_str[AMO_FETCH_XOR] = "atomic_fetch_xor";
    ishmemi_op_str[AMO_XOR] = "atomic_xor";
    ishmemi_op_str[AMO_FETCH_NBI] = "atomic_fetch_nbi";
    ishmemi_op_str[AMO_COMPARE_SWAP_NBI] = "atomic_compare_swap_nbi";
    ishmemi_op_str[AMO_SWAP_NBI] = "atomic_swap_nbi";
    ishmemi_op_str[AMO_FETCH_INC_NBI] = "atomic_fetch_inc_nbi";
    ishmemi_op_str[AMO_FETCH_ADD_NBI] = "atomic_fetch_add_nbi";
    ishmemi_op_str[AMO_FETCH_AND_NBI] = "atomic_fetch_and_nbi";
    ishmemi_op_str[AMO_FETCH_OR_NBI] = "atomic_fetch_or_nbi";
    ishmemi_op_str[AMO_FETCH_XOR_NBI] = "atomic_fetch_xor_nbi";
    ishmemi_op_str[PUT_SIGNAL] = "put_signal";
    ishmemi_op_str[PUT_SIGNAL_NBI] = "put_signal_nbi";
    ishmemi_op_str[SIGNAL_FETCH] = "signal_fetch";
    ishmemi_op_str[SIGNAL_ADD] = "signal_add";
    ishmemi_op_str[SIGNAL_SET] = "signal_set";
    ishmemi_op_str[BARRIER] = "barrier_all";
    ishmemi_op_str[SYNC] = "sync_all";
    ishmemi_op_str[ALLTOALL] = "alltoall";
    ishmemi_op_str[BCAST] = "broadcast";
    ishmemi_op_str[COLLECT] = "collect";
    ishmemi_op_str[FCOLLECT] = "fcollect";
    ishmemi_op_str[AND_REDUCE] = "and_reduce";
    ishmemi_op_str[OR_REDUCE] = "or_reduce";
    ishmemi_op_str[XOR_REDUCE] = "xor_reduce";
    ishmemi_op_str[MAX_REDUCE] = "max_reduce";
    ishmemi_op_str[MIN_REDUCE] = "min_reduce";
    ishmemi_op_str[SUM_REDUCE] = "sum_reduce";
    ishmemi_op_str[PROD_REDUCE] = "prod_reduce";
    ishmemi_op_str[WAIT] = "wait_until";
    ishmemi_op_str[WAIT_ALL] = "wait_until_all";
    ishmemi_op_str[WAIT_ANY] = "wait_until_any";
    ishmemi_op_str[WAIT_SOME] = "wait_until_some";
    ishmemi_op_str[TEST] = "test";
    ishmemi_op_str[TEST_ALL] = "test_all";
    ishmemi_op_str[TEST_ANY] = "test_any";
    ishmemi_op_str[TEST_SOME] = "test_some";
    ishmemi_op_str[FENCE] = "fence";
    ishmemi_op_str[QUIET] = "quiet";
    ishmemi_op_str[KILL] = "kill";
    ishmemi_op_str[NOP] = "nop";
    ishmemi_op_str[NOP_NO_R] = "nop_no_r";
    ishmemi_op_str[TIMESTAMP] = "timestamp";
    ishmemi_op_str[PRINT] = "print";
    ishmemi_op_str[DEBUG_TEST] = "debug_test";
    ishmemi_op_str[ISHMEMI_OP_END] = NULL;
}

void ishmemi_init_type_str()
{
    ishmemi_type_str[MEM] = "mem";
    ishmemi_type_str[UINT8] = "uint8";
    ishmemi_type_str[UINT16] = "uint16";
    ishmemi_type_str[UINT32] = "uint32";
    ishmemi_type_str[UINT64] = "uint64";
    ishmemi_type_str[ULONGLONG] = "ulonglong";
    ishmemi_type_str[INT8] = "int8";
    ishmemi_type_str[INT16] = "int16";
    ishmemi_type_str[INT32] = "int32";
    ishmemi_type_str[INT64] = "int64";
    ishmemi_type_str[LONGLONG] = "longlong";
    ishmemi_type_str[FLOAT] = "float";
    ishmemi_type_str[DOUBLE] = "double";
    ishmemi_type_str[LONGDOUBLE] = "longdouble";
    ishmemi_type_str[CHAR] = "char";
    ishmemi_type_str[SCHAR] = "schar";
    ishmemi_type_str[SHORT] = "short";
    ishmemi_type_str[INT] = "int";
    ishmemi_type_str[LONG] = "long";
    ishmemi_type_str[UCHAR] = "uchar";
    ishmemi_type_str[USHORT] = "ushort";
    ishmemi_type_str[UINT] = "uint";
    ishmemi_type_str[ULONG] = "ulong";
    ishmemi_type_str[SIZE] = "size";
    ishmemi_type_str[PTRDIFF] = "ptrdiff";
    ishmemi_type_str[SIZE8] = "size8";
    ishmemi_type_str[SIZE16] = "size16";
    ishmemi_type_str[SIZE32] = "size32";
    ishmemi_type_str[SIZE64] = "size64";
    ishmemi_type_str[SIZE128] = "size128";
    ishmemi_type_str[ISHMEMI_TYPE_END] = NULL;
}

void ishmem_init()
{
    /* Use default attributes */
    ishmemx_attr_t attr;

    /* Initialize */
    ishmemx_init_attr(&attr);
}

void ishmemx_init_attr(ishmemx_attr_t *attr)
{
    int ret = -1;
    int runtime_initialized = 0;
    int accelerator_initialized = 0;
    int memory_initialized = 0;
    int ipc_initialized = 0;
    int teams_initialized = 0;
    int collectives_initialized = 0;
    ishmemi_init_op_str();
    ishmemi_init_type_str();
    static_assert(sizeof(ishmemi_request_t) == 64, "ISHMEM request object must be 64 bytes.");
    static_assert(sizeof(ishmemi_completion_t) == 64, "ISHMEM completion object must be 64 bytes.");
    static_assert(sizeof(ishmemi_ringcompletion_t) == 64,
                  "ISHMEM ringcompletion object must be 64 bytes.");

    ishmemi_cpu_info = (ishmemi_cpu_info_t *) malloc(sizeof(ishmemi_cpu_info_t));
    ISHMEM_CHECK_GOTO_MSG(ishmemi_cpu_info == nullptr, cleanup,
                          "CPU info object allocation failed\n");
    memset(ishmemi_cpu_info, 0, sizeof(ishmemi_cpu_info_t));

    /* Currently, no support for CPU environments */
    if (!attr->gpu) {
        ISHMEM_WARN_MSG("Currently, no support for CPU-only environment\n");
        attr->gpu = true;
    }

    /* Parse environment variables */
    ret = ishmemi_parse_env();
    ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "Parsing env variables failed '%d'\n", ret);

    /* Initialize the runtime */
    ret = ishmemi_runtime_init(attr);
    ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "Runtime initialization failed '%d'\n", ret);
    runtime_initialized = 1;

    ishmemi_my_pe = ishmemi_runtime_get_rank();
    ishmemi_n_pes = ishmemi_runtime_get_size();
    ishmemi_cpu_info->my_pe = ishmemi_my_pe;
    ishmemi_cpu_info->n_pes = ishmemi_n_pes;

    if (attr->gpu) {
        ret = ishmemi_accelerator_init();
        if (ret == ISHMEMI_NO_DEVICE_ACCESS) {
            attr->gpu = false;
            /* TODO need to enable SHARED HEAP config */
        } else {
            ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "Accelerator initialization failed '%d'\n", ret);
            accelerator_initialized = 1;
        }
    }

    if (attr->gpu) {
        ret = ishmemi_memory_init();
        ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "Memory initialization failed '%d'\n", ret);
        memory_initialized = 1;
    }
    /* Register symmetric heap with host runtime */
    ishmemi_runtime_heap_create(attr, ishmemi_heap_base, ishmemi_heap_length);

    ishmemi_cpu_info->use_ipc = false;  // This will be set to true if ishmemi_ipc_init passes

    /* Initialize the attributes */
    ishmemi_cpu_info->attr = (ishmemx_attr_t *) malloc(sizeof(ishmemx_attr_t));
    ISHMEM_CHECK_GOTO_MSG(ishmemi_cpu_info->attr == nullptr, cleanup,
                          "Attribute object allocation failed\n");
    memcpy(ishmemi_cpu_info->attr, attr, sizeof(ishmemx_attr_t));

    /* info structure is allocated by memory_init */
    ishmemi_mmap_gpu_info->my_pe = ishmemi_my_pe;
    ishmemi_mmap_gpu_info->n_pes = ishmemi_n_pes;
    ishmemi_mmap_gpu_info->heap_base = ishmemi_heap_base;
    ishmemi_mmap_gpu_info->heap_length = ishmemi_heap_length;

    /* Setup local_pes info for host use */
    ishmemi_local_pes = (uint8_t *) malloc(static_cast<size_t>(ishmemi_n_pes));
    ISHMEM_CHECK_GOTO_MSG((ishmemi_local_pes == NULL), cleanup, "Local PEs allocation failed\n");

    /* For SHARED_HEAP, no IPC mechanism is available */
    if (ishmemi_params.ENABLE_ACCESSIBLE_HOST_HEAP) {
        if (ishmemi_params.ENABLE_GPU_IPC)
            ISHMEM_WARN_MSG("Disabling IPC - it is unsupported when shared heap is enabled\n");
        ishmemi_params.ENABLE_GPU_IPC = 0;
    }

    if (attr->gpu && ishmemi_params.ENABLE_GPU_IPC) {
        ret = ishmemi_ipc_init();
        ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "IPC initialization failed '%d'\n", ret);
        ipc_initialized = 1;
    } else {
        /* Populate local_pes in info */
        for (int i = 0; i < ishmemi_cpu_info->n_pes; ++i) {
            /* Note: local_pes[i] == 0 means "not local" */
            ishmemi_mmap_gpu_info->local_pes[i] = 0; /* For device use */
            ishmemi_local_pes[i] = 0;                /* For host use */
            ISHMEM_DEBUG_MSG("local_pes[%d] = NA\n", i);
        }
        ishmemi_local_pes[ishmemi_my_pe] = 1;
        ishmemi_mmap_gpu_info->local_pes[ishmemi_my_pe] = 1;
        ishmemi_ipc_buffer_delta[1] = 0;
        ishmemi_mmap_gpu_info->ipc_buffer_delta[1] = 0;
        ishmemi_mmap_gpu_info->only_intra_node = false;
    }

    ret = ishmemi_team_init();
    ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "Teams initialization failed '%d'\n", ret);
    teams_initialized = 1;

    ret = ishmemi_collectives_init();
    ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "Collectives initialization failed '%d'\n", ret);
    collectives_initialized = 1;

    /* proxy_init will initialize ring data structures */
    ret = ishmemi_proxy_init();
    ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "Proxy initialization failed '%d'\n", ret);

    ishmemi_mmap_gpu_info->is_initialized = true;
    return;

cleanup:
    if (collectives_initialized) {
        ishmemi_collectives_fini();
    }

    if (teams_initialized) {
        ishmemi_team_fini();
    }

    if (ipc_initialized) {
        ishmemi_ipc_fini();
    }

    ISHMEMI_FREE(free, ishmemi_cpu_info->attr);
    ISHMEMI_FREE(free, ishmemi_cpu_info);
    ISHMEMI_FREE(free, ishmemi_local_pes);

    if (memory_initialized) {
        ishmemi_memory_fini();
    }

    if (accelerator_initialized) {
        ishmemi_accelerator_fini();
    }

    if (runtime_initialized) {
        ishmemi_runtime_fini();
    }

    exit(1);
}

void ishmem_finalize()
{
    int ret = -1;
    ishmem_barrier_all();

    ret = ishmemi_proxy_fini();
    ISHMEM_CHECK_GOTO_MSG(ret, fail, "Proxy finalize failed '%d'\n", ret);

    ret = ishmemi_team_fini();
    ISHMEM_CHECK_GOTO_MSG(ret, fail, "Teams finalize failed '%d'\n", ret);

    ret = ishmemi_collectives_fini();
    ISHMEM_CHECK_GOTO_MSG(ret, fail, "Collectives finalize failed '%d'\n", ret);

    if (ishmemi_cpu_info->use_ipc) {
        ret = ishmemi_ipc_fini();
        ISHMEM_CHECK_GOTO_MSG(ret, fail, "IPC finalize failed '%d'\n", ret);
    }

    if (ishmemi_cpu_info->attr->gpu) {
        ret = ishmemi_memory_fini();
        ISHMEM_CHECK_GOTO_MSG(ret, fail, "Memory finalize failed '%d'\n", ret);

        ret = ishmemi_accelerator_fini();
        ISHMEM_CHECK_GOTO_MSG(ret, fail, "Accelerator finalize failed '%d'\n", ret);
    }

    ret = ishmemi_runtime_fini();
    ISHMEM_CHECK_GOTO_MSG(ret, fail, "Runtime finalize failed '%d'\n", ret);

    ISHMEMI_FREE(free, ishmemi_local_pes);
    ISHMEMI_FREE(free, ishmemi_cpu_info->attr);
    ISHMEMI_FREE(free, ishmemi_cpu_info);

    return;

fail:
    // TODO: Need to do any cleanup?
    return;
}

/* these are host memory copies */
uint8_t *ishmemi_local_pes;
void *ishmemi_ipc_buffers[MAX_LOCAL_PES + 1];
ptrdiff_t ishmemi_ipc_buffer_delta[MAX_LOCAL_PES + 1];

/* These functions have different behavior on device and on host */
int ishmem_my_pe()
{
#ifdef __SYCL_DEVICE_ONLY__
    return global_info->my_pe;
#else   // end __SYCL_DEVICE_ONLY__ start !__SYCL_DEVICE_ONLY__
    return ishmemi_cpu_info->my_pe;
#endif  // end !__SYCL_DEVICE_ONLY__
}

int ishmem_n_pes()
{
#ifdef __SYCL_DEVICE_ONLY__
    return global_info->n_pes;
#else   // end __SYCL_DEVICE_ONLY__ start !__SYCL_DEVICE_ONLY__
    return ishmemi_cpu_info->n_pes;
#endif  // end !__SYCL_DEVICE_ONLY__
}

void ishmem_info_get_version(int *major, int *minor)
{
    *major = ISHMEM_MAJOR_VERSION;
    *minor = ISHMEM_MINOR_VERSION;
}

void ishmem_info_get_name(char *name)
{
    char const *vendor_string = ISHMEM_VENDOR_STRING;
    while (*vendor_string) {
        *name++ = *vendor_string++;
    }
    *name = '\0';
}

void *ishmem_ptr(const void *dest, int pe)
{
    if constexpr (enable_error_checking) validate_parameters(pe);
    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
    if (local_index != 0) {
        return ISHMEMI_ADJUST_PTR(void, local_index, dest);
    } else {
        return nullptr;
    }
}

int ishmem_team_my_pe(ishmem_team_t team)
{
    if (team <= ISHMEM_TEAM_INVALID || team >= ISHMEMI_N_TEAMS) return -1;
    else
#ifdef __SYCL_DEVICE_ONLY__
        return global_info->team_pool[team]->my_pe;
#else
        return ishmemi_mmap_gpu_info->team_pool[team]->my_pe;
#endif
}

int ishmem_team_n_pes(ishmem_team_t team)
{
    if (team <= ISHMEM_TEAM_INVALID || team >= ISHMEMI_N_TEAMS) return -1;
    else
#ifdef __SYCL_DEVICE_ONLY__
        return global_info->team_pool[team]->size;
#else
        return ishmemi_mmap_gpu_info->team_pool[team]->size;
#endif
}

int ishmem_team_get_config(ishmem_team_t team, long config_mask, ishmem_team_config_t *config)
{
    if (team <= ISHMEM_TEAM_INVALID || team >= ISHMEMI_N_TEAMS) return -1;

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_team_t *myteam = global_info->team_pool[team];
#else
    ishmemi_team_t *myteam = ishmemi_mmap_gpu_info->team_pool[team];
#endif
    if (config_mask != 0) {
        if (config_mask != ISHMEM_TEAM_NUM_CONTEXTS) {
            ISHMEM_WARN_MSG("Invalid team config mask (%ld)\n", config_mask);
            return -1;
        }
        if (config == NULL) {
            ISHMEM_WARN_MSG("NULL config pointer passed to shmem_team_get_config\n");
            return -1;
        }
        memcpy(config, &myteam->config, sizeof(ishmem_team_config_t));
    } else if (config != NULL) {
        ISHMEM_WARN_MSG("%s %s\n", "ishmem_team_get_config encountered an unexpected",
                        "non-NULL config structure passed with a config_mask of 0.");
    }
    return 0;
}

int ishmem_team_translate_pe(ishmem_team_t src_team, int src_pe, ishmem_team_t dest_team)
{
    if (src_team <= ISHMEM_TEAM_INVALID || dest_team <= ISHMEM_TEAM_INVALID ||
        src_team >= ISHMEMI_N_TEAMS || dest_team >= ISHMEMI_N_TEAMS)
        return -1;

#ifdef __SYCL_DEVICE_ONLY__
    return ishmemi_team_translate_pe(global_info->team_pool[src_team], src_pe,
                                     global_info->team_pool[dest_team]);
#else
    return ishmemi_team_translate_pe(ishmemi_mmap_gpu_info->team_pool[src_team], src_pe,
                                     ishmemi_mmap_gpu_info->team_pool[dest_team]);
#endif
}

/* Teams Management Routines */
int ishmem_team_split_strided(ishmem_team_t parent_team, int PE_start, int PE_stride, int PE_size,
                              const ishmem_team_config_t *config, long config_mask,
                              ishmem_team_t *new_team)
{
    if (parent_team <= ISHMEM_TEAM_INVALID || parent_team >= ISHMEMI_N_TEAMS ||
        (PE_stride == 0 && PE_size != 1))
        return -1;

    return ishmemi_team_split_strided(ishmemi_mmap_gpu_info->team_pool[parent_team], PE_start,
                                      PE_stride, PE_size, config, config_mask, new_team);
}

int ishmem_team_split_2d(ishmem_team_t parent_team, int xrange,
                         const ishmem_team_config_t *xaxis_config, long xaxis_mask,
                         ishmem_team_t *xaxis_team, const ishmem_team_config_t *yaxis_config,
                         long yaxis_mask, ishmem_team_t *yaxis_team)
{
    if (parent_team <= ISHMEM_TEAM_INVALID || parent_team >= ISHMEMI_N_TEAMS) return -1;

    return ishmemi_team_split_2d(ishmemi_mmap_gpu_info->team_pool[parent_team], xrange,
                                 xaxis_config, xaxis_mask, xaxis_team, yaxis_config, yaxis_mask,
                                 yaxis_team);
}

void ishmem_team_destroy(ishmem_team_t team)
{
    if (team <= ISHMEM_TEAM_INVALID || team >= ISHMEMI_N_TEAMS) return;

    if (team == ISHMEM_TEAM_WORLD || team == ISHMEM_TEAM_SHARED || team == ISHMEMX_TEAM_NODE) {
        ISHMEM_WARN_MSG("User attempted to destroy a pre-defined team.\n");
        return;
    }

    int ret = ishmemi_team_destroy(ishmemi_mmap_gpu_info->team_pool[team]);
    if (ret != 0) {
        RAISE_ERROR_MSG("ishmem_team_destroy failed\n");
    }
}

int ishmem_team_sync(ishmem_team_t team)
{
    if (team <= ISHMEM_TEAM_INVALID || team >= ISHMEMI_N_TEAMS) return -1;

#ifdef __SYCL_DEVICE_ONLY__
    return ishmemi_team_sync(global_info->team_pool[team]);
#else
    return ishmemi_team_sync(ishmemi_mmap_gpu_info->team_pool[team]);
#endif
}

void ishmem_fence()
{
#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_request_t req;
    req.op = FENCE;
    req.type = MEM;

    ishmemi_proxy_blocking_request(req);
    atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
#else
    ishmemi_runtime_fence();
#endif
}

template void ishmemx_fence_work_group<sycl::group<1>>(const sycl::group<1> &grp);
template void ishmemx_fence_work_group<sycl::group<2>>(const sycl::group<2> &grp);
template void ishmemx_fence_work_group<sycl::group<3>>(const sycl::group<3> &grp);
template void ishmemx_fence_work_group<sycl::sub_group>(const sycl::sub_group &grp);
template <typename Group>
void ishmemx_fence_work_group(const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        sycl::group_barrier(grp);
        if (grp.leader()) {
            ishmemi_request_t req;
            req.op = FENCE;
            req.type = MEM;

            ishmemi_proxy_blocking_request(req);
        }
        atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_fence_work_group not callable from host\n");
    }
}

void ishmem_quiet()
{
#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_request_t req;
    req.op = QUIET;
    req.type = MEM;

    ishmemi_proxy_blocking_request(req);
    atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
#else
    ishmemi_runtime_quiet();
#endif
}

template void ishmemx_quiet_work_group<sycl::group<1>>(const sycl::group<1> &grp);
template void ishmemx_quiet_work_group<sycl::group<2>>(const sycl::group<2> &grp);
template void ishmemx_quiet_work_group<sycl::group<3>>(const sycl::group<3> &grp);
template void ishmemx_quiet_work_group<sycl::sub_group>(const sycl::sub_group &grp);
template <typename Group>
void ishmemx_quiet_work_group(const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        sycl::group_barrier(grp);
        if (grp.leader()) {
            ishmemi_request_t req;
            req.op = QUIET;
            req.type = MEM;

            ishmemi_proxy_blocking_request(req);
        }
        atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_quiet_work_group not callable from host\n");
    }
}
