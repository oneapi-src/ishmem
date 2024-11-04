/* Copyright (C) 2024 Intel Corporation
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
#include "collectives/reduce_impl.h"
#include "collectives/broadcast_impl.h"
#include "collectives/collect_impl.h"
#include "collectives/alltoall_impl.h"
#include <cstdlib>  // abort

int ishmemi_my_pe;
int ishmemi_n_pes;

int ishmemi_initialized;

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

ishmemi_on_queue_map ishmemi_on_queue_events_map;

#define ISHMEMI_API_IMPL(SUFFIX)                                                                   \
    void ishmemx_##SUFFIX()                                                                        \
    {                                                                                              \
        if constexpr (ishmemi_is_device) {                                                         \
            ishmemi_request_t req;                                                                 \
            req.op = ISHMEMI_OP_##SUFFIX;                                                          \
            req.type = NONE;                                                                       \
            ishmemi_proxy_blocking_request(req);                                                   \
        }                                                                                          \
    }

#define ISHMEMI_API_IMPL_NO_R(SUFFIX)                                                              \
    void ishmemx_##SUFFIX()                                                                        \
    {                                                                                              \
        if constexpr (ishmemi_is_device) {                                                         \
            ishmemi_request_t req;                                                                 \
            req.op = ISHMEMI_OP_##SUFFIX;                                                          \
            req.type = NONE;                                                                       \
            ishmemi_proxy_nonblocking_request(req);                                                \
        }                                                                                          \
    }

/* This API is only for test purposes */
ISHMEMI_API_IMPL(debug_test)
ISHMEMI_API_IMPL(nop)
ISHMEMI_API_IMPL_NO_R(nop_no_r)

void ishmemi_init_op_str()
{
    ishmemi_op_str[UNDEFINED] = "undefined";
    ishmemi_op_str[PUT] = "put";
    ishmemi_op_str[IPUT] = "iput";
    ishmemi_op_str[IBPUT] = "ibput";
    ishmemi_op_str[P] = "p";
    ishmemi_op_str[GET] = "get";
    ishmemi_op_str[IGET] = "iget";
    ishmemi_op_str[IBGET] = "ibget";
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
    ishmemi_op_str[SIGNAL_WAIT_UNTIL] = "signal_wait_until";
    ishmemi_op_str[TEST] = "test";
    ishmemi_op_str[TEST_ALL] = "test_all";
    ishmemi_op_str[TEST_ANY] = "test_any";
    ishmemi_op_str[TEST_SOME] = "test_some";
    ishmemi_op_str[SIGNAL_WAIT_UNTIL] = "signal_wait_until";
    ishmemi_op_str[FENCE] = "fence";
    ishmemi_op_str[QUIET] = "quiet";
    ishmemi_op_str[TEAM_MY_PE] = "team_my_pe";
    ishmemi_op_str[TEAM_N_PES] = "team_n_pes";
    ishmemi_op_str[TEAM_SYNC] = "team_sync";
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
    ishmemi_type_str[NONE] = "none";
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

void ishmem_query_thread(int *provided)
{
    if (!ishmemi_initialized) {
        RAISE_ERROR_MSG("Library is not initialized\n");
    }

    if (provided) *provided = ISHMEM_THREAD_MULTIPLE;
}

void ishmemx_query_initialized(int *initialized)
{
#ifdef __SYCL_DEVICE_ONLY__
    if (initialized) *initialized = global_info->is_initialized ? 1 : 0;
#else   // end __SYCL_DEVICE_ONLY__ start !__SYCL_DEVICE_ONLY__
    if (initialized) *initialized = ishmemi_initialized;
#endif  // end !__SYCL_DEVICE_ONLY__
}

static void ishmemi_init(ishmemx_attr_t *attr, bool user_attr)
{
    int ret = -1;
    int runtime_initialized = 0;
    int accelerator_initialized = 0;
    int memory_initialized = 0;
    int ipc_initialized = 0;
    int teams_initialized = 0;
    int collectives_initialized = 0;
    ishmemx_runtime_type_t env_runtime;

    ishmemi_init_op_str();
    ishmemi_init_type_str();
    static_assert(sizeof(ishmemi_request_t) == 64, "ISHMEM request object must be 64 bytes.");
    static_assert(sizeof(ishmemi_completion_t) == 64, "ISHMEM completion object must be 64 bytes.");
    static_assert(sizeof(ishmemi_ringcompletion_t) == 64,
                  "ISHMEM ringcompletion object must be 64 bytes.");

    ishmemi_cpu_info = (ishmemi_cpu_info_t *) ::malloc(sizeof(ishmemi_cpu_info_t));
    ISHMEM_CHECK_GOTO_MSG(ishmemi_cpu_info == nullptr, cleanup,
                          "CPU info object allocation failed\n");
    ::memset(ishmemi_cpu_info, 0, sizeof(ishmemi_cpu_info_t));

    /* Confirm valid runtime is provided */
    ret = (attr->runtime >= ISHMEMX_RUNTIME_INVALID);
    ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "Invalid runtime provided '%u' in ishmemx_attr_t\n",
                          attr->runtime);

    /* Currently, no support for CPU environments */
    if (!attr->gpu) {
        ISHMEM_WARN_MSG("Currently, no support for CPU-only environment\n");
        attr->gpu = true;
    }

    /* Parse environment variables */
    ret = ishmemi_parse_env();
    ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "Parsing env variables failed '%d'\n", ret);

    /* Check if environment ISHMEM_RUNTIME and value in attr conflict */
    env_runtime = ishmemi_env_get_runtime();
    if (env_runtime != attr->runtime) {
        if (user_attr) {
            ISHMEM_WARN_MSG(
                "ISHMEM_RUNTIME environment variable and attr.runtime conflict! Using "
                "ISHMEM_RUNTIME.\n");
        }
        attr->runtime = env_runtime;
    }

    /* Initialize the runtime */
    ret = ishmemi_runtime_init(attr);
    ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "Runtime initialization failed '%d'\n", ret);
    runtime_initialized = 1;

    ishmemi_my_pe = ishmemi_runtime->get_rank();
    ishmemi_n_pes = ishmemi_runtime->get_size();
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
    ishmemi_runtime->heap_create(ishmemi_heap_base, ishmemi_heap_length);

    ishmemi_cpu_info->use_ipc = false;  // This will be set to true if ishmemi_ipc_init passes

    /* Initialize the attributes */
    ishmemi_cpu_info->attr = (ishmemx_attr_t *) ::malloc(sizeof(ishmemx_attr_t));
    ISHMEM_CHECK_GOTO_MSG(ishmemi_cpu_info->attr == nullptr, cleanup,
                          "Attribute object allocation failed\n");
    memcpy(ishmemi_cpu_info->attr, attr, sizeof(ishmemx_attr_t));

    /* info structure is allocated by memory_init */
    ishmemi_mmap_gpu_info->my_pe = ishmemi_my_pe;
    ishmemi_mmap_gpu_info->n_pes = ishmemi_n_pes;
    ishmemi_mmap_gpu_info->heap_base = ishmemi_heap_base;
    ishmemi_mmap_gpu_info->heap_length = ishmemi_heap_length;

    /* Setup local_pes info for host use */
    ishmemi_local_pes = (uint8_t *) ::malloc(static_cast<size_t>(ishmemi_n_pes));
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
    ishmemi_cpu_info->is_initialized = true;
    ishmemi_initialized = 1;

    /* we cannot start running until all PEs have finished initializing */
    ishmemi_runtime->barrier_all();

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

    ISHMEMI_FREE(::free, ishmemi_cpu_info->attr);
    ISHMEMI_FREE(::free, ishmemi_cpu_info);
    ISHMEMI_FREE(::free, ishmemi_local_pes);

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

void ishmemx_init_attr(ishmemx_attr_t *attr)
{
    /* Initialize */
    ishmemi_init(attr, true);
}

void ishmem_init()
{
    if (ishmemi_initialized) {
        RAISE_ERROR_MSG("Attempt to re-initialize library\n");
    }

    /* Use default attributes */
    ishmemx_attr_t attr;

    /* Initialize */
    ishmemi_init(&attr, false);
}

int ishmem_init_thread(int requested, int *provided)
{
    if (ISHMEM_THREAD_MULTIPLE != requested) {
        ISHMEM_WARN_MSG("Only ISHMEM_THREAD_MULTIPLE is supported\n");
    }

    ishmem_init();
    if (provided) *provided = ISHMEM_THREAD_MULTIPLE;

    return 0;
}

void ishmem_finalize()
{
    int ret = -1;

    if (!ishmemi_initialized) {
        RAISE_ERROR_MSG("Attempt to finalize library without initializing\n");
        return;
    }

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

    ret = ishmemi_runtime_fini();
    ISHMEM_CHECK_GOTO_MSG(ret, fail, "Runtime finalize failed '%d'\n", ret);

    if (ishmemi_cpu_info->attr->gpu) {
        ret = ishmemi_memory_fini();
        ISHMEM_CHECK_GOTO_MSG(ret, fail, "Memory finalize failed '%d'\n", ret);

        ret = ishmemi_accelerator_fini();
        ISHMEM_CHECK_GOTO_MSG(ret, fail, "Accelerator finalize failed '%d'\n", ret);
    }

    ISHMEMI_FREE(::free, ishmemi_local_pes);
    ISHMEMI_FREE(::free, ishmemi_cpu_info->attr);
    ISHMEMI_FREE(::free, ishmemi_cpu_info);

    ishmemi_initialized = 0;

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
    if constexpr (enable_error_checking) validate_init();

#ifdef __SYCL_DEVICE_ONLY__
    return global_info->my_pe;
#else   // end __SYCL_DEVICE_ONLY__ start !__SYCL_DEVICE_ONLY__
    return ishmemi_cpu_info->my_pe;
#endif  // end !__SYCL_DEVICE_ONLY__
}

int ishmem_n_pes()
{
    if constexpr (enable_error_checking) validate_init();

#ifdef __SYCL_DEVICE_ONLY__
    return global_info->n_pes;
#else   // end __SYCL_DEVICE_ONLY__ start !__SYCL_DEVICE_ONLY__
    return ishmemi_cpu_info->n_pes;
#endif  // end !__SYCL_DEVICE_ONLY__
}

void ishmem_info_get_version(int *major, int *minor)
{
    if constexpr (enable_error_checking) validate_init();

    *major = ISHMEM_MAJOR_VERSION;
    *minor = ISHMEM_MINOR_VERSION;
}

void ishmem_info_get_name(char *name)
{
    if constexpr (enable_error_checking) validate_init();

    char const *vendor_string = ISHMEM_VENDOR_STRING;
    while (*vendor_string) {
        *name++ = *vendor_string++;
    }
    *name = '\0';
}
