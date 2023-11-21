/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "internal.h"
#include "accelerator.h"
#include "collectives.h"
#include "impl_proxy.h"
#include "internal.h"
#include "ipc.h"
#include "memory.h"
#include "proxy.h"
#include "runtime.h"
#include "collectives/reduce_impl.h"
#include "collectives/broadcast_impl.h"
#include "collectives/collect_impl.h"
#include "collectives/alltoall_impl.h"
#include <stdlib.h>  // abort

int ishmemi_my_pe;
int ishmemi_n_pes;

/* Note: this order should match ishmemi_op_t */
const char *ishmemi_op_str[] = {
    "put",
    "put_work_group",
    "iput",
    "iput_work_group",
    "p",
    "get",
    "get_work_group",
    "iget",
    "iget_work_group",
    "g",
    "put_nbi",
    "put_nbi_work_group",
    "get_nbi",
    "get_nbi_work_group",
    "amo_fetch",
    "amo_set",
    "amo_compare_swap",
    "amo_swap",
    "amo_fetch_inc",
    "amo_inc",
    "amo_fetch_add",
    "amo_add",
    "amo_fetch_and",
    "amo_and",
    "amo_fetch_or",
    "amo_or",
    "amo_fetch_xor",
    "amo_xor",
    "put_signal",
    "put_signal_work_group",
    "put_signal_nbi",
    "put_signal_nbi_work_group",
    "signal_fetch",
    "barrier",
    "barrier_work_group",
    "sync",
    "sync_work_group",
    "alltoall",
    "alltoall_work_group",
    "bcast",
    "bcast_work_group",
    "collect",
    "collect_work_group",
    "fcollect",
    "fcollect_work_group",
    "and_reduce",
    "and_reduce_work_group",
    "or_reduce",
    "or_reduce_work_group",
    "xor_reduce",
    "xor_reduce_work_group",
    "max_reduce",
    "max_reduce_work_group",
    "min_reduce",
    "min_reduce_work_group",
    "sum_reduce",
    "sum_reduce_work_group",
    "prod_reduce",
    "prod_reduce_work_group",
    "test",
    "wait",
    "fence",
    "quiet",
    "kill",
    "nop",
    "nop_no_r",
    "debug_test",
};

/* Note: this order should match ishmemi_type_t */
const char *ishmemi_type_str[] = {
    "mem",   "uint8",    "uint16", "uint32", "uint64",     "ulonglong", "int8",    "int16", "int32",
    "int64", "longlong", "float",  "double", "longdouble", "char",      "schar",   "short", "int",
    "long",  "uchar",    "ushort", "uint",   "ulong",      "size",      "ptrdiff",
};

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
sycl::ext::oneapi::experimental::device_global<ishmem_info_t *> global_info;

#define ISHMEMI_API_IMPL(SUFFIX)                                                                   \
    void ishmemx_##SUFFIX()                                                                        \
    {                                                                                              \
        if constexpr (ishmemi_is_device) {                                                         \
            ishmemi_request_t req = {                                                              \
                .op = ISHMEMI_OP_##SUFFIX,                                                         \
                .type = MEM,                                                                       \
            };                                                                                     \
            ishmemi_proxy_blocking_request(&req);                                                  \
        }                                                                                          \
    }

#define ISHMEMI_API_IMPL_NO_R(SUFFIX)                                                              \
    void ishmemx_##SUFFIX()                                                                        \
    {                                                                                              \
        if constexpr (ishmemi_is_device) {                                                         \
            ishmemi_request_t req = {                                                              \
                .op = ISHMEMI_OP_##SUFFIX,                                                         \
                .type = MEM,                                                                       \
            };                                                                                     \
            ishmemi_proxy_nonblocking_request(&req);                                               \
        }                                                                                          \
    }

/* This API is only for test purposes */
ISHMEMI_API_IMPL(debug_test)
ISHMEMI_API_IMPL(nop)
ISHMEMI_API_IMPL_NO_R(nop_no_r)

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
    int collectives_initialized = 0;
    static_assert(sizeof(ishmemi_request_t) == 64, "ISHMEM request object must be 64 bytes.");

    ishmemi_cpu_info = (ishmem_cpu_info_t *) malloc(sizeof(ishmem_cpu_info_t));
    ISHMEM_CHECK_GOTO_MSG(ishmemi_cpu_info == nullptr, cleanup,
                          "CPU info object allocation failed\n");
    memset(ishmemi_cpu_info, 0, sizeof(ishmem_cpu_info_t));

    /* Currently, no support for CPU environments */
    if (!attr->gpu) {
        ISHMEM_WARN_MSG("Currently, no support for CPU-only environment\n");
        attr->gpu = true;
    }

    /* Currently, no support for PMI, MPI */
    if (attr->runtime != ISHMEMX_RUNTIME_OPENSHMEM) {
        ISHMEM_WARN_MSG("Currently, no support for runtimes other than OpenSHMEM\n");
        attr->runtime = ISHMEMX_RUNTIME_OPENSHMEM;
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
    ishmemi_runtime_heap_create(ishmemi_heap_base, ishmemi_heap_length);

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

void ishmem_fence()
{
#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_request_t req = {
        .op = FENCE,
        .type = MEM,
    };

    ishmemi_proxy_blocking_request(&req);
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
            ishmemi_request_t req = {
                .op = FENCE,
                .type = MEM,
            };

            ishmemi_proxy_blocking_request(&req);
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
    ishmemi_request_t req = {
        .op = QUIET,
        .type = MEM,
    };

    ishmemi_proxy_blocking_request(&req);
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
            ishmemi_request_t req = {
                .op = QUIET,
                .type = MEM,
            };

            ishmemi_proxy_blocking_request(&req);
        }
        atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_quiet_work_group not callable from host\n");
    }
}
