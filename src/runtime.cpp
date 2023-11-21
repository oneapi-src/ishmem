/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "runtime.h"

ishmemi_runtime_proxy_func_t **ishmemi_proxy_funcs;

int (*ishmemi_runtime_fini)(void);
void (*ishmemi_runtime_abort)(int, const char[]);
int (*ishmemi_runtime_get_rank)(void);
int (*ishmemi_runtime_get_size)(void);
int (*ishmemi_runtime_get_node_rank)(int pe);
int (*ishmemi_runtime_get_node_size)(void);
void (*ishmemi_runtime_sync)(void);
void (*ishmemi_runtime_fence)(void);
void (*ishmemi_runtime_quiet)(void);
void (*ishmemi_runtime_barrier)(void);
void (*ishmemi_runtime_node_barrier)(void);
void (*ishmemi_runtime_bcast)(void *buf, size_t count, int root);
void (*ishmemi_runtime_node_bcast)(void *buf, size_t count, int root);
void (*ishmemi_runtime_fcollect)(void *dst, void *src, size_t count);
void (*ishmemi_runtime_node_fcollect)(void *dst, void *src, size_t count);
bool (*ishmemi_runtime_is_local)(int pe);
void *(*ishmemi_runtime_malloc)(size_t);
void *(*ishmemi_runtime_calloc)(size_t, size_t);
void (*ishmemi_runtime_free)(void *);
int (*ishmemi_runtime_get)(int pe, char *key, void *value, size_t valuelen);

int ishmemi_runtime_init(ishmemx_attr_t *attr)
{
    int ret = 0;

    switch (attr->runtime) {
        case ISHMEMX_RUNTIME_MPI:
#if defined(ENABLE_MPI)
            ret = ishmemi_runtime_mpi_init(attr->initialize_runtime);
#else
            ISHMEM_ERROR_MSG(
                "MPI runtime was selected, but Intel® SHMEM is not built with MPI support\n");
            ret = -1;
#endif
            break;
        case ISHMEMX_RUNTIME_OPENSHMEM:
#if defined(ENABLE_OPENSHMEM)
            ret = ishmemi_runtime_openshmem_init(attr->initialize_runtime);
#else
            ISHMEM_ERROR_MSG(
                "OpenSHMEM runtime was selected, but Intel® SHMEM is not built with OpenSHMEM "
                "support\n");
            ret = -1;
#endif
            break;
        case ISHMEMX_RUNTIME_PMI:
#if defined(ENABLE_PMI)
            ret = ishmemi_runtime_pmi_init(attr->initialize_runtime);
#else
            ISHMEM_ERROR_MSG(
                "PMI runtime was selected, but Intel® SHMEM is not built with PMI support\n");
            ret = -1;
#endif
            break;
        default:
            ISHMEM_ERROR_MSG("Invalid runtime selection\n");
            ret = -1;
    }

    return ret;
}

void ishmemi_runtime_heap_create(void *base, size_t size)
{
/* Currently only support SHMEM */
#if defined(ENABLE_MPI)
#error Heap preinit is not supported yet with MPI runtime
#elif defined(ENABLE_OPENSHMEM)
    ishmemi_runtime_openshmem_heap_create(base, size);
#elif defined(ENABLE_PMI)
#error Heap preinit is not supported yet with PMI runtime
#else
#error No valid runtime configured
#endif
}
