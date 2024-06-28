/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_H
#define ISHMEM_RUNTIME_H

#include "ishmemx.h"
#include "proxy_impl.h"

/* This sums the total number of base types used in defining the proxy function pointers:
 * - void, uint8_t, uint16_t, uint32_t, uint64_t
 */
constexpr size_t ishmemi_runtime_proxy_func_num_types = 14;

/* Runtime proxy routines */
typedef void (*ishmemi_runtime_proxy_func_t)(ishmemi_request_t *, ishmemi_ringcompletion_t *);
extern ishmemi_runtime_proxy_func_t **ishmemi_proxy_funcs;

/* Runtime routines */

/* Initialize runtime */
int ishmemi_runtime_init(ishmemx_attr_t *);

/* Pre-initialize Heap */
void ishmemi_runtime_heap_create(ishmemx_attr_t *, void *base, size_t size);

/* Finalize runtime */
extern int (*ishmemi_runtime_fini)(void);

/* Abort runtime */
extern void (*ishmemi_runtime_abort)(int, const char[]);

/* Get rank */
extern int (*ishmemi_runtime_get_rank)(void);

/* Get number of ranks */
extern int (*ishmemi_runtime_get_size)(void);

/* Get node-local rank */
extern int (*ishmemi_runtime_get_node_rank)(int pe);

/* Get node-local size */
extern int (*ishmemi_runtime_get_node_size)(void);

/* Sync */
extern void (*ishmemi_runtime_sync)(void);

/* Fence */
extern void (*ishmemi_runtime_fence)(void);

/* Quiet */
extern void (*ishmemi_runtime_quiet)(void);

/* Global barrier */
extern void (*ishmemi_runtime_barrier_all)(void);

/* Node-local barrier */
extern void (*ishmemi_runtime_node_barrier)(void);

/* Global broadcast */
extern void (*ishmemi_runtime_bcast)(void *buf, size_t count, int root);

/* Node-local broadcast */
extern void (*ishmemi_runtime_node_bcast)(void *buf, size_t count, int root);

/* Global fcollect */
extern void (*ishmemi_runtime_fcollect)(void *dst, void *src, size_t count);

/* Node-local fcollect */
extern void (*ishmemi_runtime_node_fcollect)(void *dst, void *src, size_t count);

/* Check if rank is node-local */
extern bool (*ishmemi_runtime_is_local)(int pe);

/* Get KVS value */
extern int (*ishmemi_runtime_get)(int pe, char *key, void *value, size_t valuelen);

/* Runtime-specific memory functions */
extern void *(*ishmemi_runtime_malloc)(size_t);
extern void *(*ishmemi_runtime_calloc)(size_t, size_t);
extern void (*ishmemi_runtime_free)(void *);

/* Runtime-specific initialization functions */
int ishmemi_runtime_mpi_init(bool initialize_runtime);
int ishmemi_runtime_openshmem_init(bool initialize_runtime);
int ishmemi_runtime_pmi_init(bool initialize_runtime);
void ishmemi_runtime_openshmem_heap_create(void *base, size_t size);

/* Team management functions */
typedef enum {
    WORLD,
    SHARED,
    NODE
} ishmemi_runtime_team_predefined_t;

inline const char *ishmemi_runtime_team_predefined_string(ishmemi_runtime_team_predefined_t val)
{
    switch (val) {
        case WORLD:
            return "WORLD";
        case SHARED:
            return "SHARED";
        case NODE:
            return "NODE";
        default:
            ISHMEM_ERROR_MSG("Unknown team passed to ishmemi_runtime_team_predefined_string\n");
            return "";
    }
}

extern int (*ishmemi_runtime_team_split_strided)(ishmemi_runtime_team_t parent_team, int PE_start,
                                                 int PE_stride, int PE_size,
                                                 const ishmemi_runtime_team_config_t *config,
                                                 long config_mask,
                                                 ishmemi_runtime_team_t *new_team);
extern int (*ishmemi_runtime_uchar_and_reduce)(ishmemi_runtime_team_t team, unsigned char *dest,
                                               const unsigned char *source, size_t nreduce);
extern int (*ishmemi_runtime_int_max_reduce)(ishmemi_runtime_team_t team, int *dest,
                                             const int *source, size_t nreduce);
extern int (*ishmemi_runtime_team_sync)(ishmemi_runtime_team_t team);
extern int (*ishmemi_runtime_team_predefined_set)(
    ishmemi_runtime_team_t *team, ishmemi_runtime_team_predefined_t predefined_team_name,
    int expected_team_size, int expected_world_pe, int expected_team_pe);

/* Runtime-specific function pointer setup */
void ishmemi_runtime_mpi_funcptr_init(void);
void ishmemi_runtime_openshmem_funcptr_init(void);
void ishmemi_runtime_pmi_funcptr_init(void);

#endif /* ISHMEM_RUNTIME_H */
