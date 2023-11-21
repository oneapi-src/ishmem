/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_H
#define ISHMEM_RUNTIME_H

#include "internal.h"
#include "impl_proxy.h"

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
void ishmemi_runtime_heap_create(void *base, size_t size);

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
extern void (*ishmemi_runtime_barrier)(void);

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

/* Runtime-specific function pointer setup */
void ishmemi_runtime_mpi_funcptr_init(void);
void ishmemi_runtime_openshmem_funcptr_init(void);
void ishmemi_runtime_pmi_funcptr_init(void);

#endif /* ISHMEM_RUNTIME_H */
