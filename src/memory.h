/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_MEMORY_H
#define ISHMEM_MEMORY_H

#include "internal.h"
#include "accelerator.h"

#define ISHMEMI_HEAP_OVERHEAD 1024

#ifdef __cplusplus
extern "C" {
#endif

typedef void *mspace;

#define ISHMEMI_ALLOC_ALIGN ((size_t) 64)
void *ishmemi_get_next(size_t incr, size_t alignment = ISHMEMI_ALLOC_ALIGN);

/* mspace routines */
mspace create_mspace_with_base(void *, size_t, int);
void *mspace_malloc(mspace, size_t);
void *mspace_calloc(mspace, size_t, size_t);
void *mspace_memalign(mspace, size_t, size_t);
void mspace_free(mspace, void *);

/* Memory routines */
/* Initialize memory */
int ishmemi_memory_init();

/* Finalize memory */
int ishmemi_memory_fini();

/* Copy utility function */
void *ishmem_copy(void *dst, void *src, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* ISHMEM_MEMORY_H */
