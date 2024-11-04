/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_MEMORY_H
#define ISHMEM_MEMORY_H

#include <cstddef>
#include <cstdint>

#define ISHMEMI_HEAP_OVERHEAD 1024

#ifdef __cplusplus
extern "C" {
#endif

typedef void *mspace;
/* mspace routines */
mspace create_mspace_with_base(void *, size_t, int);
void *mspace_memalign(mspace, size_t, size_t);
void mspace_free(mspace, void *);

#define ISHMEMI_ALLOC_ALIGN ((size_t) 64)
void *ishmemi_get_next(size_t incr, size_t alignment = ISHMEMI_ALLOC_ALIGN);

/* Memory routines */
/* Initialize memory */
int ishmemi_memory_init();

/* Finalize memory */
int ishmemi_memory_fini();

void *ishmemi_calloc(size_t count, size_t size);
void *ishmemi_ptr(const void *dest, int pe);

#ifdef __cplusplus
}
#endif

#define ISHMEMI_FAST_ADJUST(TYPENAME, info, index, p)                                              \
    ((TYPENAME *) (reinterpret_cast<ptrdiff_t>(p) +                                                \
                   static_cast<ptrdiff_t>(info->ipc_buffer_delta[(index)])))

#ifdef __SYCL_DEVICE_ONLY__
#define ISHMEMI_ADJUST_PTR(TYPENAME, index, p)                                                     \
    ((TYPENAME *) (reinterpret_cast<ptrdiff_t>(p) +                                                \
                   static_cast<ptrdiff_t>(global_info->ipc_buffer_delta[(index)])))
#define ISHMEMI_DEVICE_TO_MMAP_ADDR(TYPENAME, p_device) ((TYPENAME *) (p_device))
#else
#define ISHMEMI_ADJUST_PTR(TYPENAME, index, p)                                                     \
    ((TYPENAME *) (reinterpret_cast<ptrdiff_t>(p) +                                                \
                   static_cast<ptrdiff_t>(ishmemi_ipc_buffer_delta[(index)])))
#define ISHMEMI_DEVICE_TO_MMAP_ADDR(TYPENAME, p_device)                                            \
    ((TYPENAME *) (((uintptr_t) ishmemi_mmap_heap_base) +                                          \
                   (((uintptr_t) p_device) - ((uintptr_t) ishmemi_heap_base))))
#endif

#define ISHMEMI_HOST_ADJUST_PTR(TYPENAME, index, p)                                                \
    ((TYPENAME *) (reinterpret_cast<ptrdiff_t>(p) +                                                \
                   static_cast<ptrdiff_t>(ishmemi_ipc_buffer_delta[(index)])))

#define ISHMEMI_HOST_IN_HEAP(p)                                                                    \
    ((((uintptr_t) p) >= ((uintptr_t) ishmemi_heap_base)) &&                                       \
     (((uintptr_t) p) < (((uintptr_t) ishmemi_heap_base) + ishmemi_heap_length)))

/* Common code for pointer arithmetic */
template <typename T>
inline T *pointer_offset(T *p, ptrdiff_t offset)
{
    return ((T *) (((intptr_t) p) + offset));
}
template <typename T>
inline T *pointer_offset(T *p, size_t offset)
{
    return ((T *) (((uintptr_t) p) + offset));
}

inline bool pointer_less_or_equal(void *a, void *b)
{
    return (((uintptr_t) a) <= ((uintptr_t) b));
}

inline bool pointer_greater_or_equal(void *a, void *b)
{
    return (((uintptr_t) a) >= ((uintptr_t) b));
}

#endif /* ISHMEM_MEMORY_H */
