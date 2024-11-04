/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem.h"
#include "ishmemx.h"
#include "runtime.h"
#include "proxy_impl.h"
#include <execinfo.h>

/* Internal validation helper functions */
static inline void validate_initialized(const char *file, long int line, const char *func)
{
#ifdef __SYCL_DEVICE_ONLY__
    if (!global_info->is_initialized)
#else
    if (!ishmemi_cpu_info->is_initialized)
#endif
    {
        ishmemx_print(file, line, func, "Library is not initialized\n",
                      ishmemx_print_msg_type_t::ERROR);
    }
}

static inline void validate_pe(const char *file, long int line, const char *func, int pe)
{
    if ((pe < 0) || (pe >= ishmem_n_pes())) {
        ishmemx_print(file, line, func,
                      "Attempting to call Intel® SHMEM API using invalid PE value\n",
                      ishmemx_print_msg_type_t::ERROR);
    }
}

static inline void validate_object_is_symmetric(const char *file, long int line, const char *func,
                                                void *ptr, size_t size)
{
    if (size == 0) return;
    uintptr_t loptr = (uintptr_t) ptr;
    uintptr_t hiptr = loptr + size - 1;
#ifdef __SYCL_DEVICE_ONLY__
    void *heap_base = global_info->heap_base;
    unsigned long heap_length = global_info->heap_length;
    uintptr_t loheap = (uintptr_t) heap_base;
    uintptr_t hiheap = loheap + heap_length - 1;

    // Check if object on Intel® SHMEM heap
    if ((loptr >= loheap) && (loptr < hiheap)) {
        if (hiptr > hiheap) {
            ishmemx_print(file, line, func,
                          "Attempting to call Intel® SHMEM API using object that exceeds device "
                          "symmetric heap "
                          "region.\n",
                          ishmemx_print_msg_type_t::ERROR);
        }
    }
#else
    void *heap_base = ishmemi_heap_base;
    unsigned long heap_length = ishmemi_heap_length;
    uintptr_t loheap = (uintptr_t) heap_base;
    uintptr_t hiheap = loheap + heap_length - 1;

    // Check if object on runtime heap
    if (ishmemi_runtime->is_symmetric_address((void *) loptr)) {
        if (!ishmemi_runtime->is_symmetric_address((void *) hiptr)) {
            ishmemx_print(
                file, line, func,
                "Attempting to call Intel® SHMEM API using object that exceeds host symmetric heap "
                "region.\n",
                ishmemx_print_msg_type_t::ERROR);
        }
    }

    // Check if object on Intel® SHMEM heap
    else if ((loptr >= loheap) && (loptr <= hiheap)) {
        if (hiptr > hiheap) {
            ishmemx_print(file, line, func,
                          "Attempting to call Intel® SHMEM API using object that exceeds device "
                          "symmetric heap "
                          "region.\n",
                          ishmemx_print_msg_type_t::ERROR);
        }
    }
#endif /* end __SYCL_DEVICE_ONLY__ */
    else {
        ishmemx_print(file, line, func,
                      "Attempting to call Intel® SHMEM API using object that is not symmetric.\n",
                      ishmemx_print_msg_type_t::ERROR);
    }
}

static inline void validate_objects_dont_overlap(const char *file, long int line, const char *func,
                                                 void *ptr1, void *ptr2, size_t size1, size_t size2)
{
    if ((size1 == 0) || (size2 == 0)) return;
    uintptr_t lo1 = (uintptr_t) ptr1;
    uintptr_t lo2 = (uintptr_t) ptr2;
    uintptr_t hi1 = lo1 + size1 - 1;
    uintptr_t hi2 = lo2 + size2 - 1;
    if ((lo1 <= hi2) && (lo2 <= hi1)) {
        ishmemx_print(file, line, func,
                      "Attempting to call Intel® SHMEM API using overlapping arguments.\n",
                      ishmemx_print_msg_type_t::ERROR);
    }
}

static inline void validate_objects_dont_overlap(const char *file, long int line, const char *func,
                                                 int pe, void *ptr1, void *ptr2, size_t size1,
                                                 size_t size2)
{
    if (pe == ishmem_my_pe()) {
        if ((size1 == 0) || (size2 == 0)) return;
        uintptr_t lo1 = (uintptr_t) ptr1;
        uintptr_t lo2 = (uintptr_t) ptr2;
        uintptr_t hi1 = lo1 + size1 - 1;
        uintptr_t hi2 = lo2 + size2 - 1;
        if ((lo1 <= hi2) && (lo2 <= hi1)) {
            ishmemx_print(file, line, func,
                          "Attempting to call Intel® SHMEM API using overlapping arguments.\n",
                          ishmemx_print_msg_type_t::ERROR);
        }
    }
}

static inline void validate_stride(const char *file, long int line, const char *func,
                                   ptrdiff_t stride, size_t bsize)
{
    if (stride <= 0) {
        ishmemx_print(file, line, func,
                      "Attempting to call ISHMEM API with nonpositive stride value.\n",
                      ishmemx_print_msg_type_t::ERROR);
    }

    if (static_cast<size_t>(stride) < bsize) {
        ishmemx_print(file, line, func,
                      "Attempting to call ISHMEM API with stride value less than block size\n",
                      ishmemx_print_msg_type_t::ERROR);
    }
}

/* Input validation functions for public APIs */
/* init */
void validate_init_internal(const char *file, long int line, const char *func)
{
    validate_initialized(file, line, func);
}

/* Ptr */
void validate_parameters_internal(const char *file, long int line, const char *func, int pe)
{
    validate_initialized(file, line, func);
    validate_pe(file, line, func, pe);
}

/* AMOs/P/G */
void validate_parameters_internal(const char *file, long int line, const char *func, int pe,
                                  void *ptr, size_t size)
{
    validate_initialized(file, line, func);
    validate_pe(file, line, func, pe);
    validate_object_is_symmetric(file, line, func, ptr, size);
}

/* RMAs */
void validate_parameters_internal(const char *file, long int line, const char *func, int pe,
                                  void *symmetric_ptr, void *local_ptr, size_t size)
{
    validate_initialized(file, line, func);
    validate_pe(file, line, func, pe);
    validate_object_is_symmetric(file, line, func, symmetric_ptr, size);
    validate_objects_dont_overlap(file, line, func, pe, symmetric_ptr, local_ptr, size, size);
}

/* Strided RMAs */
void validate_parameters_internal(const char *file, long int line, const char *func, int pe,
                                  void *symmetric_ptr, void *local_ptr, size_t size, ptrdiff_t dst,
                                  ptrdiff_t sst, size_t bsize)
{
    validate_initialized(file, line, func);
    validate_pe(file, line, func, pe);
    validate_object_is_symmetric(file, line, func, symmetric_ptr, size);
    validate_objects_dont_overlap(file, line, func, pe, symmetric_ptr, local_ptr, size, size);
    validate_stride(file, line, func, dst, bsize);
    validate_stride(file, line, func, sst, bsize);
}

/* Broadcast */
void validate_parameters_internal(const char *file, long int line, const char *func, int pe_root,
                                  void *dest, void *src, size_t dest_size, size_t src_size)
{
    validate_initialized(file, line, func);
    validate_pe(file, line, func, pe_root);
    validate_object_is_symmetric(file, line, func, dest, dest_size);
    validate_object_is_symmetric(file, line, func, src, src_size);
    validate_objects_dont_overlap(file, line, func, dest, src, dest_size, src_size);
}

/* Signaling Operations */
void validate_parameters_internal(const char *file, long int line, const char *func, int pe,
                                  void *symmetric_ptr, void *local_ptr, void *sig_addr, size_t size,
                                  size_t sig_addr_size)
{
    validate_initialized(file, line, func);
    validate_pe(file, line, func, pe);
    validate_object_is_symmetric(file, line, func, symmetric_ptr, size);
    validate_object_is_symmetric(file, line, func, sig_addr, sig_addr_size);
    validate_objects_dont_overlap(file, line, func, symmetric_ptr, local_ptr, size, size);
    validate_objects_dont_overlap(file, line, func, symmetric_ptr, sig_addr, size, sig_addr_size);
    validate_objects_dont_overlap(file, line, func, local_ptr, sig_addr, size, sig_addr_size);
}

/* Test/Wait */
void validate_parameters_internal(const char *file, long int line, const char *func, void *ivar,
                                  size_t size)
{
    validate_initialized(file, line, func);
    validate_object_is_symmetric(file, line, func, ivar, size);
}

/* Reduce */
void validate_parameters_internal(const char *file, long int line, const char *func, void *dest,
                                  void *src, size_t size)
{
    validate_initialized(file, line, func);
    validate_object_is_symmetric(file, line, func, dest, size);
    validate_object_is_symmetric(file, line, func, src, size);
    if (dest != src) {
        validate_objects_dont_overlap(file, line, func, dest, src, size, size);
    }
}

/* Test/Wait (any, all), Alltoall, [F]Collect */
void validate_parameters_internal(const char *file, long int line, const char *func, void *ptrA,
                                  void *ptrB, size_t sizeA, size_t sizeB, ishmemi_op_t type)
{
    validate_initialized(file, line, func);
    if ((type == ishmemi_op_t::ALLTOALL) || (type == ishmemi_op_t::FCOLLECT) ||
        (type == ishmemi_op_t::COLLECT)) {
        void *dest = ptrA;
        void *src = ptrB;
        size_t dest_size = sizeA;
        size_t src_size = sizeB;

        validate_object_is_symmetric(file, line, func, dest, dest_size);
        validate_object_is_symmetric(file, line, func, src, src_size);
        validate_objects_dont_overlap(file, line, func, dest, src, dest_size, src_size);
    } else /* Pt-2-pt sync. */ {
        void *ivars = ptrA;
        void *status = ptrB;
        size_t ivars_size = sizeA;
        size_t status_size = sizeB;

        validate_object_is_symmetric(file, line, func, ivars, ivars_size);
        validate_objects_dont_overlap(file, line, func, ivars, status, ivars_size, status_size);
    }
}

/* Test/Wait (some) */
void validate_parameters_internal(const char *file, long int line, const char *func, void *ivars,
                                  void *indices, void *status, size_t ivars_size,
                                  size_t indices_size, size_t status_size)
{
    validate_initialized(file, line, func);
    validate_object_is_symmetric(file, line, func, ivars, ivars_size);
    validate_objects_dont_overlap(file, line, func, ivars, indices, ivars_size, indices_size);
    validate_objects_dont_overlap(file, line, func, ivars, status, ivars_size, status_size);
    validate_objects_dont_overlap(file, line, func, indices, status, indices_size, status_size);
}

/* stack trace for RAISE_ERROR_MSG */

void ishmemi_print_trace()
{
    int maxsize = (int) ishmemi_params.STACK_PRINT_LIMIT;
    if (maxsize < 10) maxsize = 10;
    if (maxsize > 50) maxsize = 50;
    void *array[50];
    int size;
    char **strings;
    int i;
    size = backtrace(array, maxsize);
    strings = backtrace_symbols(array, size);
    if (strings == nullptr) return;
    printf("Obtained %d stack frames.\n", size);
    for (i = 0; i < size; i++)
        printf("%s\n", strings[i]);
    free(strings);
}
