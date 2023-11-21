/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "internal.h"
#include "impl_proxy.h"

void validate_pe(int pe)
{
    if ((pe < 0) || (pe >= ishmem_n_pes())) {
        ishmemx_print("Attempting to call ISHMEM API using invalid PE value\n",
                      ishmemx_print_msg_type_t::ERROR);
    }
}

void validate_object_on_symmetric_heap(void *ptr, size_t size)
{
#ifdef __SYCL_DEVICE_ONLY__
    void *heap_base = global_info->heap_base;
    size_t heap_length = global_info->heap_length;
#else
    void *heap_base = ishmemi_heap_base;
    size_t heap_length = ishmemi_heap_length;
#endif
    if (size == 0) return;
    uintptr_t loptr = (uintptr_t) ptr;
    uintptr_t hiptr = loptr + size - 1;
    uintptr_t loheap = (uintptr_t) heap_base;
    uintptr_t hiheap = loheap + heap_length - 1;
    if ((loptr < loheap) || (hiptr > hiheap))
        ishmemx_print("Attempting to call ISHMEM API using object not on symmetric heap\n",
                      ishmemx_print_msg_type_t::ERROR);
}

void validate_objects_dont_overlap(void *ptr1, void *ptr2, size_t size1, size_t size2)
{
    if ((size1 == 0) || (size2 == 0)) return;
    uintptr_t lo1 = (uintptr_t) ptr1;
    uintptr_t lo2 = (uintptr_t) ptr2;
    uintptr_t hi1 = lo1 + size1 - 1;
    uintptr_t hi2 = lo2 + size2 - 1;
    if ((lo1 <= hi2) && (lo2 <= hi1)) {
        ishmemx_print("Attempting to call ISHMEM API using overlapping arguments.\n",
                      ishmemx_print_msg_type_t::ERROR);
    }
}

void validate_stride(ptrdiff_t stride)
{
    if (stride <= 0) {
        ishmemx_print("Attempting to call ISHMEM API with nonpositive stride value.\n",
                      ishmemx_print_msg_type_t::ERROR);
    }
}

/* Ptr */
void validate_parameters(int pe)
{
    validate_pe(pe);
}

/* AMOs/P/G */
void validate_parameters(int pe, void *ptr, size_t size)
{
    validate_pe(pe);
    validate_object_on_symmetric_heap(ptr, size);
}

/* RMAs */
void validate_parameters(int pe, void *ptr1, void *ptr2, size_t size)
{
    validate_pe(pe);
    validate_object_on_symmetric_heap(ptr1, size);
    validate_objects_dont_overlap(ptr1, ptr2, size, size);
}

/* Strided RMAs */
void validate_parameters(int pe, void *ptr1, void *ptr2, size_t size, ptrdiff_t dst, ptrdiff_t sst)
{
    validate_pe(pe);
    validate_object_on_symmetric_heap(ptr1, size);
    validate_objects_dont_overlap(ptr1, ptr2, size, size);
    validate_stride(dst);
    validate_stride(sst);
}

/* Broadcast */
void validate_parameters(int pe_root, void *dest, void *src, size_t dest_size, size_t src_size)
{
    validate_pe(pe_root);
    validate_object_on_symmetric_heap(dest, dest_size);
    validate_object_on_symmetric_heap(src, src_size);
    validate_objects_dont_overlap(dest, src, dest_size, src_size);
}

/* Signaling Operations */
void validate_parameters(int pe, void *ptr1, void *ptr2, void *sig_addr, size_t size,
                         size_t sig_addr_size)
{
    validate_pe(pe);
    validate_object_on_symmetric_heap(ptr1, size);
    validate_object_on_symmetric_heap(sig_addr, sig_addr_size);
    validate_objects_dont_overlap(ptr1, ptr2, size, size);
    validate_objects_dont_overlap(ptr1, sig_addr, size, sig_addr_size);
    validate_objects_dont_overlap(ptr2, sig_addr, size, sig_addr_size);
}

/* Test/Wait */
void validate_parameters(void *ivar, size_t size)
{
    validate_object_on_symmetric_heap(ivar, size);
}

/* Reduce */
void validate_parameters(void *dest, void *src, size_t size)
{
    validate_object_on_symmetric_heap(dest, size);
    validate_object_on_symmetric_heap(src, size);
    if (dest != src) {
        validate_objects_dont_overlap(dest, src, size, size);
    }
}

/* Alltoall, Fcollect */
void validate_parameters(void *dest, void *src, size_t dest_size, size_t src_size)
{
    validate_object_on_symmetric_heap(dest, dest_size);
    validate_object_on_symmetric_heap(src, src_size);
    validate_objects_dont_overlap(dest, src, dest_size, src_size);
}
