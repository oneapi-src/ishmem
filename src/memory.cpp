/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/err.h"
#include "memory.h"
#include "ishmem/env_utils.h"
#include "accelerator.h"
#include "runtime.h"

namespace {
    /* Private immediate command list for copying data */
    ze_command_list_handle_t copy_list = {};

    /* IPC handles for mmap regions */
    ze_ipc_mem_handle_t heap_handle = {};
    ze_ipc_mem_handle_t info_handle = {};

    /* Heap vars */
    mspace ishmemi_mspace;
    char *heap_curr = nullptr;
}  // namespace

/* Heap var */
void *ishmemi_heap_base = nullptr;
void *ishmemi_mmap_heap_base = nullptr;
size_t ishmemi_heap_length = 0;
uintptr_t ishmemi_heap_last = 0;

/* Info object vars */
size_t ishmemi_info_size = 0;
ishmemi_info_t *ishmemi_gpu_info = nullptr;
ishmemi_info_t *ishmemi_mmap_gpu_info = nullptr;

int ishmemi_memory_init()
{
    int ret = 0;

    ISHMEM_DEBUG_MSG("Symmetric heap size %ld\n", ishmemi_params.SYMMETRIC_SIZE);
    ishmemi_heap_length = ishmemi_params.SYMMETRIC_SIZE + ISHMEMI_HEAP_OVERHEAD;

    /* Check for overflow */
    ISHMEM_CHECK_GOTO_MSG(
        ishmemi_heap_length < ishmemi_params.SYMMETRIC_SIZE, fn_fail,
        "Adding symmetric heap overhead to requested ISHMEM_SYMMETRIC_SIZE (%zu) caused overflow\n",
        ishmemi_params.SYMMETRIC_SIZE);

    /* Allocate symmetric heap, and create mmap access to it */
    if (ishmemi_params.ENABLE_ACCESSIBLE_HOST_HEAP) {
        /* Host memory alloc */
        ret = ishmemi_usm_alloc_host(&ishmemi_heap_base, ishmemi_heap_length);
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
        ISHMEM_CHECK_GOTO_MSG(ishmemi_heap_base == nullptr, fn_fail,
                              "Unable to allocate ishmemi_heap_base\n");
    } else {
        /* Device memory alloc */
        ret = ishmemi_usm_alloc_device(&ishmemi_heap_base, ishmemi_heap_length);
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
        ISHMEM_CHECK_GOTO_MSG(ishmemi_heap_base == nullptr, fn_fail,
                              "Unable to allocate ishmemi_heap_base\n");

        /* The idea is for the host to peek and poke the symmetric heap.  Possibly useful for
         * host initiated operations or for debugging */
        ishmemi_mmap_heap_base =
            ishmemi_get_mmap_address(ishmemi_heap_base, ishmemi_heap_length, &heap_handle);
        ISHMEM_CHECK_GOTO_MSG(ishmemi_mmap_heap_base == nullptr, fn_fail,
                              "Unable to mmap GPU symmetric heap\n");

        ::memset(ishmemi_mmap_heap_base, 0, ishmemi_heap_length);
        ishmemi_heap_last = (uintptr_t) pointer_offset(ishmemi_heap_base, ishmemi_heap_length - 1);
    }

    /* This initializes the sbrk style symmtric heap allocator */
    heap_curr = (char *) ishmemi_heap_base;

    if (ishmemi_params.DEBUG) {
        ze_device_handle_t temp_gpu_device;
        ze_memory_allocation_properties_t mem_properties = {
            .stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
            .pNext = nullptr,
            .type = ZE_MEMORY_TYPE_UNKNOWN,
            .id = 0,
            .pageSize = 0,
        };

        ZE_CHECK(zeMemGetAllocProperties(ishmemi_ze_context, ishmemi_heap_base, &mem_properties,
                                         &temp_gpu_device));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

        ISHMEM_DEBUG_MSG("Heap allocation type: %s\n",
                         (mem_properties.type == ZE_MEMORY_TYPE_SHARED ? "shared" : "device"));
    }

    /* Allocate info structure */
    ishmemi_info_size = sizeof(ishmemi_info_t) + static_cast<size_t>(ishmemi_n_pes);
    ret = ishmemi_usm_alloc_device((void **) &ishmemi_gpu_info, ishmemi_info_size);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    /* SYCL queue to initialize global_info */
    try {
        sycl::queue q;
        q.copy(&ishmemi_gpu_info, global_info).wait_and_throw();
    } catch (...) {
        ret = -1;
    }

    /* host access for device data */
    ishmemi_mmap_gpu_info =
        ishmemi_get_mmap_address(ishmemi_gpu_info, ishmemi_info_size, &info_handle);
    ISHMEM_CHECK_GOTO_MSG(ishmemi_mmap_gpu_info == nullptr, fn_fail,
                          "Unable to mmap GPU info object\n");

    ::memset(ishmemi_mmap_gpu_info, 0, ishmemi_info_size);

#ifdef ENABLE_DLMALLOC
    if (ishmemi_params.ENABLE_ACCESSIBLE_HOST_HEAP) {
        ishmemi_mspace = create_mspace_with_base(ishmemi_heap_base, ishmemi_heap_length, 0);
    } else {
        ishmemi_mspace = create_mspace_with_base(ishmemi_mmap_heap_base, ishmemi_heap_length, 0);
    }
#endif

    /* create an immediate command list for use in ishmem_copy */
    ret = ishmemi_create_command_list(COPY_QUEUE, true, &copy_list);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

fn_exit:
    return ret;
fn_fail:
    if (!ret) ret = 1;
    goto fn_exit;
}

int ishmemi_memory_fini()
{
    int ret = 0;

    if (!ishmemi_params.ENABLE_ACCESSIBLE_HOST_HEAP) {
        if (ishmemi_mmap_heap_base != nullptr) {
            ret = ishmemi_close_mmap_address(heap_handle, ishmemi_mmap_heap_base,
                                             ishmemi_heap_length);
            ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
        }

        ishmemi_mmap_heap_base = nullptr;
    }

    ret = ishmemi_usm_free(ishmemi_heap_base);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ishmemi_heap_base = nullptr;
    ishmemi_heap_length = 0;

    ret = ishmemi_close_mmap_address(info_handle, ishmemi_mmap_gpu_info, ishmemi_info_size);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListDestroy(copy_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ret = ishmemi_usm_free(ishmemi_gpu_info);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ishmemi_gpu_info = nullptr;
    ishmemi_info_size = 0;

fn_exit:
    return ret;
}

void *ishmemi_get_next(size_t incr, size_t alignment)
{
    size_t space;
    void *old = nullptr;
    void *ret = nullptr;
    ptrdiff_t used = 0;

    /* Override user input to ensure minimal alignment */
    alignment = ISHMEMI_ALLOC_ALIGN > alignment ? ISHMEMI_ALLOC_ALIGN : alignment;

    /* `used` will be negative if curr < base */
    used = ((intptr_t) heap_curr) - ((intptr_t) ishmemi_heap_base);
    ISHMEM_CHECK_GOTO_MSG(((used < 0) || (used >= (ptrdiff_t) ishmemi_heap_length)), fn_fail,
                          "Unable to allocate %zu bytes in symmetric heap\n", incr);

    space = ishmemi_heap_length - static_cast<size_t>(used); /* Guaranteed to be positive */
    old = heap_curr;
    ret = (char *) std::align(alignment, incr, old, space);
    ISHMEM_CHECK_GOTO_MSG(ret == nullptr, fn_exit,
                          "Unable to allocate %zu bytes in symmetric heap\n", incr);

    heap_curr = ((char *) old) + incr;

fn_exit:
    return ret;
fn_fail:
    ret = nullptr;
    goto fn_exit;
}

void *ishmemi_alloc(size_t size, size_t alignment)
{
    void *host_ret = nullptr;
    void *ret = nullptr;

    if (size == 0) {
        goto fn_fail;
    }

    ISHMEM_CHECK_GOTO_MSG((alignment == 0 || (alignment & (alignment - 1)) != 0), fn_fail,
                          "Alignment must be a power of 2\n");

    if (ishmemi_params.ENABLE_ACCESSIBLE_HOST_HEAP) {
#ifdef ENABLE_DLMALLOC
        ret = mspace_memalign(ishmemi_mspace, alignment, size);
        ISHMEM_CHECK_GOTO_MSG(ret == nullptr, fn_fail,
                              "Unable to allocate %zu bytes in symmetric heap\n", size);
#else
        ISHMEM_CHECK_GOTO_MSG(ret == nullptr, fn_fail,
                              "Host-accessibly heap requires dlmalloc to be enabled\n", size);
#endif
    } else {
#ifdef ENABLE_DLMALLOC
        host_ret = mspace_memalign(ishmemi_mspace, alignment, size);
        ISHMEM_CHECK_GOTO_MSG(host_ret == nullptr, fn_fail,
                              "Unable to allocate %zu bytes in symmetric heap\n", size);

        ret = (void *) (((uintptr_t) host_ret - (uintptr_t) ishmemi_mmap_heap_base) +
                        (uintptr_t) ishmemi_heap_base);
#else
        ret = ishmemi_get_next(size);
#endif
    }

    ishmemi_runtime->barrier_all();

fn_exit:
    return ret;
fn_fail:
    ret = nullptr;
    goto fn_exit;
}

void *ishmem_malloc(size_t size)
{
    if constexpr (enable_error_checking) validate_init();
    return ishmemi_alloc(size);
}

void *ishmem_align(size_t alignment, size_t size)
{
    if constexpr (enable_error_checking) validate_init();
    return ishmemi_alloc(size, alignment);
}

void *ishmem_calloc(size_t count, size_t size)
{
    if constexpr (enable_error_checking) validate_init();
    return ishmemi_calloc(count, size);
}

void *ishmemi_calloc(size_t count, size_t size)
{
    void *ptr = nullptr;

    ptr = ishmemi_alloc(count * size);
    ISHMEM_CHECK_GOTO_MSG(ptr == nullptr, fn_fail, "Failed to allocate object\n");

    ptr = ishmemi_zero(ptr, count * size);
    ISHMEM_CHECK_GOTO_MSG(ptr == nullptr, fn_fail, "Failed to zero allocation\n");

fn_exit:
    return ptr;
fn_fail:
    ptr = nullptr;
    goto fn_exit;
}

void ishmem_free(void *ptr)
{
    if constexpr (enable_error_checking) validate_init();
    ishmemi_free(ptr);
}

void ishmemi_free(void *ptr)
{
    ishmemi_runtime->barrier_all();
    if (ishmemi_params.ENABLE_ACCESSIBLE_HOST_HEAP) {
        if (ptr != nullptr) {
#ifdef ENABLE_DLMALLOC
            mspace_free(ishmemi_mspace, ptr);
#endif
        }
    } else {
        if (ptr != nullptr) {
#ifdef ENABLE_DLMALLOC
            void *host_ptr = (void *) (((uintptr_t) ptr - (uintptr_t) ishmemi_heap_base) +
                                       (uintptr_t) ishmemi_mmap_heap_base);
            mspace_free(ishmemi_mspace, host_ptr);
#endif
        }
    }
}

void *ishmem_copy(void *dest, const void *src, size_t size)
{
    if constexpr (enable_error_checking) validate_init();
    return ishmemi_copy(dest, src, size);
}

void *ishmemi_copy(void *dest, const void *src, size_t size)
{
    int ret = 0;

    ZE_CHECK(zeCommandListAppendMemoryCopy(copy_list, dest, src, size, nullptr, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    return dest;

fn_fail:
    return nullptr;
}

void *ishmem_zero(void *dest, size_t size)
{
    if constexpr (enable_error_checking) validate_init();
    return ishmemi_zero(dest, size);
}

void *ishmemi_zero(void *dest, size_t size)
{
    int ret = 0;
    uint32_t zero = 0;

    ZE_CHECK(zeCommandListAppendMemoryFill(copy_list, dest, &zero, 1, size, nullptr, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    return dest;

fn_fail:
    return nullptr;
}

void *ishmem_ptr(const void *dest, int pe)
{
    if constexpr (enable_error_checking) validate_parameters(pe);
    return ishmemi_ptr(dest, pe);
}

void *ishmemi_ptr(const void *dest, int pe)
{
    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
    if (local_index != 0) {
        return ISHMEMI_ADJUST_PTR(void, local_index, dest);
    } else {
        return nullptr;
    }
}
