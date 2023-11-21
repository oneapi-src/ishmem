/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "memory.h"
#include "env_utils.h"
#include "accelerator.h"
#include "runtime.h"

static char *ishmemi_heap_curr = nullptr;  // why static ?

void *ishmemi_heap_base = nullptr;
size_t ishmemi_heap_length = 0;
uintptr_t ishmemi_heap_last = 0;
void *ishmemi_mmap_heap_base = nullptr;
ishmem_info_t *ishmemi_gpu_info = nullptr;

size_t ishmemi_info_size = 0;
ishmem_info_t *ishmemi_mmap_gpu_info = nullptr;

mspace ishmemi_mspace;
ze_command_queue_desc_t ishmem_copy_cmd_queue_desc = {
    .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
    .pNext = nullptr,
    .ordinal = 1,
    .index = 0,
    .flags = 0,
    .mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
    .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
};
ze_command_list_handle_t ishmem_copy_cmd_list = {};

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
    if (!ishmemi_params.ENABLE_ACCESSIBLE_HOST_HEAP) {
        /* Device memory alloc */
        ret = ishmemi_usm_alloc_device(&ishmemi_heap_base, ishmemi_heap_length);
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
        /* The idea is for the host to peek and poke the symmetric heap.  Possibly useful for
         * host initiated operations or for debugging
         */
        ishmemi_mmap_heap_base = ishmemi_get_mmap_address(ishmemi_heap_base, ishmemi_heap_length);
        if (ishmemi_mmap_heap_base == nullptr) {
            RAISE_ERROR_MSG("unable to mmap symmetric heap\n");
        }
        memset(ishmemi_mmap_heap_base, 0, ishmemi_heap_length);
        ishmemi_heap_last = (uintptr_t) pointer_offset(ishmemi_heap_base, ishmemi_heap_length - 1);
    } else {
        /* Shared memory alloc */
        ret = ishmemi_usm_alloc_host(&ishmemi_heap_base, ishmemi_heap_length);
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
    }

    /* this initialized the sbrk style symmtric heap allocator */
    ishmemi_heap_curr = (char *) ishmemi_heap_base;

    if (ishmemi_params.DEBUG) {
        ze_device_handle_t temp_gpu_device;
        ze_memory_allocation_properties_t mem_properties;
        mem_properties.pNext = nullptr;
        mem_properties.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
        ZE_CHECK(zeMemGetAllocProperties(ishmemi_ze_context, ishmemi_heap_base, &mem_properties,
                                         &temp_gpu_device));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
        ISHMEM_DEBUG_MSG("Heap allocation type: %s\n",
                         (mem_properties.type == ZE_MEMORY_TYPE_SHARED ? "shared" : "device"));
    }

    /* allocate info structure */
    ishmemi_info_size =
        sizeof(ishmem_info_t) + (static_cast<size_t>(ishmemi_n_pes) * sizeof(uint8_t));
    ret = ishmemi_usm_alloc_device((void **) &ishmemi_gpu_info, ishmemi_info_size);

    /* SYCL queue to initialize global_info */
    try {
        sycl::queue q;
        q.copy(&ishmemi_gpu_info, global_info).wait_and_throw();
    } catch (...) {
        ret = -1;
    }
    ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "ishmemi_alloc_usm_device failed '%d'\n", ret);

    /* host access for device data */
    ishmemi_mmap_gpu_info = ishmemi_get_mmap_address(ishmemi_gpu_info, ishmemi_info_size);
    if (ishmemi_mmap_gpu_info == nullptr) {
        RAISE_ERROR_MSG("unable to mmap gpu info structure\n");
    }
    memset(ishmemi_mmap_gpu_info, 0, ishmemi_info_size);

    if (ishmemi_params.ENABLE_ACCESSIBLE_HOST_HEAP) {
        ishmemi_mspace = create_mspace_with_base(ishmemi_heap_base, ishmemi_heap_length, 0);
    }
    ZE_CHECK(zeCommandListCreateImmediate(ishmemi_ze_context, ishmemi_gpu_device,
                                          &ishmem_copy_cmd_queue_desc, &ishmem_copy_cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    return (nullptr == ishmemi_heap_base) ? -1 : 0;

fn_fail:
    return -1;
}

int ishmemi_memory_fini()
{
    int ret = 0;
    if (!ishmemi_params.ENABLE_ACCESSIBLE_HOST_HEAP) {
        if (ishmemi_mmap_heap_base != nullptr) munmap(ishmemi_mmap_heap_base, ishmemi_heap_length);
        ishmemi_mmap_heap_base = nullptr;
    }
    if (ishmemi_heap_base != nullptr) {
        /* TODO, should ISHMEMI_FREE check the result of the call to ishmemi_usm_free?  And do what?
         */
        ISHMEMI_FREE(ishmemi_usm_free, ishmemi_heap_base);
        ishmemi_heap_length = 0;
    }
    ishmemi_heap_base = nullptr;

    ret = munmap((void *) ishmemi_mmap_gpu_info, ishmemi_info_size);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
    ZE_CHECK(zeCommandListDestroy(ishmem_copy_cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ret = ishmemi_usm_free(ishmemi_gpu_info);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

fn_exit:
    return ret;
}

void *ishmemi_get_next(size_t incr, size_t alignment)
{
    /* Override user input to ensure minimal alignment*/
    alignment = ISHMEMI_ALLOC_ALIGN > alignment ? ISHMEMI_ALLOC_ALIGN : alignment;

    /* this is ptrdiff_t so it will be negative if curr < base */
    ptrdiff_t used = ((intptr_t) ishmemi_heap_curr) - ((intptr_t) ishmemi_heap_base);
    if ((used < 0) || (used >= ishmemi_heap_length)) {
        RAISE_ERROR_MSG("Symmetric heap out of bounds\n");
        /* Not a recoverable error, since we don't know where to move heap_curr */
    }
    size_t space = ishmemi_heap_length -
                   static_cast<size_t>(used);  // this is guaranteed positive, so unsigned is ok
    /* std:: align will bump old_ptr by the alignment adjustment */
    void *old_ptr = ishmemi_heap_curr;
    void *result = (char *) std::align(alignment, incr, old_ptr, space);
    if (result == nullptr) {
        ISHMEM_WARN_MSG("Out of symmetric space\n");
    } else {
        ishmemi_heap_curr = ((char *) old_ptr) + incr;
    }
    return result;
}

void *ishmem_malloc(size_t size)
{
    void *ret;
    if (size == 0) return (nullptr);

    // TODO: internal barrier, dlmalloc, thread-safety
    if (!ishmemi_params.ENABLE_ACCESSIBLE_HOST_HEAP) {
        ret = ishmemi_get_next(size);
    } else {
        ret = mspace_memalign(ishmemi_mspace, ISHMEMI_ALLOC_ALIGN, size);
    }

    ishmemi_runtime_barrier();
    return ret;
}

void *ishmem_align(size_t alignment, size_t size)
{
    void *ret;

    if (size == 0) return nullptr;
    // Undefined behaviour if alignment is not a power of 2
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return nullptr;

    if (!ishmemi_params.ENABLE_ACCESSIBLE_HOST_HEAP) {
        ret = ishmemi_get_next(size, alignment);
    } else {
        ret = mspace_memalign(ishmemi_mspace, alignment, size);
    }

    ishmemi_runtime_barrier();
    return ret;
}

void *ishmem_calloc(size_t count, size_t size)
{
    int ret = 0;
    void *ptr;
    if (count == 0 || size == 0) return (nullptr);

    // TODO: internal barrier, dlmalloc, thread-safety
    if (!ishmemi_params.ENABLE_ACCESSIBLE_HOST_HEAP) {
        ptr = ishmemi_get_next(size * count);
    } else {
        ptr = mspace_memalign(ishmemi_mspace, ISHMEMI_ALLOC_ALIGN, count * size);
    }

    if (ptr != nullptr) {
        ze_command_queue_desc_t cmd_queue_desc = {.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                                  .pNext = nullptr,
                                                  .ordinal = 1,
                                                  .index = 0,
                                                  .flags = 0,
                                                  .mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                                  .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
        ze_command_list_handle_t cmd_list = {};
        uint32_t zero = 0;
        ZE_CHECK(zeCommandListCreateImmediate(ishmemi_ze_context, ishmemi_gpu_device,
                                              &cmd_queue_desc, &cmd_list));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
        ZE_CHECK(zeCommandListAppendMemoryFill(cmd_list, ptr, &zero, 1, size, nullptr, 0, nullptr));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
        ZE_CHECK(zeCommandListDestroy(cmd_list));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
    }

    ishmemi_runtime_barrier();

    return ptr;

fn_fail:
    return nullptr;
}

void ishmem_free(void *ptr)
{
    // TODO
    ishmemi_runtime_barrier();
    if (ishmemi_params.ENABLE_ACCESSIBLE_HOST_HEAP) {
        if (ptr != nullptr) {
            mspace_free(ishmemi_mspace, ptr);
        }
    }
}

void *ishmem_copy(void *dst, void *src, size_t size)
{
    int ret = 0;
    ze_memory_type_t dst_type, src_type;

    /* Check the pointer type for dst and src */
    ishmemi_get_memory_type(dst, &dst_type);
    ishmemi_get_memory_type(src, &src_type);

    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
    ZE_CHECK(
        zeCommandListAppendMemoryCopy(ishmem_copy_cmd_list, dst, src, size, nullptr, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    return dst;

fn_fail:
    return nullptr;
}
