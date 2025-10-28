/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/env_utils.h"
#include "accelerator.h"
#include <level_zero/ze_api.h>
#include <atomic>

/* TODO: Workaround to resolve compiler limitation. Need to be fixed later */
#if __INTEL_CLANG_COMPILER <= 20210400
#include <CL/sycl/backend/level_zero.hpp>
#else
#include <ext/oneapi/backend/level_zero.hpp>
#endif

namespace {
    /* L0 driver */
    ze_driver_handle_t *all_drivers = nullptr;
    ze_device_handle_t **all_devices = nullptr;
    uint32_t driver_count = 0;
    uint32_t driver_idx = 0;
    bool driver_found = false;

    /* L0 device */
    ze_device_properties_t device_properties = {};

    /* L0 queues */
    uint32_t link_queue_count = 0;
    std::atomic<uint64_t> link_index = 0; /* Used for round-robining link engines */
    ze_command_queue_handle_t compute_queue = {};
    ze_command_queue_handle_t copy_queue = {};
    ze_command_queue_handle_t *link_queues = nullptr;
    uint32_t compute_ordinal = 0;
    uint32_t copy_ordinal = 0;
    uint32_t link_ordinal = 0;

    /* L0 lists */
    ishmemi_thread_safe_vector<ze_command_list_handle_t> compute_lists;
    ishmemi_thread_safe_vector<ze_command_list_handle_t> copy_lists;
    ishmemi_thread_safe_vector<ze_command_list_handle_t> *link_lists;

    /* Misc */
    bool ishmemi_accelerator_preinitialized = false;
    bool ishmemi_accelerator_initialized = false;
}  // namespace

/* L0 Context */
ze_context_handle_t ishmemi_ze_context = nullptr;
ze_context_desc_t ishmemi_ze_context_desc = {};

/* L0 device */
ze_driver_handle_t ishmemi_gpu_driver = nullptr;
ze_device_handle_t ishmemi_gpu_device = nullptr;

/* L0 events */
ze_event_pool_handle_t ishmemi_ze_event_pool;

/* this should be thread safe because we query the size, then sync
 * then destroy the first size items, then erase them from the list
 */
static int sync_cq(ze_command_queue_handle_t &queue,
                   ishmemi_thread_safe_vector<ze_command_list_handle_t> &cmd_lists)
{
    static std::atomic<size_t> size;
    size_t cur_size = 0;
    int ret = 0;
    std::vector<ze_command_list_handle_t>::iterator first, last;

    cmd_lists.mtx.lock();
    size.store(cmd_lists.size());
    cmd_lists.mtx.unlock();

    ZE_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    cmd_lists.mtx.lock();
    cur_size = size.load();
    for (size_t i = 0; i < cur_size; ++i) {
        ZE_CHECK(zeCommandListDestroy(cmd_lists[i]));
    }

    first = cmd_lists.begin();
    last = first + static_cast<long>(cur_size);
    cmd_lists.erase(first, last);
    cmd_lists.mtx.unlock();

fn_exit:
    return ret;
}

static inline uint32_t get_next_link_index()
{
    uint32_t index = link_index.fetch_add(1, std::memory_order_relaxed) % link_queue_count;
    return index;
}

int ishmemi_accelerator_preinit()
{
    int ret = 0;
    uint32_t i;
    uint32_t device_count = 0;
    ze_init_flag_t flags = ZE_INIT_FLAG_GPU_ONLY;

    if (ishmemi_accelerator_preinitialized) {
        goto fn_exit;
    }

    /* Initialize ZE */
    ret = zeInit(flags);
    if (ret == ZE_RESULT_ERROR_UNINITIALIZED) {
        ret = ISHMEMI_NO_DEVICE_ACCESS;
        goto fn_exit;
    }
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    /* Create the ZE driver */
    ZE_CHECK(zeDriverGet(&driver_count, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    if (driver_count == 0) {
        ISHMEM_ERROR_MSG("No ZE driver found (driver count: %d).\n", driver_count);
        ret = ISHMEMI_NO_DRIVERS;
        goto fn_fail;
    }

    /* Allocate device handle buffers, but don't make any PE assignments in preinit */
    all_drivers = (ze_driver_handle_t *) ::calloc(driver_count, sizeof(ze_driver_handle_t));
    ISHMEM_CHECK_GOTO_MSG(all_drivers == nullptr, fn_fail, "Allocation of all_drivers failed\n");

    all_devices = (ze_device_handle_t **) ::calloc(driver_count, sizeof(ze_device_handle_t *));
    ISHMEM_CHECK_GOTO_MSG(all_devices == nullptr, fn_fail, "Allocation of all_devices failed\n");

    ZE_CHECK(zeDriverGet(&driver_count, all_drivers));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    /* Parse the drivers for a suitable driver */
    for (i = 0; i < driver_count; i++) {
        device_count = 0;
        ZE_CHECK(zeDeviceGet(all_drivers[i], &device_count, nullptr));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
        if (device_count == 0) continue;

        /* Ensure a single device is detected */
        ISHMEM_CHECK_GOTO_MSG(device_count != 1, fn_fail, "Detected more than one device\n");
        all_devices[i] = (ze_device_handle_t *) ::malloc(device_count * sizeof(ze_device_handle_t));
        ISHMEM_CHECK_GOTO_MSG(all_devices == nullptr, fn_fail,
                              "Allocation of all_drivers[%d] failed\n", i);

        ZE_CHECK(zeDeviceGet(all_drivers[i], &device_count, all_devices[i]));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

        ZE_CHECK(zeDeviceGetProperties(all_devices[i][0], &device_properties));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

        if (ZE_DEVICE_TYPE_GPU == device_properties.type && !driver_found) {
            ishmemi_gpu_driver = all_drivers[i];
            driver_idx = i;
            driver_found = true;
        }
    }

    if (!driver_found) {
        ISHMEM_ERROR_MSG("No ZE driver found for GPU\n");
        ret = ISHMEMI_NO_DEVICES;
        goto fn_fail;
    }

    /* Create the ZE context */
    ishmemi_ze_context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;

    ZE_CHECK(zeContextCreate(ishmemi_gpu_driver, &ishmemi_ze_context_desc, &ishmemi_ze_context));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

fn_exit:
    ishmemi_accelerator_preinitialized = true;
    return ret;
fn_fail:
    ishmemi_accelerator_fini();
    if (!ret) ret = 1;
    goto fn_exit;
}

int ishmemi_accelerator_init()
{
    int ret = 0;
    uint32_t i, j;
    uint32_t cq_group_count = 0;
    ze_event_pool_desc_t event_pool_desc;
    ze_command_queue_group_properties_t *cq_group_prop = nullptr;

    ret = ishmemi_accelerator_preinit();
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    if (driver_found) {
        /* Set the default GPU */
        ishmemi_gpu_device = all_devices[driver_idx][0];

        /* Discover command queue groups */
        ZE_CHECK(
            zeDeviceGetCommandQueueGroupProperties(ishmemi_gpu_device, &cq_group_count, nullptr));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

        cq_group_prop = (ze_command_queue_group_properties_t *) ::malloc(
            cq_group_count * sizeof(ze_command_queue_group_properties_t));
        ISHMEM_CHECK_GOTO_MSG(cq_group_prop == nullptr, fn_fail,
                              "Allocation of cq_group_prop failed\n");

        for (i = 0; i < cq_group_count; ++i) {
            cq_group_prop[i] = {
                .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES,
                .pNext = nullptr,
                .flags = 0,
                .maxMemoryFillPatternSize = 0,
                .numQueues = 0,
            };
        }

        ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(ishmemi_gpu_device, &cq_group_count,
                                                        cq_group_prop));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

        /* Setup all command queues */
        for (i = 0; i < cq_group_count; ++i) {
            ze_command_queue_desc_t desc = {
                .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                .pNext = nullptr,
                .ordinal = i,
                .index = 0,
                .flags = 0,
                .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
            };

            if (cq_group_prop[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
                ZE_CHECK(zeCommandQueueCreate(ishmemi_ze_context, ishmemi_gpu_device, &desc,
                                              &compute_queue));
                ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

                compute_ordinal = i;
            } else if (cq_group_prop[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY &&
                       cq_group_prop[i].numQueues == 1) {
                ZE_CHECK(zeCommandQueueCreate(ishmemi_ze_context, ishmemi_gpu_device, &desc,
                                              &copy_queue));
                ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

                copy_ordinal = i;
            } else if (cq_group_prop[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY &&
                       cq_group_prop[i].numQueues > 1) {
                link_queues = (ze_command_queue_handle_t *) ::malloc(
                    cq_group_prop[i].numQueues * sizeof(ze_command_queue_handle_t));
                ISHMEM_CHECK_GOTO_MSG(link_queues == nullptr, fn_fail,
                                      "Allocation of link_queues failed\n");

                for (j = 0; j < cq_group_prop[i].numQueues; ++j) {
                    desc.index = j;
                    ZE_CHECK(zeCommandQueueCreate(ishmemi_ze_context, ishmemi_gpu_device, &desc,
                                                  &link_queues[j]));
                    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
                }

                link_ordinal = i;
            }
        }

        if (ishmemi_params.DEBUG) {
            ishmemi_print_device_properties(device_properties);
        }
    }

    /* Set the default interval for garbage collection for lists */
    compute_lists.reserve(ishmemi_params.NBI_COUNT);
    copy_lists.reserve(ishmemi_params.NBI_COUNT);
    link_lists = (ishmemi_thread_safe_vector<ze_command_list_handle_t> *) ::malloc(
        link_queue_count * sizeof(ishmemi_thread_safe_vector<ze_command_list_handle_t>));
    for (i = 0; i < link_queue_count; ++i) {
        link_lists[i].reserve(ishmemi_params.NBI_COUNT);
    }

    /* Create the ZE event pool */
    event_pool_desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        .pNext = nullptr,
        .flags = 0,
        .count = 1,
    };
    ZE_CHECK(zeEventPoolCreate(ishmemi_ze_context, &event_pool_desc, 0, nullptr,
                               &ishmemi_ze_event_pool));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

fn_exit:
    ishmemi_accelerator_initialized = true;
    return ret;
fn_fail:
    ishmemi_accelerator_fini();
    goto fn_exit;
}

int ishmemi_accelerator_fini(void)
{
    int ret = 0;

    if (compute_queue) {
        sync_cq(compute_queue, compute_lists);
        ZE_CHECK(zeCommandQueueDestroy(compute_queue));
        compute_queue = {};
    }

    if (copy_queue) {
        sync_cq(copy_queue, copy_lists);
        ZE_CHECK(zeCommandQueueDestroy(copy_queue));
        copy_queue = {};
    }

    for (uint32_t i = 0; i < link_queue_count; ++i) {
        if (link_queues[i]) {
            sync_cq(link_queues[i], link_lists[i]);
            ZE_CHECK(zeCommandQueueDestroy(link_queues[i]));
            link_queues[i] = {};
        }
    }
    ISHMEMI_FREE(::free, link_queues);
    link_queues = nullptr;

    for (size_t i = 0; i < driver_count; i++)
        ISHMEMI_FREE(::free, all_devices[i]);
    ISHMEMI_FREE(::free, all_devices);
    ISHMEMI_FREE(::free, all_drivers);

    ishmemi_accelerator_preinitialized = false;
    ishmemi_accelerator_initialized = false;
    driver_found = false;
    driver_idx = 0;
    driver_count = 0;

    if (ishmemi_ze_context) {
        ZE_CHECK(zeContextDestroy(ishmemi_ze_context));
        ishmemi_ze_context = nullptr;
    }

    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

fn_exit:
    return ret;
}

int ishmemi_create_command_list(ishmemi_queue_type_t queue_type, bool immediate,
                                ze_command_list_handle_t *list, ze_command_list_flags_t flags)
{
    int ret = 0;
    uint32_t ordinal = 0;
    uint32_t index = 0;
    ze_command_list_desc_t list_desc = {};
    ze_command_queue_desc_t queue_desc = {};

    ISHMEM_CHECK_GOTO_MSG(list == nullptr, fn_fail,
                          "Failed to create command list - nullptr provided\n");

    switch (queue_type) {
        case COMPUTE_QUEUE:
            ordinal = compute_ordinal;
            break;
        case COPY_QUEUE:
            ordinal = copy_ordinal;
            break;
        case LINK_QUEUE:
            if (link_queue_count == 0) {
                ordinal = copy_ordinal;
            } else {
                if (link_queue_count > 1) {
                    index = get_next_link_index();
                }
                ordinal = link_ordinal;
            }
            break;
        default:
            ISHMEM_CHECK_GOTO_MSG(
                true, fn_fail, "Failed to create command list - undefined queue type provided\n");
            break;
    }

    if (immediate) {
        /* Currently only use synchronous and normal priority - may need to extend later */
        queue_desc = {
            .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
            .pNext = nullptr,
            .ordinal = ordinal,
            .index = index,
            .flags = 0,
            .mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
            .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
        };

        ZE_CHECK(zeCommandListCreateImmediate(ishmemi_ze_context, ishmemi_gpu_device, &queue_desc,
                                              list));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
    } else {
        list_desc = {
            .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
            .pNext = nullptr,
            .commandQueueGroupOrdinal = ordinal,
            .flags = flags,
        };

        ZE_CHECK(zeCommandListCreate(ishmemi_ze_context, ishmemi_gpu_device, &list_desc, list));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
    }

fn_exit:
    return ret;
fn_fail:
    ret = 1;
    goto fn_exit;
}

int ishmemi_create_command_list_nbi(ishmemi_queue_type_t queue_type, ze_command_list_handle_t *list,
                                    ze_command_list_flags_t flags)
{
    int ret = 0;
    uint32_t ordinal = 0;
    uint32_t index = 0;
    ze_command_list_desc_t list_desc = {};

    ISHMEM_CHECK_GOTO_MSG(list == nullptr, fn_fail,
                          "Failed to create command list - nullptr provided\n");

    switch (queue_type) {
        case COMPUTE_QUEUE:
            ordinal = compute_ordinal;
            break;
        case COPY_QUEUE:
            ordinal = copy_ordinal;
            break;
        case LINK_QUEUE:
            if (link_queue_count == 0) {
                ordinal = copy_ordinal;
            } else {
                if (link_queue_count > 1) {
                    index = get_next_link_index();
                }
                ordinal = link_ordinal;
            }
            break;
        default:
            ISHMEM_CHECK_GOTO_MSG(
                true, fn_fail, "Failed to create command list - undefined queue type provided\n");
            break;
    }

    list_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        .pNext = nullptr,
        .commandQueueGroupOrdinal = ordinal,
        .flags = flags,
    };

    ZE_CHECK(zeCommandListCreate(ishmemi_ze_context, ishmemi_gpu_device, &list_desc, list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    switch (queue_type) {
        case COMPUTE_QUEUE:
            compute_lists.push_back_thread_safe(*list);
            break;
        case COPY_QUEUE:
            copy_lists.push_back_thread_safe(*list);
            break;
        case LINK_QUEUE:
            if (link_queue_count == 0) {
                copy_lists.push_back_thread_safe(*list);
            } else {
                link_lists[index].push_back_thread_safe(*list);
            }
            break;
        default:
            ISHMEM_CHECK_GOTO_MSG(true, fn_fail,
                                  "Failed to store command list - undefined queue type provided\n");
            break;
    }

fn_exit:
    return ret;
fn_fail:
    ret = 1;
    goto fn_exit;
}

int ishmemi_execute_command_lists(ishmemi_queue_type_t queue_type, uint32_t list_count,
                                  ze_command_list_handle_t *lists, ze_fence_handle_t fence)
{
    int ret = 0;
    uint32_t index = 0;
    ze_command_queue_handle_t queue = {};

    ISHMEM_CHECK_GOTO_MSG(lists == nullptr, fn_fail,
                          "Failed to execute command list - nullptr provided\n");

    switch (queue_type) {
        case COMPUTE_QUEUE:
            queue = compute_queue;
            break;
        case COPY_QUEUE:
            queue = copy_queue;
            break;
        case LINK_QUEUE:
            if (link_queue_count == 0) {
                queue = copy_queue;
            } else {
                if (link_queue_count > 1) {
                    index = get_next_link_index();
                }
                queue = link_queues[index];
            }
            break;
        default:
            ISHMEM_CHECK_GOTO_MSG(
                true, fn_fail, "Failed to execute command list - undefined queue type provided\n");
            break;
    }

    ZE_CHECK(zeCommandQueueExecuteCommandLists(queue, list_count, lists, fence));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

fn_exit:
    return ret;
fn_fail:
    ret = 1;
    goto fn_exit;
}

int ishmemi_get_memory_type(const void *ptr, ze_memory_type_t *type)
{
    int ret = 0;
    ze_memory_allocation_properties_t mem_prop;
    ze_device_handle_t device;
    mem_prop.pNext = nullptr;

    ZE_CHECK(zeMemGetAllocProperties(ishmemi_ze_context, ptr, &mem_prop, &device));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    *type = mem_prop.type;

fn_exit:
    return ret;
}

void ishmemi_level_zero_sync()
{
    sync_cq(compute_queue, compute_lists);
    sync_cq(copy_queue, copy_lists);
    for (uint32_t i = 0; i < link_queue_count; ++i) {
        sync_cq(link_queues[i], link_lists[i]);
    }
}

int ishmemi_usm_alloc_host(void **ptr, size_t size)
{
    int ret = 0;
    ze_host_mem_alloc_desc_t host_mem_desc = {
        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
        .pNext = nullptr,
        .flags = 0,
    };

    ZE_CHECK(zeMemAllocHost(ishmemi_ze_context, &host_mem_desc, size, 64, ptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

fn_exit:
    return ret;
}

int ishmemi_usm_alloc_device(void **ptr, size_t size)
{
    int ret = 0;
    ze_device_mem_alloc_desc_t device_mem_desc = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        .pNext = nullptr,
        .flags = 0,
        .ordinal = 0 /* what should this be ? */
    };

    ZE_CHECK(
        zeMemAllocDevice(ishmemi_ze_context, &device_mem_desc, size, 64, ishmemi_gpu_device, ptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

fn_exit:
    return ret;
}

int ishmemi_usm_free(void *ptr)
{
    int ret = 0;

    ZE_CHECK(zeMemFree(ishmemi_ze_context, ptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

fn_exit:
    return ret;
}
