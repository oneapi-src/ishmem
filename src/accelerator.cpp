/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "env_utils.h"
#include "accelerator.h"
#include <level_zero/ze_api.h>

/* TODO: Workaround to resolve compiler limitation. Need to be fixed later */
#if __INTEL_CLANG_COMPILER <= 20210400
#include <CL/sycl/backend/level_zero.hpp>
#else
#include <ext/oneapi/backend/level_zero.hpp>
#endif

ze_driver_handle_t *all_drivers = nullptr;
ze_device_handle_t **all_devices = nullptr;
uint32_t driver_count = 0;
bool ishmemi_accelerator_preinitialized = false;
bool ishmemi_accelerator_initialized = false;

ze_driver_handle_t ishmemi_gpu_driver = nullptr;
ze_driver_handle_t ishmemi_fpga_driver = nullptr;
ze_device_handle_t ishmemi_gpu_device = nullptr;
ze_device_handle_t ishmemi_fpga_device = nullptr;

ze_context_handle_t ishmemi_ze_context = nullptr;
ze_context_desc_t ishmemi_ze_context_desc = {};
ze_event_pool_handle_t ishmemi_ze_event_pool;

unsigned int ishmemi_link_engine_index = 0;
#ifdef USE_REDUCED_LINK_ENGINE_SET
unsigned int ishmemi_link_engine[NUM_LINK_QUEUE] = {2, 4};
#else
unsigned int ishmemi_link_engine[NUM_LINK_QUEUE] = {2, 4, 6};
#endif

ze_command_queue_handle_t ishmemi_ze_cmd_queue;
ze_command_queue_handle_t ishmemi_ze_all_cmd_queue;
ze_command_queue_handle_t ishmemi_ze_link_cmd_queue[NUM_LINK_QUEUE];
std::vector<ze_command_list_handle_t> ishmemi_ze_cmd_lists;
std::vector<ze_command_list_handle_t> ishmemi_ze_link_cmd_lists[NUM_LINK_QUEUE];

uint32_t ishmemi_gpu_driver_idx = 0;
uint32_t ishmemi_fpga_driver_idx = 0;
bool ishmemi_gpu_driver_found = false;
bool ishmemi_fpga_driver_found = false;

/* this should be thread safe because we query the size, then sync
 * then destroy the first size items, then erase them from the list
 */
static int ishmemi_sync_cmd_queue(ze_command_queue_handle_t &queue,
                                  std::vector<ze_command_list_handle_t> &cmd_lists)
{
    int ret = 0;
    size_t size = cmd_lists.size();
    std::vector<ze_command_list_handle_t>::iterator first, last;

    ZE_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    for (size_t i = 0; i < size; i += 1)
        ZE_CHECK(zeCommandListDestroy(cmd_lists[i]));

    first = cmd_lists.begin();
    last = first + static_cast<long>(size);
    cmd_lists.erase(first, last);

fn_exit:
    return ret;
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
    all_drivers = (ze_driver_handle_t *) calloc(driver_count, sizeof(ze_driver_handle_t));
    ISHMEM_CHECK_GOTO_MSG(all_drivers == nullptr, fn_fail, "Allocation of all_drivers failed\n");

    all_devices = (ze_device_handle_t **) calloc(driver_count, sizeof(ze_device_handle_t *));
    ISHMEM_CHECK_GOTO_MSG(all_devices == nullptr, fn_fail, "Allocation of all_devices failed\n");

    ZE_CHECK(zeDriverGet(&driver_count, all_drivers));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    /* Parse the drivers for a suitable driver */
    for (i = 0; i < driver_count; i++) {
        device_count = 0;
        ZE_CHECK(zeDeviceGet(all_drivers[i], &device_count, nullptr));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
        if (device_count == 0) continue;

        // Only a single device will be returned because of setting in ishmrun launcher
        ISHMEM_CHECK_GOTO_MSG(device_count != 1, fn_fail, "Detected more than one device\n");
        all_devices[i] = (ze_device_handle_t *) malloc(device_count * sizeof(ze_device_handle_t));
        ISHMEM_CHECK_GOTO_MSG(all_devices == nullptr, fn_fail,
                              "Allocation of all_drivers[%d] failed\n", i);

        ZE_CHECK(zeDeviceGet(all_drivers[i], &device_count, all_devices[i]));

        ze_device_properties_t device_properties;
        ZE_CHECK(zeDeviceGetProperties(all_devices[i][0], &device_properties));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
        /* Storing gpu and fpga devices only for now */
        if (ZE_DEVICE_TYPE_GPU == device_properties.type && !ishmemi_gpu_driver_found) {
            ishmemi_gpu_driver = all_drivers[i];
            ishmemi_gpu_driver_idx = i;
            ishmemi_gpu_driver_found = true;
        } else if (ZE_DEVICE_TYPE_FPGA == device_properties.type && !ishmemi_fpga_driver_found) {
            ishmemi_fpga_driver = all_drivers[i];
            ishmemi_fpga_driver_idx = i;
            ishmemi_fpga_driver_found = true;
        }
    }

    if (!ishmemi_gpu_driver_found && !ishmemi_fpga_driver_found) {
        ISHMEM_ERROR_MSG("No ZE driver found for GPU or FPGA\n");
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
    uint32_t device_count = 0;
    uint32_t i, j;
    int ret = 0;
    ze_command_queue_desc_t cmdq_desc;
    ze_event_pool_desc_t event_pool_desc;

    ret = ishmemi_accelerator_preinit();
    if (ret != 0) goto fn_exit;
    /* set default interval for cmd_list garbage collection */
    ishmemi_ze_cmd_lists.reserve(ishmemi_params.NBI_COUNT);
    for (int i = 0; i < NUM_LINK_QUEUE; i += 1) {
        ishmemi_ze_link_cmd_lists[i].reserve(ishmemi_params.NBI_COUNT);
    }

    if (ishmemi_gpu_driver_found) {
        /* TODO: Make default device assignment topology-aware instead of round-robin */
        /* Set the default device for GPU */
        /* TODO: This currently assumes all devices for the driver are GPU devices */
        ishmemi_gpu_device = all_devices[ishmemi_gpu_driver_idx][0];
    }

    if (ishmemi_fpga_driver_found) {
        /* Set the default device for FPGA */
        /* TODO: This currently assumes all devices for the driver are FPGA devices */
        ishmemi_fpga_device = all_devices[ishmemi_fpga_driver_idx][0];
    }

    if (ishmemi_gpu_driver_found) {
        /* Get P2P properties between the local device and each GPU device */
        for (i = 0; i < driver_count; i++) {
            device_count = 0;
            ZE_CHECK(zeDeviceGet(all_drivers[i], &device_count, nullptr));
            ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
            if (device_count == 0) continue;

            for (j = 0; j < device_count; j++) {
                ze_device_properties_t device_properties;
                ZE_CHECK(zeDeviceGetProperties(all_devices[i][j], &device_properties));
                ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
            }
        }
    }

    if (ishmemi_params.DEBUG) {
        ze_device_properties_t device_properties;

        if (ishmemi_gpu_driver_found) {
            ZE_CHECK(zeDeviceGetProperties(ishmemi_gpu_device, &device_properties));
            ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
            ishmemi_print_device_properties(device_properties);
        }

        if (ishmemi_fpga_driver_found) {
            ZE_CHECK(zeDeviceGetProperties(ishmemi_fpga_device, &device_properties));
            ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
            ishmemi_print_device_properties(device_properties);
        }
    }

    /* Create the ZE command queue */
    cmdq_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .pNext = nullptr,
        .ordinal = 1,
        .index = 0,
        .flags = 0,
        .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
    };
    ZE_CHECK(zeCommandQueueCreate(ishmemi_ze_context, ishmemi_gpu_device, &cmdq_desc,
                                  &ishmemi_ze_cmd_queue));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    /* create link queue for group command lists */
    cmdq_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .pNext = nullptr,
        .ordinal = 2,
        .index = 0,
        .flags = 0,
        .mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
    };
    ZE_CHECK(zeCommandQueueCreate(ishmemi_ze_context, ishmemi_gpu_device, &cmdq_desc,
                                  &ishmemi_ze_all_cmd_queue));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    for (uint32_t i = 0; i < NUM_LINK_QUEUE; i += 1) {
        cmdq_desc = {
            .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
            .pNext = nullptr,
            .ordinal = 2,
            .index = 2U + (i * 2),  // 2 4 6
            .flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY,
            .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
            .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
        };
        ZE_CHECK(zeCommandQueueCreate(ishmemi_ze_context, ishmemi_gpu_device, &cmdq_desc,
                                      &ishmemi_ze_link_cmd_queue[i]));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
    }
    /* Create the ZE event pool */
    event_pool_desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
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

    if (ishmemi_ze_cmd_queue) {
        ishmemi_sync_cmd_queue(ishmemi_ze_cmd_queue, ishmemi_ze_cmd_lists);
        ZE_CHECK(zeCommandQueueDestroy(ishmemi_ze_cmd_queue));
        ishmemi_ze_cmd_queue = nullptr;
    }

    if (ishmemi_ze_all_cmd_queue) {
        ZE_CHECK(zeCommandQueueDestroy(ishmemi_ze_all_cmd_queue));
        ishmemi_ze_all_cmd_queue = nullptr;
    }

    for (int i = 0; i < NUM_LINK_QUEUE; i += 1) {
        if (ishmemi_ze_link_cmd_queue[i]) {
            ishmemi_sync_cmd_queue(ishmemi_ze_link_cmd_queue[i], ishmemi_ze_link_cmd_lists[i]);
            ZE_CHECK(zeCommandQueueDestroy(ishmemi_ze_link_cmd_queue[i]));
        }
        ishmemi_ze_link_cmd_queue[i] = nullptr;
    }

    for (int i = 0; i < driver_count; i++)
        ISHMEMI_FREE(free, all_devices[i]);
    ISHMEMI_FREE(free, all_devices);
    ISHMEMI_FREE(free, all_drivers);

    ishmemi_accelerator_preinitialized = false;
    ishmemi_accelerator_initialized = false;
    ishmemi_gpu_driver_found = false;
    ishmemi_fpga_driver_found = false;
    ishmemi_gpu_driver_idx = 0;
    ishmemi_fpga_driver_idx = 0;
    driver_count = 0;

    if (ishmemi_ze_context) {
        ZE_CHECK(zeContextDestroy(ishmemi_ze_context));
        ishmemi_ze_context = nullptr;
    }

    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

fn_exit:
    return ret;
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
    ishmemi_sync_cmd_queue(ishmemi_ze_cmd_queue, ishmemi_ze_cmd_lists);
    for (int i = 0; i < NUM_LINK_QUEUE; i += 1) {
        ishmemi_sync_cmd_queue(ishmemi_ze_link_cmd_queue[i], ishmemi_ze_link_cmd_lists[i]);
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
