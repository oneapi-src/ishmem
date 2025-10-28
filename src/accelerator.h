/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_ACCELERATOR_H
#define ISHMEM_ACCELERATOR_H

#include "ishmem/err.h"
#include "ishmem/util.h"
#include <level_zero/ze_api.h>

#include <sys/mman.h>
#include <sys/unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>

#define ISHMEMI_NO_DEVICES       -1
#define ISHMEMI_NO_DRIVERS       -2
#define ISHMEMI_NO_DEVICE_ACCESS -3

extern ze_driver_handle_t ishmemi_gpu_driver;
extern ze_device_handle_t ishmemi_gpu_device;

extern ze_context_handle_t ishmemi_ze_context;
extern ze_context_desc_t ishmemi_ze_context_desc;

extern ze_event_pool_handle_t ishmemi_ze_event_pool;

typedef enum : uint32_t {
    COMPUTE_QUEUE = 0,
    COPY_QUEUE,
    LINK_QUEUE,
    UNDEFINED_QUEUE,
} ishmemi_queue_type_t;

static inline void ishmemi_print_device_properties(const ze_device_properties_t &props)
{
    std::stringstream stream;
    stream << "PE : " << ishmemi_my_pe << " Device info: " << std::endl
           << "    name : " << props.name << std::endl
           << "    type : " << ((props.type == ZE_DEVICE_TYPE_GPU) ? "GPU" : "Unknown") << std::endl
           << "    vendorId : " << props.vendorId << std::endl
           << "    deviceId : " << props.deviceId << std::endl
           << "    subdeviceId : " << props.subdeviceId << std::endl
           << "    coreClockRate : " << props.coreClockRate << std::endl;
    char bytestr[8];
    stream << "    uuid : ";
    for (int i = 0; i < ZE_MAX_DEVICE_UUID_SIZE; i += 1) {
        uint8_t b = props.uuid.id[i];
        snprintf(bytestr, sizeof(bytestr), " %02x", b);
        stream << bytestr;
    }
    stream << std::endl;
    std::cout << stream.str();
}

/* Accelerator routines */

/* Initialize accelerator */
int ishmemi_accelerator_preinit(void);
int ishmemi_accelerator_init(void);

/* Finalize accelerator */
int ishmemi_accelerator_fini(void);
int ishmemi_accelerator_postfini(void);

/* Query allocation memory type */
int ishmemi_get_memory_type(const void *ptr, ze_memory_type_t *type);

/* synchronize level_zero command queues */
void ishmemi_level_zero_sync();

/* USM memory functions */
int ishmemi_usm_alloc_host(void **, size_t);
int ishmemi_usm_alloc_device(void **, size_t);
int ishmemi_usm_free(void *);

/* List/queue helper functions */
int ishmemi_create_command_list(ishmemi_queue_type_t, bool, ze_command_list_handle_t *,
                                ze_command_list_flags_t flags = 0);
int ishmemi_create_command_list_nbi(ishmemi_queue_type_t, ze_command_list_handle_t *,
                                    ze_command_list_flags_t flags = 0);
int ishmemi_execute_command_lists(ishmemi_queue_type_t, uint32_t, ze_command_list_handle_t *,
                                  ze_fence_handle_t fence = nullptr);

template <typename T>
T *ishmemi_get_mmap_address(T *device_ptr, size_t size, ze_ipc_mem_handle_t *ze_ipc_handle)
{
    int ret = 0;
    int fd;
    int flags = MAP_SHARED;
    void *base;
    ZE_CHECK(zeMemGetIpcHandle(ishmemi_ze_context, device_ptr, ze_ipc_handle));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    memcpy(&fd, ze_ipc_handle, sizeof(fd));
    base = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, fd, 0);
    if (base == (void *) MAP_FAILED) {
        ISHMEM_CHECK_GOTO_MSG(1, fn_fail, "mmap failed with description: %s\n", strerror(errno));
        /* there is a level zero implicit scaling issue with mmap, such that mmap will fail when
         * memory is mapped across two tiles.  If this triggers that is probably the reason. */
    }
    return (T *) base;
fn_fail:
    return (T *) nullptr;
}

template <typename T>
int ishmemi_close_mmap_address(ze_ipc_mem_handle_t ze_ipc_handle, T *host_ptr, size_t size)
{
    int ret;
    ret = munmap(host_ptr, size);
    ISHMEM_CHECK_GOTO_MSG(ret != 0, fn_fail, "munmap failed with description: %s\n",
                          strerror(errno));

    ZE_CHECK(zeMemPutIpcHandle(ishmemi_ze_context, ze_ipc_handle));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

fn_exit:
    return ret;
fn_fail:
    ret = -1;
    goto fn_exit;
}

#endif /* ISHMEM_ACCELERATOR_H */
