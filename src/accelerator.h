/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_ACCELERATOR_H
#define ISHMEM_ACCELERATOR_H

#include "internal.h"
#include <level_zero/ze_api.h>

#include <sys/mman.h>  // used by get_mmap_address
#include <sys/unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>  // for stringstream below

#define ISHMEMI_NO_DEVICES       -1
#define ISHMEMI_NO_DRIVERS       -2
#define ISHMEMI_NO_DEVICE_ACCESS -3

extern ze_driver_handle_t *all_drivers;
extern ze_device_handle_t **all_devices;

extern ze_driver_handle_t ishmemi_gpu_driver;
extern ze_driver_handle_t ishmemi_fpga_driver;
extern ze_device_handle_t ishmemi_gpu_device;
extern ze_device_handle_t ishmemi_fpga_device;

extern ze_context_handle_t ishmemi_ze_context;
extern ze_context_desc_t ishmemi_ze_context_desc;
/* ishmemi_ze_cmd_queue is the main copy engine */
extern ze_command_queue_handle_t ishmemi_ze_cmd_queue;
extern ze_command_queue_handle_t ishmemi_ze_all_cmd_queue;

/* ishmemi_ze_link_cmd_queue are the bandwidth link copy engines */
#ifdef USE_REDUCED_LINK_ENGINE_SET
constexpr int NUM_LINK_QUEUE = 2;
#else
constexpr int NUM_LINK_QUEUE = 3;
#endif
extern unsigned int ishmemi_link_engine_index;
extern ze_command_queue_handle_t ishmemi_ze_link_cmd_queue[NUM_LINK_QUEUE];
extern unsigned int ishmemi_link_engine[NUM_LINK_QUEUE];

/* used for garbage collecting nbi cmd lists on synchronize */
/* pre-size this according to ishmem_nbi_count environment, with default 1000 */
/* then auto-cleanup on synchronize or when the number gets to the limit */
extern std::vector<ze_command_list_handle_t> ishmemi_ze_link_cmd_lists[NUM_LINK_QUEUE];
extern std::vector<ze_command_list_handle_t> ishmemi_ze_cmd_lists;
extern ze_event_pool_handle_t ishmemi_ze_event_pool;
extern uint32_t ishmemi_gpu_driver_idx;

static inline unsigned int ishmemi_next_link_engine_index()
{
    unsigned int next = ishmemi_link_engine_index + 1;
    if (next >= NUM_LINK_QUEUE) next = 0;
    ishmemi_link_engine_index = next;
    return (next);
}

static inline void ishmemi_print_device_properties(const ze_device_properties_t &props)
{
    std::stringstream stream;
    stream << "PE : " << ishmemi_my_pe << " Device info: " << std::endl
           << "    name : " << props.name << std::endl
           << "    type : " << ((props.type == ZE_DEVICE_TYPE_GPU) ? "GPU" : "FPGA") << std::endl
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

template <typename T>
T *ishmemi_get_mmap_address(T *device_ptr, size_t size)
{
    int ret = 0;
    int fd;
    int flags = MAP_SHARED;
    void *base;
    ze_ipc_mem_handle_t ze_ipc_handle;
    ZE_CHECK(zeMemGetIpcHandle(ishmemi_ze_context, device_ptr, &ze_ipc_handle));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    memcpy(&fd, &ze_ipc_handle, sizeof(fd));
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

#endif /* ISHMEM_ACCELERATOR_H */
