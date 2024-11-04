/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef RUNTIME_IPC_H
#define RUNTIME_IPC_H

#include <stdlib.h>
#include <level_zero/ze_api.h>

#include "ishmem/err.h"
#include "accelerator.h"
#include "memory.h"

typedef enum {
    IPC_ALGORITHM_IMMEDIATE_CL,
    IPC_ALGORITHM_REGULAR_CL
} ishmemi_ipc_algorithm_t;

constexpr ishmemi_ipc_algorithm_t ishmemi_ipc_algorithm = IPC_ALGORITHM_REGULAR_CL;

/* get_ipc_buffer will return null unless the target PE is local and the given pointer is in the
 * ishmem symmetric heap */

static inline void *get_ipc_buffer(int pe, void *buf)
{
    uint8_t lindex = ishmemi_local_pes[pe];
    if (lindex == 0) return (nullptr);
    if (((uintptr_t) buf) < ((uintptr_t) ishmemi_heap_base)) return (nullptr);
    if (((uintptr_t) buf) > ishmemi_heap_last) return (nullptr);
    return (pointer_offset(buf, ishmemi_ipc_buffer_delta[lindex]));
}

template <typename TYPENAME>
inline size_t size_of()
{
    return sizeof(TYPENAME);
}

template <>
inline size_t size_of<void>()
{
    return 1;
}

template <typename TYPENAME>
int ishmemi_ipc_put_immediate_cl(TYPENAME *dst, const TYPENAME *src, size_t nelems, int pe)
{
    int ret = 0;
    size_t bytes = nelems * size_of<TYPENAME>();
    ze_command_queue_desc_t cmd_queue_desc = {.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                              .pNext = nullptr,
                                              .ordinal = 2,
                                              .index = 0,
                                              .flags = 0,
                                              .mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                              .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL};

    ze_command_list_handle_t cmd_list = {};

    if ((pe == ishmemi_my_pe) || (pe == (ishmemi_my_pe ^ 1))) {
        // use main copy engine
        cmd_queue_desc.ordinal = 1;
        cmd_queue_desc.index = 0;
    } else {
        // rotate through link copy engines
        cmd_queue_desc.ordinal = 2;
        cmd_queue_desc.index = ishmemi_link_engine[ishmemi_next_link_engine_index()];
    }
    void *ipc_dst = get_ipc_buffer(pe, (void *) dst);
    if (ipc_dst == nullptr) return (1); /* dest is not ipc-able */

    ZE_CHECK(zeCommandListCreateImmediate(ishmemi_ze_context, ishmemi_gpu_device, &cmd_queue_desc,
                                          &cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
    ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, ipc_dst, src, bytes, nullptr, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
    ZE_CHECK(zeCommandListDestroy(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
    return (ret);

fn_exit:
    return ret;
}

template <typename TYPENAME>
int ishmemi_ipc_put_regular_cl(TYPENAME *dst, const TYPENAME *src, size_t nelems, int pe)
{
    int ret = 0;
    size_t bytes = nelems * size_of<TYPENAME>();
    ze_command_queue_handle_t cmd_queue;
    ze_command_list_desc_t cmd_list_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        .pNext = nullptr,
        .commandQueueGroupOrdinal = 2,
        .flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY,
    };

    ze_command_list_handle_t cmd_list;

    ze_event_desc_t event_desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_DESC,
        .pNext = nullptr,
        .index = 0,
        .signal = 0,
        .wait = 0,
    };
    ze_event_handle_t event;

    void *ipc_dst = get_ipc_buffer(pe, (void *) dst);
    if (ipc_dst == nullptr) return (1); /* dest is not ipc-able */

    if ((pe == ishmemi_my_pe) || (pe == (ishmemi_my_pe ^ 1))) {
        // use main copy engine
        cmd_queue = ishmemi_ze_cmd_queue;
        cmd_list_desc.commandQueueGroupOrdinal = 1; /* main copy engine ordinal */
        cmd_list_desc.flags = 0;
        /* create command list for the main command queue */
    } else {
        // rotate through link copy engines
        unsigned int idx = ishmemi_next_link_engine_index();
        cmd_queue = ishmemi_ze_link_cmd_queue[idx];
        cmd_list_desc.commandQueueGroupOrdinal = 2; /* link engines ordinal */
        cmd_list_desc.flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;
    }
    ZE_CHECK(
        zeCommandListCreate(ishmemi_ze_context, ishmemi_gpu_device, &cmd_list_desc, &cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeEventCreate(ishmemi_ze_event_pool, &event_desc, &event));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, ipc_dst, src, bytes, event, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListClose(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandQueueExecuteCommandLists(cmd_queue, 1, &cmd_list, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeEventHostSynchronize(event, UINT64_MAX));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListDestroy(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    return (ret);

fn_exit:
    return ret;
}

template <typename TYPENAME>
inline int ishmemi_ipc_put(TYPENAME *dst, const TYPENAME *src, size_t nelems, int pe)
{
    if constexpr (ishmemi_ipc_algorithm == IPC_ALGORITHM_IMMEDIATE_CL) {
        return ishmemi_ipc_put_immediate_cl(dst, src, nelems, pe);
    } else if constexpr (ishmemi_ipc_algorithm == IPC_ALGORITHM_REGULAR_CL) {
        return ishmemi_ipc_put_regular_cl(dst, src, nelems, pe);
    }
}

template <typename TYPENAME>
int ishmemi_ipc_put_nbi(TYPENAME *dst, const TYPENAME *src, size_t nelems, int pe)
{
    int ret = 0;
    size_t outstanding = 0;
    size_t bytes = nelems * size_of<TYPENAME>();
    void *ipc_dst = get_ipc_buffer(pe, (void *) dst);
    if (ipc_dst == nullptr) return (1); /* dest is not ipc-able */
    /* Check if src is a GPU buffer */
    ze_command_queue_handle_t cmd_queue;
    ze_command_list_handle_t cmd_list = {};
    ze_command_list_desc_t cmd_list_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        .pNext = nullptr,
        .commandQueueGroupOrdinal = 2,
        .flags = 0,
    };

    if ((pe == ishmemi_my_pe) || (pe == (ishmemi_my_pe ^ 1))) {
        // use main copy engine
        cmd_queue = ishmemi_ze_cmd_queue;
        cmd_list_desc.commandQueueGroupOrdinal = 1; /* main copy engine ordinal */
        /* create command list for the main command queue */
        ZE_CHECK(
            zeCommandListCreate(ishmemi_ze_context, ishmemi_gpu_device, &cmd_list_desc, &cmd_list));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
        /* save the command list for later destruction on synchronize */
        outstanding = ishmemi_ze_cmd_lists.push_back_thread_safe(cmd_list);
    } else {
        // rotate through link copy engines
        unsigned int idx = ishmemi_next_link_engine_index();
        cmd_queue = ishmemi_ze_link_cmd_queue[idx];
        cmd_list_desc.commandQueueGroupOrdinal = 2; /* link engines ordinal */
        /* create command list for the chosen link command queue */
        ZE_CHECK(
            zeCommandListCreate(ishmemi_ze_context, ishmemi_gpu_device, &cmd_list_desc, &cmd_list));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
        /* save the command list for later destruction on synchronize */
        outstanding = ishmemi_ze_link_cmd_lists[idx].push_back_thread_safe(cmd_list);
    }

    /* We can assume that dst is a GPU buffer since it has to be on the symmetric heap */

    ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, ipc_dst, src, bytes, nullptr, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListClose(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandQueueExecuteCommandLists(cmd_queue, 1, &cmd_list, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    if (outstanding >= ishmemi_params.NBI_COUNT) ishmemi_level_zero_sync();

fn_exit:
    return ret;
}

template <typename TYPENAME>
int ishmemi_ipc_get_immediate_cl(TYPENAME *dst, const TYPENAME *src, size_t nelems, int pe)
{
    int ret = 0;
    size_t bytes = nelems * size_of<TYPENAME>();
    ze_command_queue_desc_t cmd_queue_desc = {.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                              .pNext = nullptr,
                                              .ordinal = 1,
                                              .index = 0,
                                              .flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY,
                                              .mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                              .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t cmd_list = {};

    if ((pe == ishmemi_my_pe) || (pe == (ishmemi_my_pe ^ 1))) {
        // use main copy engine
        cmd_queue_desc.ordinal = 1;
        cmd_queue_desc.index = 0;
    } else {
        // rotate through link copy engines<
        cmd_queue_desc.ordinal = 2;
        cmd_queue_desc.index = ishmemi_link_engine[ishmemi_next_link_engine_index()];
    }
    void *ipc_src = get_ipc_buffer(pe, (void *) src);
    if (ipc_src == nullptr) return (1); /* src is not ipc-able */

    ZE_CHECK(zeCommandListCreateImmediate(ishmemi_ze_context, ishmemi_gpu_device, &cmd_queue_desc,
                                          &cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
    ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, dst, ipc_src, bytes, nullptr, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
    ZE_CHECK(zeCommandListDestroy(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
    return (ret);

fn_exit:
    return ret;
}

template <typename TYPENAME>
int ishmemi_ipc_get_regular_cl(TYPENAME *dst, const TYPENAME *src, size_t nelems, int pe)
{
    int ret = 0;
    size_t bytes = nelems * size_of<TYPENAME>();
    ze_command_queue_handle_t cmd_queue;
    ze_command_list_desc_t cmd_list_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        .pNext = nullptr,
        .commandQueueGroupOrdinal = 2,
        .flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY,
    };

    ze_command_list_handle_t cmd_list;

    ze_event_desc_t event_desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_DESC,
        .pNext = nullptr,
        .index = 0,
        .signal = 0,
        .wait = 0,
    };
    ze_event_handle_t event;

    void *ipc_src = get_ipc_buffer(pe, (void *) src);
    if (ipc_src == nullptr) return (1); /* src is not ipc-able */

    if ((pe == ishmemi_my_pe) || (pe == (ishmemi_my_pe ^ 1))) {
        // use main copy engine
        cmd_queue = ishmemi_ze_cmd_queue;
        cmd_list_desc.commandQueueGroupOrdinal = 1; /* main copy engine ordinal */
        cmd_list_desc.flags = 0;
        /* create command list for the main command queue */
    } else {
        // rotate through link copy engines
        unsigned int idx = ishmemi_next_link_engine_index();
        cmd_queue = ishmemi_ze_link_cmd_queue[idx];
        cmd_list_desc.commandQueueGroupOrdinal = 2; /* link engines ordinal */
        cmd_list_desc.flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;
    }
    ZE_CHECK(
        zeCommandListCreate(ishmemi_ze_context, ishmemi_gpu_device, &cmd_list_desc, &cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeEventCreate(ishmemi_ze_event_pool, &event_desc, &event));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, dst, ipc_src, bytes, event, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListClose(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandQueueExecuteCommandLists(cmd_queue, 1, &cmd_list, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeEventHostSynchronize(event, UINT64_MAX));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListDestroy(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    return (ret);

fn_exit:
    return ret;
}

template <typename TYPENAME>
inline int ishmemi_ipc_get(TYPENAME *dst, const TYPENAME *src, size_t nelems, int pe)
{
    if constexpr (ishmemi_ipc_algorithm == IPC_ALGORITHM_IMMEDIATE_CL) {
        return ishmemi_ipc_get_immediate_cl(dst, src, nelems, pe);
    } else if constexpr (ishmemi_ipc_algorithm == IPC_ALGORITHM_REGULAR_CL) {
        return ishmemi_ipc_get_regular_cl(dst, src, nelems, pe);
    }
}

template <typename TYPENAME>
int ishmemi_ipc_get_nbi(TYPENAME *dst, const TYPENAME *src, size_t nelems, int pe)
{
    int ret = 0;
    size_t outstanding = 0;
    size_t bytes = nelems * size_of<TYPENAME>();
    void *ipc_src = get_ipc_buffer(pe, (void *) src);
    if (ipc_src == nullptr) return (1); /* src is not ipc-able */
    /* Check if src is a GPU buffer */
    ze_command_queue_handle_t cmd_queue;
    ze_command_list_handle_t cmd_list = {};
    ze_command_list_desc_t cmd_list_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        .pNext = nullptr,
        .commandQueueGroupOrdinal = 2,
        .flags = 0,
    };

    if ((pe == ishmemi_my_pe) || (pe == (ishmemi_my_pe ^ 1))) {
        // use main copy engine
        cmd_queue = ishmemi_ze_cmd_queue;
        cmd_list_desc.commandQueueGroupOrdinal = 1; /* main copy engine ordinal */
        /* create command list for the main command queue */
        ZE_CHECK(
            zeCommandListCreate(ishmemi_ze_context, ishmemi_gpu_device, &cmd_list_desc, &cmd_list));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
        /* save the command list for later destruction on synchronize */
        outstanding = ishmemi_ze_cmd_lists.push_back_thread_safe(cmd_list);
    } else {
        // rotate through link copy engines
        unsigned int idx = ishmemi_next_link_engine_index();
        cmd_queue = ishmemi_ze_link_cmd_queue[idx];
        cmd_list_desc.commandQueueGroupOrdinal = 2; /* link engines ordinal */
        /* create command list for the chosen link command queue */
        ZE_CHECK(
            zeCommandListCreate(ishmemi_ze_context, ishmemi_gpu_device, &cmd_list_desc, &cmd_list));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
        /* save the command list for later destruction on synchronize */
        outstanding = ishmemi_ze_link_cmd_lists[idx].push_back_thread_safe(cmd_list);
    }

    /* We can assume that dst is a GPU buffer since it has to be on the symmetric heap */

    ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, dst, ipc_src, bytes, nullptr, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListClose(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandQueueExecuteCommandLists(cmd_queue, 1, &cmd_list, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    if (outstanding >= ishmemi_params.NBI_COUNT) ishmemi_level_zero_sync();
    goto fn_exit;

fn_exit:
    return ret;
}

/* does a vector of puts in the same command list
 * used by fcollect, collect, and alltoall
 */

struct put_item {
    void *dst;
    const void *src;
    size_t size;
    int pe;
};

int ishmemi_ipc_put_v(int nitems, struct put_item *items);

#endif /* RUNTIME_IPC_H */
