/* Copyright (C) 2025 Intel Corporation
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

    ze_command_list_handle_t cmd_list = {};
    ishmemi_queue_type_t queue_type = UNDEFINED_QUEUE;

    if ((pe == ishmemi_my_pe) || (pe == (ishmemi_my_pe ^ 1))) {
        queue_type = COPY_QUEUE;
    } else {
        queue_type = LINK_QUEUE;
    }

    void *ipc_dst = get_ipc_buffer(pe, (void *) dst);
    ISHMEMI_CHECK_RESULT((ipc_dst == nullptr), 0, fn_exit);

    ret = ishmemi_create_command_list(queue_type, true, &cmd_list);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, ipc_dst, src, bytes, nullptr, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListDestroy(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

fn_exit:
    return ret;
}

template <typename TYPENAME>
int ishmemi_ipc_put_regular_cl(TYPENAME *dst, const TYPENAME *src, size_t nelems, int pe)
{
    int ret = 0;
    size_t bytes = nelems * size_of<TYPENAME>();

    ze_command_list_handle_t cmd_list = {};
    ishmemi_queue_type_t queue_type = UNDEFINED_QUEUE;

    ze_event_handle_t event;
    ze_event_desc_t event_desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_DESC,
        .pNext = nullptr,
        .index = 0,
        .signal = 0,
        .wait = 0,
    };

    void *ipc_dst = get_ipc_buffer(pe, (void *) dst);
    ISHMEMI_CHECK_RESULT((ipc_dst == nullptr), 0, fn_exit);

    if ((pe == ishmemi_my_pe) || (pe == (ishmemi_my_pe ^ 1))) {
        queue_type = COPY_QUEUE;
    } else {
        queue_type = LINK_QUEUE;
    }

    ret = ishmemi_create_command_list(queue_type, false, &cmd_list);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeEventCreate(ishmemi_ze_event_pool, &event_desc, &event));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, ipc_dst, src, bytes, event, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListClose(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ret = ishmemi_execute_command_lists(queue_type, 1, &cmd_list);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeEventHostSynchronize(event, UINT64_MAX));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListDestroy(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

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
    size_t bytes = nelems * size_of<TYPENAME>();

    ze_command_list_handle_t cmd_list = {};
    ishmemi_queue_type_t queue_type = UNDEFINED_QUEUE;

    void *ipc_dst = get_ipc_buffer(pe, (void *) dst);
    ISHMEMI_CHECK_RESULT((ipc_dst == nullptr), 0, fn_exit);

    if ((pe == ishmemi_my_pe) || (pe == (ishmemi_my_pe ^ 1))) {
        queue_type = COPY_QUEUE;
    } else {
        queue_type = LINK_QUEUE;
    }

    ret = ishmemi_create_command_list_nbi(queue_type, &cmd_list);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, ipc_dst, src, bytes, nullptr, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListClose(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ret = ishmemi_execute_command_lists(queue_type, 1, &cmd_list);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    /* TODO: Should we sync here or check periodically in proxy thread? */

fn_exit:
    return ret;
}

template <typename TYPENAME>
int ishmemi_ipc_get_immediate_cl(TYPENAME *dst, const TYPENAME *src, size_t nelems, int pe)
{
    int ret = 0;
    size_t bytes = nelems * size_of<TYPENAME>();

    ze_command_list_handle_t cmd_list = {};
    ishmemi_queue_type_t queue_type = UNDEFINED_QUEUE;

    if ((pe == ishmemi_my_pe) || (pe == (ishmemi_my_pe ^ 1))) {
        queue_type = COPY_QUEUE;
    } else {
        queue_type = LINK_QUEUE;
    }

    void *ipc_src = get_ipc_buffer(pe, (void *) src);
    ISHMEMI_CHECK_RESULT((ipc_src == nullptr), 0, fn_exit);

    ret = ishmemi_create_command_list(queue_type, true, &cmd_list);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, dst, ipc_src, bytes, nullptr, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListDestroy(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

fn_exit:
    return ret;
}

template <typename TYPENAME>
int ishmemi_ipc_get_regular_cl(TYPENAME *dst, const TYPENAME *src, size_t nelems, int pe)
{
    int ret = 0;
    size_t bytes = nelems * size_of<TYPENAME>();

    ze_command_list_handle_t cmd_list = {};
    ishmemi_queue_type_t queue_type = UNDEFINED_QUEUE;

    ze_event_handle_t event;
    ze_event_desc_t event_desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_DESC,
        .pNext = nullptr,
        .index = 0,
        .signal = 0,
        .wait = 0,
    };

    void *ipc_src = get_ipc_buffer(pe, (void *) src);
    ISHMEMI_CHECK_RESULT((ipc_src == nullptr), 0, fn_exit);

    if ((pe == ishmemi_my_pe) || (pe == (ishmemi_my_pe ^ 1))) {
        queue_type = COPY_QUEUE;
    } else {
        queue_type = LINK_QUEUE;
    }

    ret = ishmemi_create_command_list(queue_type, false, &cmd_list);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeEventCreate(ishmemi_ze_event_pool, &event_desc, &event));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, dst, ipc_src, bytes, event, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListClose(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ret = ishmemi_execute_command_lists(queue_type, 1, &cmd_list);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeEventHostSynchronize(event, UINT64_MAX));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListDestroy(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

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
    size_t bytes = nelems * size_of<TYPENAME>();

    ze_command_list_handle_t cmd_list = {};
    ishmemi_queue_type_t queue_type = UNDEFINED_QUEUE;

    void *ipc_src = get_ipc_buffer(pe, (void *) src);
    ISHMEMI_CHECK_RESULT((ipc_src == nullptr), 0, fn_exit);

    if ((pe == ishmemi_my_pe) || (pe == (ishmemi_my_pe ^ 1))) {
        queue_type = COPY_QUEUE;
    } else {
        queue_type = LINK_QUEUE;
    }

    ret = ishmemi_create_command_list_nbi(queue_type, &cmd_list);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, dst, ipc_src, bytes, nullptr, 0, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListClose(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ret = ishmemi_execute_command_lists(queue_type, 1, &cmd_list);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    /* TODO: Should we sync here or check periodically in proxy thread? */

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
