/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "ishmem/err.h"
#include "accelerator.h"
#include "runtime_ipc.h"
#include "memory.h"

int ishmemi_ipc_put_v(int nitems, struct put_item *items)
{
    int ret = 0;
    void *ipc_dst = nullptr;
    ze_command_list_handle_t cmd_list = {};

    ze_event_handle_t event;
    ze_event_desc_t event_desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_DESC,
        .pNext = nullptr,
        .index = 0,
        .signal = 0,
        .wait = 0,
    };

    ret = ishmemi_create_command_list(COPY_QUEUE, false, &cmd_list);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    for (int i = 0; i < nitems; ++i) {
        ipc_dst = get_ipc_buffer(items[i].pe, (void *) items[i].dst);
        ISHMEMI_CHECK_RESULT((ipc_dst == nullptr), 0, fn_exit);

        ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, ipc_dst, items[i].src, items[i].size,
                                               nullptr, 0, nullptr));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
    }

    ZE_CHECK(zeEventCreate(ishmemi_ze_event_pool, &event_desc, &event));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListAppendSignalEvent(cmd_list, event));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListClose(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ret = ishmemi_execute_command_lists(COPY_QUEUE, 1, &cmd_list);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeEventHostSynchronize(event, UINT64_MAX));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListDestroy(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

fn_exit:
    return ret;
}
