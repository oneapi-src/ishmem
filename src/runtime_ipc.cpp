/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "ishmem/err.h"
#include "accelerator.h"
#include "runtime_ipc.h"
#include "memory.h"

int ishmemi_ipc_put_v(int nitems, struct put_item *items)
{
    int ret = 0;
    ze_command_list_desc_t cmd_list_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        .pNext = nullptr,
        .commandQueueGroupOrdinal = 2,
        .flags = 0,
    };

    ze_command_list_handle_t cmd_list;

    ZE_CHECK(
        zeCommandListCreate(ishmemi_ze_context, ishmemi_gpu_device, &cmd_list_desc, &cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    for (size_t i = 0; i < nitems; i += 1) {
        void *ipc_dst = get_ipc_buffer(items[i].pe, (void *) items[i].dst);
        if (ipc_dst == nullptr) return (1); /* dest is not ipc-able */

        ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, ipc_dst, items[i].src, items[i].size,
                                               nullptr, 0, nullptr));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
    }
    ZE_CHECK(zeCommandListClose(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandQueueExecuteCommandLists(ishmemi_ze_all_cmd_queue, 1, &cmd_list, nullptr));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);

    ZE_CHECK(zeCommandListDestroy(cmd_list));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);
fn_exit:
    return ret;
}
