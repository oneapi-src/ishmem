/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "internal.h"
#include "impl_proxy.h"

ISHMEM_DEVICE_ATTRIBUTES void ishmemx_print(const char *out)
{
    ishmemx_print(out, ishmemx_print_msg_type_t::DEBUG);
}

unsigned int message_allocate()
{
    ishmem_info_t *info = global_info;
    unsigned int my_index = 0;
    for (;;) {
        sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                         sycl::access::address_space::global_space>
            atomic_lock(info->message_buffer_lock[my_index]);
        if (atomic_lock.exchange(1) == 0) break;
        my_index = my_index + 1;
        if (my_index >= NUM_MESSAGES) my_index = 0;
    }
    return (my_index);
}

void message_free(unsigned int index)
{
    ishmem_info_t *info = global_info;
    sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        atomic_lock(info->message_buffer_lock[index]);
    atomic_lock.store(0);
}

#ifdef __SYCL_DEVICE_ONLY__
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_print(const char *out, ishmemx_print_msg_type_t msg_type)
{
    ishmem_info_t *info = global_info;

    /* linear search for a message buffer */
    unsigned int message_index = message_allocate();

    /* Setup buffer */
    size_t len = 0;
    const char *p = out;
    while (*p++ != 0)
        len += 1;
    if (len > MAX_PROXY_MSG_SIZE) len = MAX_PROXY_MSG_SIZE;
    char *msg_ptr = &info->messages[message_index].message[0];
    memcpy(msg_ptr, out, len);
    msg_ptr[MAX_PROXY_MSG_SIZE - 1] = 0; /* terminate string in case of truncation */

    ishmemi_request_t req = {
        .op = PRINT,
        .type = MEM,
        .dest_pe = msg_type,
        .src = msg_ptr,
    };

    /* Initiate request */
    ishmemi_proxy_blocking_request(&req);

    message_free(message_index);
}
#else
void ishmemx_print(const char *out, ishmemx_print_msg_type_t msg_type)
{
    if (msg_type == ishmemx_print_msg_type_t::DEBUG) {
        ISHMEM_DEBUG_MSG("%s\n", out);
    } else if (msg_type == ishmemx_print_msg_type_t::WARNING) {
        ISHMEM_WARN_MSG("%s\n", out);
    } else {
        RAISE_ERROR_MSG("%s\n", out);
    }
}
#endif
