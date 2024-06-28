/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/err.h"
#include "proxy_impl.h"

unsigned int message_allocate()
{
    ishmemi_info_t *info = global_info;
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
    ishmemi_info_t *info = global_info;
    sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        atomic_lock(info->message_buffer_lock[index]);
    atomic_lock.store(0);
}

void copy_str(char *d, const char *s)
{
    for (int i = 0; i < MAX_PROXY_MSG_SIZE; i += 1) {
        char c = s[i];
        d[i] = c;
        if (c == 0) break;
    }
    d[MAX_PROXY_MSG_SIZE - 1] = 0;
}

#ifdef __SYCL_DEVICE_ONLY__

ISHMEM_DEVICE_ATTRIBUTES void ishmemx_print(const char *file, long int line, const char *func,
                                            const char *out, ishmemx_print_msg_type_t msg_type)
{
    ishmemi_info_t *info = global_info;

    /* linear search for a message buffer */
    unsigned int message_index = message_allocate();
    ishmemi_message_t *m = &info->messages[message_index];

    copy_str(m->file, file);
    m->line = line;
    copy_str(m->func, func);
    copy_str(m->message, out);

    ishmemi_request_t req;
    req.dest_pe = msg_type;
    req.src = m;
    req.op = PRINT;
    req.type = MEM;

    /* Initiate request */
    ishmemi_proxy_blocking_request(req);

    message_free(message_index);
}
#else
void ishmemx_print(const char *file, long int line, const char *func, const char *out,
                   ishmemx_print_msg_type_t msg_type)
{
    switch (msg_type) {
        case DEBUG:
            if (ishmemi_params.DEBUG) {
                ISHMEM_COMMON_MSG_INTERNAL("DEBUG", file, line, func, "%s", out);
                fflush(stderr);
            }
            break;
        case WARNING:
            ISHMEM_COMMON_MSG_INTERNAL("WARN", file, line, func, "%s", out);
            fflush(stderr);
            break;
        case ERROR:
            ISHMEM_COMMON_MSG_INTERNAL("ERROR", file, line, func, "%s", out);
            fflush(stderr);
            RAISE_ERROR_MSG("host error\n");
            break;
        case STDOUT:
            printf("%s", out);
            fflush(stdout);
            break;
        case STDERR:
            fprintf(stderr, "%s", out);
            fflush(stderr);
            break;
        default:
            break;
    }
}
#endif

ISHMEM_DEVICE_ATTRIBUTES void ishmemx_print(const char *out)
{
    ishmemx_print(__FILE__, __LINE__, __func__, out, ishmemx_print_msg_type_t::DEBUG);
}

ISHMEM_DEVICE_ATTRIBUTES void ishmemx_print(const char *out, ishmemx_print_msg_type_t msg_type)
{
    ishmemx_print(__FILE__, __LINE__, __func__, out, msg_type);
}
