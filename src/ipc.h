/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_IPC_H
#define ISHMEM_IPC_H

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <iostream>
#include <level_zero/ze_api.h>
#include "ishmem.h"

typedef struct ishmemi_socket_payload_t {
    int src_pe;
    ze_ipc_mem_handle_t handle;
} ishmemi_socket_payload_t;

int ishmemi_ipc_init();
int ishmemi_ipc_fini();

#endif
