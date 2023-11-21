/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef PROXY_FUNC_H
#define PROXY_FUNC_H

#include "runtime.h"

extern ishmemi_runtime_proxy_func_t **ishmemi_upcall_funcs;

int ishmemi_proxy_func_init();

int ishmemi_proxy_func_fini();

#endif /* PROXY_FUNC_H */
