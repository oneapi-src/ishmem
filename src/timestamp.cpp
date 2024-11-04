/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/util.h"
#include "proxy_impl.h"
#include "timestamp.h"
#include <time.h>

void ishmemx_timestamp(ishmemx_ts_handle_t dst)
{
#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_request_t req;
    req.dst = (void *) dst;
    req.op = TIMESTAMP;
    req.type = NONE;

    ishmemi_proxy_blocking_request(req);
#else
    *(unsigned long *) dst = rdtsc();
#endif
}

void ishmemx_timestamp_nbi(ishmemx_ts_handle_t dst)
{
#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_request_t req;
    req.dst = (void *) dst;
    req.op = TIMESTAMP;
    req.type = NONE;

    ishmemi_proxy_nonblocking_request(req);
#else
    *(unsigned long *) dst = rdtsc();
#endif
}
