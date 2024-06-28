/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_GPUPRINT_H
#define ISHMEM_GPUPRINT_H

#define ISHMEMI_PRINT(out)                                                                         \
    ishmemi_print(__FILE__, (long int) __LINE__, __func__, out, ishmemi_msg_type_t::DEBUG)
/* this version you specify */
ISHMEM_DEVICE_ATTRIBUTES void ishmemi_print(const char *file, long int line, const char *func,
                                            const char *out, ishmemi_msg_type_t msg_type);

#endif
