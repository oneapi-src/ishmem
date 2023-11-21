/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_EXP_H
#define ISHMEM_EXP_H

/* experimental interfaces */

/* different versions of barrier */
ISHMEM_DEVICE_ATTRIBUTES void ishmem_exp_sync_atomic_exchange();

ISHMEM_DEVICE_ATTRIBUTES void ishmem_exp_sync_bitmap();

ISHMEM_DEVICE_ATTRIBUTES void ishmem_exp_sync_atomic_add();

ISHMEM_DEVICE_ATTRIBUTES void ishmem_exp_sync_store();

#endif /* ISHMEM_EXP_H */
