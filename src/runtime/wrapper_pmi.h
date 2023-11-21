/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_WRAPPER_PMI_H
#define ISHMEM_RUNTIME_WRAPPER_PMI_H

#include <pmi.h>

extern int (*PMI_WRAPPER_Abort)(int, char *);
extern int (*PMI_WRAPPER_Barrier)();
extern int (*PMI_WRAPPER_Finalize)();
extern int (*PMI_WRAPPER_Get_rank)(int *);
extern int (*PMI_WRAPPER_Get_size)(int *);
extern int (*PMI_WRAPPER_Init)(int *);
extern int (*PMI_WRAPPER_Initialized)(int *);
extern int (*PMI_WRAPPER_KVS_Get_key_length_max)(int *);
extern int (*PMI_WRAPPER_KVS_Get_my_name)(char *, int);
extern int (*PMI_WRAPPER_KVS_Get_name_length_max)(int *);
extern int (*PMI_WRAPPER_KVS_Get_value_length_max)(int *);

int ishmemi_pmi_wrapper_init();

#endif
