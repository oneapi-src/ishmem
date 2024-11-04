/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_WRAPPER_PMI_H
#define ISHMEM_RUNTIME_WRAPPER_PMI_H

#include <pmi.h>

namespace ishmemi_pmi_wrappers {
    int init_wrappers(void);
    int fini_wrappers(void);

    extern int (*Abort)(int, char *);
    extern int (*Barrier)();
    extern int (*Finalize)();
    extern int (*Get_rank)(int *);
    extern int (*Get_size)(int *);
    extern int (*Init)(int *);
    extern int (*Initialized)(int *);
    extern int (*KVS_Get_key_length_max)(int *);
    extern int (*KVS_Get_my_name)(char *, int);
    extern int (*KVS_Get_name_length_max)(int *);
    extern int (*KVS_Get_value_length_max)(int *);
}  // namespace ishmemi_pmi_wrappers

#endif
