/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "wrapper.h"
#include "env_utils.h"
#include <pmi.h>
#include <dlfcn.h>

int (*PMI_WRAPPER_Abort)(int, char *);
int (*PMI_WRAPPER_Barrier)();
int (*PMI_WRAPPER_Finalize)();
int (*PMI_WRAPPER_Get_rank)(int *);
int (*PMI_WRAPPER_Get_size)(int *);
int (*PMI_WRAPPER_Init)(int *);
int (*PMI_WRAPPER_Initialized)(int *);
int (*PMI_WRAPPER_KVS_Get_key_length_max)(int *);
int (*PMI_WRAPPER_KVS_Get_my_name)(char *, int);
int (*PMI_WRAPPER_KVS_Get_name_length_max)(int *);
int (*PMI_WRAPPER_KVS_Get_value_length_max)(int *);

int ishmemi_pmi_wrapper_init()
{
    int ret = 0;
    void *pmi_handle = nullptr;

    const char *pmi_libname = ishmemi_params.PMI_LIB_NAME.c_str();

    pmi_handle = dlopen(pmi_libname, RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);

    if (pmi_handle == nullptr) {
        RAISE_ERROR_MSG("could not find pmi library '%s' in environment\n", pmi_libname);
    }

    ISHMEMI_LINK_SYMBOL(pmi_handle, PMI, Abort);
    ISHMEMI_LINK_SYMBOL(pmi_handle, PMI, Barrier);
    ISHMEMI_LINK_SYMBOL(pmi_handle, PMI, Finalize);
    ISHMEMI_LINK_SYMBOL(pmi_handle, PMI, Get_rank);
    ISHMEMI_LINK_SYMBOL(pmi_handle, PMI, Get_size);
    ISHMEMI_LINK_SYMBOL(pmi_handle, PMI, Init);
    ISHMEMI_LINK_SYMBOL(pmi_handle, PMI, Initialized);
    ISHMEMI_LINK_SYMBOL(pmi_handle, PMI, KVS_Get_key_length_max);
    ISHMEMI_LINK_SYMBOL(pmi_handle, PMI, KVS_Get_my_name);
    ISHMEMI_LINK_SYMBOL(pmi_handle, PMI, KVS_Get_name_length_max);
    ISHMEMI_LINK_SYMBOL(pmi_handle, PMI, KVS_Get_value_length_max);

fn_exit:
    return ret;
fn_fail:
    goto fn_exit;
}
