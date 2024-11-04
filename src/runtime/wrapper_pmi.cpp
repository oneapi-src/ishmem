/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/err.h"
#include "ishmem/env_utils.h"
#include "wrapper.h"
#include "wrapper_pmi.h"
#include <pmi.h>
#include <dlfcn.h>

namespace ishmemi_pmi_wrappers {
    static bool initialized = false;

    int (*Abort)(int, char *);
    int (*Barrier)();
    int (*Finalize)();
    int (*Get_rank)(int *);
    int (*Get_size)(int *);
    int (*Init)(int *);
    int (*Initialized)(int *);
    int (*KVS_Get_key_length_max)(int *);
    int (*KVS_Get_my_name)(char *, int);
    int (*KVS_Get_name_length_max)(int *);
    int (*KVS_Get_value_length_max)(int *);

    /* dl handle */
    void *pmi_handle = nullptr;
    std::vector<void **> wrapper_list;

    int fini_wrappers(void)
    {
        int ret = 0;
        for (auto p : wrapper_list)
            *p = nullptr;
        if (pmi_handle != nullptr) {
            ret = dlclose(pmi_handle);
            pmi_handle = nullptr;
            ISHMEM_CHECK_GOTO_MSG(ret, fn_exit, "dlclose failed %s\n", dlerror());
        }
        return (0);
    fn_exit:
        return (1);
    }

    int init_wrappers(void)
    {
        int ret = 0;
        void *pmi_handle = nullptr;

        const char *pmi_libname = ishmemi_params.PMI_LIB_NAME.c_str();

        pmi_handle = dlopen(pmi_libname, RTLD_NOW | RTLD_GLOBAL);

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
}  // namespace ishmemi_pmi_wrappers
