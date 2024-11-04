/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_WRAPPER_H
#define ISHMEM_RUNTIME_WRAPPER_H

#include <vector>

#define ISHMEMI_LINK_STRINGIFY(INPUT) #INPUT

#define ISHMEMI_LINK_SYMBOL(lib_handle, prefix, suffix)                                            \
    do {                                                                                           \
        void **var_ptr = (void **) &suffix;                                                        \
        void *tmp = (void *) dlsym(lib_handle, ISHMEMI_LINK_STRINGIFY(prefix##_##suffix));         \
        if (tmp == nullptr) {                                                                      \
            ISHMEM_ERROR_MSG("link symbol failed for '%s'\n",                                      \
                             ISHMEMI_LINK_STRINGIFY(prefix##_##suffix));                           \
            ret = -1;                                                                              \
        } else {                                                                                   \
            *var_ptr = tmp;                                                                        \
            wrapper_list.push_back(var_ptr);                                                       \
        }                                                                                          \
    } while (0);

#if defined(ENABLE_MPI)
#include "wrapper_mpi.h"
#endif
#if defined(ENABLE_OPENSHMEM)
#include "wrapper_openshmem.h"
#endif
#if defined(ENABLE_PMI)
#include "wrapper_pmi.h"
#endif

#endif
