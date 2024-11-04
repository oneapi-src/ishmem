/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_HELPER_H
#define ISHMEM_RUNTIME_HELPER_H

#include <ishmem.h>
#include <vector>

/* Runtime class */
class ishmemi_test_runtime_type {
  public:
    /* Base class constructor */
    ishmemi_test_runtime_type(void) {}

    /* Base class destructor */
    virtual ~ishmemi_test_runtime_type(void) {}

    /* Helper APIs */
    virtual ishmemx_runtime_type_t get_type(void) = 0;

    /* Setup APIs */
    virtual void init(void) = 0;
    virtual void finalize(void) = 0;

    /* Memory APIs */
    virtual void *calloc(size_t, size_t) = 0;
    virtual void *malloc(size_t) = 0;
    virtual void free(void *) = 0;

    /* Operation APIs */
    virtual void sync(void) = 0;
    virtual void broadcast(void *, void *, size_t, int) = 0;
    virtual void uint64_sum_reduce(uint64_t *, uint64_t *, size_t) = 0;
    virtual void float_sum_reduce(float *, float *, size_t) = 0;
};

extern ishmemi_test_runtime_type *ishmemi_test_runtime;

/* Symbol macros */
#define ISHMEMI_TEST_LINK_STRINGIFY(INPUT) #INPUT

#define ISHMEMI_TEST_LINK_SYMBOL(lib_handle, prefix, suffix)                                       \
    do {                                                                                           \
        void **var_ptr = (void **) &suffix;                                                        \
        void *tmp = (void *) dlsym(lib_handle, ISHMEMI_TEST_LINK_STRINGIFY(prefix##_##suffix));    \
        if (tmp == nullptr) {                                                                      \
            ISHMEM_ERROR_MSG("link symbol failed for '%s'\n",                                      \
                             ISHMEMI_TEST_LINK_STRINGIFY(prefix##_##suffix));                      \
            ret = -1;                                                                              \
        } else {                                                                                   \
            *var_ptr = tmp;                                                                        \
            wrapper_list.push_back(var_ptr);                                                       \
        }                                                                                          \
    } while (0);

#if defined(ENABLE_OPENSHMEM)
#include "runtime_openshmem.h"
#endif
#if defined(ENABLE_MPI)
#include "runtime_mpi.h"
#endif

#endif
