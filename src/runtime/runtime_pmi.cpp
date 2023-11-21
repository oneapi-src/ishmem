/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government
 * retains certain rights in this software.
 *
 * This file is part of the Sandia OpenSHMEM software package. For license
 * information, see the LICENSE file in the top level directory of the
 * distribution.
 *
 */

/* Wrappers to interface with PMI runtime */
#include "ishmem_config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "pmi.h"

#include "runtime.h"
#include "wrapper.h"
#include "uthash.h"

static int rank = -1;
static int size = 0, node_size = 0;
static char *kvs_name, *kvs_key, *kvs_value;
static int max_name_len, max_key_len, max_val_len;
static bool initialized_pmi = false;
static int *location_array = nullptr;

#define SINGLETON_KEY_LEN 128
#define SINGLETON_VAL_LEN 1024

typedef struct {
    char key[SINGLETON_KEY_LEN];
    char val[SINGLETON_VAL_LEN];
    UT_hash_handle hh;
} singleton_kvs_t;

singleton_kvs_t *singleton_kvs = nullptr;

int ishmemi_runtime_pmi_fini(void)
{
    free(location_array);
    free(kvs_name);
    free(kvs_key);
    free(kvs_value);

    if (initialized_pmi) {
        PMI_Finalize();
        initialized_pmi = false;
    }

    return 0;
}

void ishmemi_runtime_pmi_abort(int exit_code, const char msg[])
{
    if (size == 1) {
        fprintf(stderr, "%s\n", msg);
        exit(exit_code);
    }

    PMI_Abort(exit_code, msg);

    /* PMI_Abort should not return */
    abort();
}

int ishmemi_runtime_pmi_get_rank(void)
{
    return rank;
}

int ishmemi_runtime_pmi_get_size(void)
{
    return size;
}

/* FIXME - Not currently being used */
int ishmemi_runtime_pmi_get_node_rank(int pe)
{
    if (pe >= size || pe < 0) {
        std::cout << "[ERROR] Wrong PE value " << pe << std::endl;
        ishmemi_runtime_pmi_abort(1, "Wrong PE value");
    }

    if (size == 1) {
        return 0;
    } else {
        return location_array[pe];
    }
}

/* FIXME - Not currently being used */
int ishmemi_runtime_pmi_get_node_size(void)
{
    if (size == 1) {
        return 1;
    } else {
        return node_size;
    }
}

void ishmemi_runtime_pmi_barrier(void)
{
    PMI_Barrier();
}

void ishmemi_runtime_pmi_node_barrier(void)
{
    // FIXME: No node barrier supported
    PMI_Barrier();
}

void ishmemi_runtime_pmi_bcast(void *buf, size_t count, int root)
{
    // FIXME: No bcast supported
}

void ishmemi_runtime_pmi_node_bcast(void *buf, size_t count, int root)
{
    // FIXME: No node bcast supported
}

void ishmemi_runtime_pmi_unsupported(ishmem_info_t *info)
{
    ISHMEM_ERROR_MSG("Encountered type '%s' unsupported for operation '%s'\n",
                     ishmemi_type_str[info->req.type], ishmemi_op_str[info->req.op]);
    info->proxy_state = EXIT;
}

bool ishmemi_runtime_pmi_is_local(int pe)
{
    /* TODO: Implement */
    return false;
}

void ishmemi_runtime_pmi_funcptr_init()
{
    ishmemi_proxy_funcs = (ishmemi_runtime_proxy_func_t *) malloc(
        sizeof(ishmemi_runtime_proxy_func_t) * ISHMEMI_OP_END);

    /* Initialize every function with the "unsupported op" function */
    /* Don't include KILL operation - it is the same for all backends currently */
    for (int i = 0; i < ISHMEMI_OP_END - 1; ++i) {
        ishmemi_proxy_funcs[i] = ishmemi_runtime_pmi_unsupported;
    }
}

int ishmemi_runtime_pmi_init(bool initialize_runtime)  // TODO: Fix later for XPMEM
{
    int initialized, ret = 0;

    /* Setup PMI dlsym links */
    ret = ishmemi_pmi_wrapper_init();

    if (initialize_runtime) {
        PMI_Init(&initialized);
        initialized_pmi = true;
    }

    if (PMI_SUCCESS != PMI_Initialized(&initialized)) {
        return 1;
    }

    if (!initialized) {
        if (PMI_SUCCESS != PMI_Init(&initialized)) {
            return 2;
        } else {
            initialized_pmi = true;
        }
    }

    if (PMI_SUCCESS != PMI_Get_rank(&rank)) {
        return 3;
    }

    if (PMI_SUCCESS != PMI_Get_size(&size)) {
        return 4;
    }

    if (size > 1) {
        if (PMI_SUCCESS != PMI_KVS_Get_name_length_max(&max_name_len)) {
            return 5;
        }

        kvs_name = (char *) malloc(max_name_len);
        if (nullptr == kvs_name) return 6;

        if (PMI_SUCCESS != PMI_KVS_Get_key_length_max(&max_key_len)) {
            return 7;
        }

        if (PMI_SUCCESS != PMI_KVS_Get_value_length_max(&max_val_len)) {
            return 8;
        }

        if (PMI_SUCCESS != PMI_KVS_Get_my_name(kvs_name, max_name_len)) {
            return 9;
        }

        // TODO: Fix later for XPMEM
        /*if (enable_node_ranks) {
            location_array = (int *) malloc(sizeof(int) * size);
            if (nullptr == location_array) return 10;
        }*/
    } else {
        /* Use a local KVS for singleton runs */
        max_key_len = SINGLETON_KEY_LEN;
        max_val_len = SINGLETON_VAL_LEN;
        kvs_name = nullptr;
        max_name_len = 0;
    }

    kvs_key = (char *) malloc(max_key_len);
    if (nullptr == kvs_key) return 11;

    kvs_value = (char *) malloc(max_val_len);
    if (nullptr == kvs_value) return 12;

    /* Setup runtime function pointers */
    ishmemi_runtime_fini = ishmemi_runtime_pmi_fini;
    ishmemi_runtime_abort = ishmemi_runtime_pmi_abort;
    ishmemi_runtime_get_rank = ishmemi_runtime_pmi_get_rank;
    ishmemi_runtime_get_size = ishmemi_runtime_pmi_get_size;
    ishmemi_runtime_get_node_rank = ishmemi_runtime_pmi_get_node_rank;
    ishmemi_runtime_get_node_size = ishmemi_runtime_pmi_get_node_size;
    ishmemi_runtime_barrier = ishmemi_runtime_pmi_barrier;
    ishmemi_runtime_node_barrier = ishmemi_runtime_pmi_node_barrier;
    ishmemi_runtime_bcast = ishmemi_runtime_pmi_bcast;
    ishmemi_runtime_node_bcast = ishmemi_runtime_pmi_node_bcast;
    ishmemi_runtime_is_local = ishmemi_runtime_pmi_is_local;
    ishmemi_runtime_malloc = malloc;
    ishmemi_runtime_calloc = calloc;
    ishmemi_runtime_free = free;

    ishmemi_runtime_pmi_funcptr_init();

    return 0;
}
