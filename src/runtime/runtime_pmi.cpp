/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portions derived from Sandia OpenSHMEM (https://github.com/Sandia-OpenSHMEM/SOS)
 * For license and copyright information, see the LICENSE and third-party-programs.txt
 * files in the top level directory of this distribution.
 */

/* Wrappers to interface with PMI runtime */
#include "ishmem/config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "pmi.h"

#include "runtime.h"
#include "runtime_pmi.h"
#include "wrapper.h"
#include "uthash.h"

#define PMI_CHECK(call)                                                                            \
    do {                                                                                           \
        int pmi_err = call;                                                                        \
        if (pmi_err != PMI_SUCCESS) {                                                              \
            fprintf(stderr, "PMI FAIL: call = '%s' result = '%d'\n", #call, pmi_err);              \
            ret = pmi_err;                                                                         \
        }                                                                                          \
    } while (0)

ishmemi_runtime_pmi::ishmemi_runtime_pmi(bool initialize_runtime)
{
    int ret = 0, initialized = 0;

    /* Setup PMI dlsym links */
    ret = ishmemi_pmi_wrappers::init_wrappers();

    /* Initialize the runtime if requested */
    if (initialize_runtime) {
        PMI_CHECK(ishmemi_pmi_wrappers::Init(&initialized));
        this->initialized = true;
    }

    PMI_CHECK(ishmemi_pmi_wrappers::Initialized(&initialized));

    if (!initialized) {
        PMI_CHECK(ishmemi_pmi_wrappers::Init(&initialized));
        this->initialized = true;
    }

    PMI_CHECK(ishmemi_pmi_wrappers::Get_rank(&this->rank));
    PMI_CHECK(ishmemi_pmi_wrappers::Get_size(&this->size));

    if (size > 1) {
        PMI_CHECK(ishmemi_pmi_wrappers::KVS_Get_name_length_max(&this->max_name_len));

        this->kvs_name = (char *) ::malloc(this->max_name_len);
        ISHMEM_CHECK_GOTO_MSG(this->kvs_name == nullptr, fn_fail,
                              "Allocation of kvs_name failed\n");

        PMI_CHECK(ishmemi_pmi_wrappers::KVS_Get_key_length_max(&this->max_key_len));
        PMI_CHECK(ishmemi_pmi_wrappers::KVS_Get_value_length_max(&this->max_val_len));
        PMI_CHECK(ishmemi_pmi_wrappers::KVS_Get_my_name(kvs_name, max_name_len));

        /* TODO: Fix later for XPMEM */
        /*if (enable_node_ranks) {
            location_array = (int *) ::malloc(sizeof(int) * size);
            if (nullptr == location_array) return 10;
        }*/
    } else {
        /* Use a local KVS for singleton runs */
        this->max_key_len = this->SINGLETON_KEY_LEN;
        this->max_val_len = this->SINGLETON_VAL_LEN;
        this->kvs_name = nullptr;
        this->max_name_len = 0;
    }

    this->kvs_key = (char *) ::malloc(this->max_key_len);
    ISHMEM_CHECK_GOTO_MSG(this->kvs_key == nullptr, fn_fail, "Allocation of kvs_key failed\n");

    this->kvs_value = (char *) ::malloc(this->max_val_len);
    ISHMEM_CHECK_GOTO_MSG(this->kvs_value == nullptr, fn_fail, "Allocation of kvs_value failed\n");

    /* Initialize the function pointer table */
    this->funcptr_init();

fn_fail:
    return;
}

ishmemi_runtime_pmi::~ishmemi_runtime_pmi(void)
{
    int ret = 0;

    /* Cleanup the internal runtime info */
    ::free(this->location_array);
    ::free(this->kvs_name);
    ::free(this->kvs_key);
    ::free(this->kvs_value);

    /* Finalize the runtime if necessary */
    if (this->initialized) {
        PMI_CHECK(ishmemi_pmi_wrappers::Finalize());
        this->initialized = false;
    }

    /* Cleanup the function pointer table */
    this->funcptr_fini();
}

void ishmemi_runtime_pmi::heap_create(void *base, size_t size)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
}

/* Query APIs */
int ishmemi_runtime_pmi::get_rank(void)
{
    return this->rank;
}

int ishmemi_runtime_pmi::get_size(void)
{
    return this->size;
}

int ishmemi_runtime_pmi::get_node_rank(int pe)
{
    /* TODO - Fix implementation */
    if (pe >= size || pe < 0) {
        std::cout << "[ERROR] Wrong PE value " << pe << std::endl;
        this->abort(1, "Wrong PE value");
    }

    if (size == 1) {
        return 0;
    } else {
        return location_array[pe];
    }
}

int ishmemi_runtime_pmi::get_node_size(void)
{
    /* TODO - Fix implementation */
    if (size == 1) {
        return 1;
    } else {
        return node_size;
    }
}

bool ishmemi_runtime_pmi::is_local(int pe)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
    return false;
}

bool ishmemi_runtime_pmi::is_symmetric_address(const void *addr)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
    return false;
}

/* Memory APIs */
void *ishmemi_runtime_pmi::malloc(size_t size)
{
    return ::malloc(size);
}

void *ishmemi_runtime_pmi::calloc(size_t num, size_t size)
{
    return ::calloc(num, size);
}

void ishmemi_runtime_pmi::free(void *ptr)
{
    ::free(ptr);
}

/* Team APIs */
int ishmemi_runtime_pmi::team_sync(ishmemi_runtime_team_t team)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
    return -1;
}

int ishmemi_runtime_pmi::team_predefined_set(ishmemi_runtime_team_t *team,
                                             ishmemi_runtime_team_predefined_t predefined_team_name,
                                             int expected_team_size, int expected_world_pe,
                                             int expected_team_pe)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
    return -1;
}

int ishmemi_runtime_pmi::team_split_strided(ishmemi_runtime_team_t parent_team, int PE_start,
                                            int PE_stride, int PE_size,
                                            const ishmemi_runtime_team_config_t *config,
                                            long config_mask, ishmemi_runtime_team_t *new_team)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
    return -1;
}

void ishmemi_runtime_pmi::team_destroy(ishmemi_runtime_team_t team)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
    return -1;
}

/* Operation APIs */
void ishmemi_runtime_pmi::abort(int exit_code, const char msg[])
{
    int ret = 0;

    if (size == 1) {
        fprintf(stderr, "%s\n", msg);
        exit(exit_code);
    }

    PMI_CHECK(ishmemi_pmi_wrappers::Abort(exit_code, msg));

    /* ishmemi_pmi_wrappers::Abort should not return */
    ::abort();
}

int ishmemi_runtime_pmi::get_kvs(int pe, char *key, void *value, size_t valuelen)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
    return -1;
}

int ishmemi_runtime_pmi::uchar_and_reduce(ishmemi_runtime_team_t team, unsigned char *dest,
                                          const unsigned char *source, size_t nreduce)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
    return -1;
}

int ishmemi_runtime_pmi::int_max_reduce(ishmemi_runtime_team_t team, int *dest, const int *source,
                                        size_t nreduce)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
    return -1;
}

void ishmemi_runtime_pmi::bcast(void *buf, size_t count, int root)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
}

void ishmemi_runtime_pmi::node_bcast(void *buf, size_t count, int root)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
}

void ishmemi_runtime_pmi::fcollect(void *dst, void *src, size_t count)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
}

void ishmemi_runtime_pmi::node_fcollect(void *dst, void *src, size_t count)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
}

void ishmemi_runtime_pmi::barrier_all(void)
{
    int ret = 0;
    PMI_CHECK(ishmemi_pmi_wrappers::Barrier());
}

void ishmemi_runtime_pmi::node_barrier(void)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
}

void ishmemi_runtime_pmi::fence(void)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
}

void ishmemi_runtime_pmi::quiet(void)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
}

void ishmemi_runtime_pmi::sync(void)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
}

void ishmemi_runtime_pmi::progress(void) {}

/* Private functions */
void ishmemi_runtime_pmi::funcptr_init(void)
{
    proxy_funcs = (ishmemi_runtime_proxy_func_t **) ::malloc(
        sizeof(ishmemi_runtime_proxy_func_t *) * ISHMEMI_OP_END);
    ISHMEM_CHECK_GOTO_MSG(proxy_funcs == nullptr, fn_exit, "Allocation of proxy_funcs failed\n");

    /* Initialize every function with the "unsupported op" function */
    /* Don't include KILL operation - it is the same for all backends currently */
    for (int i = 0; i < ISHMEMI_OP_END - 1; ++i) {
        for (int j = 0; j < ishmemi_runtime_type::proxy_func_num_types; ++j) {
            proxy_funcs[i][j] = ishmemi_runtime_type::unsupported;
        }
    }

fn_exit:
    return;
}

void ishmemi_runtime_pmi::funcptr_fini(void)
{
    for (int i = 0; i < ISHMEMI_OP_END; ++i) {
        for (int j = 0; j < ishmemi_runtime_type::proxy_func_num_types; ++j) {
            proxy_funcs[i][j] = &ishmemi_runtime_type::unsupported;
        }
        ISHMEMI_FREE(::free, proxy_funcs[i]);
    }
    ISHMEMI_FREE(::free, proxy_funcs);
}
