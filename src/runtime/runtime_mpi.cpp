/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* Wrappers to interface with MPI runtime */
#include "ishmem_config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "runtime.h"
#include "wrapper.h"

#define MPI_CHECK(call)                                                                            \
    do {                                                                                           \
        int mpi_err = call;                                                                        \
        if (mpi_err != MPI_SUCCESS) {                                                              \
            fprintf(stderr, "MPI FAIL: call = '%s' result = '%d'\n", #call, mpi_err);              \
            ret = mpi_err;                                                                         \
        }                                                                                          \
    } while (0)

static int rank = -1;
static int size = 0, node_size = 0;
static bool initialized_mpi = false;
static MPI_Comm node_comm;
static MPI_Group group, node_group;

int ishmemi_runtime_mpi_funcptr_fini();

int ishmemi_runtime_mpi_fini(void)
{
    int ret = 0;

    MPI_CHECK(MPI_WRAPPER_Group_free(&group));
    MPI_CHECK(MPI_WRAPPER_Group_free(&node_group));
    MPI_CHECK(MPI_WRAPPER_Comm_free(&node_comm));

    if (initialized_mpi) {
        MPI_CHECK(MPI_WRAPPER_Finalize());
        initialized_mpi = false;
    }

    ishmemi_runtime_mpi_funcptr_fini(); /* free function tables */
    ishmemi_mpi_wrapper_fini();         /* close shared library */

    return ret;
}

void ishmemi_runtime_mpi_abort(int exit_code, const char msg[])
{
    std::cerr << "[ABORT] " << msg << std::endl;
    MPI_WRAPPER_Abort(MPI_COMM_WORLD, exit_code);
}

int ishmemi_runtime_mpi_get_rank(void)
{
    return rank;
}

int ishmemi_runtime_mpi_get_size(void)
{
    return size;
}

int ishmemi_runtime_mpi_get_node_rank(int pe)
{
    int ret = 0;
    int node_pe;

    if (size == 1) {
        node_pe = 0;
    } else {
        MPI_CHECK(MPI_WRAPPER_Group_translate_ranks(group, 1, &pe, node_group, &node_pe));
        if (ret != 0) {
            node_pe = -1;
        }
    }

    return node_pe;
}

int ishmemi_runtime_mpi_get_node_size(void)
{
    return node_size;
}

void ishmemi_runtime_mpi_barrier(void)
{
    MPI_WRAPPER_Barrier(MPI_COMM_WORLD);
}

int ishmemi_runtime_mpi_team_sync(ishmemi_runtime_team_t team)
{
    /* TODO: Implement */
    return -1;
}

int ishmemi_runtime_mpi_team_predefined_set(ishmemi_runtime_team_t *team,
                                            ishmemi_runtime_team_predefined_t predefined_team_name,
                                            int expected_team_size, int expected_world_pe,
                                            int expected_team_pe)
{
    return -1;
}

void ishmemi_runtime_mpi_node_barrier(void)
{
    MPI_WRAPPER_Barrier(node_comm);
}

void ishmemi_runtime_mpi_bcast(void *buf, size_t count, int root)
{
    MPI_WRAPPER_Bcast(buf, count, MPI_BYTE, root, MPI_COMM_WORLD);
}

void ishmemi_runtime_mpi_node_bcast(void *buf, size_t count, int root)
{
    MPI_WRAPPER_Bcast(buf, count, MPI_BYTE, root, node_comm);
}

/* Barrier operation */
void ishmemi_runtime_mpi_barrier_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ishmemi_runtime_mpi_barrier();
}

bool ishmemi_runtime_mpi_is_local(int pe)
{
    return (ishmemi_runtime_mpi_get_node_rank(pe) != -1);
}

void ishmemi_runtime_mpi_heap_create(void *base, size_t size)
{
    /* TODO: Implement */
}

int ishmemi_runtime_mpi_team_split_strided(ishmemi_runtime_team_t parent_team, int PE_start,
                                           int PE_stride, int PE_size,
                                           const ishmemi_runtime_team_config_t *config,
                                           long config_mask, ishmemi_runtime_team_t *new_team)
{
    /* TODO: Implement */
    return -1;
}

int ishmemi_runtime_mpi_uchar_and_reduce(ishmemi_runtime_team_t team, unsigned char *dest,
                                         const unsigned char *source, size_t nreduce)
{
    /* TODO: Implement */
    return -1;
}

int ishmemi_runtime_mpi_int_max_reduce(ishmemi_runtime_team_t team, int *dest, const int *source,
                                       size_t nreduce)
{
    /* TODO: Implement */
    return -1;
}

void ishmemi_runtime_mpi_unsupported(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEM_ERROR_MSG("Encountered type '%s' unsupported for operation '%s'\n",
                     ishmemi_type_str[msg->type], ishmemi_op_str[msg->op]);
    ishmemi_cpu_info->proxy_state = EXIT;
}

void ishmemi_runtime_mpi_funcptr_init()
{
    ishmemi_proxy_funcs = (ishmemi_runtime_proxy_func_t **) malloc(
        sizeof(ishmemi_runtime_proxy_func_t *) * ISHMEMI_OP_END);
    ISHMEM_CHECK_GOTO_MSG(ishmemi_proxy_funcs == nullptr, fn_exit,
                          "Allocation of ishmemi_proxy_funcs failed\n");

    /* Initialize every function with the "unsupported op" function */
    /* Note: KILL operation is covered inside the proxy directly - it is the same for all backends
     * currently */
    for (int i = 0; i < ISHMEMI_OP_END; ++i) {
        ishmemi_proxy_funcs[i] = (ishmemi_runtime_proxy_func_t *) malloc(
            sizeof(ishmemi_runtime_proxy_func_t) * ishmemi_runtime_proxy_func_num_types);
        for (int j = 0; j < ishmemi_runtime_proxy_func_num_types; ++j) {
            ishmemi_proxy_funcs[i][j] = ishmemi_runtime_mpi_unsupported;
        }
    }

    /* Fill in the supported functions */
    ishmemi_proxy_funcs[BARRIER][0] = ishmemi_runtime_mpi_barrier_all;

fn_exit:
    return;
}

int ishmemi_runtime_mpi_funcptr_fini()
{
    for (int i = 0; i < ISHMEMI_OP_END; ++i) {
        for (int j = 0; j < ishmemi_runtime_proxy_func_num_types; ++j) {
            ishmemi_proxy_funcs[i][j] = ishmemi_runtime_mpi_unsupported;
        }
        ISHMEMI_FREE(free, ishmemi_proxy_funcs[i]);
    }
    ISHMEMI_FREE(free, ishmemi_proxy_funcs);
    return (0);
}

int ishmemi_runtime_mpi_init(bool initialize_runtime)
{
    int ret = 0;

    /* Setup MPI dlsym links */
    ret = ishmemi_mpi_wrapper_init();
    if (ret != 0) return ret;

    if (initialize_runtime) {
        MPI_CHECK(MPI_WRAPPER_Init(nullptr, nullptr));
        initialized_mpi = true;
    }

    /* Setup internal runtime info */
    MPI_CHECK(MPI_WRAPPER_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_WRAPPER_Comm_size(MPI_COMM_WORLD, &size));
    MPI_CHECK(MPI_WRAPPER_Comm_group(MPI_COMM_WORLD, &group));

    if (size > 1) {
        MPI_CHECK(MPI_WRAPPER_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                              MPI_INFO_NULL, &node_comm));

        if (node_comm == MPI_COMM_NULL) {
            RAISE_ERROR_MSG("MPI FAILURE: node_comm was not correctly created\n");
        }

        MPI_CHECK(MPI_WRAPPER_Comm_size(node_comm, &node_size));
        MPI_CHECK(MPI_WRAPPER_Comm_group(node_comm, &node_group));

        if (node_size > size) {
            RAISE_ERROR_MSG("MPI FAILURE: node_comm was not correctly created\n");
        }

    } else {
        node_size = size;
    }

    /* Setup runtime function pointers */
    ishmemi_runtime_fini = ishmemi_runtime_mpi_fini;
    ishmemi_runtime_abort = ishmemi_runtime_mpi_abort;
    ishmemi_runtime_get_rank = ishmemi_runtime_mpi_get_rank;
    ishmemi_runtime_get_size = ishmemi_runtime_mpi_get_size;
    ishmemi_runtime_get_node_rank = ishmemi_runtime_mpi_get_node_rank;
    ishmemi_runtime_get_node_size = ishmemi_runtime_mpi_get_node_size;
    ishmemi_runtime_barrier_all = ishmemi_runtime_mpi_barrier;
    ishmemi_runtime_node_barrier = ishmemi_runtime_mpi_node_barrier;
    ishmemi_runtime_bcast = ishmemi_runtime_mpi_bcast;
    ishmemi_runtime_node_bcast = ishmemi_runtime_mpi_node_bcast;
    ishmemi_runtime_is_local = ishmemi_runtime_mpi_is_local;
    ishmemi_runtime_malloc = malloc;
    ishmemi_runtime_calloc = calloc;
    ishmemi_runtime_free = free;
    ishmemi_runtime_team_split_strided = ishmemi_runtime_mpi_team_split_strided;
    ishmemi_runtime_team_sync = ishmemi_runtime_mpi_team_sync;
    ishmemi_runtime_team_predefined_set = ishmemi_runtime_mpi_team_predefined_set;
    ishmemi_runtime_uchar_and_reduce = ishmemi_runtime_mpi_uchar_and_reduce;
    ishmemi_runtime_int_max_reduce = ishmemi_runtime_mpi_int_max_reduce;

    ishmemi_runtime_mpi_funcptr_init();

    return ret;
}
