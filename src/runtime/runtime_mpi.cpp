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
void ishmemi_mpi_type_barrier()
{
    ishmemi_runtime_mpi_barrier();
}

bool ishmemi_runtime_mpi_is_local(int pe)
{
    return (ishmemi_runtime_mpi_get_node_rank(pe) != -1);
}

#error "code wrong"
void ishmemi_runtime_mpi_unsupported()
{
    ISHMEM_ERROR_MSG("Encountered type '%s' unsupported for operation '%s'\n",
                     ishmemi_type_str[info->req.type], ishmemi_op_str[info->req.op]);
    info->proxy_state = EXIT;
}

void ishmemi_runtime_mpi_funcptr_init()
{
    ishmemi_proxy_funcs = (ishmemi_runtime_proxy_func_t *) malloc(
        sizeof(ishmemi_runtime_proxy_func_t) * ISHMEMI_OP_END);

    /* Initialize every function with the "unsupported op" function */
    /* Don't include KILL operation - it is the same for all backends currently */
    for (int i = 0; i < ISHMEMI_OP_END - 1; ++i) {
        ishmemi_proxy_funcs[i] = ishmemi_runtime_mpi_unsupported;
    }

    ishmemi_proxy_funcs[BARRIER] = ishmemi_mpi_type_barrier;
}

int ishmemi_runtime_mpi_init(bool initialize_runtime)
{
    int ret = 0;

    /* Setup MPI dlsym links */
    ret = ishmemi_mpi_wrapper_init();

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
    ishmemi_runtime_barrier = ishmemi_runtime_mpi_barrier;
    ishmemi_runtime_node_barrier = ishmemi_runtime_mpi_node_barrier;
    ishmemi_runtime_bcast = ishmemi_runtime_mpi_bcast;
    ishmemi_runtime_node_bcast = ishmemi_runtime_mpi_node_bcast;
    ishmemi_runtime_is_local = ishmemi_runtime_mpi_is_local;
    ishmemi_runtime_malloc = malloc;
    ishmemi_runtime_calloc = calloc;
    ishmemi_runtime_free = free;

    ishmemi_runtime_mpi_funcptr_init();

    return ret;
}
