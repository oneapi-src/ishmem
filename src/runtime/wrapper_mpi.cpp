/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "wrapper.h"
#include "env_utils.h"
#include "internal.h"
#include <mpi.h>
#include <dlfcn.h>

int (*MPI_WRAPPER_Init)(int *, char ***);
int (*MPI_WRAPPER_Finalize)(void);
int (*MPI_WRAPPER_Abort)(MPI_Comm, int);
int (*MPI_WRAPPER_Barrier)(MPI_Comm);
int (*MPI_WRAPPER_Bcast)(void *, int, MPI_Datatype, int, MPI_Comm);
int (*MPI_WRAPPER_Comm_group)(MPI_Comm, MPI_Group *);
int (*MPI_WRAPPER_Comm_rank)(MPI_Comm, int *);
int (*MPI_WRAPPER_Comm_size)(MPI_Comm, int *);
int (*MPI_WRAPPER_Comm_split_type)(MPI_Comm, int, int, MPI_Info, MPI_Comm *);
int (*MPI_WRAPPER_Comm_free)(MPI_Comm *);
int (*MPI_WRAPPER_Group_translate_ranks)(MPI_Group, int, const int[], MPI_Group, int[]);
int (*MPI_WRAPPER_Group_free)(MPI_Group *);

int ishmemi_mpi_wrapper_init()
{
    int ret = 0;
    void *mpi_handle = nullptr;

    const char *mpi_libname = ishmemi_params.MPI_LIB_NAME.c_str();

    mpi_handle = dlopen(mpi_libname, RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);

    if (mpi_handle == nullptr) {
        RAISE_ERROR_MSG("could not find mpi library '%s' in environment\n", mpi_libname);
    }

    ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Init);
    ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Finalize);
    ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Abort);
    ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Barrier);
    ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Bcast);
    ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Comm_group);
    ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Comm_rank);
    ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Comm_size);
    ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Comm_split_type);
    ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Comm_free);
    ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Group_translate_ranks);
    ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Group_free);

fn_exit:
    return ret;
fn_fail:
    goto fn_exit;
}
