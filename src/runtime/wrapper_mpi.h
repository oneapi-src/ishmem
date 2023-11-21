/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_WRAPPER_MPI_H
#define ISHMEM_RUNTIME_WRAPPER_MPI_H

#include <mpi.h>

extern int (*MPI_WRAPPER_Init)(int *, char ***);
extern int (*MPI_WRAPPER_Finalize)(void);
extern int (*MPI_WRAPPER_Abort)(MPI_Comm, int);
extern int (*MPI_WRAPPER_Barrier)(MPI_Comm);
extern int (*MPI_WRAPPER_Bcast)(void *, int, MPI_Datatype, int, MPI_Comm);
extern int (*MPI_WRAPPER_Comm_group)(MPI_Comm, MPI_Group *);
extern int (*MPI_WRAPPER_Comm_rank)(MPI_Comm, int *);
extern int (*MPI_WRAPPER_Comm_size)(MPI_Comm, int *);
extern int (*MPI_WRAPPER_Comm_split_type)(MPI_Comm, int, int, MPI_Info, MPI_Comm *);
extern int (*MPI_WRAPPER_Comm_free)(MPI_Comm *);
extern int (*MPI_WRAPPER_Group_translate_ranks)(MPI_Group, int, const int[], MPI_Group, int[]);
extern int (*MPI_WRAPPER_Group_free)(MPI_Group *);

int ishmemi_mpi_wrapper_init();

#endif
