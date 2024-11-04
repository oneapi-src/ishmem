/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_WRAPPER_MPI_H
#define ISHMEM_RUNTIME_WRAPPER_MPI_H

#include <mpi.h>

namespace ishmemi_mpi_wrappers {
    int init_wrappers(void);
    int fini_wrappers(void);

    /* Runtime */
    extern int (*Init)(int *, char ***);
    extern int (*Init_thread)(int *, char ***, int, int *);
    extern int (*Finalize)(void);
    extern int (*Abort)(MPI_Comm, int);
    extern int (*Comm_group)(MPI_Comm, MPI_Group *);
    extern int (*Comm_rank)(MPI_Comm, int *);
    extern int (*Comm_size)(MPI_Comm, int *);
    extern int (*Comm_split)(MPI_Comm, int, int, MPI_Comm *);
    extern int (*Comm_split_type)(MPI_Comm, int, int, MPI_Info, MPI_Comm *);
    extern int (*Comm_dup)(MPI_Comm, MPI_Comm *);
    extern int (*Comm_free)(MPI_Comm *);
    extern int (*Group_translate_ranks)(MPI_Group, int, const int[], MPI_Group, int[]);
    extern int (*Group_free)(MPI_Group *);

    /* Info hints */
    extern int (*Info_create)(MPI_Info *);
    extern int (*Info_set)(MPI_Info, const char *, const char *);
    extern int (*Info_free)(MPI_Info *);

    /* Window management */
    extern int (*Win_create)(void *, MPI_Aint, int, MPI_Info, MPI_Comm, MPI_Win *);
    extern int (*Win_lock_all)(int, MPI_Win);
    extern int (*Win_unlock_all)(MPI_Win);
    extern int (*Win_lock)(int, int, int, MPI_Win);
    extern int (*Win_unlock)(int, MPI_Win);
    extern int (*Win_flush_local)(int, MPI_Win);
    extern int (*Win_flush_local_all)(MPI_Win);
    extern int (*Win_flush_all)(MPI_Win);
    extern int (*Win_sync)(MPI_Win);
    extern int (*Win_free)(MPI_Win *);

    /* Datatypes */
    extern int (*Type_vector)(int, int, int, MPI_Datatype, MPI_Datatype *);
    extern int (*Type_create_resized)(MPI_Datatype, MPI_Aint, MPI_Aint, MPI_Datatype *);
    extern int (*Type_commit)(MPI_Datatype *);
    extern int (*Type_free)(MPI_Datatype *);

    /* RMA */
    extern int (*Put)(const void *, int, MPI_Datatype, int, MPI_Aint, int, MPI_Datatype, MPI_Win);
    extern int (*Get)(void *, int, MPI_Datatype, int, MPI_Aint, int, MPI_Datatype, MPI_Win);
    extern int (*Accumulate)(const void *, int, MPI_Datatype, int, MPI_Aint, int, MPI_Datatype,
                             MPI_Op, MPI_Win);
    extern int (*Fetch_and_op)(const void *, void *, MPI_Datatype, int, MPI_Aint, MPI_Op, MPI_Win);
    extern int (*Compare_and_swap)(const void *, const void *, void *, MPI_Datatype, int, MPI_Aint,
                                   MPI_Win);
    extern int (*Iprobe)(int, int, MPI_Comm, int *, MPI_Status *);

    /* Collectives */
    extern int (*Allgather)(const void *, int, MPI_Datatype, void *, int, MPI_Datatype, MPI_Comm);
    extern int (*Allgatherv)(const void *, int, MPI_Datatype, void *, const int[], const int[],
                             MPI_Datatype, MPI_Comm);
    extern int (*Allreduce)(const void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm);
    extern int (*Alltoall)(const void *, int, MPI_Datatype, void *, int, MPI_Datatype, MPI_Comm);
    extern int (*Barrier)(MPI_Comm);
    extern int (*Bcast)(void *, int, MPI_Datatype, int, MPI_Comm);
}  // namespace ishmemi_mpi_wrappers

#endif
