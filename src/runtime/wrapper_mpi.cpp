/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/err.h"
#include "wrapper.h"
#include "wrapper_mpi.h"
#include "ishmem/env_utils.h"
#include <mpi.h>
#include <dlfcn.h>

namespace ishmemi_mpi_wrappers {
    static bool initialized = false;

    /* Runtime */
    int (*Init)(int *, char ***);
    int (*Init_thread)(int *, char ***, int, int *);
    int (*Finalize)(void);
    int (*Abort)(MPI_Comm, int);
    int (*Comm_group)(MPI_Comm, MPI_Group *);
    int (*Comm_rank)(MPI_Comm, int *);
    int (*Comm_size)(MPI_Comm, int *);
    int (*Comm_split)(MPI_Comm, int, int, MPI_Comm *);
    int (*Comm_split_type)(MPI_Comm, int, int, MPI_Info, MPI_Comm *);
    int (*Comm_dup)(MPI_Comm, MPI_Comm *);
    int (*Comm_free)(MPI_Comm *);
    int (*Group_translate_ranks)(MPI_Group, int, const int[], MPI_Group, int[]);
    int (*Group_free)(MPI_Group *);

    /* Info hints */
    int (*Info_create)(MPI_Info *);
    int (*Info_set)(MPI_Info, const char *, const char *);
    int (*Info_free)(MPI_Info *);

    /* Window management */
    int (*Win_create)(void *, MPI_Aint, int, MPI_Info, MPI_Comm, MPI_Win *);
    int (*Win_lock_all)(int, MPI_Win);
    int (*Win_unlock_all)(MPI_Win);
    int (*Win_lock)(int, int, int, MPI_Win);
    int (*Win_unlock)(int, MPI_Win);
    int (*Win_flush_local)(int, MPI_Win);
    int (*Win_flush_local_all)(MPI_Win);
    int (*Win_flush_all)(MPI_Win);
    int (*Win_sync)(MPI_Win);
    int (*Win_free)(MPI_Win *);

    /* Datatypes */
    int (*Type_vector)(int, int, int, MPI_Datatype, MPI_Datatype *);
    int (*Type_create_resized)(MPI_Datatype, MPI_Aint, MPI_Aint, MPI_Datatype *);
    int (*Type_commit)(MPI_Datatype *);
    int (*Type_free)(MPI_Datatype *);

    /* RMA */
    int (*Put)(const void *, int, MPI_Datatype, int, MPI_Aint, int, MPI_Datatype, MPI_Win);
    int (*Get)(void *, int, MPI_Datatype, int, MPI_Aint, int, MPI_Datatype, MPI_Win);
    int (*Accumulate)(const void *, int, MPI_Datatype, int, MPI_Aint, int, MPI_Datatype, MPI_Op,
                      MPI_Win);
    int (*Fetch_and_op)(const void *, void *, MPI_Datatype, int, MPI_Aint, MPI_Op, MPI_Win);
    int (*Compare_and_swap)(const void *, const void *, void *, MPI_Datatype, int, MPI_Aint,
                            MPI_Win);
    int (*Iprobe)(int, int, MPI_Comm, int *, MPI_Status *);

    /* Collectives */
    int (*Allgather)(const void *, int, MPI_Datatype, void *, int, MPI_Datatype, MPI_Comm);
    int (*Allgatherv)(const void *, int, MPI_Datatype, void *, const int[], const int[],
                      MPI_Datatype, MPI_Comm);
    int (*Allreduce)(const void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm);
    int (*Scan)(const void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm);
    int (*Exscan)(const void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm);
    int (*Alltoall)(const void *, int, MPI_Datatype, void *, int, MPI_Datatype, MPI_Comm);
    int (*Barrier)(MPI_Comm);
    int (*Bcast)(void *, int, MPI_Datatype, int, MPI_Comm);

    /* dl handle */
    void *mpi_handle = nullptr;
    std::vector<void **> wrapper_list;

    int fini_wrappers(void)
    {
        int ret = 0;
        for (auto p : wrapper_list)
            *p = nullptr;
        if (mpi_handle != nullptr) {
            ret = dlclose(mpi_handle);
            mpi_handle = nullptr;
            ISHMEM_CHECK_GOTO_MSG(ret, fn_exit, "dlclose failed %s\n", dlerror());
        }
        return (0);
    fn_exit:
        return (1);
    }

    int init_wrappers(void)
    {
        int ret = 0;

        const char *mpi_libname = ishmemi_params.MPI_LIB_NAME.c_str();

        /* don't initialize twice */
        if (initialized) goto fn_exit;
        initialized = true;

        mpi_handle = dlopen(mpi_libname, RTLD_NOW | RTLD_GLOBAL);

        if (mpi_handle == nullptr) {
            RAISE_ERROR_MSG("could not find mpi library '%s' in environment\n", mpi_libname);
        }

        /* Runtime */
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Init);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Init_thread);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Finalize);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Abort);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Comm_group);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Comm_rank);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Comm_size);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Comm_split);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Comm_split_type);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Comm_dup);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Comm_free);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Group_translate_ranks);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Group_free);

        /* Info hints */
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Info_create);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Info_set);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Info_free);

        /* Window Management */
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Win_create);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Win_lock_all);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Win_unlock_all);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Win_lock);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Win_unlock);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Win_flush_local);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Win_flush_local_all);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Win_flush_all);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Win_sync);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Win_free);

        /* Datatypes */
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Type_vector);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Type_create_resized);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Type_commit);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Type_free);

        /* RMA */
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Put);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Get);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Accumulate);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Fetch_and_op);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Compare_and_swap);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Iprobe);

        /* Collectives */
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Allgather);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Allgatherv);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Allreduce);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Scan);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Exscan);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Alltoall);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Barrier);
        ISHMEMI_LINK_SYMBOL(mpi_handle, MPI, Bcast);

    fn_exit:
        return ret;
    }
}  // namespace ishmemi_mpi_wrappers
