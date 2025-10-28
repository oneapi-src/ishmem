/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* Wrappers to interface with MPI runtime */
#include "ishmem/config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "accelerator.h"
#include "runtime.h"
#include "runtime_mpi.h"
#include "wrapper.h"

#define MPI_CHECK_GOTO(label, call)                                                                \
    do {                                                                                           \
        int mpi_err = call;                                                                        \
        if (mpi_err != MPI_SUCCESS) {                                                              \
            fprintf(stderr, "MPI FAIL: call = '%s' result = '%d'\n", #call, mpi_err);              \
            ret = mpi_err;                                                                         \
            goto label;                                                                            \
        }                                                                                          \
    } while (0)

#define MPI_CHECK(call)                                                                            \
    do {                                                                                           \
        int mpi_err = call;                                                                        \
        if (mpi_err != MPI_SUCCESS) {                                                              \
            fprintf(stderr, "MPI FAIL: call = '%s' result = '%d'\n", #call, mpi_err);              \
            ret = mpi_err;                                                                         \
        }                                                                                          \
    } while (0)

#define ISHMEMI_RUNTIME_MPI_COMM_HELPER_IMPL(OP, TEAM_PTR)                                         \
    MPI_Win win __attribute__((unused)) = ishmemi_runtime_mpi::global_win;                         \
    MPI_Comm comm __attribute__((unused)) =                                                        \
        ishmemi_runtime_mpi::teams[ishmemi_runtime_mpi::world_team].comm;                          \
    int rank __attribute__((unused)) =                                                             \
        ishmemi_runtime_mpi::teams[ishmemi_runtime_mpi::world_team].rank;                          \
    int size __attribute__((unused)) =                                                             \
        ishmemi_runtime_mpi::teams[ishmemi_runtime_mpi::world_team].size;                          \
    if constexpr (ishmemi_op_uses_team<OP>()) {                                                    \
        comm = ishmemi_runtime_mpi::teams[TEAM_PTR->runtime_team.mpi].comm;                        \
        rank = ishmemi_runtime_mpi::teams[TEAM_PTR->runtime_team.mpi].rank;                        \
        size = ishmemi_runtime_mpi::teams[TEAM_PTR->runtime_team.mpi].size;                        \
    }

#define ISHMEMI_RUNTIME_MPI_COMM_HELPER(OP)                                                        \
    ISHMEMI_RUNTIME_MPI_COMM_HELPER_IMPL(OP, (&ishmemi_cpu_info->team_host_pool[msg->team]))

#define ISHMEMI_RUNTIME_MPI_REQUEST_HELPER(T, OP)                                                  \
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, OP)                                                          \
    MPI_Datatype dt __attribute__((unused)) = get_datatype<T, OP>();                               \
    ISHMEMI_RUNTIME_MPI_COMM_HELPER_IMPL(OP, team_ptr)

#define ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, DISP)                                       \
    ISHMEMI_RUNTIME_MPI_REQUEST_HELPER(T, OP)                                                      \
    MPI_Aint disp __attribute__((unused)) =                                                        \
        CALC_DISP(DISP, ishmemi_runtime_mpi::global_win_base_addr);

#define CALC_DISP(target, base) (intptr_t) target - (ptrdiff_t) base

#define CONVERT_GPU_BUFFER(QUALIFIER, TYPE, var, size, constexpr_check)                            \
    QUALIFIER TYPE *var##_host = var;                                                              \
    bool var##_gpu = false;                                                                        \
    ze_ipc_mem_handle_t var##_handle = {};                                                         \
    if constexpr (constexpr_check) {                                                               \
        if (ISHMEMI_HOST_IN_HEAP(var)) {                                                           \
            var##_host = ISHMEMI_DEVICE_TO_MMAP_ADDR(TYPE, var);                                   \
        } else if ((var##_gpu == is_gpu_buffer(var)) && var##_gpu) {                               \
            var##_host = ishmemi_get_mmap_address(var, size, &var##_handle);                       \
        }                                                                                          \
    }

#define CLEANUP_GPU_BUFFER(TYPE, var, size, constexpr_check)                                       \
    if constexpr (constexpr_check) {                                                               \
        if (var##_gpu) {                                                                           \
            ret = ishmemi_close_mmap_address(var##_handle, (TYPE *) var##_host, size);             \
            ISHMEMI_CHECK_RESULT(ret, 0, fn_exit);                                                 \
        }                                                                                          \
    }

/* Runtime generic implementations */
namespace {
    template <typename T>
    static inline int compare(int cmp, T a, T b)
    {
        switch (cmp) {
            case ISHMEM_CMP_EQ:
                return ((a == b) ? 1 : 0);
            case ISHMEM_CMP_NE:
                return ((a != b) ? 1 : 0);
            case ISHMEM_CMP_GT:
                return ((a > b) ? 1 : 0);
            case ISHMEM_CMP_GE:
                return ((a >= b) ? 1 : 0);
            case ISHMEM_CMP_LT:
                return ((a < b) ? 1 : 0);
            case ISHMEM_CMP_LE:
                return ((a <= b) ? 1 : 0);
            default:
                return -1;
        }
    }

    template <typename T>
    static inline bool is_gpu_buffer(const T *ptr)
    {
        int ret;
        ze_memory_type_t type;

        if (ptr == nullptr) return false;

        ret = ishmemi_get_memory_type(ptr, &type);
        ISHMEM_CHECK_GOTO_MSG(ret == -1, fn_exit, "Failed to check memory type of pointer\n");

        return (type == ZE_MEMORY_TYPE_DEVICE);

    fn_exit:
        return true;
    }

    inline int barrier_impl(MPI_Comm comm, MPI_Win win)
    {
        int ret = 0;

        /* Ensure L0 operations are finished */
        ishmemi_level_zero_sync();

        /* Ensure all local RMA facilitated by MPI backend are finished */
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_flush_all(win));

        /* Syncronize the private and public windows */
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_sync(win));

        /* Synchronize with other PEs */
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Barrier(comm));

    fn_exit:
        return ret;
    }

    inline int fence_impl(MPI_Win win)
    {
        int ret = 0;

        /* Ensure L0 operations are finished */
        ishmemi_level_zero_sync();

        /* Ensure all local RMA facilitated by MPI backend are finished */
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_flush_all(win));

        /* Syncronize the private and public windows */
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_sync(win));

    fn_exit:
        return ret;
    }

    inline int quiet_impl(MPI_Win win)
    {
        return fence_impl(win);
    }

    inline int sync_impl(MPI_Comm comm, MPI_Win win)
    {
        int ret = 0;

        /* Syncronize the private and public windows */
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_sync(win));

        /* Synchronize with other PEs */
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Barrier(comm));

    fn_exit:
        return ret;
    }

    template <typename T>
    inline int test_multi_impl(T cmp_value, int cmp, MPI_Datatype dt, int rank, MPI_Aint start,
                               size_t nelems, const int *status, size_t *indices, size_t &complete,
                               MPI_Win win)
    {
        int ret = 0;
        MPI_Op op = MPI_NO_OP;
        MPI_Aint disp = start;
        T *results = (T *) ::calloc(nelems, sizeof(T));
        ISHMEM_CHECK_GOTO_MSG(results == nullptr, fn_fail, "Unable to allocate host memory\n");

        for (size_t i = 0; i < nelems; ++i) {
            if (status && status[i]) {
                disp = disp + (MPI_Aint) sizeof(T);
                continue;
            }

            MPI_CHECK_GOTO(fn_fail, ishmemi_mpi_wrappers::Fetch_and_op(nullptr, &results[i], dt,
                                                                       rank, disp, op, win));

            disp = disp + (MPI_Aint) sizeof(T);
        }

        MPI_CHECK_GOTO(fn_fail, ishmemi_mpi_wrappers::Win_flush_local(rank, win));

        for (size_t i = 0; i < nelems; ++i) {
            int tmp;
            if (status && status[i]) {
                continue;
            }

            tmp = compare(cmp, results[i], cmp_value);
            ISHMEM_CHECK_GOTO_MSG(tmp == -1, fn_fail, "Unknown or unsupported comparison op\n");
            if (tmp == 1) {
                indices[complete] = i;
                ++complete;
            }
        }

    fn_exit:
        ::free(results);
        return ret;
    fn_fail:
        ret = -1;
        goto fn_exit;
    }

    template <typename T>
    inline int test_multi_impl(const T *cmp_values, int cmp, MPI_Datatype dt, int rank,
                               MPI_Aint start, size_t nelems, const int *status, size_t *indices,
                               size_t &complete, MPI_Win win)
    {
        int ret = 0;
        MPI_Op op = MPI_NO_OP;
        MPI_Aint disp = start;
        T *results = (T *) ::calloc(nelems, sizeof(T));
        ISHMEM_CHECK_GOTO_MSG(results == nullptr, fn_fail, "Unable to allocate host memory\n");

        for (size_t i = 0; i < nelems; ++i) {
            if (status && status[i]) {
                disp = disp + (MPI_Aint) sizeof(T);
                continue;
            }

            MPI_CHECK_GOTO(fn_fail, ishmemi_mpi_wrappers::Fetch_and_op(nullptr, &results[i], dt,
                                                                       rank, disp, op, win));

            disp = disp + (MPI_Aint) sizeof(T);
        }

        MPI_CHECK_GOTO(fn_fail, ishmemi_mpi_wrappers::Win_flush_local(rank, win));

        for (size_t i = 0; i < nelems; ++i) {
            int tmp;
            if (status && status[i]) {
                continue;
            }

            tmp = compare(cmp, results[i], cmp_values[i]);
            ISHMEM_CHECK_GOTO_MSG(tmp == -1, fn_fail, "Unknown or unsupported comparison op\n");
            if (tmp == 1) {
                indices[complete] = i;
                ++complete;
            }
        }

    fn_exit:
        ::free(results);
        return ret;
    fn_fail:
        ret = -1;
        goto fn_exit;
    }

    template <typename T>
    inline int test_impl(T cmp_value, int cmp, MPI_Datatype dt, int rank, MPI_Aint disp,
                         MPI_Win win)
    {
        int ret = 0;
        MPI_Op op = MPI_NO_OP;
        T result;

        MPI_CHECK_GOTO(
            fn_fail, ishmemi_mpi_wrappers::Fetch_and_op(nullptr, &result, dt, rank, disp, op, win));

        MPI_CHECK_GOTO(fn_fail, ishmemi_mpi_wrappers::Win_flush_local(rank, win));

        ret = compare(cmp, result, cmp_value);
        ISHMEM_CHECK_GOTO_MSG(ret == -1, fn_fail, "Unknown or unsupported comparison op\n");

    fn_exit:
        return ret;
    fn_fail:
        ret = -1;
        goto fn_exit;
    }

    /* Some MPI routines may skip internal progress. This can be used to manually poll MPI progress
     * in those cases. */
    inline void force_progress(MPI_Comm comm)
    {
        int ret __attribute__((unused)) = 0;
        int iprobe_flag = 0;

        MPI_CHECK(ishmemi_mpi_wrappers::Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &iprobe_flag,
                                               MPI_STATUS_IGNORE));
    }

    int translate_rank(int pe, ishmemi_runtime_mpi::team_t source_team,
                       ishmemi_runtime_mpi::team_t dest_team)
    {
        int ret __attribute__((unused)) = 0;
        int node_pe;

        MPI_CHECK_GOTO(fn_fail, ishmemi_mpi_wrappers::Group_translate_ranks(
                                    source_team.group, 1, &pe, dest_team.group, &node_pe));
        return ((node_pe == MPI_UNDEFINED) ? -1 : node_pe);
    fn_fail:
        return -1;
    }

    void team_destroy_impl(int idx)
    {
        int ret __attribute__((unused)) = 0;
        std::map<ishmemi_runtime_mpi_types::team_t, ishmemi_runtime_mpi::team_t>::iterator it;

        ishmemi_runtime_mpi::team_t mpi_team = ishmemi_runtime_mpi::teams[idx];
        if (mpi_team.comm == MPI_COMM_WORLD || mpi_team.comm == MPI_COMM_SELF) {
            /* Only free the group */
            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Group_free(&mpi_team.group));
            return;
        }

        /* Otherwise free the communicator and the group */
        it = ishmemi_runtime_mpi::teams.find(idx);
        if (it != ishmemi_runtime_mpi::teams.end()) {
            mpi_team = it->second;

            if (mpi_team.comm != MPI_COMM_NULL) {
                MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_free(&mpi_team.comm));
            }

            if (mpi_team.group != MPI_GROUP_NULL) {
                MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Group_free(&mpi_team.group));
            }

            ishmemi_runtime_mpi::teams.erase(it);
        }

    fn_exit:
        return;
    }
}  // namespace

/* Proxy function implementations */
namespace impl {
    template <typename T, ishmemi_op_t OP>
    static constexpr MPI_Datatype get_datatype()
    {
        /* Floating-point types */
        if constexpr (ishmemi_op_floating_point_matters<OP>() && std::is_floating_point_v<T>) {
            if constexpr (std::is_same_v<T, float>) return MPI_FLOAT;
            else if constexpr (std::is_same_v<T, double>) return MPI_DOUBLE;
            else static_assert(false, "Unknown or unsupported type");
        }

        /* Signed types */
        if constexpr (ishmemi_op_sign_matters<OP>() && std::is_signed_v<T>) {
            if constexpr (sizeof(T) == sizeof(int8_t)) return MPI_INT8_T;
            else if constexpr (sizeof(T) == sizeof(int16_t)) return MPI_INT16_T;
            else if constexpr (sizeof(T) == sizeof(int32_t)) return MPI_INT32_T;
            else if constexpr (sizeof(T) == sizeof(int64_t)) return MPI_INT64_T;
            else if constexpr (sizeof(T) == sizeof(long long)) return MPI_LONG_LONG;
            else static_assert(false, "Unknown or unsupported type");
        }

        /* Unsigned types */
        if constexpr (sizeof(T) == sizeof(uint8_t)) return MPI_UINT8_T;
        else if constexpr (sizeof(T) == sizeof(uint16_t)) return MPI_UINT16_T;
        else if constexpr (sizeof(T) == sizeof(uint32_t)) return MPI_UINT32_T;
        else if constexpr (sizeof(T) == sizeof(uint64_t)) return MPI_UINT64_T;
        else if constexpr (sizeof(T) == sizeof(unsigned long long)) return MPI_UNSIGNED_LONG_LONG;
        else static_assert(false, "Unknown or unsupported type");
    }

    template <ishmemi_op_t OP>
    static constexpr MPI_Op get_reduction_op()
    {
        if constexpr (OP == AND_REDUCE) return MPI_BAND;
        else if constexpr (OP == OR_REDUCE) return MPI_BOR;
        else if constexpr (OP == XOR_REDUCE) return MPI_BXOR;
        else if constexpr (OP == MAX_REDUCE) return MPI_MAX;
        else if constexpr (OP == MIN_REDUCE) return MPI_MIN;
        else if constexpr (OP == SUM_REDUCE) return MPI_SUM;
        else if constexpr (OP == PROD_REDUCE) return MPI_PROD;
        else static_assert(false, "Unknown or unsupported reduction op");
    }

    template <ishmemi_op_t OP>
    static constexpr MPI_Op get_amo_op()
    {
        if constexpr (OP == AMO_FETCH) return MPI_NO_OP;
        else if constexpr (OP == AMO_SET) return MPI_REPLACE;
        else if constexpr (OP == AMO_SWAP) return MPI_REPLACE;
        else if constexpr (OP == AMO_FETCH_INC) return MPI_SUM;
        else if constexpr (OP == AMO_INC) return MPI_SUM;
        else if constexpr (OP == AMO_FETCH_ADD) return MPI_SUM;
        else if constexpr (OP == AMO_ADD) return MPI_SUM;
        else if constexpr (OP == AMO_FETCH_AND) return MPI_BAND;
        else if constexpr (OP == AMO_AND) return MPI_BAND;
        else if constexpr (OP == AMO_FETCH_OR) return MPI_BOR;
        else if constexpr (OP == AMO_OR) return MPI_BOR;
        else if constexpr (OP == AMO_FETCH_XOR) return MPI_BXOR;
        else if constexpr (OP == AMO_XOR) return MPI_BXOR;
        else if constexpr (OP == AMO_FETCH_NBI) return MPI_NO_OP;
        else if constexpr (OP == AMO_SWAP_NBI) return MPI_REPLACE;
        else if constexpr (OP == AMO_FETCH_INC_NBI) return MPI_SUM;
        else if constexpr (OP == AMO_FETCH_ADD_NBI) return MPI_SUM;
        else if constexpr (OP == AMO_FETCH_AND_NBI) return MPI_BAND;
        else if constexpr (OP == AMO_FETCH_OR_NBI) return MPI_BOR;
        else if constexpr (OP == AMO_FETCH_XOR_NBI) return MPI_BXOR;
        else static_assert(false, "Unknown or unsupported atomic op");
    }

    /* RMA */
    template <typename T, ishmemi_op_t OP, bool FLUSH = true, bool SIGNAL = false>
    static int put(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, dest);

        if constexpr (OP == P) {
            nelems = 1;
            src = &val;
        }

        if constexpr (OP == IBPUT) {
            nelems = nelems * bsize;
        }

        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Put(src, (int) nelems, dt, pe, disp,
                                                          (int) nelems, dt, win));
        if constexpr (SIGNAL) {
            MPI_Aint sig_disp = CALC_DISP(sig_addr, ishmemi_runtime_mpi::global_win_base_addr);
            MPI_Datatype sig_dt = MPI_UINT64_T;
            MPI_Op op = MPI_NO_OP;

            if (sig_op == ISHMEM_SIGNAL_SET) {
                op = MPI_REPLACE;
            } else if (sig_op == ISHMEM_SIGNAL_ADD) {
                op = MPI_SUM;
            }

            /* Blocking AMO case */
            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Accumulate(&signal, 1, sig_dt, pe,
                                                                     sig_disp, 1, sig_dt, op, win));
        }

        if constexpr (FLUSH) {
            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_flush_local(pe, win));
        }

    fn_exit:
        return ret;
    }

    template <typename T, ishmemi_op_t OP>
    static int iput_datatype(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, dest);

        if constexpr (OP == IPUT) {
            bsize = 1;
        }

        if (dst == (ptrdiff_t) bsize && sst == (ptrdiff_t) bsize) {
            put<T, OP>(msg, comp);
        } else {
            MPI_Datatype sdt = MPI_DATATYPE_NULL;
            MPI_Datatype ddt = MPI_DATATYPE_NULL;

            ishmemi_runtime_mpi::get_strided_dt(nelems, sst, bsize, 0, dt, &sdt);

            if (sst != dst) {
                ishmemi_runtime_mpi::get_strided_dt(nelems, dst, bsize, 0, dt, &ddt);
            } else {
                ddt = sdt;
            }

            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Put(src, 1, sdt, pe, disp, 1, ddt, win));
            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_flush_local(pe, win));
        }

    fn_exit:
        return ret;
    }

    template <typename T, ishmemi_op_t OP>
    static int iput_fallback(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, dest);

        if constexpr (OP == IPUT) {
            bsize = 1;
        }

        if (dst == (ptrdiff_t) bsize && sst == (ptrdiff_t) bsize) {
            put<T, OP>(msg, comp);
        } else {
            T *src_offset = (T *) src;
            MPI_Aint disp_offset = disp;

            for (size_t i = 0; i < nelems; ++i) {
                MPI_CHECK_GOTO(
                    fn_exit, ishmemi_mpi_wrappers::Put(src_offset, (int) bsize, dt, pe, disp_offset,
                                                       (int) bsize, dt, win));
                src_offset = pointer_offset<T>(src_offset, (unsigned long) sst * sizeof(T));
                disp_offset += (MPI_Aint) ((unsigned long) dst * sizeof(T));
            }

            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_flush_local(pe, win));
        }

    fn_exit:
        return ret;
    }

    template <typename T, ishmemi_op_t OP, bool FLUSH = true>
    static int get(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        T result;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, src);

        if constexpr (OP == G) {
            dest = &result;
            nelems = 1;
        }

        if constexpr (OP == IBGET) {
            nelems = nelems * bsize;
        }

        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Get(dest, (int) nelems, dt, pe, disp,
                                                          (int) nelems, dt, win));
        if constexpr (FLUSH) {
            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_flush_local(pe, win));
        }

        if constexpr (OP == G) {
            ishmemi_union_set_field_value<T, G>(comp->completion.ret, result);
        }

    fn_exit:
        return ret;
    }

    template <typename T, ishmemi_op_t OP>
    static int iget_datatype(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, src);

        if constexpr (OP == IGET) {
            bsize = 1;
        }

        if (dst == (ptrdiff_t) bsize && sst == (ptrdiff_t) bsize) {
            get<T, OP>(msg, comp);
        } else {
            MPI_Datatype sdt = MPI_DATATYPE_NULL;
            MPI_Datatype ddt = MPI_DATATYPE_NULL;

            ishmemi_runtime_mpi::get_strided_dt(nelems, sst, bsize, 0, dt, &sdt);

            if (sst != dst) {
                ishmemi_runtime_mpi::get_strided_dt(nelems, dst, bsize, 0, dt, &ddt);
            } else {
                ddt = sdt;
            }

            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Get(dest, 1, sdt, pe, disp, 1, ddt, win));
            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_flush_local(pe, win));
        }

    fn_exit:
        return ret;
    }

    template <typename T, ishmemi_op_t OP>
    static int iget_fallback(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, src);

        if constexpr (OP == IGET) {
            bsize = 1;
        }

        if (dst == (ptrdiff_t) bsize && sst == (ptrdiff_t) bsize) {
            get<T, OP>(msg, comp);
        } else {
            T *dest_offset = (T *) dest;
            MPI_Aint disp_offset = disp;

            for (size_t i = 0; i < nelems; ++i) {
                MPI_CHECK_GOTO(fn_exit,
                               ishmemi_mpi_wrappers::Get(dest_offset, (int) bsize, dt, pe,
                                                         disp_offset, (int) bsize, dt, win));
                dest_offset = pointer_offset<T>(dest_offset, (unsigned long) dst * sizeof(T));
                disp_offset += (MPI_Aint) ((unsigned long) sst * sizeof(T));
            }

            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_flush_local(pe, win));
        }

    fn_exit:
        return ret;
    }

    /* AMOs */
    template <typename T, ishmemi_op_t OP, bool FLUSH = true>
    static int amo_fetch_op(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_REQUEST_HELPER(T, OP);
        MPI_Aint disp = 0;
        MPI_Op op = get_amo_op<OP>();

        if constexpr (OP == AMO_FETCH) {
            disp = CALC_DISP(src, ishmemi_runtime_mpi::global_win_base_addr);
        } else {
            disp = CALC_DISP(dest, ishmemi_runtime_mpi::global_win_base_addr);
        }

        if constexpr (OP == AMO_FETCH_INC || OP == AMO_FETCH_INC_NBI) {
            val = static_cast<T>(1);
        }

        if constexpr (FLUSH) {
            /* Blocking AMO case */
            T result;
            MPI_CHECK_GOTO(
                fn_exit, ishmemi_mpi_wrappers::Fetch_and_op(&val, &result, dt, pe, disp, op, win));

            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_flush_local(pe, win));
            ishmemi_union_set_field_value<T, OP>(comp->completion.ret, result);
        } else {
            /* Non-blocking AMO case */
            MPI_CHECK_GOTO(fn_exit,
                           ishmemi_mpi_wrappers::Fetch_and_op(&val, fetch, dt, pe, disp, op, win));
        }

    fn_exit:
        return ret;
    }

    template <typename T, ishmemi_op_t OP, bool FLUSH = true>
    static int amo_op(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, dest);
        MPI_Op op = get_amo_op<OP>();

        if constexpr (OP == AMO_INC) {
            val = static_cast<T>(1);
        }

        if constexpr (FLUSH) {
            /* Blocking AMO case */
            MPI_CHECK_GOTO(fn_exit,
                           ishmemi_mpi_wrappers::Accumulate(&val, 1, dt, pe, disp, 1, dt, op, win));

            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_flush_local(pe, win));
        } else {
            /* Non-blocking AMO case */
            MPI_CHECK_GOTO(fn_exit,
                           ishmemi_mpi_wrappers::Fetch_and_op(&val, fetch, dt, pe, disp, op, win));
        }

    fn_exit:
        return ret;
    }

    template <typename T, ishmemi_op_t OP, bool FLUSH = true>
    static int amo_compare_swap(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, dest);

        if constexpr (FLUSH) {
            /* Blocking AMO case */
            T result;
            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Compare_and_swap(&val, &cond, &result, dt,
                                                                           pe, disp, win));

            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_flush_local(pe, win));
            ishmemi_union_set_field_value<T, OP>(comp->completion.ret, result);
        } else {
            /* Non-blocking AMO case */
            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Compare_and_swap(&val, &cond, fetch, dt,
                                                                           pe, disp, win));
        }

    fn_exit:
        return ret;
    }

    /* Collectives */
    static int barrier(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        ISHMEMI_RUNTIME_MPI_COMM_HELPER(BARRIER);
        return barrier_impl(comm, win);
    }

    template <ishmemi_op_t OP>
    static int sync(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        ISHMEMI_RUNTIME_MPI_COMM_HELPER(OP);
        return sync_impl(comm, win);
    }

    static int alltoall(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_REQUEST_HELPER(uint8_t, ALLTOALL);

        MPI_CHECK(
            ishmemi_mpi_wrappers::Alltoall(src, (int) nelems, dt, dest, (int) nelems, dt, comm));
        return ret;
    }

    static int broadcast(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_REQUEST_HELPER(uint8_t, BCAST);

        if (rank == root) {
            ishmemi_copy(dest, src, nelems);
        }
        MPI_CHECK(ishmemi_mpi_wrappers::Bcast(dest, (int) nelems, dt, root, comm));
        return ret;
    }

    static int collect(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_REQUEST_HELPER(uint8_t, COLLECT);

        int *recvcounts = nullptr;
        int *displs = nullptr;
        int rcount = (int) nelems;
        int world_team_size = ishmemi_runtime_mpi::teams[ishmemi_runtime_mpi::world_team].size;

        recvcounts = (int *) ::malloc(sizeof(int) * (size_t) world_team_size);
        ISHMEM_CHECK_GOTO_MSG(recvcounts == nullptr, fn_fail, "Unable to allocate host memory\n");

        displs = (int *) ::malloc(sizeof(int) * (size_t) world_team_size);
        ISHMEM_CHECK_GOTO_MSG(displs == nullptr, fn_fail, "Unable to allocate host memory\n");

        /* Allgather nelems */
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Allgather(&rcount, 1, MPI_INT, recvcounts, 1,
                                                                MPI_INT, comm));

        /* Calculate displacements */
        displs[0] = 0;

        for (int i = 1; i < world_team_size; ++i) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        /* Perform the collect */
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Allgatherv(src, (int) nelems, dt, dest,
                                                                 recvcounts, displs, dt, comm));
    fn_exit:
        comp->completion.ret.i = ret;

        ::free(displs);
        ::free(recvcounts);

        return ret;
    fn_fail:
        ret = -1;
        goto fn_exit;
    }

    static int fcollect(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_REQUEST_HELPER(uint8_t, FCOLLECT);

        MPI_CHECK(
            ishmemi_mpi_wrappers::Allgather(src, (int) nelems, dt, dest, (int) nelems, dt, comm));
        comp->completion.ret.i = ret;
        return ret;
    }

    /* Reductions */
    template <typename T, ishmemi_op_t OP>
    static int reduce(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_REQUEST_HELPER(T, OP);
        MPI_Op op = get_reduction_op<OP>();

        MPI_CHECK(ishmemi_mpi_wrappers::Allreduce(src, dest, (int) nelems, dt, op, comm));
        comp->completion.ret.i = ret;
        return ret;
    }

    /* SCAN */
    template <typename T>
    int inscan(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_REQUEST_HELPER(T, INSCAN);

        MPI_CHECK(ishmemi_mpi_wrappers::Scan(src, dest, (int) nelems, dt, MPI_SUM, comm));
        comp->completion.ret.i = ret;
        return ret;
    }

    template <typename T>
    int exscan(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_REQUEST_HELPER(T, EXSCAN);

        MPI_CHECK(ishmemi_mpi_wrappers::Exscan(src, dest, (int) nelems, dt, MPI_SUM, comm));
        comp->completion.ret.i = ret;
        return ret;
    }

    /* Point-to-point Synchronization */
    template <typename T, ishmemi_op_t OP>
    static int test(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, dest);

        ret = test_impl(cmp_value, cmp, dt, rank, disp, win);
        ISHMEM_CHECK_GOTO_MSG(ret == -1, fn_exit, "Failed to execute test");
        comp->completion.ret.i = ret;
        ret = 0;

    fn_exit:
        return ret;
    }

    template <typename T, ishmemi_op_t OP, bool VECTOR>
    static int test_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, dest);

        if (nelems == 0) ret = 1;

        /* Get host buffers */
        CONVERT_GPU_BUFFER(const, int, status, sizeof(int) * nelems, true);
        CONVERT_GPU_BUFFER(const, T, cmp_values, sizeof(T) * nelems, VECTOR);

        for (size_t i = 0; i < nelems; ++i) {
            if (status_host && status_host[i]) {
                disp = disp + (MPI_Aint) sizeof(T);
                continue;
            }

            if constexpr (VECTOR) {
                ret = test_impl(cmp_values_host[i], cmp, dt, rank, disp, win);
            } else {
                ret = test_impl(cmp_value, cmp, dt, rank, disp, win);
            }

            ISHMEM_CHECK_GOTO_MSG(ret == -1, fn_exit, "Failed to execute test_all");
            if (ret == 0) break;
            disp = disp + (MPI_Aint) sizeof(T);
        }

        CLEANUP_GPU_BUFFER(int, status, sizeof(int) * nelems, true);
        CLEANUP_GPU_BUFFER(T, cmp_values, sizeof(T) * nelems, VECTOR);

        comp->completion.ret.i = ret;
        ret = 0;

    fn_exit:
        return ret;
    }

    template <typename T, ishmemi_op_t OP, bool VECTOR>
    static int test_any(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        size_t complete = SIZE_MAX;
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, dest);

        /* Get host buffers */
        CONVERT_GPU_BUFFER(const, int, status, sizeof(int) * nelems, true);
        CONVERT_GPU_BUFFER(const, T, cmp_values, sizeof(T) * nelems, VECTOR);

        for (size_t i = 0; i < nelems; ++i) {
            if (status_host && status_host[i]) {
                disp = disp + (MPI_Aint) sizeof(T);
                continue;
            }

            if constexpr (VECTOR) {
                ret = test_impl(cmp_values_host[i], cmp, dt, rank, disp, win);
            } else {
                ret = test_impl(cmp_value, cmp, dt, rank, disp, win);
            }

            ISHMEM_CHECK_GOTO_MSG(ret == -1, fn_exit, "Failed to execute test_any");
            if (ret == 1) {
                complete = i;
                break;
            }
            disp = disp + (MPI_Aint) sizeof(T);
        }

        CLEANUP_GPU_BUFFER(int, status, sizeof(int) * nelems, true);
        CLEANUP_GPU_BUFFER(T, cmp_values, sizeof(T) * nelems, VECTOR);

        comp->completion.ret.szt = complete;
        ret = 0;

    fn_exit:
        return ret;
    }

    template <typename T, ishmemi_op_t OP, bool VECTOR>
    static int test_some(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        size_t complete = 0;
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, dest);

        /* Get host buffers */
        CONVERT_GPU_BUFFER(, size_t, indices, sizeof(size_t) * nelems, true);
        CONVERT_GPU_BUFFER(const, int, status, sizeof(int) * nelems, true);
        CONVERT_GPU_BUFFER(const, T, cmp_values, sizeof(T) * nelems, VECTOR);

        if constexpr (VECTOR) {
            ret = test_multi_impl(cmp_values_host, cmp, dt, rank, disp, nelems, status_host,
                                  indices_host, complete, win);
        } else {
            ret = test_multi_impl(cmp_value, cmp, dt, rank, disp, nelems, status_host, indices_host,
                                  complete, win);
        }
        ISHMEM_CHECK_GOTO_MSG(ret == -1, fn_exit, "Failed to run multiple test ops\n");

        CLEANUP_GPU_BUFFER(size_t, indices, sizeof(size_t) * nelems, true);
        CLEANUP_GPU_BUFFER(int, status, sizeof(int) * nelems, true);
        CLEANUP_GPU_BUFFER(T, cmp_values, sizeof(T) * nelems, VECTOR);

        comp->completion.ret.szt = complete;
        ret = 0;

    fn_exit:
        return ret;
    }

    static int signal_wait_until(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        MPI_Op op = MPI_NO_OP;
        uint64_t result;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(uint64_t, SIGNAL_WAIT_UNTIL, sig_addr);

        while (true) {
            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Fetch_and_op(nullptr, &result, dt, rank,
                                                                       disp, op, win));

            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_flush_local(rank, win));

            ret = compare(cmp, result, cmp_value);
            ISHMEM_CHECK_GOTO_MSG(ret == -1, fn_exit, "Unknown or unsupported comparison op\n");

            if (ret == 1) break;
            force_progress(comm);
        }

        comp->completion.ret.ui64 = result;

        ret = 0;

    fn_exit:
        return ret;
    }

    template <typename T, ishmemi_op_t OP>
    static int wait_until(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, dest);

        while (true) {
            ret = test_impl(cmp_value, cmp, dt, rank, disp, win);
            ISHMEM_CHECK_GOTO_MSG(ret == -1, fn_exit, "Failed to execute wait_until");
            if (ret == 1) break;
            force_progress(comm);
        }

        ret = 0;

    fn_exit:
        return ret;
    }

    template <typename T, ishmemi_op_t OP, bool VECTOR>
    static int wait_until_all(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, dest);

        /* Get host buffers */
        CONVERT_GPU_BUFFER(const, int, status, sizeof(int) * nelems, true);
        CONVERT_GPU_BUFFER(const, T, cmp_values, sizeof(T) * nelems, VECTOR);

        size_t num_skip = 0;
        if (status_host) {
            for (size_t i = 0; i < nelems; ++i) {
                num_skip += (status_host[i] == 0) ? 0 : 1;
            }
        }

        if (num_skip < nelems) {
            /* Iteratively wait_until on each ivar */
            for (size_t i = 0; i < nelems; ++i) {
                if (status_host && status_host[i]) {
                    disp = disp + (MPI_Aint) sizeof(T);
                    continue;
                }
                while (true) {
                    if constexpr (VECTOR) {
                        ret = test_impl(cmp_values_host[i], cmp, dt, rank, disp, win);
                    } else {
                        ret = test_impl(cmp_value, cmp, dt, rank, disp, win);
                    }

                    ISHMEM_CHECK_GOTO_MSG(ret == -1, fn_exit, "Failed to execute wait_until_all");
                    if (ret == 1) break;
                    force_progress(comm);
                }
                disp = disp + (MPI_Aint) sizeof(T);
            }
        }

        CLEANUP_GPU_BUFFER(int, status, sizeof(int) * nelems, true);
        CLEANUP_GPU_BUFFER(T, cmp_values, sizeof(T) * nelems, VECTOR);

        ret = 0;

    fn_exit:
        return ret;
    }

    template <typename T, ishmemi_op_t OP, bool VECTOR>
    static int wait_until_any(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        size_t complete = SIZE_MAX;
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, dest);
        MPI_Aint tmp_disp = disp;

        /* Get host buffers */
        CONVERT_GPU_BUFFER(const, int, status, sizeof(int) * nelems, true);
        CONVERT_GPU_BUFFER(const, T, cmp_values, sizeof(T) * nelems, VECTOR);

        size_t num_skip = 0;
        if (status_host) {
            for (size_t i = 0; i < nelems; ++i) {
                num_skip += (status_host[i] == 0) ? 0 : 1;
            }
        }

        if (num_skip < nelems) {
            while (true) {
                /* Iteratively test each ivar */
                for (size_t i = 0; i < nelems; ++i) {
                    if (status_host && status_host[i]) {
                        tmp_disp = tmp_disp + (MPI_Aint) sizeof(T);
                        continue;
                    }

                    if constexpr (VECTOR) {
                        ret = test_impl(cmp_values_host[i], cmp, dt, rank, tmp_disp, win);
                    } else {
                        ret = test_impl(cmp_value, cmp, dt, rank, tmp_disp, win);
                    }

                    ISHMEM_CHECK_GOTO_MSG(ret == -1, fn_exit, "Failed to execute wait_until_any");
                    if (ret == 1) {
                        complete = i;
                        break;
                    }
                    tmp_disp = tmp_disp + (MPI_Aint) sizeof(T);
                    force_progress(comm);
                }

                /* Completion condition */
                if (complete != SIZE_MAX) break;
                tmp_disp = disp;
            }
        }

        CLEANUP_GPU_BUFFER(int, status, sizeof(int) * nelems, true);
        CLEANUP_GPU_BUFFER(T, cmp_values, sizeof(T) * nelems, VECTOR);

        comp->completion.ret.szt = complete;
        ret = 0;

    fn_exit:
        return ret;
    }

    template <typename T, ishmemi_op_t OP, bool VECTOR>
    static int wait_until_some(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        size_t complete = 0;
        int ret = 0;
        ISHMEMI_RUNTIME_MPI_DISP_REQUEST_HELPER(T, OP, dest);
        MPI_Aint tmp_disp = disp;

        /* Get host buffers */
        CONVERT_GPU_BUFFER(, size_t, indices, sizeof(size_t) * nelems, true);
        CONVERT_GPU_BUFFER(const, int, status, sizeof(int) * nelems, true);
        CONVERT_GPU_BUFFER(const, T, cmp_values, sizeof(T) * nelems, VECTOR);

        size_t num_skip = 0;
        if (status_host) {
            for (size_t i = 0; i < nelems; ++i) {
                num_skip += (status_host[i] == 0) ? 0 : 1;
            }
        }

        if (num_skip < nelems) {
            while (true) {
                /* Iteratively test each ivar */
                for (size_t i = 0; i < nelems; ++i) {
                    if (status_host && status_host[i]) {
                        tmp_disp = tmp_disp + (MPI_Aint) sizeof(T);
                        continue;
                    }

                    if constexpr (VECTOR) {
                        ret = test_impl(cmp_values_host[i], cmp, dt, rank, tmp_disp, win);
                    } else {
                        ret = test_impl(cmp_value, cmp, dt, rank, tmp_disp, win);
                    }

                    ISHMEM_CHECK_GOTO_MSG(ret == -1, fn_exit, "Failed to execute wait_until_some");
                    if (ret == 1) {
                        indices_host[complete] = i;
                        ++complete;
                    }

                    tmp_disp = tmp_disp + (MPI_Aint) sizeof(T);
                    force_progress(comm);
                }

                /* Completion condition */
                if (complete) break;
                tmp_disp = disp;
            }
        }

        CLEANUP_GPU_BUFFER(size_t, indices, sizeof(size_t) * nelems, true);
        CLEANUP_GPU_BUFFER(int, status, sizeof(int) * nelems, true);
        CLEANUP_GPU_BUFFER(T, cmp_values, sizeof(T) * nelems, VECTOR);

        comp->completion.ret.szt = complete;
        ret = 0;

    fn_exit:
        return ret;
    }

    /* Memory Ordering */
    static int fence(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        ISHMEMI_RUNTIME_MPI_COMM_HELPER(FENCE);
        return fence_impl(win);
    }

    static int quiet(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        ISHMEMI_RUNTIME_MPI_COMM_HELPER(QUIET);
        return quiet_impl(win);
    }

    static int team_my_pe(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret __attribute__((unused)) = 0;
        int pe = 0;
        ISHMEMI_RUNTIME_MPI_COMM_HELPER(TEAM_MY_PE);
        MPI_CHECK_GOTO(fn_fail, ishmemi_mpi_wrappers::Comm_rank(comm, &pe));
        comp->completion.ret.i = pe;

    fn_exit:
        return comp->completion.ret.i;
    fn_fail:
        comp->completion.ret.i = ret;
        goto fn_exit;
    }

    static int team_n_pes(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
    {
        int ret __attribute__((unused)) = 0;
        int npes = 0;
        ISHMEMI_RUNTIME_MPI_COMM_HELPER(TEAM_N_PES);
        MPI_CHECK_GOTO(fn_fail, ishmemi_mpi_wrappers::Comm_size(comm, &npes));
        comp->completion.ret.i = npes;

    fn_exit:
        return comp->completion.ret.i;
    fn_fail:
        comp->completion.ret.i = ret;
        goto fn_exit;
    }
}  // namespace impl

/* Default initialization of static members */

ishmemi_runtime_mpi_types::team_t ishmemi_runtime_mpi::world_team =
    ishmemi_runtime_mpi::team_undefined;
ishmemi_runtime_mpi_types::team_t ishmemi_runtime_mpi::node_team =
    ishmemi_runtime_mpi::team_undefined;
ishmemi_runtime_mpi_types::team_t ishmemi_runtime_mpi::shared_team =
    ishmemi_runtime_mpi::team_undefined;
ishmemi_runtime_mpi_types::team_t ishmemi_runtime_mpi::team_idx = 0;

std::map<ishmemi_runtime_mpi_types::team_t, ishmemi_runtime_mpi::team_t> ishmemi_runtime_mpi::teams;

MPI_Win ishmemi_runtime_mpi::global_win = MPI_WIN_NULL;
void *ishmemi_runtime_mpi::global_win_base_addr = nullptr;
size_t ishmemi_runtime_mpi::global_win_size = 0;
ishmemi_runtime_mpi::datatype_entry_t *ishmemi_runtime_mpi::datatype_map = nullptr;

/* Class method implementations */
ishmemi_runtime_mpi::ishmemi_runtime_mpi(bool initialize_runtime, void *mpi_comm)
{
    int ret = 0;
    team_t temp_team = {};

    teams[team_undefined] = temp_team;

    /* Setup MPI dlsym links */
    ret = ishmemi_mpi_wrappers::init_wrappers();
    ISHMEM_CHECK_GOTO_MSG(ret, fn_exit, "Failed to load MPI library\n");

    /* Initialize the runtime if requested */
    if (initialize_runtime) {
        int required = MPI_THREAD_MULTIPLE;
        int provided = 0;
        MPI_CHECK_GOTO(fn_exit,
                       ishmemi_mpi_wrappers::Init_thread(nullptr, nullptr, required, &provided));
        ISHMEM_CHECK_GOTO_MSG(required != provided, fn_exit,
                              "Failed to initialize MPI with MPI_THREAD_MULTIPLE\n");
        this->initialized = true;
    }

    /* Setup internal runtime info */
    if (mpi_comm) {
        temp_team.comm = *((MPI_Comm *) (mpi_comm));
    } else {
        temp_team.comm = MPI_COMM_WORLD;
    }
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_rank(temp_team.comm, &temp_team.rank));
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_size(temp_team.comm, &temp_team.size));
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_group(temp_team.comm, &temp_team.group));

    world_team = team_idx++;
    teams[world_team] = std::move(temp_team);

    temp_team = {};

    if (teams[world_team].size > 1) {
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_split_type(
                                    teams[world_team].comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                    &temp_team.comm));

        if (temp_team.comm == MPI_COMM_NULL) {
            RAISE_ERROR_MSG("MPI FAILURE: node_team was not correctly created\n");
        }
    } else {
        /* Duplicate the communicator to ensure different teams can run ops concurrently */
        MPI_CHECK_GOTO(fn_exit,
                       ishmemi_mpi_wrappers::Comm_dup(teams[world_team].comm, &temp_team.comm));
    }

    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_rank(temp_team.comm, &temp_team.rank));
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_size(temp_team.comm, &temp_team.size));
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_group(temp_team.comm, &temp_team.group));

    if (temp_team.size > teams[world_team].size) {
        RAISE_ERROR_MSG("MPI FAILURE: node_team was not correctly created\n");
    }

    /* Save node_team */
    node_team = team_idx++;
    teams[node_team] = std::move(temp_team);

    temp_team = {};

    /* Duplicate the communicator to ensure teams can run ops concurrently */
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_dup(teams[node_team].comm, &temp_team.comm));
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_rank(temp_team.comm, &temp_team.rank));
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_size(temp_team.comm, &temp_team.size));
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_group(temp_team.comm, &temp_team.group));

    if (temp_team.size > teams[world_team].size) {
        RAISE_ERROR_MSG("MPI FAILURE: shared_team was not correctly created\n");
    }

    shared_team = team_idx++;
    teams[shared_team] = temp_team;

    /* Initialize the function pointer table */
    this->funcptr_init();

fn_exit:
    return;
}

ishmemi_runtime_mpi::~ishmemi_runtime_mpi(void)
{
    int ret __attribute__((unused)) = 0;

    /* Cleanup the datatype map */
    if (datatype_map) {
        datatype_entry_t *entry = nullptr, *tmp = nullptr;
        HASH_ITER(hh, datatype_map, entry, tmp)
        {
            MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Type_free(&entry->datatype));
            ::free(entry);
        }
    }

    /* Cleanup the runtime-created teams */
    for (auto iter = teams.begin(); iter != teams.end();) {
        auto curr = iter++;
        team_destroy_impl(curr->first);
    }

    world_team = shared_team = node_team = team_undefined;

    /* Complete the access epoch */
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_unlock_all(global_win));

    /* Close the global window */
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_free(&global_win));
    global_win_base_addr = nullptr;
    global_win_size = 0;

    /* Finalize MPI */
    if (this->initialized) {
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Finalize());
        this->initialized = false;
    }

    /* Cleanup the function pointer table */
    this->funcptr_fini();

    /* Close the shared library */
    ishmemi_mpi_wrappers::fini_wrappers();

fn_exit:
    return;
}

void ishmemi_runtime_mpi::heap_create(void *base, size_t size)
{
    int ret __attribute__((unused)) = 0;

    /* Setup info hints for window */
    MPI_Info info = MPI_INFO_NULL;
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Info_create(&info));
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Info_set(info, "accumulate_ordering", "none"));

    /* Create the window */
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_create(base, (MPI_Aint) size, 1, info,
                                                             teams[world_team].comm, &global_win));

    /* Start an access epoch */
    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Win_lock_all(MPI_MODE_NOCHECK, global_win));

    /* Store the window base */
    global_win_base_addr = base;
    global_win_size = size;

    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Info_free(&info));

fn_exit:
    return;
}

/* Query APIs */
int ishmemi_runtime_mpi::get_rank(void)
{
    return teams[world_team].rank;
}

int ishmemi_runtime_mpi::get_size(void)
{
    return teams[world_team].size;
}

int ishmemi_runtime_mpi::get_node_rank(int pe)
{
    return translate_rank(pe, teams[world_team], teams[node_team]);
}

int ishmemi_runtime_mpi::get_node_size(void)
{
    return teams[node_team].size;
}

bool ishmemi_runtime_mpi::is_local(int pe)
{
    return (this->get_node_rank(pe) != -1);
}

bool ishmemi_runtime_mpi::is_symmetric_address(const void *addr)
{
    return ((uintptr_t) addr >= (uintptr_t) global_win_base_addr) &&
           ((uintptr_t) addr < ((uintptr_t) global_win_base_addr + global_win_size));
}

/* Memory APIs */
void *ishmemi_runtime_mpi::malloc(size_t size)
{
    return ::malloc(size);
}

void *ishmemi_runtime_mpi::calloc(size_t num, size_t size)
{
    return ::calloc(num, size);
}

void ishmemi_runtime_mpi::free(void *ptr)
{
    ::free(ptr);
}

/* Team APIs */
int ishmemi_runtime_mpi::team_sync(ishmemi_runtime_team_t team)
{
    return sync_impl(teams[team.mpi].comm, global_win);
}

int ishmemi_runtime_mpi::team_predefined_set(ishmemi_runtime_team_t *team,
                                             ishmemi_runtime_team_predefined_t predefined_team_name,
                                             int expected_team_size, int expected_world_pe,
                                             int expected_team_pe)
{
    int ret = 0;
    team_t temp_team = {};

    switch (predefined_team_name) {
        case WORLD:
            team->mpi = world_team;
            break;
        case SHARED:
            /* Catch the mis-match between SHARED and NODE team type */
            /* This can happen when ISHMEM_ENABLE_GPU_IPC=0 */
            if (expected_team_size == 1 && teams[shared_team].size != 1) {
                /* Cleanup the previous shared_team objects */
                team_destroy_impl(shared_team);

                /* Initialize a new shared_team with MPI_COMM_SELF */
                temp_team.comm = MPI_COMM_SELF;
                MPI_CHECK_GOTO(fn_fail,
                               ishmemi_mpi_wrappers::Comm_rank(temp_team.comm, &temp_team.rank));
                MPI_CHECK_GOTO(fn_fail,
                               ishmemi_mpi_wrappers::Comm_size(temp_team.comm, &temp_team.size));
                MPI_CHECK_GOTO(fn_fail,
                               ishmemi_mpi_wrappers::Comm_group(temp_team.comm, &temp_team.group));

                teams[shared_team] = std::move(temp_team);
            }

            team->mpi = shared_team;
            break;
        case NODE:
            team->mpi = node_team;
            break;
        default:
            return -3;
    }

fn_exit:
    if (teams[team->mpi].size != expected_team_size) {
        ret = -1;
    } else if (teams[team->mpi].rank != expected_team_pe) {
        ret = -2;
    }

    return ret;
fn_fail:
    teams[team->mpi].size = -1;
    teams[team->mpi].rank = -1;
    goto fn_exit;
}

int ishmemi_runtime_mpi::team_split_strided(ishmemi_runtime_team_t parent_team, int PE_start,
                                            int PE_stride, int PE_size,
                                            const ishmemi_runtime_team_config_t *config,
                                            long config_mask, ishmemi_runtime_team_t *new_team)
{
    int ret = 0;
    int included = 0;
    int start = 0;
    int team_rank_order = 0;
    team_t temp_team;

    new_team->mpi = team_undefined;

    ishmemi_runtime_mpi::team_t parent = teams[parent_team.mpi];

    /* Validate inputs */
    ISHMEM_CHECK_GOTO_MSG((PE_start < 0), fn_exit, "invalid start - provided %d - expected >= 0\n",
                          PE_start);
    ISHMEM_CHECK_GOTO_MSG((PE_start > parent.size), fn_exit,
                          "invalid start - provided %d - expected <= %d\n", PE_start, parent.size);
    ISHMEM_CHECK_GOTO_MSG((parent.size < PE_size), fn_exit,
                          "invalid size - provided %d - expected <= %d\n", PE_size, parent.size);

    /* Determine if this pe is included */
    start = translate_rank(PE_start, parent, teams[world_team]);
    team_rank_order = ishmemi_pe_in_active_set(teams[world_team].rank, start, PE_stride, PE_size);
    included = team_rank_order == -1 ? MPI_UNDEFINED : 1;

    MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_split(parent.comm, included, team_rank_order,
                                                             &temp_team.comm));
    if (temp_team.comm != MPI_COMM_NULL) {
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_rank(temp_team.comm, &temp_team.rank));
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_size(temp_team.comm, &temp_team.size));
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Comm_group(temp_team.comm, &temp_team.group));

        new_team->mpi = team_idx++;
        teams[new_team->mpi] = std::move(temp_team);
    }

fn_exit:
    return ret;
}

void ishmemi_runtime_mpi::team_destroy(ishmemi_runtime_team_t team)
{
    team_destroy_impl(team.mpi);
}

/* Operation APIs */
void ishmemi_runtime_mpi::abort(int exit_code, const char msg[])
{
    std::cerr << "[ABORT] " << msg << std::endl;
    ishmemi_mpi_wrappers::Abort(teams[world_team].comm, exit_code);
}

int ishmemi_runtime_mpi::get_kvs(int pe, char *key, void *value, size_t valuelen)
{
    RAISE_ERROR_MSG("This API is not yet implemented\n");
    return -1;
}

int ishmemi_runtime_mpi::uchar_and_reduce(ishmemi_runtime_team_t team, unsigned char *dest,
                                          const unsigned char *source, size_t nreduce)
{
    int ret = 0;
    MPI_Datatype dt = impl::get_datatype<unsigned char, AND_REDUCE>();
    MPI_Op op = impl::get_reduction_op<AND_REDUCE>();
    MPI_Comm comm = teams[team.mpi].comm;

    MPI_CHECK(ishmemi_mpi_wrappers::Allreduce(source, dest, (int) nreduce, dt, op, comm));
    return ret;
}

int ishmemi_runtime_mpi::int_max_reduce(ishmemi_runtime_team_t team, int *dest, const int *source,
                                        size_t nreduce)
{
    int ret = 0;
    MPI_Datatype dt = impl::get_datatype<int, MAX_REDUCE>();
    MPI_Op op = impl::get_reduction_op<MAX_REDUCE>();
    MPI_Comm comm = teams[team.mpi].comm;

    MPI_CHECK(ishmemi_mpi_wrappers::Allreduce(source, dest, (int) nreduce, dt, op, comm));
    return ret;
}

void ishmemi_runtime_mpi::bcast(void *buf, size_t count, int root)
{
    ishmemi_mpi_wrappers::Bcast(buf, (int) count, MPI_BYTE, root, teams[world_team].comm);
}

void ishmemi_runtime_mpi::node_bcast(void *buf, size_t count, int root)
{
    ishmemi_mpi_wrappers::Bcast(buf, (int) count, MPI_BYTE, root, teams[node_team].comm);
}

void ishmemi_runtime_mpi::fcollect(void *dst, void *src, size_t count)
{
    ishmemi_mpi_wrappers::Allgather(src, (int) count, MPI_BYTE, dst, (int) count, MPI_BYTE,
                                    teams[world_team].comm);
}

void ishmemi_runtime_mpi::node_fcollect(void *dst, void *src, size_t count)
{
    ishmemi_mpi_wrappers::Allgather(src, (int) count, MPI_BYTE, dst, (int) count, MPI_BYTE,
                                    teams[node_team].comm);
}

void ishmemi_runtime_mpi::barrier_all(void)
{
    barrier_impl(teams[world_team].comm, global_win);
}

void ishmemi_runtime_mpi::node_barrier(void)
{
    barrier_impl(teams[node_team].comm, global_win);
}

void ishmemi_runtime_mpi::fence(void)
{
    fence_impl(global_win);
}

void ishmemi_runtime_mpi::quiet(void)
{
    quiet_impl(global_win);
}

void ishmemi_runtime_mpi::sync(void)
{
    int ret __attribute__((unused)) = 0;
    MPI_CHECK(ishmemi_mpi_wrappers::Barrier(teams[world_team].comm));
}

void ishmemi_runtime_mpi::progress(void)
{
    force_progress(teams[world_team].comm);
}

/* Private functions */
void ishmemi_runtime_mpi::get_strided_dt(size_t nelems, ptrdiff_t stride, size_t block_size,
                                         int extent, MPI_Datatype base, MPI_Datatype *datatype)
{
    int ret __attribute__((unused)) = 0;
    MPI_Datatype tmp = MPI_DATATYPE_NULL;
    datatype_key_t key = {(int) nelems, (int) stride, (int) block_size, extent, base};
    datatype_entry_t *entry = nullptr;

    HASH_FIND(hh, datatype_map, &key.nelems, sizeof(datatype_key_t), entry);

    if (entry) {
        *datatype = entry->datatype;
    } else {
        /* Create the vector type based on the provided base datatype */
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Type_vector((int) nelems, (int) block_size,
                                                                  (int) stride, base, &tmp));

        if (extent > 0) {
            /* If the extent is non-zero, we need to extend the vector type. This will occur when
             * the operation needs to cover multiple elements of the strided type (i.e. strided
             * collectives)
             */
            /* TODO */
        } else {
            *datatype = tmp;
        }

        /* Commit the datatype */
        MPI_CHECK_GOTO(fn_exit, ishmemi_mpi_wrappers::Type_commit(datatype));

        if (extent > 0) {
            /* Cleanup the vector type if it needed to be extended */
            /* TODO */
        }

        /* Insert into the datatype cache */
        entry = (datatype_entry_t *) ::malloc(sizeof(datatype_entry_t));
        ISHMEM_CHECK_GOTO_MSG(entry == nullptr, fn_exit, "Allocation of datatype entry failed\n");
        memset(entry, 0, sizeof(datatype_entry_t));
        memcpy(&entry->key, &key, sizeof(datatype_key_t));
        entry->datatype = *datatype;

        HASH_ADD(hh, datatype_map, key, sizeof(datatype_key_t), entry);
    }

fn_exit:
    return;
}

void ishmemi_runtime_mpi::funcptr_init(void)
{
    proxy_funcs = (ishmemi_runtime_proxy_func_t **) ::malloc(
        sizeof(ishmemi_runtime_proxy_func_t *) * ISHMEMI_OP_END);
    ISHMEM_CHECK_GOTO_MSG(proxy_funcs == nullptr, fn_exit, "Allocation of proxy_funcs failed\n");

    /* Initialize every function with the "unsupported op" function */
    /* Note: KILL operation is covered inside the proxy directly - it is the same for all backends
     * currently */
    for (size_t i = 0; i < ISHMEMI_OP_END; ++i) {
        proxy_funcs[i] = (ishmemi_runtime_proxy_func_t *) ::malloc(
            sizeof(ishmemi_runtime_proxy_func_t) * ishmemi_runtime_type::proxy_func_num_types);
        for (size_t j = 0; j < ishmemi_runtime_type::proxy_func_num_types; ++j) {
            proxy_funcs[i][j] = ishmemi_runtime_type::unsupported;
        }
    }

    /* Fill in the supported functions */
    /* RMA */
    proxy_funcs[PUT][UINT8] = impl::put<uint8_t, PUT>;
    proxy_funcs[P][UINT8] = impl::put<uint8_t, P>;
    proxy_funcs[P][UINT16] = impl::put<uint16_t, P>;
    proxy_funcs[P][UINT32] = impl::put<uint32_t, P>;
    proxy_funcs[P][UINT64] = impl::put<uint64_t, P>;
    proxy_funcs[P][ULONGLONG] = impl::put<unsigned long long, P>;
    proxy_funcs[P][FLOAT] = impl::put<float, P>;
    proxy_funcs[P][DOUBLE] = impl::put<double, P>;
    proxy_funcs[PUT_NBI][UINT8] = impl::put<uint8_t, PUT, false>;

    proxy_funcs[GET][UINT8] = impl::get<uint8_t, GET>;
    proxy_funcs[G][UINT8] = impl::get<uint8_t, G>;
    proxy_funcs[G][UINT16] = impl::get<uint16_t, G>;
    proxy_funcs[G][UINT32] = impl::get<uint32_t, G>;
    proxy_funcs[G][UINT64] = impl::get<uint64_t, G>;
    proxy_funcs[G][ULONGLONG] = impl::get<unsigned long long, G>;
    proxy_funcs[G][FLOAT] = impl::get<float, G>;
    proxy_funcs[G][DOUBLE] = impl::get<double, G>;
    proxy_funcs[GET_NBI][UINT8] = impl::get<uint8_t, GET, false>;

    if (gpu_non_contig_support) {
        proxy_funcs[IPUT][UINT8] = impl::iput_datatype<uint8_t, IPUT>;
        proxy_funcs[IPUT][UINT16] = impl::iput_datatype<uint16_t, IPUT>;
        proxy_funcs[IPUT][UINT32] = impl::iput_datatype<uint32_t, IPUT>;
        proxy_funcs[IPUT][UINT64] = impl::iput_datatype<uint64_t, IPUT>;
        proxy_funcs[IPUT][ULONGLONG] = impl::iput_datatype<unsigned long long, IPUT>;
        proxy_funcs[IBPUT][UINT8] = impl::iput_datatype<uint8_t, IBPUT>;
        proxy_funcs[IBPUT][UINT16] = impl::iput_datatype<uint16_t, IBPUT>;
        proxy_funcs[IBPUT][UINT32] = impl::iput_datatype<uint32_t, IBPUT>;
        proxy_funcs[IBPUT][UINT64] = impl::iput_datatype<uint64_t, IBPUT>;
        proxy_funcs[IGET][UINT8] = impl::iget_datatype<uint8_t, IGET>;
        proxy_funcs[IGET][UINT16] = impl::iget_datatype<uint16_t, IGET>;
        proxy_funcs[IGET][UINT32] = impl::iget_datatype<uint32_t, IGET>;
        proxy_funcs[IGET][UINT64] = impl::iget_datatype<uint64_t, IGET>;
        proxy_funcs[IGET][ULONGLONG] = impl::iget_datatype<unsigned long long, IGET>;
        proxy_funcs[IBGET][UINT8] = impl::iget_datatype<uint8_t, IBGET>;
        proxy_funcs[IBGET][UINT16] = impl::iget_datatype<uint16_t, IBGET>;
        proxy_funcs[IBGET][UINT32] = impl::iget_datatype<uint32_t, IBGET>;
        proxy_funcs[IBGET][UINT64] = impl::iget_datatype<uint64_t, IBGET>;
    } else {
        proxy_funcs[IPUT][UINT8] = impl::iput_fallback<uint8_t, IPUT>;
        proxy_funcs[IPUT][UINT16] = impl::iput_fallback<uint16_t, IPUT>;
        proxy_funcs[IPUT][UINT32] = impl::iput_fallback<uint32_t, IPUT>;
        proxy_funcs[IPUT][UINT64] = impl::iput_fallback<uint64_t, IPUT>;
        proxy_funcs[IPUT][ULONGLONG] = impl::iput_fallback<unsigned long long, IPUT>;
        proxy_funcs[IBPUT][UINT8] = impl::iput_fallback<uint8_t, IBPUT>;
        proxy_funcs[IBPUT][UINT16] = impl::iput_fallback<uint16_t, IBPUT>;
        proxy_funcs[IBPUT][UINT32] = impl::iput_fallback<uint32_t, IBPUT>;
        proxy_funcs[IBPUT][UINT64] = impl::iput_fallback<uint64_t, IBPUT>;
        proxy_funcs[IGET][UINT8] = impl::iget_fallback<uint8_t, IGET>;
        proxy_funcs[IGET][UINT16] = impl::iget_fallback<uint16_t, IGET>;
        proxy_funcs[IGET][UINT32] = impl::iget_fallback<uint32_t, IGET>;
        proxy_funcs[IGET][UINT64] = impl::iget_fallback<uint64_t, IGET>;
        proxy_funcs[IGET][ULONGLONG] = impl::iget_fallback<unsigned long long, IGET>;
        proxy_funcs[IBGET][UINT8] = impl::iget_fallback<uint8_t, IBGET>;
        proxy_funcs[IBGET][UINT16] = impl::iget_fallback<uint16_t, IBGET>;
        proxy_funcs[IBGET][UINT32] = impl::iget_fallback<uint32_t, IBGET>;
        proxy_funcs[IBGET][UINT64] = impl::iget_fallback<uint64_t, IBGET>;
    }

    /* AMO */
    proxy_funcs[AMO_FETCH][UINT32] = impl::amo_fetch_op<uint32_t, AMO_FETCH>;
    proxy_funcs[AMO_SET][UINT32] = impl::amo_op<uint32_t, AMO_SET>;
    proxy_funcs[AMO_COMPARE_SWAP][UINT32] = impl::amo_compare_swap<uint32_t, AMO_COMPARE_SWAP>;
    proxy_funcs[AMO_SWAP][UINT32] = impl::amo_fetch_op<uint32_t, AMO_SWAP>;
    proxy_funcs[AMO_FETCH_INC][UINT32] = impl::amo_fetch_op<uint32_t, AMO_FETCH_INC>;
    proxy_funcs[AMO_INC][UINT32] = impl::amo_op<uint32_t, AMO_INC>;
    proxy_funcs[AMO_FETCH_ADD][UINT32] = impl::amo_fetch_op<uint32_t, AMO_FETCH_ADD>;
    proxy_funcs[AMO_ADD][UINT32] = impl::amo_op<uint32_t, AMO_ADD>;
    proxy_funcs[AMO_FETCH_AND][UINT32] = impl::amo_fetch_op<uint32_t, AMO_FETCH_AND>;
    proxy_funcs[AMO_AND][UINT32] = impl::amo_op<uint32_t, AMO_AND>;
    proxy_funcs[AMO_FETCH_OR][UINT32] = impl::amo_fetch_op<uint32_t, AMO_FETCH_OR>;
    proxy_funcs[AMO_OR][UINT32] = impl::amo_op<uint32_t, AMO_OR>;
    proxy_funcs[AMO_FETCH_XOR][UINT32] = impl::amo_fetch_op<uint32_t, AMO_FETCH_XOR>;
    proxy_funcs[AMO_XOR][UINT32] = impl::amo_op<uint32_t, AMO_XOR>;

    proxy_funcs[AMO_FETCH][UINT64] = impl::amo_fetch_op<uint64_t, AMO_FETCH>;
    proxy_funcs[AMO_SET][UINT64] = impl::amo_op<uint64_t, AMO_SET>;
    proxy_funcs[AMO_COMPARE_SWAP][UINT64] = impl::amo_compare_swap<uint64_t, AMO_COMPARE_SWAP>;
    proxy_funcs[AMO_SWAP][UINT64] = impl::amo_fetch_op<uint64_t, AMO_SWAP>;
    proxy_funcs[AMO_FETCH_INC][UINT64] = impl::amo_fetch_op<uint64_t, AMO_FETCH_INC>;
    proxy_funcs[AMO_INC][UINT64] = impl::amo_op<uint64_t, AMO_INC>;
    proxy_funcs[AMO_FETCH_ADD][UINT64] = impl::amo_fetch_op<uint64_t, AMO_FETCH_ADD>;
    proxy_funcs[AMO_ADD][UINT64] = impl::amo_op<uint64_t, AMO_ADD>;
    proxy_funcs[AMO_FETCH_AND][UINT64] = impl::amo_fetch_op<uint64_t, AMO_FETCH_AND>;
    proxy_funcs[AMO_AND][UINT64] = impl::amo_op<uint64_t, AMO_AND>;
    proxy_funcs[AMO_FETCH_OR][UINT64] = impl::amo_fetch_op<uint64_t, AMO_FETCH_OR>;
    proxy_funcs[AMO_OR][UINT64] = impl::amo_op<uint64_t, AMO_OR>;
    proxy_funcs[AMO_FETCH_XOR][UINT64] = impl::amo_fetch_op<uint64_t, AMO_FETCH_XOR>;
    proxy_funcs[AMO_XOR][UINT64] = impl::amo_op<uint64_t, AMO_XOR>;

    proxy_funcs[AMO_FETCH][ULONGLONG] = impl::amo_fetch_op<unsigned long long, AMO_FETCH>;
    proxy_funcs[AMO_SET][ULONGLONG] = impl::amo_op<unsigned long long, AMO_SET>;
    proxy_funcs[AMO_COMPARE_SWAP][ULONGLONG] =
        impl::amo_compare_swap<unsigned long long, AMO_COMPARE_SWAP>;
    proxy_funcs[AMO_SWAP][ULONGLONG] = impl::amo_fetch_op<unsigned long long, AMO_SWAP>;
    proxy_funcs[AMO_FETCH_INC][ULONGLONG] = impl::amo_fetch_op<unsigned long long, AMO_FETCH_INC>;
    proxy_funcs[AMO_INC][ULONGLONG] = impl::amo_op<unsigned long long, AMO_INC>;
    proxy_funcs[AMO_FETCH_ADD][ULONGLONG] = impl::amo_fetch_op<unsigned long long, AMO_FETCH_ADD>;
    proxy_funcs[AMO_ADD][ULONGLONG] = impl::amo_op<unsigned long long, AMO_ADD>;
    proxy_funcs[AMO_FETCH_AND][ULONGLONG] = impl::amo_fetch_op<unsigned long long, AMO_FETCH_AND>;
    proxy_funcs[AMO_AND][ULONGLONG] = impl::amo_op<unsigned long long, AMO_AND>;
    proxy_funcs[AMO_FETCH_OR][ULONGLONG] = impl::amo_fetch_op<unsigned long long, AMO_FETCH_OR>;
    proxy_funcs[AMO_OR][ULONGLONG] = impl::amo_op<unsigned long long, AMO_OR>;
    proxy_funcs[AMO_FETCH_XOR][ULONGLONG] = impl::amo_fetch_op<unsigned long long, AMO_FETCH_XOR>;
    proxy_funcs[AMO_XOR][ULONGLONG] = impl::amo_op<unsigned long long, AMO_XOR>;

    proxy_funcs[AMO_FETCH][INT32] = impl::amo_fetch_op<int32_t, AMO_FETCH>;
    proxy_funcs[AMO_SET][INT32] = impl::amo_op<int32_t, AMO_SET>;
    proxy_funcs[AMO_COMPARE_SWAP][INT32] = impl::amo_compare_swap<int32_t, AMO_COMPARE_SWAP>;
    proxy_funcs[AMO_SWAP][INT32] = impl::amo_fetch_op<int32_t, AMO_SWAP>;
    proxy_funcs[AMO_FETCH_INC][INT32] = impl::amo_fetch_op<int32_t, AMO_FETCH_INC>;
    proxy_funcs[AMO_INC][INT32] = impl::amo_op<int32_t, AMO_INC>;
    proxy_funcs[AMO_FETCH_ADD][INT32] = impl::amo_fetch_op<int32_t, AMO_FETCH_ADD>;
    proxy_funcs[AMO_ADD][INT32] = impl::amo_op<int32_t, AMO_ADD>;
    proxy_funcs[AMO_FETCH_AND][INT32] = impl::amo_fetch_op<int32_t, AMO_FETCH_AND>;
    proxy_funcs[AMO_AND][INT32] = impl::amo_op<int32_t, AMO_AND>;
    proxy_funcs[AMO_FETCH_OR][INT32] = impl::amo_fetch_op<int32_t, AMO_FETCH_OR>;
    proxy_funcs[AMO_OR][INT32] = impl::amo_op<int32_t, AMO_OR>;
    proxy_funcs[AMO_FETCH_XOR][INT32] = impl::amo_fetch_op<int32_t, AMO_FETCH_XOR>;
    proxy_funcs[AMO_XOR][INT32] = impl::amo_op<int32_t, AMO_XOR>;

    proxy_funcs[AMO_FETCH][INT64] = impl::amo_fetch_op<int64_t, AMO_FETCH>;
    proxy_funcs[AMO_SET][INT64] = impl::amo_op<int64_t, AMO_SET>;
    proxy_funcs[AMO_COMPARE_SWAP][INT64] = impl::amo_compare_swap<int64_t, AMO_COMPARE_SWAP>;
    proxy_funcs[AMO_SWAP][INT64] = impl::amo_fetch_op<int64_t, AMO_SWAP>;
    proxy_funcs[AMO_FETCH_INC][INT64] = impl::amo_fetch_op<int64_t, AMO_FETCH_INC>;
    proxy_funcs[AMO_INC][INT64] = impl::amo_op<int64_t, AMO_INC>;
    proxy_funcs[AMO_FETCH_ADD][INT64] = impl::amo_fetch_op<int64_t, AMO_FETCH_ADD>;
    proxy_funcs[AMO_ADD][INT64] = impl::amo_op<int64_t, AMO_ADD>;
    proxy_funcs[AMO_FETCH_AND][INT64] = impl::amo_fetch_op<int64_t, AMO_FETCH_AND>;
    proxy_funcs[AMO_AND][INT64] = impl::amo_op<int64_t, AMO_AND>;
    proxy_funcs[AMO_FETCH_OR][INT64] = impl::amo_fetch_op<int64_t, AMO_FETCH_OR>;
    proxy_funcs[AMO_OR][INT64] = impl::amo_op<int64_t, AMO_OR>;
    proxy_funcs[AMO_FETCH_XOR][INT64] = impl::amo_fetch_op<int64_t, AMO_FETCH_XOR>;
    proxy_funcs[AMO_XOR][INT64] = impl::amo_op<int64_t, AMO_XOR>;

    proxy_funcs[AMO_FETCH][LONGLONG] = impl::amo_fetch_op<long long, AMO_FETCH>;
    proxy_funcs[AMO_SET][LONGLONG] = impl::amo_op<long long, AMO_SET>;
    proxy_funcs[AMO_COMPARE_SWAP][LONGLONG] = impl::amo_compare_swap<long long, AMO_COMPARE_SWAP>;
    proxy_funcs[AMO_SWAP][LONGLONG] = impl::amo_fetch_op<long long, AMO_SWAP>;
    proxy_funcs[AMO_FETCH_INC][LONGLONG] = impl::amo_fetch_op<long long, AMO_FETCH_INC>;
    proxy_funcs[AMO_INC][LONGLONG] = impl::amo_op<long long, AMO_INC>;
    proxy_funcs[AMO_FETCH_ADD][LONGLONG] = impl::amo_fetch_op<long long, AMO_FETCH_ADD>;
    proxy_funcs[AMO_ADD][LONGLONG] = impl::amo_op<long long, AMO_ADD>;

    proxy_funcs[AMO_FETCH][FLOAT] = impl::amo_fetch_op<float, AMO_FETCH>;
    proxy_funcs[AMO_SET][FLOAT] = impl::amo_op<float, AMO_SET>;
    proxy_funcs[AMO_SWAP][FLOAT] = impl::amo_fetch_op<float, AMO_SWAP>;

    proxy_funcs[AMO_FETCH][DOUBLE] = impl::amo_fetch_op<double, AMO_FETCH>;
    proxy_funcs[AMO_SET][DOUBLE] = impl::amo_op<double, AMO_SET>;
    proxy_funcs[AMO_SWAP][DOUBLE] = impl::amo_fetch_op<double, AMO_SWAP>;

    /* AMO NBI */
    proxy_funcs[AMO_FETCH_NBI][UINT32] = impl::amo_fetch_op<uint32_t, AMO_FETCH_NBI, false>;
    proxy_funcs[AMO_FETCH_NBI][INT32] = impl::amo_fetch_op<int32_t, AMO_FETCH_NBI, false>;
    proxy_funcs[AMO_FETCH_NBI][UINT64] = impl::amo_fetch_op<uint64_t, AMO_FETCH_NBI, false>;
    proxy_funcs[AMO_FETCH_NBI][INT64] = impl::amo_fetch_op<int64_t, AMO_FETCH_NBI, false>;
    proxy_funcs[AMO_FETCH_NBI][ULONGLONG] =
        impl::amo_fetch_op<unsigned long long, AMO_FETCH_NBI, false>;
    proxy_funcs[AMO_FETCH_NBI][LONGLONG] = impl::amo_fetch_op<long long, AMO_FETCH_NBI, false>;
    proxy_funcs[AMO_FETCH_NBI][FLOAT] = impl::amo_fetch_op<float, AMO_FETCH_NBI, false>;
    proxy_funcs[AMO_FETCH_NBI][DOUBLE] = impl::amo_fetch_op<double, AMO_FETCH_NBI, false>;

    proxy_funcs[AMO_COMPARE_SWAP_NBI][UINT32] =
        impl::amo_compare_swap<uint32_t, AMO_COMPARE_SWAP_NBI, false>;
    proxy_funcs[AMO_COMPARE_SWAP_NBI][INT32] =
        impl::amo_compare_swap<int32_t, AMO_COMPARE_SWAP_NBI, false>;
    proxy_funcs[AMO_COMPARE_SWAP_NBI][UINT64] =
        impl::amo_compare_swap<uint64_t, AMO_COMPARE_SWAP_NBI, false>;
    proxy_funcs[AMO_COMPARE_SWAP_NBI][INT64] =
        impl::amo_compare_swap<int64_t, AMO_COMPARE_SWAP_NBI, false>;
    proxy_funcs[AMO_COMPARE_SWAP_NBI][ULONGLONG] =
        impl::amo_compare_swap<unsigned long long, AMO_COMPARE_SWAP_NBI, false>;
    proxy_funcs[AMO_COMPARE_SWAP_NBI][LONGLONG] =
        impl::amo_compare_swap<long long, AMO_COMPARE_SWAP_NBI, false>;

    proxy_funcs[AMO_SWAP_NBI][UINT32] = impl::amo_fetch_op<uint32_t, AMO_SWAP_NBI, false>;
    proxy_funcs[AMO_SWAP_NBI][INT32] = impl::amo_fetch_op<int32_t, AMO_SWAP_NBI, false>;
    proxy_funcs[AMO_SWAP_NBI][UINT64] = impl::amo_fetch_op<uint64_t, AMO_SWAP_NBI, false>;
    proxy_funcs[AMO_SWAP_NBI][INT64] = impl::amo_fetch_op<int64_t, AMO_SWAP_NBI, false>;
    proxy_funcs[AMO_SWAP_NBI][ULONGLONG] =
        impl::amo_fetch_op<unsigned long long, AMO_SWAP_NBI, false>;
    proxy_funcs[AMO_SWAP_NBI][LONGLONG] = impl::amo_fetch_op<long long, AMO_SWAP_NBI, false>;
    proxy_funcs[AMO_SWAP_NBI][FLOAT] = impl::amo_fetch_op<float, AMO_SWAP_NBI, false>;
    proxy_funcs[AMO_SWAP_NBI][DOUBLE] = impl::amo_fetch_op<double, AMO_SWAP_NBI, false>;

    proxy_funcs[AMO_FETCH_INC_NBI][UINT32] = impl::amo_fetch_op<uint32_t, AMO_FETCH_INC_NBI, false>;
    proxy_funcs[AMO_FETCH_INC_NBI][INT32] = impl::amo_fetch_op<int32_t, AMO_FETCH_INC_NBI, false>;
    proxy_funcs[AMO_FETCH_INC_NBI][UINT64] = impl::amo_fetch_op<uint64_t, AMO_FETCH_INC_NBI, false>;
    proxy_funcs[AMO_FETCH_INC_NBI][INT64] = impl::amo_fetch_op<int64_t, AMO_FETCH_INC_NBI, false>;
    proxy_funcs[AMO_FETCH_INC_NBI][ULONGLONG] =
        impl::amo_fetch_op<unsigned long long, AMO_FETCH_INC_NBI, false>;
    proxy_funcs[AMO_FETCH_INC_NBI][LONGLONG] =
        impl::amo_fetch_op<long long, AMO_FETCH_INC_NBI, false>;

    proxy_funcs[AMO_FETCH_ADD_NBI][UINT32] = impl::amo_fetch_op<uint32_t, AMO_FETCH_ADD_NBI, false>;
    proxy_funcs[AMO_FETCH_ADD_NBI][INT32] = impl::amo_fetch_op<int32_t, AMO_FETCH_ADD_NBI, false>;
    proxy_funcs[AMO_FETCH_ADD_NBI][UINT64] = impl::amo_fetch_op<uint64_t, AMO_FETCH_ADD_NBI, false>;
    proxy_funcs[AMO_FETCH_ADD_NBI][INT64] = impl::amo_fetch_op<int64_t, AMO_FETCH_ADD_NBI, false>;
    proxy_funcs[AMO_FETCH_ADD_NBI][ULONGLONG] =
        impl::amo_fetch_op<unsigned long long, AMO_FETCH_ADD_NBI, false>;
    proxy_funcs[AMO_FETCH_ADD_NBI][LONGLONG] =
        impl::amo_fetch_op<long long, AMO_FETCH_ADD_NBI, false>;

    proxy_funcs[AMO_FETCH_AND_NBI][UINT32] = impl::amo_fetch_op<uint32_t, AMO_FETCH_AND_NBI, false>;
    proxy_funcs[AMO_FETCH_AND_NBI][INT32] = impl::amo_fetch_op<int32_t, AMO_FETCH_AND_NBI, false>;
    proxy_funcs[AMO_FETCH_AND_NBI][UINT64] = impl::amo_fetch_op<uint64_t, AMO_FETCH_AND_NBI, false>;
    proxy_funcs[AMO_FETCH_AND_NBI][INT64] = impl::amo_fetch_op<int64_t, AMO_FETCH_AND_NBI, false>;
    proxy_funcs[AMO_FETCH_AND_NBI][ULONGLONG] =
        impl::amo_fetch_op<unsigned long long, AMO_FETCH_AND_NBI, false>;

    proxy_funcs[AMO_FETCH_OR_NBI][UINT32] = impl::amo_fetch_op<uint32_t, AMO_FETCH_OR_NBI, false>;
    proxy_funcs[AMO_FETCH_OR_NBI][INT32] = impl::amo_fetch_op<int32_t, AMO_FETCH_OR_NBI, false>;
    proxy_funcs[AMO_FETCH_OR_NBI][UINT64] = impl::amo_fetch_op<uint64_t, AMO_FETCH_OR_NBI, false>;
    proxy_funcs[AMO_FETCH_OR_NBI][INT64] = impl::amo_fetch_op<int64_t, AMO_FETCH_OR_NBI, false>;
    proxy_funcs[AMO_FETCH_OR_NBI][ULONGLONG] =
        impl::amo_fetch_op<unsigned long long, AMO_FETCH_OR_NBI, false>;

    proxy_funcs[AMO_FETCH_XOR_NBI][UINT32] = impl::amo_fetch_op<uint32_t, AMO_FETCH_XOR_NBI, false>;
    proxy_funcs[AMO_FETCH_XOR_NBI][INT32] = impl::amo_fetch_op<int32_t, AMO_FETCH_XOR_NBI, false>;
    proxy_funcs[AMO_FETCH_XOR_NBI][UINT64] = impl::amo_fetch_op<uint64_t, AMO_FETCH_XOR_NBI, false>;
    proxy_funcs[AMO_FETCH_XOR_NBI][INT64] = impl::amo_fetch_op<int64_t, AMO_FETCH_XOR_NBI, false>;
    proxy_funcs[AMO_FETCH_XOR_NBI][ULONGLONG] =
        impl::amo_fetch_op<unsigned long long, AMO_FETCH_XOR_NBI, false>;

    /* Signaling */
    proxy_funcs[PUT_SIGNAL][UINT8] = impl::put<uint8_t, PUT, true, true>;
    proxy_funcs[PUT_SIGNAL_NBI][UINT8] = impl::put<uint8_t, PUT, false, true>;
    proxy_funcs[SIGNAL_ADD][UINT64] = impl::amo_op<uint64_t, AMO_ADD>;
    proxy_funcs[SIGNAL_SET][UINT64] = impl::amo_op<uint64_t, AMO_SET>;
    proxy_funcs[SIGNAL_WAIT_UNTIL][UINT64] = impl::signal_wait_until;

    /* Teams */
    proxy_funcs[TEAM_MY_PE][0] = impl::team_my_pe;
    proxy_funcs[TEAM_N_PES][0] = impl::team_n_pes;
    proxy_funcs[TEAM_SYNC][0] = impl::sync<TEAM_SYNC>;

    /* Collectives */
    proxy_funcs[BARRIER][0] = impl::barrier;
    proxy_funcs[SYNC][0] = impl::sync<SYNC>;
    proxy_funcs[ALLTOALL][UINT8] = impl::alltoall;
    proxy_funcs[BCAST][UINT8] = impl::broadcast;
    proxy_funcs[COLLECT][UINT8] = impl::collect;
    proxy_funcs[FCOLLECT][UINT8] = impl::fcollect;

    /* Reductions */
    proxy_funcs[AND_REDUCE][UINT8] = impl::reduce<uint8_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][UINT8] = impl::reduce<uint8_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][UINT8] = impl::reduce<uint8_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][UINT8] = impl::reduce<uint8_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][UINT8] = impl::reduce<uint8_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][UINT8] = impl::reduce<uint8_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][UINT8] = impl::reduce<uint8_t, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][UINT16] = impl::reduce<uint16_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][UINT16] = impl::reduce<uint16_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][UINT16] = impl::reduce<uint16_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][UINT16] = impl::reduce<uint16_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][UINT16] = impl::reduce<uint16_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][UINT16] = impl::reduce<uint16_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][UINT16] = impl::reduce<uint16_t, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][UINT32] = impl::reduce<uint32_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][UINT32] = impl::reduce<uint32_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][UINT32] = impl::reduce<uint32_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][UINT32] = impl::reduce<uint32_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][UINT32] = impl::reduce<uint32_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][UINT32] = impl::reduce<uint32_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][UINT32] = impl::reduce<uint32_t, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][UINT64] = impl::reduce<uint64_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][UINT64] = impl::reduce<uint64_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][UINT64] = impl::reduce<uint64_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][UINT64] = impl::reduce<uint64_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][UINT64] = impl::reduce<uint64_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][UINT64] = impl::reduce<uint64_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][UINT64] = impl::reduce<uint64_t, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][ULONGLONG] = impl::reduce<unsigned long long, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][ULONGLONG] = impl::reduce<unsigned long long, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][ULONGLONG] = impl::reduce<unsigned long long, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][ULONGLONG] = impl::reduce<unsigned long long, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][ULONGLONG] = impl::reduce<unsigned long long, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][ULONGLONG] = impl::reduce<unsigned long long, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][ULONGLONG] = impl::reduce<unsigned long long, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][INT8] = impl::reduce<int8_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][INT8] = impl::reduce<int8_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][INT8] = impl::reduce<int8_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][INT8] = impl::reduce<int8_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][INT8] = impl::reduce<int8_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][INT8] = impl::reduce<int8_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][INT8] = impl::reduce<int8_t, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][INT16] = impl::reduce<int16_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][INT16] = impl::reduce<int16_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][INT16] = impl::reduce<int16_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][INT16] = impl::reduce<int16_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][INT16] = impl::reduce<int16_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][INT16] = impl::reduce<int16_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][INT16] = impl::reduce<int16_t, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][INT32] = impl::reduce<int32_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][INT32] = impl::reduce<int32_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][INT32] = impl::reduce<int32_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][INT32] = impl::reduce<int32_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][INT32] = impl::reduce<int32_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][INT32] = impl::reduce<int32_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][INT32] = impl::reduce<int32_t, PROD_REDUCE>;

    proxy_funcs[AND_REDUCE][INT64] = impl::reduce<int64_t, AND_REDUCE>;
    proxy_funcs[OR_REDUCE][INT64] = impl::reduce<int64_t, OR_REDUCE>;
    proxy_funcs[XOR_REDUCE][INT64] = impl::reduce<int64_t, XOR_REDUCE>;
    proxy_funcs[MAX_REDUCE][INT64] = impl::reduce<int64_t, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][INT64] = impl::reduce<int64_t, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][INT64] = impl::reduce<int64_t, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][INT64] = impl::reduce<int64_t, PROD_REDUCE>;

    proxy_funcs[MAX_REDUCE][LONGLONG] = impl::reduce<long long, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][LONGLONG] = impl::reduce<long long, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][LONGLONG] = impl::reduce<long long, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][LONGLONG] = impl::reduce<long long, PROD_REDUCE>;

    proxy_funcs[MAX_REDUCE][FLOAT] = impl::reduce<float, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][FLOAT] = impl::reduce<float, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][FLOAT] = impl::reduce<float, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][FLOAT] = impl::reduce<float, PROD_REDUCE>;

    proxy_funcs[MAX_REDUCE][DOUBLE] = impl::reduce<double, MAX_REDUCE>;
    proxy_funcs[MIN_REDUCE][DOUBLE] = impl::reduce<double, MIN_REDUCE>;
    proxy_funcs[SUM_REDUCE][DOUBLE] = impl::reduce<double, SUM_REDUCE>;
    proxy_funcs[PROD_REDUCE][DOUBLE] = impl::reduce<double, PROD_REDUCE>;

    /* Scan */
    proxy_funcs[INSCAN][UINT8] = impl::inscan<uint8_t>;
    proxy_funcs[INSCAN][UINT16] = impl::inscan<uint16_t>;
    proxy_funcs[INSCAN][UINT32] = impl::inscan<uint32_t>;
    proxy_funcs[INSCAN][UINT64] = impl::inscan<uint64_t>;
    proxy_funcs[INSCAN][ULONGLONG] = impl::inscan<unsigned long long>;
    proxy_funcs[INSCAN][INT8] = impl::inscan<int8_t>;
    proxy_funcs[INSCAN][INT16] = impl::inscan<int16_t>;
    proxy_funcs[INSCAN][INT32] = impl::inscan<int32_t>;
    proxy_funcs[INSCAN][INT64] = impl::inscan<int64_t>;
    proxy_funcs[INSCAN][LONGLONG] = impl::inscan<long long>;
    proxy_funcs[INSCAN][FLOAT] = impl::inscan<float>;
    proxy_funcs[INSCAN][DOUBLE] = impl::inscan<double>;

    proxy_funcs[EXSCAN][UINT8] = impl::exscan<uint8_t>;
    proxy_funcs[EXSCAN][UINT16] = impl::exscan<uint16_t>;
    proxy_funcs[EXSCAN][UINT32] = impl::exscan<uint32_t>;
    proxy_funcs[EXSCAN][UINT64] = impl::exscan<uint64_t>;
    proxy_funcs[EXSCAN][ULONGLONG] = impl::exscan<unsigned long long>;
    proxy_funcs[EXSCAN][INT8] = impl::exscan<int8_t>;
    proxy_funcs[EXSCAN][INT16] = impl::exscan<int16_t>;
    proxy_funcs[EXSCAN][INT32] = impl::exscan<int32_t>;
    proxy_funcs[EXSCAN][INT64] = impl::exscan<int64_t>;
    proxy_funcs[EXSCAN][LONGLONG] = impl::exscan<long long>;
    proxy_funcs[EXSCAN][FLOAT] = impl::exscan<float>;
    proxy_funcs[EXSCAN][DOUBLE] = impl::exscan<double>;

    /* Point-to-point Synchronization */
    proxy_funcs[TEST][INT32] = impl::test<int32_t, TEST>;
    proxy_funcs[TEST_ALL][INT32] = impl::test_all<int32_t, TEST_ALL, false>;
    proxy_funcs[TEST_ANY][INT32] = impl::test_any<int32_t, TEST_ANY, false>;
    proxy_funcs[TEST_SOME][INT32] = impl::test_some<int32_t, TEST_SOME, false>;
    proxy_funcs[WAIT][INT32] = impl::wait_until<int32_t, WAIT>;
    proxy_funcs[WAIT_ALL][INT32] = impl::wait_until_all<int32_t, WAIT_ALL, false>;
    proxy_funcs[WAIT_ANY][INT32] = impl::wait_until_any<int32_t, WAIT_ANY, false>;
    proxy_funcs[WAIT_SOME][INT32] = impl::wait_until_some<int32_t, WAIT_SOME, false>;
    proxy_funcs[TEST_ALL_VECTOR][INT32] = impl::test_all<int32_t, TEST_ALL_VECTOR, true>;
    proxy_funcs[TEST_ANY_VECTOR][INT32] = impl::test_any<int32_t, TEST_ANY_VECTOR, true>;
    proxy_funcs[TEST_SOME_VECTOR][INT32] = impl::test_some<int32_t, TEST_SOME_VECTOR, true>;
    proxy_funcs[WAIT_ALL_VECTOR][INT32] = impl::wait_until_all<int32_t, WAIT_ALL_VECTOR, true>;
    proxy_funcs[WAIT_ANY_VECTOR][INT32] = impl::wait_until_any<int32_t, WAIT_ANY_VECTOR, true>;
    proxy_funcs[WAIT_SOME_VECTOR][INT32] = impl::wait_until_some<int32_t, WAIT_SOME_VECTOR, true>;

    proxy_funcs[TEST][INT64] = impl::test<int64_t, TEST>;
    proxy_funcs[TEST_ALL][INT64] = impl::test_all<int64_t, TEST_ALL, false>;
    proxy_funcs[TEST_ANY][INT64] = impl::test_any<int64_t, TEST_ANY, false>;
    proxy_funcs[TEST_SOME][INT64] = impl::test_some<int64_t, TEST_SOME, false>;
    proxy_funcs[WAIT][INT64] = impl::wait_until<int64_t, WAIT>;
    proxy_funcs[WAIT_ALL][INT64] = impl::wait_until_all<int64_t, WAIT_ALL, false>;
    proxy_funcs[WAIT_ANY][INT64] = impl::wait_until_any<int64_t, WAIT_ANY, false>;
    proxy_funcs[WAIT_SOME][INT64] = impl::wait_until_some<int64_t, WAIT_SOME, false>;
    proxy_funcs[TEST_ALL_VECTOR][INT64] = impl::test_all<int64_t, TEST_ALL_VECTOR, true>;
    proxy_funcs[TEST_ANY_VECTOR][INT64] = impl::test_any<int64_t, TEST_ANY_VECTOR, true>;
    proxy_funcs[TEST_SOME_VECTOR][INT64] = impl::test_some<int64_t, TEST_SOME_VECTOR, true>;
    proxy_funcs[WAIT_ALL_VECTOR][INT64] = impl::wait_until_all<int64_t, WAIT_ALL_VECTOR, true>;
    proxy_funcs[WAIT_ANY_VECTOR][INT64] = impl::wait_until_any<int64_t, WAIT_ANY_VECTOR, true>;
    proxy_funcs[WAIT_SOME_VECTOR][INT64] = impl::wait_until_some<int64_t, WAIT_SOME_VECTOR, true>;

    proxy_funcs[TEST][LONGLONG] = impl::test<long long, TEST>;
    proxy_funcs[TEST_ALL][LONGLONG] = impl::test_all<long long, TEST_ALL, false>;
    proxy_funcs[TEST_ANY][LONGLONG] = impl::test_any<long long, TEST_ANY, false>;
    proxy_funcs[TEST_SOME][LONGLONG] = impl::test_some<long long, TEST_SOME, false>;
    proxy_funcs[WAIT][LONGLONG] = impl::wait_until<long long, WAIT>;
    proxy_funcs[WAIT_ALL][LONGLONG] = impl::wait_until_all<long long, WAIT_ALL, false>;
    proxy_funcs[WAIT_ANY][LONGLONG] = impl::wait_until_any<long long, WAIT_ANY, false>;
    proxy_funcs[WAIT_SOME][LONGLONG] = impl::wait_until_some<long long, WAIT_SOME, false>;
    proxy_funcs[TEST_ALL_VECTOR][LONGLONG] = impl::test_all<long long, TEST_ALL_VECTOR, true>;
    proxy_funcs[TEST_ANY_VECTOR][LONGLONG] = impl::test_any<long long, TEST_ANY_VECTOR, true>;
    proxy_funcs[TEST_SOME_VECTOR][LONGLONG] = impl::test_some<long long, TEST_SOME_VECTOR, true>;
    proxy_funcs[WAIT_ALL_VECTOR][LONGLONG] = impl::wait_until_all<long long, WAIT_ALL_VECTOR, true>;
    proxy_funcs[WAIT_ANY_VECTOR][LONGLONG] = impl::wait_until_any<long long, WAIT_ANY_VECTOR, true>;
    proxy_funcs[WAIT_SOME_VECTOR][LONGLONG] =
        impl::wait_until_some<long long, WAIT_SOME_VECTOR, true>;

    proxy_funcs[TEST][UINT32] = impl::test<uint32_t, TEST>;
    proxy_funcs[TEST_ALL][UINT32] = impl::test_all<uint32_t, TEST_ALL, false>;
    proxy_funcs[TEST_ANY][UINT32] = impl::test_any<uint32_t, TEST_ANY, false>;
    proxy_funcs[TEST_SOME][UINT32] = impl::test_some<uint32_t, TEST_SOME, false>;
    proxy_funcs[WAIT][UINT32] = impl::wait_until<uint32_t, WAIT>;
    proxy_funcs[WAIT_ALL][UINT32] = impl::wait_until_all<uint32_t, WAIT_ALL, false>;
    proxy_funcs[WAIT_ANY][UINT32] = impl::wait_until_any<uint32_t, WAIT_ANY, false>;
    proxy_funcs[WAIT_SOME][UINT32] = impl::wait_until_some<uint32_t, WAIT_SOME, false>;
    proxy_funcs[TEST_ALL_VECTOR][UINT32] = impl::test_all<uint32_t, TEST_ALL_VECTOR, true>;
    proxy_funcs[TEST_ANY_VECTOR][UINT32] = impl::test_any<uint32_t, TEST_ANY_VECTOR, true>;
    proxy_funcs[TEST_SOME_VECTOR][UINT32] = impl::test_some<uint32_t, TEST_SOME_VECTOR, true>;
    proxy_funcs[WAIT_ALL_VECTOR][UINT32] = impl::wait_until_all<uint32_t, WAIT_ALL_VECTOR, true>;
    proxy_funcs[WAIT_ANY_VECTOR][UINT32] = impl::wait_until_any<uint32_t, WAIT_ANY_VECTOR, true>;
    proxy_funcs[WAIT_SOME_VECTOR][UINT32] = impl::wait_until_some<uint32_t, WAIT_SOME_VECTOR, true>;

    proxy_funcs[TEST][UINT64] = impl::test<uint64_t, TEST>;
    proxy_funcs[TEST_ALL][UINT64] = impl::test_all<uint64_t, TEST_ALL, false>;
    proxy_funcs[TEST_ANY][UINT64] = impl::test_any<uint64_t, TEST_ANY, false>;
    proxy_funcs[TEST_SOME][UINT64] = impl::test_some<uint64_t, TEST_SOME, false>;
    proxy_funcs[WAIT][UINT64] = impl::wait_until<uint64_t, WAIT>;
    proxy_funcs[WAIT_ALL][UINT64] = impl::wait_until_all<uint64_t, WAIT_ALL, false>;
    proxy_funcs[WAIT_ANY][UINT64] = impl::wait_until_any<uint64_t, WAIT_ANY, false>;
    proxy_funcs[WAIT_SOME][UINT64] = impl::wait_until_some<uint64_t, WAIT_SOME, false>;
    proxy_funcs[TEST_ALL_VECTOR][UINT64] = impl::test_all<uint64_t, TEST_ALL_VECTOR, true>;
    proxy_funcs[TEST_ANY_VECTOR][UINT64] = impl::test_any<uint64_t, TEST_ANY_VECTOR, true>;
    proxy_funcs[TEST_SOME_VECTOR][UINT64] = impl::test_some<uint64_t, TEST_SOME_VECTOR, true>;
    proxy_funcs[WAIT_ALL_VECTOR][UINT64] = impl::wait_until_all<uint64_t, WAIT_ALL_VECTOR, true>;
    proxy_funcs[WAIT_ANY_VECTOR][UINT64] = impl::wait_until_any<uint64_t, WAIT_ANY_VECTOR, true>;
    proxy_funcs[WAIT_SOME_VECTOR][UINT64] = impl::wait_until_some<uint64_t, WAIT_SOME_VECTOR, true>;

    proxy_funcs[TEST][ULONGLONG] = impl::test<unsigned long long, TEST>;
    proxy_funcs[TEST_ALL][ULONGLONG] = impl::test_all<unsigned long long, TEST_ALL, false>;
    proxy_funcs[TEST_ANY][ULONGLONG] = impl::test_any<unsigned long long, TEST_ANY, false>;
    proxy_funcs[TEST_SOME][ULONGLONG] = impl::test_some<unsigned long long, TEST_SOME, false>;
    proxy_funcs[WAIT][ULONGLONG] = impl::wait_until<unsigned long long, WAIT>;
    proxy_funcs[WAIT_ALL][ULONGLONG] = impl::wait_until_all<unsigned long long, WAIT_ALL, false>;
    proxy_funcs[WAIT_ANY][ULONGLONG] = impl::wait_until_any<unsigned long long, WAIT_ANY, false>;
    proxy_funcs[WAIT_SOME][ULONGLONG] = impl::wait_until_some<unsigned long long, WAIT_SOME, false>;
    proxy_funcs[TEST_ALL_VECTOR][ULONGLONG] =
        impl::test_all<unsigned long long, TEST_ALL_VECTOR, true>;
    proxy_funcs[TEST_ANY_VECTOR][ULONGLONG] =
        impl::test_any<unsigned long long, TEST_ANY_VECTOR, true>;
    proxy_funcs[TEST_SOME_VECTOR][ULONGLONG] =
        impl::test_some<unsigned long long, TEST_SOME_VECTOR, true>;
    proxy_funcs[WAIT_ALL_VECTOR][ULONGLONG] =
        impl::wait_until_all<unsigned long long, WAIT_ALL_VECTOR, true>;
    proxy_funcs[WAIT_ANY_VECTOR][ULONGLONG] =
        impl::wait_until_any<unsigned long long, WAIT_ANY_VECTOR, true>;
    proxy_funcs[WAIT_SOME_VECTOR][ULONGLONG] =
        impl::wait_until_some<unsigned long long, WAIT_SOME_VECTOR, true>;

    /* Memory Ordering */
    proxy_funcs[FENCE][0] = impl::fence;
    proxy_funcs[QUIET][0] = impl::quiet;

fn_exit:
    return;
}

void ishmemi_runtime_mpi::funcptr_fini(void)
{
    for (size_t i = 0; i < ISHMEMI_OP_END; ++i) {
        for (size_t j = 0; j < ishmemi_runtime_type::proxy_func_num_types; ++j) {
            proxy_funcs[i][j] = ishmemi_runtime_type::unsupported;
        }
        ISHMEMI_FREE(::free, proxy_funcs[i]);
    }
    ISHMEMI_FREE(::free, proxy_funcs);
}
