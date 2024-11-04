/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ishmem/err.h>
#include "runtime_mpi.h"
#include <dlfcn.h>
#include <mpi.h>

namespace ishmemi_test_mpi_wrappers {
    static bool initialized = false;

    /* Setup APIs */
    void (*Init)(int *, char ***);
    void (*Finalize)(void);

    /* Operation APIs */
    void (*Barrier)(MPI_Comm);
    void (*Bcast)(void *, int, MPI_Datatype, int, MPI_Comm);
    void (*Allreduce)(void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm);

    /* dl handle */
    void *mpi_handle = nullptr;
    std::vector<void **> wrapper_list;

    static int fini_wrappers(void)
    {
        int ret = 0;
        for (auto p : wrapper_list)
            *p = nullptr;
        if (mpi_handle != nullptr) {
            ret = dlclose(mpi_handle);
            mpi_handle = nullptr;
            ISHMEM_CHECK_GOTO_MSG(ret, fn_exit, "dlclose failed %s\n", dlerror());
        }
        return 0;
    fn_exit:
        return 1;
    }

    static int init_wrappers(void)
    {
        int ret = 0;

        const char *mpi_libname = getenv("ISHMEM_MPI_LIB_NAME");
        if (mpi_libname == nullptr) {
            /* Default value */
            mpi_libname = "libmpi.so";
        }

        if (initialized) goto fn_exit;
        initialized = true;

        mpi_handle = dlopen(mpi_libname, RTLD_NOW | RTLD_GLOBAL);

        if (mpi_handle == nullptr) {
            RAISE_ERROR_MSG("could not find MPI library '%s' in environment\n", mpi_libname);
        }

        ISHMEMI_TEST_LINK_SYMBOL(mpi_handle, MPI, Init);
        ISHMEMI_TEST_LINK_SYMBOL(mpi_handle, MPI, Finalize);
        ISHMEMI_TEST_LINK_SYMBOL(mpi_handle, MPI, Barrier);
        ISHMEMI_TEST_LINK_SYMBOL(mpi_handle, MPI, Bcast);
        ISHMEMI_TEST_LINK_SYMBOL(mpi_handle, MPI, Allreduce);

    fn_exit:
        return ret;
    }
}  // namespace ishmemi_test_mpi_wrappers

ishmemi_test_runtime_mpi::ishmemi_test_runtime_mpi(void)
{
    ishmemi_test_mpi_wrappers::init_wrappers();
}

ishmemi_test_runtime_mpi::~ishmemi_test_runtime_mpi(void)
{
    ishmemi_test_mpi_wrappers::fini_wrappers();
}

ishmemx_runtime_type_t ishmemi_test_runtime_mpi::get_type(void)
{
    return ISHMEMX_RUNTIME_MPI;
}

void ishmemi_test_runtime_mpi::init(void)
{
    ishmemi_test_mpi_wrappers::Init(nullptr, nullptr);
}

void ishmemi_test_runtime_mpi::finalize(void)
{
    ishmemi_test_mpi_wrappers::Finalize();
}

void *ishmemi_test_runtime_mpi::calloc(size_t num, size_t size)
{
    return ::calloc(num, size);
}

void *ishmemi_test_runtime_mpi::malloc(size_t size)
{
    return ::malloc(size);
}

void ishmemi_test_runtime_mpi::free(void *ptr)
{
    ::free(ptr);
}

void ishmemi_test_runtime_mpi::sync()
{
    ishmemi_test_mpi_wrappers::Barrier(MPI_COMM_WORLD);
}

void ishmemi_test_runtime_mpi::broadcast(void *dst, void *src, size_t size, int root)
{
    int my_pe = ishmem_my_pe();
    if (my_pe == root) {
        ::memcpy(dst, src, size);
    }
    ishmemi_test_mpi_wrappers::Bcast(dst, (int) size, MPI_BYTE, root, MPI_COMM_WORLD);
}

void ishmemi_test_runtime_mpi::uint64_sum_reduce(uint64_t *dst, uint64_t *src, size_t num)
{
    ishmemi_test_mpi_wrappers::Allreduce(src, dst, (int) num, MPI_UINT64_T, MPI_SUM,
                                         MPI_COMM_WORLD);
}

void ishmemi_test_runtime_mpi::float_sum_reduce(float *dst, float *src, size_t num)
{
    ishmemi_test_mpi_wrappers::Allreduce(src, dst, (int) num, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}
