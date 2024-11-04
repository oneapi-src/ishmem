/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ishmem/err.h>
#include "runtime_openshmem.h"
#include <dlfcn.h>
#include <shmem.h>

extern shmem_team_t __attribute__((weak)) SHMEM_TEAM_WORLD;

namespace ishmemi_test_openshmem_wrappers {
    static bool initialized = false;

    shmem_team_t *TEAM_WORLD;
    shmem_team_t SHMEM_TEAM_WORLD;

    /* Setup APIs */
    void (*init)(void);
    void (*finalize)(void);

    /* Memory APIs */
    void *(*calloc)(size_t, size_t);
    void *(*malloc)(size_t);
    void (*free)(void *);

    /* Operation APIs */
    void (*sync_all)(void);
    void (*broadcastmem)(shmem_team_t, void *, void *, size_t, int);
    void (*uint64_sum_reduce)(shmem_team_t, uint64_t *, uint64_t *, size_t);
    void (*float_sum_reduce)(shmem_team_t, float *, float *, size_t);

    /* dl handle */
    void *shmem_handle = nullptr;
    std::vector<void **> wrapper_list;

    static int fini_wrappers(void)
    {
        int ret = 0;
        for (auto p : wrapper_list)
            *p = nullptr;
        if (shmem_handle != nullptr) {
            ret = dlclose(shmem_handle);
            shmem_handle = nullptr;
            ISHMEM_CHECK_GOTO_MSG(ret, fn_exit, "dlclose failed %s\n", dlerror());
        }
        return 0;
    fn_exit:
        return 1;
    }

    static int init_wrappers(void)
    {
        int ret = 0;

        const char *shmem_libname = getenv("ISHMEM_SHMEM_LIB_NAME");
        if (shmem_libname == nullptr) {
            /* Default value */
            shmem_libname = "libsma.so";
        }

        if (initialized) goto fn_exit;
        initialized = true;

        shmem_handle = dlopen(shmem_libname, RTLD_NOW | RTLD_GLOBAL);

        if (shmem_handle == nullptr) {
            RAISE_ERROR_MSG("could not find shmem library '%s' in environment\n", shmem_libname);
        }

        /* Load the SHMEM_TEAM symbols */
        ISHMEMI_TEST_LINK_SYMBOL(shmem_handle, SHMEM, TEAM_WORLD);
        SHMEM_TEAM_WORLD = *TEAM_WORLD;

        /* Load the SHMEM APIs */
        ISHMEMI_TEST_LINK_SYMBOL(shmem_handle, shmem, init);
        ISHMEMI_TEST_LINK_SYMBOL(shmem_handle, shmem, finalize);
        ISHMEMI_TEST_LINK_SYMBOL(shmem_handle, shmem, calloc);
        ISHMEMI_TEST_LINK_SYMBOL(shmem_handle, shmem, malloc);
        ISHMEMI_TEST_LINK_SYMBOL(shmem_handle, shmem, free);
        ISHMEMI_TEST_LINK_SYMBOL(shmem_handle, shmem, sync_all);
        ISHMEMI_TEST_LINK_SYMBOL(shmem_handle, shmem, broadcastmem);
        ISHMEMI_TEST_LINK_SYMBOL(shmem_handle, shmem, uint64_sum_reduce);
        ISHMEMI_TEST_LINK_SYMBOL(shmem_handle, shmem, float_sum_reduce);

    fn_exit:
        return ret;
    }
}  // namespace ishmemi_test_openshmem_wrappers

ishmemi_test_runtime_openshmem::ishmemi_test_runtime_openshmem(void)
{
    ishmemi_test_openshmem_wrappers::init_wrappers();
}

ishmemi_test_runtime_openshmem::~ishmemi_test_runtime_openshmem(void)
{
    ishmemi_test_openshmem_wrappers::fini_wrappers();
}

ishmemx_runtime_type_t ishmemi_test_runtime_openshmem::get_type(void)
{
    return ISHMEMX_RUNTIME_OPENSHMEM;
}

void ishmemi_test_runtime_openshmem::init(void)
{
    ishmemi_test_openshmem_wrappers::init();
}

void ishmemi_test_runtime_openshmem::finalize(void)
{
    ishmemi_test_openshmem_wrappers::finalize();
}

void *ishmemi_test_runtime_openshmem::calloc(size_t num, size_t size)
{
    return ishmemi_test_openshmem_wrappers::calloc(num, size);
}

void *ishmemi_test_runtime_openshmem::malloc(size_t size)
{
    return ishmemi_test_openshmem_wrappers::malloc(size);
}

void ishmemi_test_runtime_openshmem::free(void *ptr)
{
    ishmemi_test_openshmem_wrappers::free(ptr);
}

void ishmemi_test_runtime_openshmem::sync()
{
    ishmemi_test_openshmem_wrappers::sync_all();
}

void ishmemi_test_runtime_openshmem::broadcast(void *dst, void *src, size_t size, int root)
{
    ishmemi_test_openshmem_wrappers::broadcastmem(ishmemi_test_openshmem_wrappers::SHMEM_TEAM_WORLD,
                                                  dst, src, size, root);
}

void ishmemi_test_runtime_openshmem::uint64_sum_reduce(uint64_t *dst, uint64_t *src, size_t num)
{
    ishmemi_test_openshmem_wrappers::uint64_sum_reduce(
        ishmemi_test_openshmem_wrappers::SHMEM_TEAM_WORLD, dst, src, num);
}

void ishmemi_test_runtime_openshmem::float_sum_reduce(float *dst, float *src, size_t num)
{
    ishmemi_test_openshmem_wrappers::float_sum_reduce(
        ishmemi_test_openshmem_wrappers::SHMEM_TEAM_WORLD, dst, src, num);
}
