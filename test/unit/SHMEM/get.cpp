/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem_tester.h"

#ifdef ISHMEM_TYPE_BRANCH
#undef ISHMEM_TYPE_BRANCH
#endif

#define ISHMEM_TYPE_BRANCH(enumname, name, type)                                                   \
    ishmem_##name##_get((type *) dest, (type *) src, nelems, pe);                                  \
    break;

ISHMEM_GEN_TEST_FUNCTION_SINGLE(int res = 0; int pe = ((ishmem_my_pe() == 0)
                                                           ? (ishmem_n_pes() - 1)
                                                           : (ishmem_my_pe() - 1) % ishmem_n_pes());
                                , ishmem_getmem(dest COMMA src COMMA nelems COMMA pe); break;)

#ifdef ISHMEM_TYPE_BRANCH
#undef ISHMEM_TYPE_BRANCH
#endif

#define ISHMEM_TYPE_BRANCH(enumname, name, type)                                                   \
    ishmemx_##name##_get_work_group((type *) dest, (type *) src, nelems, pe, grp);                 \
    break;

ISHMEM_GEN_TEST_FUNCTION_WORK_GROUP(
    int res = 0; int pe = (ishmem_my_pe() + 1) % ishmem_n_pes();
    , ishmemx_getmem_work_group(dest COMMA src COMMA nelems COMMA pe COMMA grp); break;)

int main(int argc, char **argv)
{
    class ishmem_tester t(argc, argv);

    size_t bufsize = (t.max_nelems * sizeof(uint64_t)) + 4096;
    t.alloc_memory(bufsize);
    size_t errors = 0;

    errors += t.run_aligned_tests(NOP);
    errors += t.run_offset_tests(NOP);
    return (t.finalize_and_report(errors));
}
