/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <time.h>
#include <ishmem_tester.h>
#include "rma_test.h"

#undef TEST_BRANCH_ON_QUEUE

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    *res = ishmem_##typename##_##testname(ISHMEM_TEAM_WORLD, (type *) dest, (type *) src, nelems);

#define TEST_BRANCH_ON_QUEUE(testname, typeenum, typename, type, op, opname)                       \
    ishmemx_##typename##_##testname##_on_queue((type *) dest, (type *) src, nelems, res, q);

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    *res = ishmemx_##typename##_##testname##_work_group(ISHMEM_TEAM_WORLD, (type *) dest,          \
                                                        (type *) src, nelems, grp);

GEN_FNS(collect, NOP, nop)

#undef TEST_BRANCH_SINGLE
#undef TEST_BRANCH_ON_QUEUE
#undef TEST_BRANCH_WORK_GROUP

#define TEST_BRANCH_SINGLE(testname, typeenum, typename, type, op, opname)                         \
    *res = ishmem_##testname(ISHMEM_TEAM_WORLD, (type *) dest, (type *) src, nelems);

#define TEST_BRANCH_ON_QUEUE(testname, typeenum, typename, type, op, opname)                       \
    ishmemx_##testname##_on_queue((type *) dest, (type *) src, nelems, res, q);

#define TEST_BRANCH_WORK_GROUP(testname, typeenum, typename, type, op, opname)                     \
    *res = ishmemx_##testname##_work_group(ISHMEM_TEAM_WORLD, (type *) dest, (type *) src, nelems, \
                                           grp);

GEN_MEM_FNS(collectmem, NOP, nop)

class collect_tester : public ishmem_tester {
  public:
    size_t *collect_nelems_source = nullptr;
    size_t *collect_nelems_dest = nullptr;

    collect_tester(int argc, char *argv[]) : ishmem_tester(argc, argv, true)
    {
        collect_nelems_source =
            (size_t *) ishmemi_test_runtime->malloc((size_t) n_pes * sizeof(size_t));
        collect_nelems_dest =
            (size_t *) ishmemi_test_runtime->malloc((size_t) n_pes * sizeof(size_t));
    }
    ~collect_tester()
    {
        assert(collect_nelems_source);
        ishmemi_test_runtime->free(collect_nelems_source);
        collect_nelems_source = nullptr;
        assert(collect_nelems_dest);
        ishmemi_test_runtime->free(collect_nelems_dest);
        collect_nelems_dest = nullptr;
    }
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
    virtual size_t run_offset_tests(ishmemi_op_t op);
};

/* result written into aligned_source, using host_source as a temp buffer if needed */
size_t collect_tester::create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                             size_t nelems)
{
    size_t test_size = nelems * typesize(t);
    size_t from_pe = (size_t) my_pe;
    for (size_t idx = 0; idx < ((test_size / sizeof(long)) + 1); idx += 1) {
        aligned_source[idx] =
            (long) ((nelems << 48) + ((0x80L + from_pe) << 40) + (0xffL << 32) + idx);
        if (patterndebugflag && (idx < 16)) {
            printf("[%d] source pattern idx %lu val %016lx\n", my_pe, idx, aligned_source[idx]);
        }
    }

    return (test_size);
}

/* check pattern written into host_check, using host_source as a temp buffer */
size_t collect_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                            size_t nelems)
{
    size_t test_size_per_pe = 0;
    size_t total_nelems = 0;
    for (size_t from_pe = 0; from_pe < n_pes; from_pe += 1) {
        test_size_per_pe = (size_t) collect_nelems_dest[from_pe] * typesize(t);
        for (size_t idx = 0; idx < ((test_size_per_pe / sizeof(long)) + 1); idx += 1) {
            host_source[idx] = (long) ((collect_nelems_dest[from_pe] << 48) +
                                       ((0x80L + from_pe) << 40) + (0xffL << 32) + idx);
        }
        memcpy((void *) (((uintptr_t) host_check) + (total_nelems * typesize(t))), host_source,
               test_size_per_pe);
        total_nelems += collect_nelems_dest[from_pe];
    }
    if (patterndebugflag) {
        for (size_t idx = 0; (idx < 16) && (idx < ((test_size_per_pe / sizeof(long)) + 1)); idx++) {
            printf("[%d] check pattern idx %lu val %016lx\n", my_pe, idx, host_check[idx]);
        }
    }

    return (total_nelems * typesize(t));
}

size_t collect_tester::run_offset_tests(ishmemi_op_t op)
{
    size_t errors = 0;
    /* quick tests of different source and destination offsets and small lengths */
    /* could be sped up by making the numbers of cases datatype dependent */
    printf("[%d] Run Offset Tests op %s\n", my_pe, ishmemi_op_str[op]);
    for (int i = 0; i < num_test_modes; i += 1) {
        testmode_t mode = test_modes[i];
        printf("[%d] Testing %s\n", my_pe, mode_to_str(mode));
        for (int typeindex = 0; typeindex < num_test_types; typeindex += 1) {
            ishmemi_type_t t = test_types[typeindex];
            if (my_pe == 0) {
                for (size_t i = 0; i < n_pes; ++i) {
                    collect_nelems_source[i] = (size_t) rand() % max_nelems + 1;
                }
            }
            ishmemi_test_runtime->broadcast(collect_nelems_dest, collect_nelems_source,
                                            (size_t) n_pes * sizeof(size_t), 0);
            /* offsets run from 0 to 15 in units of the datatype size */
            for (unsigned long source_offset = 0; source_offset < 15;
                 source_offset += typesize(t)) {
                for (unsigned long dest_offset = 0; dest_offset < 15; dest_offset += typesize(t)) {
                    if (verboseflag) {
                        printf("[%d] Test %s %s nelems %ld os %ld od %ld\n", my_pe,
                               mode_to_str(mode), type_to_str(t), collect_nelems_dest[my_pe],
                               source_offset, dest_offset);
                    }
                    errors += do_test(t, op, mode, collect_nelems_dest[my_pe], source_offset,
                                      dest_offset);
                }
            }
        }
    }
    return (errors);
}

int main(int argc, char **argv)
{
    class collect_tester t(argc, argv);

    size_t bufsize = (t.max_nelems * t.typesize(SIZE128) * (size_t) t.n_pes) + 4096L;
    t.alloc_memory(bufsize);
    size_t errors = 0;

    GEN_TABLES(collect, NOP, nop)
    GEN_MEM_TABLES(collectmem, NOP, nop)

    /* Only running offset test since aligned test is the same as zero offset */
    if (!t.test_types_set) t.add_test_type_list(collectives_copy_types);
    errors += t.run_offset_tests(NOP);

    return (t.finalize_and_report(errors));
}
