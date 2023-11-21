/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ishmem_tester.h>
#include <time.h>

#ifdef ISHMEM_TYPE_BRANCH
#undef ISHMEM_TYPE_BRANCH
#endif

#define ISHMEM_TYPE_BRANCH(enumname, name, type)                                                   \
    res = ishmem_##name##_collect((type *) dest, (type *) src, nelems);                            \
    break;

ISHMEM_GEN_TEST_FUNCTION_SINGLE(int res = 0;, res = ishmem_collectmem(dest COMMA src COMMA nelems);
                                break;)

#ifdef ISHMEM_TYPE_BRANCH
#undef ISHMEM_TYPE_BRANCH
#endif

#define ISHMEM_TYPE_BRANCH(enumname, name, type)                                                   \
    res = ishmemx_##name##_collect_work_group((type *) dest, (type *) src, nelems, grp);           \
    break;

ISHMEM_GEN_TEST_FUNCTION_WORK_GROUP(
    int res = 0;, res = ishmemx_collectmem_work_group(dest COMMA src COMMA nelems COMMA grp);
    break;)

class collect_tester : public ishmem_tester {
  public:
    size_t *collect_nelems_source = nullptr;
    size_t *collect_nelems_dest = nullptr;

    collect_tester(int argc, char *argv[]) : ishmem_tester(argc, argv)
    {
        collect_nelems_source = (size_t *) shmem_malloc((size_t) n_pes * sizeof(long));
        collect_nelems_dest = (size_t *) shmem_malloc((size_t) n_pes * sizeof(long));
    }
    ~collect_tester()
    {
        shmem_free(collect_nelems_source);
        shmem_free(collect_nelems_dest);
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
        printf("[%d] Testing %s\n", my_pe, modestr(mode));
        for (ishmemi_type_t t : ishmemi_type_t_Iterator()) {
            if (my_pe == 0) {
                for (size_t i = 0; i < n_pes; ++i) {
                    collect_nelems_source[i] = (size_t) rand() % max_nelems + 1;
                }
            }
            shmem_size_broadcast(SHMEM_TEAM_WORLD, collect_nelems_dest, collect_nelems_source,
                                 (size_t) n_pes, 0);
            /* offsets run from 0 to 15 in units of the datatype size */
            for (unsigned long source_offset = 0; source_offset < 15;
                 source_offset += typesize(t)) {
                for (unsigned long dest_offset = 0; dest_offset < 15; dest_offset += typesize(t)) {
                    if (verboseflag) {
                        printf("[%d] Test %s %s nelems %ld os %ld od %ld\n", my_pe, modestr(mode),
                               typestr(t), collect_nelems_dest[my_pe], source_offset, dest_offset);
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

    size_t bufsize = ((t.max_nelems * sizeof(uint64_t) * (size_t) t.n_pes) + 4096);
    t.alloc_memory(bufsize);
    size_t errors = 0;

    /* Only running offset test since aligned test is the same as zero offset */
    errors += t.run_offset_tests(NOP);

    return (t.finalize_and_report(errors));
}
