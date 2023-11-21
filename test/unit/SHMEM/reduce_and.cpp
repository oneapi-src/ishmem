/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include "ishmem_tester.h"

#ifdef ISHMEM_GEN_TYPE_FUNCTION
#undef ISHMEM_GEN_TYPE_FUNCTION
#endif
#define ISHMEM_GEN_TYPE_FUNCTION(function, returnvar, memcase)                                     \
    function                                                                                       \
    {                                                                                              \
        *test_run = true;                                                                          \
        returnvar switch (t)                                                                       \
        {                                                                                          \
            case UCHAR:                                                                            \
                ISHMEM_TYPE_BRANCH(UCHAR, uchar, unsigned char)                                    \
            case USHORT:                                                                           \
                ISHMEM_TYPE_BRANCH(USHORT, ushort, unsigned short)                                 \
            case UINT:                                                                             \
                ISHMEM_TYPE_BRANCH(UINT, uint, unsigned int)                                       \
            case ULONG:                                                                            \
                ISHMEM_TYPE_BRANCH(ULONG, ulong, unsigned long)                                    \
            case ULONGLONG:                                                                        \
                ISHMEM_TYPE_BRANCH(ULONGLONG, ulonglong, unsigned long long)                       \
            case INT8:                                                                             \
                ISHMEM_TYPE_BRANCH(INT8, int8, int8_t)                                             \
            case INT16:                                                                            \
                ISHMEM_TYPE_BRANCH(INT16, int16, int16_t)                                          \
            case INT32:                                                                            \
                ISHMEM_TYPE_BRANCH(INT32, int32, int32_t)                                          \
            case INT64:                                                                            \
                ISHMEM_TYPE_BRANCH(INT64, int64, int64_t)                                          \
            case UINT8:                                                                            \
                ISHMEM_TYPE_BRANCH(UINT8, uint8, uint8_t)                                          \
            case UINT16:                                                                           \
                ISHMEM_TYPE_BRANCH(UINT16, uint16, uint16_t)                                       \
            case UINT32:                                                                           \
                ISHMEM_TYPE_BRANCH(UINT32, uint32, uint32_t)                                       \
            case UINT64:                                                                           \
                ISHMEM_TYPE_BRANCH(UINT64, uint64, uint64_t)                                       \
            case SIZE:                                                                             \
                ISHMEM_TYPE_BRANCH(SIZE, size, size_t)                                             \
            default:                                                                               \
                *test_run = false;                                                                 \
                return (res);                                                                      \
        }                                                                                          \
        return (res);                                                                              \
    }

#ifdef ISHMEM_TYPE_BRANCH
#undef ISHMEM_TYPE_BRANCH
#endif

#define ISHMEM_TYPE_BRANCH(enumname, name, type)                                                   \
    res = ishmem_##name##_and_reduce((type *) dest, (type *) src, nelems);                         \
    break;

ISHMEM_GEN_TEST_FUNCTION_SINGLE(int res = 0;, break;)

#ifdef ISHMEM_TYPE_BRANCH
#undef ISHMEM_TYPE_BRANCH
#endif

#define ISHMEM_TYPE_BRANCH(enumname, name, type)                                                   \
    res = ishmemx_##name##_and_reduce_work_group((type *) dest, (type *) src, nelems, grp);        \
    break;

ISHMEM_GEN_TEST_FUNCTION_WORK_GROUP(int res = 0;, break;)

class reduce_and_tester : public ishmem_tester {
  public:
    reduce_and_tester(int argc, char *argv[]) : ishmem_tester(argc, argv) {}
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
};

/* result written into aligned_source, using host_source as a temp buffer if needed */
size_t reduce_and_tester::create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                                size_t nelems)
{
    /* initialize our source buffer */
    /* the source pattern is 64 bits, and does not change with datatype or buffer offsets, so
     *  a matching correction is made in the checking code
     *
     * because the pattern is 64 bit aligned, but the source buffer may be unaligned, we compose
     * the pattern in the aligned_source buffer. The do_test code will memcpy it to the host_source
     * or device_source buffer as needed.
     * This code can use host_source as a temp buffer
     */
    size_t test_size = nelems * typesize(t);
    int from_pe = my_pe;
    /* test pattern includes nelems, source pe, and index so you can tell apart values from other
     * tests or other PEs */
    for (size_t idx = 0; idx < ((test_size / sizeof(long)) + 1); idx += 1) {
        aligned_source[idx] = static_cast<long>(
            (nelems << 48) + static_cast<size_t>((0x80L + from_pe) << 40) + (0xffL << 32) + idx);
        if (patterndebugflag && (idx < 16)) {
            printf("[%d] source pattern idx %lu val %016lx\n", my_pe, idx, aligned_source[idx]);
        }
    }

    return (test_size);
}

/* check pattern written into host_check, using host_source as a temp buffer */
size_t reduce_and_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                               size_t nelems)
{
    /* initialize check buffer */
    /* all to all concatenates the source buffers from all PEs, so the check buffer does the same */
    /* this is not offset, because the copy from test_dest to host_result does the alignment */
    size_t test_size = nelems * typesize(t);
    long a = 0x80L;
    for (int i = 1; i < n_pes; i++) {
        a &= (0x80L + i);
    }
    for (size_t idx = 0; idx < ((test_size / sizeof(long)) + 1); idx += 1) {
        host_check[idx] =
            static_cast<long>((nelems << 48) + static_cast<size_t>(a << 40) + (0xffL << 32) + idx);
        if (patterndebugflag && (idx < 16)) {
            printf("[%d] check pattern idx %lu val %016lx\n", my_pe, idx, host_check[idx]);
        }
    }

    return (test_size);
}

int main(int argc, char **argv)
{
    class reduce_and_tester t(argc, argv);

    size_t bufsize =
        2 * ((t.max_nelems * sizeof(uint64_t) * static_cast<unsigned long>(t.n_pes)) + 4096);
    t.alloc_memory(bufsize);
    size_t errors = 0;

    errors += t.run_aligned_tests(NOP);
    errors += t.run_offset_tests(NOP);

    return (t.finalize_and_report(errors));
}
