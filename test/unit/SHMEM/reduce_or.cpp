/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include "ishmem_tester.h"
#include "reduce_test.h"

GEN_BITWISE_FNS(reduce, OR_REDUCE, or)

class reduce_or_tester : public ishmem_tester {
  public:
    reduce_or_tester(int argc, char *argv[]) : ishmem_tester(argc, argv) {}
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
};

/* result written into aligned_source, using host_source as a temp buffer if needed */
size_t reduce_or_tester::create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
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
size_t reduce_or_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                              size_t nelems)
{
    /* initialize check buffer */
    /* all to all concatenates the source buffers from all PEs, so the check buffer does the same */
    /* this is not offset, because the copy from test_dest to host_result does the alignment */
    size_t test_size = nelems * typesize(t);
    long a = 0x80L;
    for (int i = 1; i < n_pes; i++) {
        a |= (0x80L + i);
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
    class reduce_or_tester t(argc, argv);

    size_t bufsize =
        2 * ((t.max_nelems * sizeof(uint64_t) * static_cast<unsigned long>(t.n_pes)) + 4096);
    t.alloc_memory(bufsize);
    size_t errors = 0;

    GEN_BITWISE_TABLES(reduce, OR_REDUCE, or)

    if (!t.test_types_set) t.add_test_type_list(bitwise_reduction_types);
    errors += t.run_aligned_tests(OR_REDUCE);
    errors += t.run_offset_tests(OR_REDUCE);

    return (t.finalize_and_report(errors));
}
