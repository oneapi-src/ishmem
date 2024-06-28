/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include <bitset>
#include "ishmem_tester.h"
#include "reduce_test.h"

#define BITS_PER_BYTE 8

GEN_COMPARISON_FNS(reduce, MAX_REDUCE, max)

unsigned long tmax(void *a, void *b, ishmemi_type_t t, size_t index)
{
    switch (t) {
        case FLOAT: {
            float my_max = std::max(((float *) a)[index], ((float *) b)[index]);
            return (unsigned long) std::bitset<sizeof(float) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(*reinterpret_cast<unsigned long *>(&my_max)))
                .to_ulong();
        }
        case DOUBLE: {
            double my_max = std::max(((double *) a)[index], ((double *) b)[index]);
            return (unsigned long) std::bitset<sizeof(double) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(*reinterpret_cast<unsigned long *>(&my_max)))
                .to_ulong();
        }
        case LONGDOUBLE:
            return (unsigned long) std::bitset<sizeof(long) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((long *) a)[index], ((long *) b)[index])))
                .to_ulong();
        case CHAR:
            return (unsigned long) std::bitset<sizeof(char) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((char *) a)[index], ((char *) b)[index])))
                .to_ulong();
        case SCHAR:
            return (unsigned long) std::bitset<sizeof(signed char) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((signed char *) a)[index], ((signed char *) b)[index])))
                .to_ulong();
        case SHORT:
            return (unsigned long) std::bitset<sizeof(short) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((short *) a)[index], ((short *) b)[index])))
                .to_ulong();
        case INT:
            return (unsigned long) std::bitset<sizeof(int) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((int *) a)[index], ((int *) b)[index])))
                .to_ulong();
        case LONG:
            return (unsigned long) std::bitset<sizeof(long) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((long *) a)[index], ((long *) b)[index])))
                .to_ulong();
        case LONGLONG:
            return (unsigned long) std::bitset<sizeof(long long) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((long long *) a)[index], ((long long *) b)[index])))
                .to_ulong();
        case UCHAR:
            return (unsigned long) std::bitset<sizeof(unsigned char) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((unsigned char *) a)[index], ((unsigned char *) b)[index])))
                .to_ulong();
        case USHORT:
            return (unsigned long) std::bitset<sizeof(unsigned short) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((unsigned short *) a)[index], ((unsigned short *) b)[index])))
                .to_ulong();
        case UINT:
            return (unsigned long) std::bitset<sizeof(unsigned int) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((unsigned int *) a)[index], ((unsigned int *) b)[index])))
                .to_ulong();
        case ULONG:
            return (unsigned long) std::bitset<sizeof(unsigned long) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((unsigned long *) a)[index], ((unsigned long *) b)[index])))
                .to_ulong();
        case ULONGLONG:
            return (unsigned long) std::bitset<sizeof(unsigned long long) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(std::max(((unsigned long long *) a)[index],
                                                                ((unsigned long long *) b)[index])))
                .to_ulong();
        case INT8:
            return (unsigned long) std::bitset<sizeof(int8_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((int8_t *) a)[index], ((int8_t *) b)[index])))
                .to_ulong();
        case INT16:
            return (unsigned long) std::bitset<sizeof(int16_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((int16_t *) a)[index], ((int16_t *) b)[index])))
                .to_ulong();
        case INT32:
            return (unsigned long) std::bitset<sizeof(int32_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((int32_t *) a)[index], ((int32_t *) b)[index])))
                .to_ulong();
        case INT64:
            return (unsigned long) std::bitset<sizeof(int64_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((int64_t *) a)[index], ((int64_t *) b)[index])))
                .to_ulong();
        case UINT8:
            return (unsigned long) std::bitset<sizeof(uint8_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((uint8_t *) a)[index], ((uint8_t *) b)[index])))
                .to_ulong();
        case UINT16:
            return (unsigned long) std::bitset<sizeof(uint16_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((uint16_t *) a)[index], ((uint16_t *) b)[index])))
                .to_ulong();
        case UINT32:
            return (unsigned long) std::bitset<sizeof(uint32_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((uint32_t *) a)[index], ((uint32_t *) b)[index])))
                .to_ulong();
        case UINT64:
            return (unsigned long) std::bitset<sizeof(uint64_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((uint64_t *) a)[index], ((uint64_t *) b)[index])))
                .to_ulong();
        case SIZE:
            return (unsigned long) std::bitset<sizeof(size_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((size_t *) a)[index], ((size_t *) b)[index])))
                .to_ulong();
        case PTRDIFF:
            return (unsigned long) std::bitset<sizeof(ptrdiff_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(
                           std::max(((ptrdiff_t *) a)[index], ((ptrdiff_t *) b)[index])))
                .to_ulong();
        default:
            return ULONG_MAX;
    }
}

class reduce_max_tester : public ishmem_tester {
  public:
    reduce_max_tester(int argc, char *argv[]) : ishmem_tester(argc, argv) {}
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
};

/* result written into aligned_source, using host_source as a temp buffer if needed */
size_t reduce_max_tester::create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
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
    /* test pattern includes nelems, source pe, and index so you can tell apart values from other
     * tests or other PEs */
    if (t == FLOAT) {
        float *asrc = (float *) &aligned_source[0];
        for (size_t idx = 0; idx < ((test_size / sizeof(float)) + 1); idx += 1) {
            asrc[idx] = (float) ((((double) my_pe) * 100.0) + ((double) idx / 128.0));
            if (patterndebugflag && (idx < 16)) {
                printf("[%d] source pattern idx %lu val %f\n", my_pe, idx, asrc[idx]);
            }
        }
    } else if (t == DOUBLE) {
        double *asrc = (double *) &aligned_source[0];
        for (size_t idx = 0; idx < ((test_size / sizeof(double)) + 1); idx += 1) {
            asrc[idx] = (double) ((((double) my_pe) * 100.0) + ((double) idx / 128.0));
            if (patterndebugflag && (idx < 16)) {
                printf("[%d] source pattern idx %lu val %f\n", my_pe, idx, asrc[idx]);
            }
        }
    } else {
        for (size_t idx = 0; idx < ((test_size / sizeof(long)) + 1); idx += 1) {
            aligned_source[idx] =
                static_cast<long>(((idx % static_cast<size_t>((my_pe + 2))) << 48) +
                                  (((idx + 1) % static_cast<size_t>((my_pe + 2))) << 40) +
                                  (((idx + 2) % static_cast<size_t>((my_pe + 2))) << 32) +
                                  ((idx + 3) % static_cast<size_t>((my_pe + 2))));
            if (patterndebugflag && (idx < 16)) {
                printf("[%d] source pattern idx %lu val %016lx\n", my_pe, idx, aligned_source[idx]);
            }
        }
    }

    return (test_size);
}

/* check pattern written into host_check, using host_source as a temp buffer */
size_t reduce_max_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                               size_t nelems)
{
    /* initialize check buffer */
    /* all to all concatenates the source buffers from all PEs, so the check buffer does the same */
    /* this is not offset, because the copy from test_dest to host_result does the alignment */
    size_t test_size = nelems * typesize(t);
    if (t == FLOAT) {
        float *asrc = (float *) &host_check[0];
        for (size_t idx = 0; idx < ((test_size / sizeof(float)) + 1); idx += 1) {
            asrc[idx] = (float) ((0.0 * 100.0) + ((double) idx / 128.0));
            for (int i = 1; i < n_pes; i++) {
                asrc[idx] =
                    std::max(asrc[idx], (float) (((double) i * 100.0) + ((double) idx / 128.0)));
            }
            if (patterndebugflag && (idx < 16)) {
                printf("[%d] source pattern idx %lu val %f\n", my_pe, idx, asrc[idx]);
            }
        }
    } else if (t == DOUBLE) {
        double *asrc = (double *) &host_check[0];
        for (size_t idx = 0; idx < ((test_size / sizeof(double)) + 1); idx += 1) {
            asrc[idx] = (double) ((0.0 * 100.0) + ((double) idx / 128.0));
            for (int i = 1; i < n_pes; i++) {
                asrc[idx] = std::max(asrc[idx], ((double) i * 100.0) + ((double) idx / 128.0));
            }
            if (patterndebugflag && (idx < 16)) {
                printf("[%d] source pattern idx %lu val %f\n", my_pe, idx, asrc[idx]);
            }
        }
    } else {
        for (size_t idx = 0; idx < ((test_size / sizeof(long)) + 1); idx += 1) {
            long expected = 0L;
            for (size_t j = 0; j < sizeof(long) / typesize(t); j++) {
                // Construct mask for divinding long type into smaller chunks
                long mask = 0L;
                for (size_t k = 0; k < (typesize(t) * BITS_PER_BYTE); k++) {
                    mask |= 1;
                    if (k != (typesize(t) * BITS_PER_BYTE) - 1) mask <<= 1;
                }
                mask <<= ((sizeof(long) / typesize(t)) - (j + 1)) * (typesize(t) * BITS_PER_BYTE);

                long cur_max = static_cast<long>(((idx % 2) << 48) + (((idx + 1) % 2) << 40) +
                                                 (((idx + 2) % 2) << 32) + ((idx + 3) % 2));
                for (int i = 1; i < n_pes; i++) {
                    long cmp =
                        static_cast<long>(((idx % static_cast<size_t>((i + 2))) << 48) +
                                          (((idx + 1) % static_cast<size_t>((i + 2))) << 40) +
                                          (((idx + 2) % static_cast<size_t>((i + 2))) << 32) +
                                          ((idx + 3) % static_cast<size_t>((i + 2))));

                    cur_max = static_cast<long>(
                        tmax(&cur_max, &cmp, t, ((sizeof(long) / typesize(t)) - (j + 1))));
                    cur_max <<=
                        ((sizeof(long) / typesize(t)) - (j + 1)) * ((typesize(t)) * BITS_PER_BYTE);
                }
                expected |= (cur_max & mask);
            }
            host_check[idx] = expected;
            if (patterndebugflag && (idx < 16)) {
                printf("[%d] check pattern idx %lu val %016lx\n", my_pe, idx, host_check[idx]);
            }
        }
    }

    return (test_size);
}

int main(int argc, char **argv)
{
    class reduce_max_tester t(argc, argv);

    size_t bufsize =
        2 * ((t.max_nelems * sizeof(uint64_t) * static_cast<unsigned long>(t.n_pes)) + 4096);
    t.alloc_memory(bufsize);
    size_t errors = 0;

    GEN_COMPARISON_TABLES(reduce, MAX_REDUCE, max)

    if (!t.test_types_set) {
        t.add_test_type(LONG);
    }
    if (!t.test_types_set) t.add_test_type_list(compare_reduction_types);
    errors += t.run_aligned_tests(MAX_REDUCE);
    errors += t.run_offset_tests(MAX_REDUCE);

    return (t.finalize_and_report(errors));
}
