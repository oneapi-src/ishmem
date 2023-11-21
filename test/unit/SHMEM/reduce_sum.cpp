/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <CL/sycl.hpp>
#include "ishmem_tester.h"
#include <bitset>

#define BITS_PER_BYTE 8

#ifdef ISHMEM_GEN_TYPE_FUNCTION
#undef ISHMEM_GEN_TYPE_FUNCTION
#endif
#define ISHMEM_GEN_TYPE_FUNCTION(function, returnvar, memcase)                                     \
    function                                                                                       \
    {                                                                                              \
        *test_run = true;                                                                          \
        returnvar switch (t)                                                                       \
        {                                                                                          \
            case FLOAT:                                                                            \
                ISHMEM_TYPE_BRANCH(FLOAT, float, float)                                            \
            case DOUBLE:                                                                           \
                ISHMEM_TYPE_BRANCH(DOUBLE, double, double)                                         \
            case LONGDOUBLE:                                                                       \
                ISHMEM_TYPE_BRANCH(LONG, long, long)                                               \
            case CHAR:                                                                             \
                ISHMEM_TYPE_BRANCH(CHAR, char, char)                                               \
            case SCHAR:                                                                            \
                ISHMEM_TYPE_BRANCH(SCHAR, schar, signed char)                                      \
            case SHORT:                                                                            \
                ISHMEM_TYPE_BRANCH(SHORT, short, short)                                            \
            case INT:                                                                              \
                ISHMEM_TYPE_BRANCH(INT, int, int)                                                  \
            case LONG:                                                                             \
                ISHMEM_TYPE_BRANCH(LONG, long, long)                                               \
            case LONGLONG:                                                                         \
                ISHMEM_TYPE_BRANCH(LONGLONG, longlong, long long)                                  \
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
            case PTRDIFF:                                                                          \
                ISHMEM_TYPE_BRANCH(PTRDIFF, ptrdiff, ptrdiff_t);                                   \
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
    res = ishmem_##name##_sum_reduce((type *) dest, (type *) src, nelems);                         \
    break;

ISHMEM_GEN_TEST_FUNCTION_SINGLE(int res = 0;, break;)

#ifdef ISHMEM_TYPE_BRANCH
#undef ISHMEM_TYPE_BRANCH
#endif

#define ISHMEM_TYPE_BRANCH(enumname, name, type)                                                   \
    res = ishmemx_##name##_sum_reduce_work_group((type *) dest, (type *) src, nelems, grp);        \
    break;

ISHMEM_GEN_TEST_FUNCTION_WORK_GROUP(int res = 0;, break;)

unsigned long tsum(void *a, void *b, ishmemi_type_t t, size_t index)
{
    switch (t) {
        case FLOAT: {
            float my_sum = ((float *) a)[index] + ((float *) b)[index];
            return (unsigned long) std::bitset<sizeof(float) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(*reinterpret_cast<unsigned long *>(&my_sum)))
                .to_ulong();
        }
        case DOUBLE: {
            double my_sum = ((double *) a)[index] + ((double *) b)[index];
            return (unsigned long) std::bitset<sizeof(double) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(*reinterpret_cast<unsigned long *>(&my_sum)))
                .to_ulong();
        }
        case LONGDOUBLE:
            return (unsigned long) std::bitset<sizeof(long) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((long *) a)[index] + ((long *) b)[index]))
                .to_ulong();
        case CHAR:
            return (unsigned long) std::bitset<sizeof(char) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((char *) a)[index] + ((char *) b)[index]))
                .to_ulong();
        case SCHAR:
            return (unsigned long) std::bitset<sizeof(signed char) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((signed char *) a)[index] +
                                                       ((signed char *) b)[index]))
                .to_ulong();
        case SHORT:
            return (unsigned long) std::bitset<sizeof(short) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((short *) a)[index] + ((short *) b)[index]))
                .to_ulong();
        case INT:
            return (unsigned long) std::bitset<sizeof(int) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((int *) a)[index] + ((int *) b)[index]))
                .to_ulong();
        case LONG:
            return (unsigned long) std::bitset<sizeof(long) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((long *) a)[index] + ((long *) b)[index]))
                .to_ulong();
        case LONGLONG:
            return (unsigned long) std::bitset<sizeof(long long) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((long long *) a)[index] +
                                                       ((long long *) b)[index]))
                .to_ulong();
        case UCHAR:
            return (unsigned long) std::bitset<sizeof(unsigned char) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((unsigned char *) a)[index] +
                                                       ((unsigned char *) b)[index]))
                .to_ulong();
        case USHORT:
            return (unsigned long) std::bitset<sizeof(unsigned short) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((unsigned short *) a)[index] +
                                                       ((unsigned short *) b)[index]))
                .to_ulong();
        case UINT:
            return (unsigned long) std::bitset<sizeof(unsigned int) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((unsigned int *) a)[index] +
                                                       ((unsigned int *) b)[index]))
                .to_ulong();
        case ULONG:
            return (unsigned long) std::bitset<sizeof(unsigned long) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((unsigned long *) a)[index] +
                                                       ((unsigned long *) b)[index]))
                .to_ulong();
        case ULONGLONG:
            return (unsigned long) std::bitset<sizeof(unsigned long long) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((unsigned long long *) a)[index] +
                                                       ((unsigned long long *) b)[index]))
                .to_ulong();
        case INT8:
            return (unsigned long) std::bitset<sizeof(int8_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((int8_t *) a)[index] +
                                                       ((int8_t *) b)[index]))
                .to_ulong();
        case INT16:
            return (unsigned long) std::bitset<sizeof(int16_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((int16_t *) a)[index] +
                                                       ((int16_t *) b)[index]))
                .to_ulong();
        case INT32:
            return (unsigned long) std::bitset<sizeof(int32_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((int32_t *) a)[index] +
                                                       ((int32_t *) b)[index]))
                .to_ulong();
        case INT64:
            return (unsigned long) std::bitset<sizeof(int64_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((int64_t *) a)[index] +
                                                       ((int64_t *) b)[index]))
                .to_ulong();
        case UINT8:
            return (unsigned long) std::bitset<sizeof(uint8_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((uint8_t *) a)[index] +
                                                       ((uint8_t *) b)[index]))
                .to_ulong();
        case UINT16:
            return (unsigned long) std::bitset<sizeof(uint16_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((uint16_t *) a)[index] +
                                                       ((uint16_t *) b)[index]))
                .to_ulong();
        case UINT32:
            return (unsigned long) std::bitset<sizeof(uint32_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((uint32_t *) a)[index] +
                                                       ((uint32_t *) b)[index]))
                .to_ulong();
        case UINT64:
            return (unsigned long) std::bitset<sizeof(uint64_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((uint64_t *) a)[index] +
                                                       ((uint64_t *) b)[index]))
                .to_ulong();
        case SIZE:
            return (unsigned long) std::bitset<sizeof(size_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((size_t *) a)[index] +
                                                       ((size_t *) b)[index]))
                .to_ulong();
        case PTRDIFF:
            return (unsigned long) std::bitset<sizeof(ptrdiff_t) * BITS_PER_BYTE>(
                       static_cast<unsigned long long>(((ptrdiff_t *) a)[index] +
                                                       ((ptrdiff_t *) b)[index]))
                .to_ulong();
        default:
            return ULONG_MAX;
    }
}

class reduce_sum_tester : public ishmem_tester {
  public:
    reduce_sum_tester(int argc, char *argv[]) : ishmem_tester(argc, argv) {}
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
};

/* result written into aligned_source, using host_source as a temp buffer if needed */
size_t reduce_sum_tester::create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
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

    return (test_size);
}

/* check pattern written into host_check, using host_source as a temp buffer */
size_t reduce_sum_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                               size_t nelems)
{
    /* initialize check buffer */
    /* all to all concatenates the source buffers from all PEs, so the check buffer does the same */
    /* this is not offset, because the copy from test_dest to host_result does the alignment */
    size_t test_size = nelems * typesize(t);
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

            long cur_sum = static_cast<long>(((idx % 2) << 48) + (((idx + 1) % 2) << 40) +
                                             (((idx + 2) % 2) << 32) + ((idx + 3) % 2));
            for (int i = 1; i < n_pes; i++) {
                long cmp = static_cast<long>(((idx % static_cast<size_t>((i + 2))) << 48) +
                                             (((idx + 1) % static_cast<size_t>((i + 2))) << 40) +
                                             (((idx + 2) % static_cast<size_t>((i + 2))) << 32) +
                                             ((idx + 3) % static_cast<size_t>((i + 2))));

                cur_sum = static_cast<long>(
                    tsum(&cur_sum, &cmp, t, ((sizeof(long) / typesize(t)) - (j + 1))));
                cur_sum <<=
                    ((sizeof(long) / typesize(t)) - (j + 1)) * ((typesize(t)) * BITS_PER_BYTE);
            }
            expected |= (cur_sum & mask);
        }
        host_check[idx] = expected;
        if (patterndebugflag && (idx < 16)) {
            printf("[%d] check pattern idx %lu val %016lx\n", my_pe, idx, host_check[idx]);
        }
    }

    return (test_size);
}

int main(int argc, char **argv)
{
    class reduce_sum_tester t(argc, argv);

    size_t bufsize =
        2 * ((t.max_nelems * sizeof(uint64_t) * static_cast<unsigned long>(t.n_pes)) + 4096);
    t.alloc_memory(bufsize);
    size_t errors = 0;

    errors += t.run_aligned_tests(NOP);
    errors += t.run_offset_tests(NOP);

    return (t.finalize_and_report(errors));
}
