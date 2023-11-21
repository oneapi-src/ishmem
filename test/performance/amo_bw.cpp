/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "amo_test.h"
#include "common.h"  //  needed for forward declaration of test_amo

/*   forward declaration */
SYCL_EXTERNAL void test_amo(ishmemi_type_t t, ishmemi_op_t op, size_t iterations, void *dest,
                            int pe);

#define BW_TEST_HEADER   int pe = n_pes - 1;
#define BW_TEST_FUNCTION test_amo(t, op, iterations, dest, pe)
#define BW_TEST_FUNCTION_WORK_GROUP

#include "ishmem_tester.h"

#define GEN_EXTENDED(name)                                                                         \
    ISHMEM_GEN_AMO_EXTENDED_FUNCTION(test_##name##(void *dest COMMA int pe),                       \
                                     int res __attribute__((unused)) = 0;)
#define GEN_STANDARD(name)                                                                         \
    ISHMEM_GEN_AMO_STANDARD_FUNCTION(test_##name##(void *dest COMMA int pe),                       \
                                     int res __attribute__((unused)) = 0;)
#define GEN_BITWISE(name)                                                                          \
    ISHMEM_GEN_AMO_BITWISE_FUNCTION(test_##name##(void *dest COMMA int pe),                        \
                                    int res __attribute__((unused)) = 0;)

#define SIGNATURE                                                                                  \
    ishmemi_type_t t COMMA ishmemi_op_t op COMMA size_t iterations COMMA void *dest COMMA int pe
// FETCH
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type)                                                        \
    for (size_t iter = 0; iter < iterations; iter++) {                                             \
        ishmem_##name##_atomic_fetch((type *) dest, pe);                                           \
    }                                                                                              \
    break;

ISHMEM_GEN_AMO_EXTENDED_FUNCTION(void test_fetch(SIGNATURE), int res __attribute__((unused)) = 0;)

// SET
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type)                                                        \
    for (size_t iter = 0; iter < iterations; iter++) {                                             \
        ishmem_##name##_atomic_set((type *) dest, (type) 0, pe);                                   \
    }                                                                                              \
    break;

ISHMEM_GEN_AMO_EXTENDED_FUNCTION(void test_set(SIGNATURE), int res __attribute__((unused)) = 0;)

// COMPARE_SWAP
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type)                                                        \
    for (size_t iter = 0; iter < iterations; iter++) {                                             \
        ishmem_##name##_atomic_compare_swap((type *) dest, (type) 0, (type) 0, pe);                \
    }                                                                                              \
    break;

ISHMEM_GEN_AMO_STANDARD_FUNCTION(void test_compare_swap(SIGNATURE),
                                 int res __attribute__((unused)) = 0;)

// SWAP
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type)                                                        \
    for (size_t iter = 0; iter < iterations; iter++) {                                             \
        ishmem_##name##_atomic_swap((type *) dest, (type) 0, pe);                                  \
    }                                                                                              \
    break;

ISHMEM_GEN_AMO_STANDARD_FUNCTION(void test_swap(SIGNATURE), int res __attribute__((unused)) = 0;)

// FETCH_INC
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type)                                                        \
    for (size_t iter = 0; iter < iterations; iter++) {                                             \
        ishmem_##name##_atomic_fetch_inc((type *) dest, pe);                                       \
    }                                                                                              \
    break;

ISHMEM_GEN_AMO_STANDARD_FUNCTION(void test_fetch_inc(SIGNATURE),
                                 int res __attribute__((unused)) = 0;)

// INC

#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type)                                                        \
    for (size_t iter = 0; iter < iterations; iter++) {                                             \
        ishmem_##name##_atomic_inc((type *) dest, pe);                                             \
    }                                                                                              \
    break;

ISHMEM_GEN_AMO_STANDARD_FUNCTION(void test_inc(SIGNATURE), int res __attribute__((unused)) = 0;)

// FETCH_ADD

#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type)                                                        \
    for (size_t iter = 0; iter < iterations; iter++) {                                             \
        ishmem_##name##_atomic_fetch_add((type *) dest, (type) 0, pe);                             \
    }                                                                                              \
    break;

ISHMEM_GEN_AMO_STANDARD_FUNCTION(void test_fetch_add(SIGNATURE),
                                 int res __attribute__((unused)) = 0;)

// ADD

#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type)                                                        \
    for (size_t iter = 0; iter < iterations; iter++) {                                             \
        ishmem_##name##_atomic_add((type *) dest, (type) 0, pe);                                   \
    }                                                                                              \
    break;

ISHMEM_GEN_AMO_STANDARD_FUNCTION(void test_add(SIGNATURE), int res __attribute__((unused)) = 0;)

#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type)                                                        \
    for (size_t iter = 0; iter < iterations; iter++) {                                             \
        ishmem_##name##_atomic_fetch_and((type *) dest, (type) 0, pe);                             \
    }                                                                                              \
    break;

ISHMEM_GEN_AMO_BITWISE_FUNCTION(void test_fetch_and(SIGNATURE),
                                int res __attribute__((unused)) = 0;)

// AND

#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type)                                                        \
    for (size_t iter = 0; iter < iterations; iter++) {                                             \
        ishmem_##name##_atomic_and((type *) dest, (type) 0, pe);                                   \
    }                                                                                              \
    break;

ISHMEM_GEN_AMO_BITWISE_FUNCTION(void test_and(SIGNATURE), int res __attribute__((unused)) = 0;)

// FETCH_OR

#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type)                                                        \
    for (size_t iter = 0; iter < iterations; iter++) {                                             \
        ishmem_##name##_atomic_fetch_or((type *) dest, (type) 0, pe);                              \
    }                                                                                              \
    break;

ISHMEM_GEN_AMO_BITWISE_FUNCTION(void test_fetch_or(SIGNATURE), int res __attribute__((unused)) = 0;)

// OR

#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type)                                                        \
    for (size_t iter = 0; iter < iterations; iter++) {                                             \
        ishmem_##name##_atomic_or((type *) dest, (type) 0, pe);                                    \
    }                                                                                              \
    break;

ISHMEM_GEN_AMO_BITWISE_FUNCTION(void test_or(SIGNATURE), int res __attribute__((unused)) = 0;)

// FETCH_XOR

#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type)                                                        \
    for (size_t iter = 0; iter < iterations; iter++) {                                             \
        ishmem_##name##_atomic_fetch_xor((type *) dest, (type) 0, pe);                             \
    }                                                                                              \
    break;

ISHMEM_GEN_AMO_BITWISE_FUNCTION(void test_fetch_xor(SIGNATURE),
                                int res __attribute__((unused)) = 0;)

// XOR

#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type)                                                        \
    for (size_t iter = 0; iter < iterations; iter++) {                                             \
        ishmem_##name##_atomic_xor((type *) dest, (type) 0, pe);                                   \
    }                                                                                              \
    break;

ISHMEM_GEN_AMO_BITWISE_FUNCTION(void test_xor(SIGNATURE), int res __attribute__((unused)) = 0;)

SYCL_EXTERNAL void test_amo(ishmemi_type_t t, ishmemi_op_t op, size_t iterations, void *dest,
                            int pe)
{
    switch (op) {
        case AMO_FETCH:
            test_fetch(t, op, iterations, dest, pe);
            break;
        case AMO_SET:
            test_set(t, op, iterations, dest, pe);
            break;
        case AMO_COMPARE_SWAP:
            test_compare_swap(t, op, iterations, dest, pe);
            break;
        case AMO_SWAP:
            test_swap(t, op, iterations, dest, pe);
            break;
        case AMO_FETCH_INC:
            test_fetch_inc(t, op, iterations, dest, pe);
            break;
        case AMO_INC:
            test_inc(t, op, iterations, dest, pe);
            break;
        case AMO_FETCH_ADD:
            test_fetch_add(t, op, iterations, dest, pe);
            break;
        case AMO_ADD:
            test_add(t, op, iterations, dest, pe);
            break;
        case AMO_FETCH_AND:
            test_fetch_and(t, op, iterations, dest, pe);
            break;
        case AMO_AND:
            test_and(t, op, iterations, dest, pe);
            break;
        case AMO_FETCH_OR:
            test_fetch_or(t, op, iterations, dest, pe);
            break;
        case AMO_OR:
            test_or(t, op, iterations, dest, pe);
            break;
        case AMO_FETCH_XOR:
            test_fetch_xor(t, op, iterations, dest, pe);
            break;
        case AMO_XOR:
            test_xor(t, op, iterations, dest, pe);
            break;
        default:
            break;
    }
}

STUB_UNIT_TESTS

typedef Iterator<ishmemi_op_t, ishmemi_op_t::AMO_FETCH, ishmemi_op_t::AMO_XOR>
    ishmemi_amo_op_Iterator;

typedef Iterator<ishmemi_type_t, ishmemi_type_t::FLOAT, ishmemi_type_t::PTRDIFF>
    ishmemi_amo_t_Iterator;

int main(int argc, char **argv)
{
    class ishmem_tester test(argc, argv);
    test.max_nelems = 1;  // because the atomics are only single
    size_t bufsize = (test.max_nelems * sizeof(uint64_t)) + 4096;
    test.alloc_memory(bufsize);
    size_t errors = 0;
    for (ishmemi_type_t t : ishmemi_amo_t_Iterator()) {
        for (ishmemi_op_t op : ishmemi_amo_op_Iterator()) {
            test.run_bw_tests(t, op, device, 1, false);
        }
    }
    for (ishmemi_type_t t : ishmemi_amo_t_Iterator()) {
        for (ishmemi_op_t op : ishmemi_amo_op_Iterator()) {
            test.run_bw_tests(t, op, host_host_host, 1, false);
        }
    }
    for (ishmemi_type_t t : ishmemi_amo_t_Iterator()) {
        for (ishmemi_op_t op : ishmemi_amo_op_Iterator()) {
            test.run_bw_tests(t, op, host_host_device, 1, false);
        }
    }
    ishmem_sync_all();
    return (test.finalize_and_report(errors));
}
