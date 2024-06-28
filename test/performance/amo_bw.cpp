/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "amo_test.h"
#include "common.h"  //  needed for forward declaration of test_amo
#include <ishmem/types.h>

/*   forward declaration */
SYCL_EXTERNAL void test_amo(ishmemi_type_t t, ishmemi_op_t op, size_t iterations, void *dest,
                            int pe);

#define BW_TEST_HEADER   int pe = n_pes - 1;
#define BW_TEST_FUNCTION test_amo(t, op, iterations, dest, pe)
#define BW_TEST_FUNCTION_WORK_GROUP

#include "ishmem_tester.h"

/* Create functions for each operation, that contain switch statements for each type */

#define SIGNATURE                                                                                  \
    ishmemi_type_t t COMMA ishmemi_op_t op COMMA size_t iterations COMMA void *dest COMMA int pe
#define RESULT int res __attribute__((unused)) = 0;
#define ITER   for (size_t iter = 0; iter < iterations; iter++)

/* There are three signatures for amo functions,
 * DP  dest,pe
 * DZP dest,0,pe
 * DZZP dest,0,0,pe
 * which stand for "(type *) dest, (type) 0, (type) 0, int pe), etc.
 *
 * These are deeply nested macros, the purpose of which is to minimize the length of the source file
 * and to generate the necessary switch statements without much risk of inconsistent branches.
 */
#define LOOP_DP(name, type, operator)                                                              \
    ITER ishmem_##name##_atomic_##operator((type *) dest, pe);                                     \
    break;
#define LOOP_DZP(name, type, operator)                                                             \
    ITER ishmem_##name##_atomic_##operator((type *) dest, (type) 0, pe);                           \
    break;
#define LOOP_DZZP(name, type, operator)                                                            \
    ITER ishmem_##name##_atomic_##operator((type *) dest, (type) 0, (type) 0, pe);                 \
    break;

#undef ISHMEM_AMO_BRANCH

/////////////
// DZZP cases
/////////////
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator) LOOP_DZZP(name, type, operator)

// STANDARD dest,0,0,pe
ISHMEM_GEN_AMO_STANDARD_FUNCTION(int test_compare_swap(SIGNATURE), RESULT, compare_swap)

////////////
// DZP cases
////////////
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator) LOOP_DZP(name, type, operator)

// EXTENDED dest,0,pe
ISHMEM_GEN_AMO_EXTENDED_FUNCTION(int test_set(SIGNATURE), RESULT, set)

// STANDARD dest,0,pe
ISHMEM_GEN_AMO_STANDARD_FUNCTION(int test_swap(SIGNATURE), RESULT, swap)
ISHMEM_GEN_AMO_STANDARD_FUNCTION(int test_fetch_add(SIGNATURE), RESULT, fetch_add)
ISHMEM_GEN_AMO_STANDARD_FUNCTION(int test_add(SIGNATURE), RESULT, add)

// BITWISE dest, 0, pe
ISHMEM_GEN_AMO_BITWISE_FUNCTION(int test_fetch_and(SIGNATURE), RESULT, fetch_and)
ISHMEM_GEN_AMO_BITWISE_FUNCTION(int test_and(SIGNATURE), RESULT, and)
ISHMEM_GEN_AMO_BITWISE_FUNCTION(int test_fetch_or(SIGNATURE), RESULT, fetch_or)
ISHMEM_GEN_AMO_BITWISE_FUNCTION(int test_or(SIGNATURE), RESULT, or)
ISHMEM_GEN_AMO_BITWISE_FUNCTION(int test_fetch_xor(SIGNATURE), RESULT, fetch_xor)
ISHMEM_GEN_AMO_BITWISE_FUNCTION(int test_xor(SIGNATURE), RESULT, xor)

///////////
// DP cases
///////////
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator) LOOP_DP(name, type, operator)

// EXTENDED dest,pe
ISHMEM_GEN_AMO_EXTENDED_FUNCTION(int test_fetch(SIGNATURE), RESULT, fetch)

// STANDARD dest,pe
ISHMEM_GEN_AMO_STANDARD_FUNCTION(int test_fetch_inc(SIGNATURE), RESULT, fetch_inc)
ISHMEM_GEN_AMO_STANDARD_FUNCTION(int test_inc(SIGNATURE), RESULT, inc)

/* now create a function to select which test_ function to call */

#define BRANCH(name)                                                                               \
    test_##name(t, op, iterations, dest, pe);                                                      \
    break

SYCL_EXTERNAL void test_amo(ishmemi_type_t t, ishmemi_op_t op, size_t iterations, void *dest,
                            int pe)
{
    switch (op) {
        case AMO_FETCH:
            BRANCH(fetch);
        case AMO_SET:
            BRANCH(set);
        case AMO_COMPARE_SWAP:
            BRANCH(compare_swap);
        case AMO_SWAP:
            BRANCH(swap);
        case AMO_FETCH_INC:
            BRANCH(fetch_inc);
        case AMO_INC:
            BRANCH(inc);
        case AMO_FETCH_ADD:
            BRANCH(fetch_add);
        case AMO_ADD:
            BRANCH(add);
        case AMO_FETCH_AND:
            BRANCH(fetch_and);
        case AMO_AND:
            BRANCH(and);
        case AMO_FETCH_OR:
            BRANCH(fetch_or);
        case AMO_OR:
            BRANCH(or);
        case AMO_FETCH_XOR:
            BRANCH(fetch_xor);
        case AMO_XOR:
            BRANCH(xor);
        default:
            break;
    }
}

STUB_UNIT_TESTS

int main(int argc, char **argv)
{
    class ishmem_tester test(argc, argv);
    test.max_nelems = 1;  // because the atomics are only single
    size_t bufsize = (test.max_nelems * sizeof(uint64_t)) + 4096;
    test.alloc_memory(bufsize);
    size_t errors = 0;
    // run bitwise tests on types and ops

    if (!test.test_types_set && !test.test_ops_set) {
        test.add_test_type_list(bitwise_amo_types);
        test.add_test_op_list(bitwise_amo_ops);
        test.run_bw_tests(1, false);

        test.reset_test_types();
        test.reset_test_ops();
        test.add_test_type_list(standard_amo_types);
        test.add_test_op_list(standard_amo_ops);
        test.run_bw_tests(1, false);

        test.reset_test_types();
        test.reset_test_ops();
        test.add_test_type_list(extended_amo_types);
        test.add_test_op_list(extended_amo_ops);
        test.run_bw_tests(1, false);
    } else {
        test.run_bw_tests(1, false);
    }

    ishmem_sync_all();
    return (test.finalize_and_report(errors));
}
