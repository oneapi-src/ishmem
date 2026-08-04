/* Copyright (C) 2023-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "amo_test.h"
#include "proxy_types.h"

/*   forward declaration */
SYCL_EXTERNAL int test_amo(ishmemi_type_t t, ishmemi_op_t op, size_t iterations, void *dest,
                           int pe);

#define BW_TEST_HEADER   int pe = n_pes - 1;
#define BW_TEST_FUNCTION test_amo(t, op, iterations, dest, pe)
#define BW_TEST_FUNCTION_WORK_GROUP

#include "ishmem_tester.h"

#define SIGNATURE                                                                                  \
    ishmemi_type_t t COMMA ishmemi_op_t op COMMA size_t iterations COMMA void *dest COMMA int pe
#define RESULT int res __attribute__((unused)) = 0;

// FETCH
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator)                                              \
    {                                                                                              \
        type *ptr = (type *) ishmem_ptr(dest, pe);                                                 \
        volatile type val __attribute__((unused)) = 0;                                             \
        for (size_t iter = 0; iter < iterations; iter++) {                                         \
            sycl::atomic_ref<type, sycl::memory_order::relaxed, sycl::memory_scope::device,        \
                             sycl::access::address_space::global_space>                            \
                atomic_d(*ptr);                                                                    \
            val = atomic_d.load();                                                                 \
        }                                                                                          \
        break;                                                                                     \
    }

ISHMEM_GEN_AMO_EXTENDED_FUNCTION(int test_fetch(SIGNATURE), RESULT, fetch)

// SET
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator)                                              \
    {                                                                                              \
        type *ptr = (type *) ishmem_ptr(dest, pe);                                                 \
        for (size_t iter = 0; iter < iterations; iter++) {                                         \
            sycl::atomic_ref<type, sycl::memory_order::relaxed, sycl::memory_scope::device,        \
                             sycl::access::address_space::global_space>                            \
                atomic_d(*ptr);                                                                    \
            atomic_d = (type) 0;                                                                   \
        }                                                                                          \
        break;                                                                                     \
    }

ISHMEM_GEN_AMO_EXTENDED_FUNCTION(int test_set(SIGNATURE), RESULT, set)

// COMPARE_SWAP
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator)                                              \
    {                                                                                              \
        type *ptr = (type *) ishmem_ptr(dest, pe);                                                 \
        volatile type val __attribute__((unused)) = 0;                                             \
        type expected = 0;                                                                         \
        for (size_t iter = 0; iter < iterations; iter++) {                                         \
            sycl::atomic_ref<type, sycl::memory_order::relaxed, sycl::memory_scope::device,        \
                             sycl::access::address_space::global_space>                            \
                atomic_d(*ptr);                                                                    \
            atomic_d.compare_exchange_strong(expected, val, sycl::memory_order::relaxed,           \
                                             sycl::memory_scope::device);                          \
        }                                                                                          \
        break;                                                                                     \
    }

ISHMEM_GEN_AMO_STANDARD_FUNCTION(int test_compare_swap(SIGNATURE), RESULT, compare_swap)

// SWAP
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator)                                              \
    {                                                                                              \
        type *ptr = (type *) ishmem_ptr(dest, pe);                                                 \
        volatile type val __attribute__((unused)) = 0;                                             \
        for (size_t iter = 0; iter < iterations; iter++) {                                         \
            sycl::atomic_ref<type, sycl::memory_order::relaxed, sycl::memory_scope::device,        \
                             sycl::access::address_space::global_space>                            \
                atomic_d(*ptr);                                                                    \
            val = atomic_d.exchange(val, sycl::memory_order::relaxed, sycl::memory_scope::device); \
        }                                                                                          \
        break;                                                                                     \
    }

ISHMEM_GEN_AMO_STANDARD_FUNCTION(int test_swap(SIGNATURE), RESULT, swap)

// FETCH_INC
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator)                                              \
    {                                                                                              \
        type *ptr = (type *) ishmem_ptr(dest, pe);                                                 \
        volatile type val __attribute__((unused)) = 0;                                             \
        for (size_t iter = 0; iter < iterations; iter++) {                                         \
            sycl::atomic_ref<type, sycl::memory_order::relaxed, sycl::memory_scope::device,        \
                             sycl::access::address_space::global_space>                            \
                atomic_d(*ptr);                                                                    \
            val = atomic_d.fetch_add((type) 1, sycl::memory_order::relaxed,                        \
                                     sycl::memory_scope::device);                                  \
        }                                                                                          \
        break;                                                                                     \
    }

ISHMEM_GEN_AMO_STANDARD_FUNCTION(int test_fetch_inc(SIGNATURE), RESULT, fetch)

// INC
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator)                                              \
    {                                                                                              \
        type *ptr = (type *) ishmem_ptr(dest, pe);                                                 \
        for (size_t iter = 0; iter < iterations; iter++) {                                         \
            sycl::atomic_ref<type, sycl::memory_order::relaxed, sycl::memory_scope::device,        \
                             sycl::access::address_space::global_space>                            \
                atomic_d(*ptr);                                                                    \
            atomic_d += 1;                                                                         \
        }                                                                                          \
        break;                                                                                     \
    }

ISHMEM_GEN_AMO_STANDARD_FUNCTION(int test_inc(SIGNATURE), RESULT, inc)

// FETCH_ADD
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator)                                              \
    {                                                                                              \
        type *ptr = (type *) ishmem_ptr(dest, pe);                                                 \
        volatile type val __attribute__((unused)) = 0;                                             \
        for (size_t iter = 0; iter < iterations; iter++) {                                         \
            sycl::atomic_ref<type, sycl::memory_order::relaxed, sycl::memory_scope::device,        \
                             sycl::access::address_space::global_space>                            \
                atomic_d(*ptr);                                                                    \
            val = atomic_d.fetch_add((type) 1, sycl::memory_order::relaxed,                        \
                                     sycl::memory_scope::device);                                  \
        }                                                                                          \
        break;                                                                                     \
    }

ISHMEM_GEN_AMO_STANDARD_FUNCTION(int test_fetch_add(SIGNATURE), RESULT, fetch_add)

// ADD
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator)                                              \
    {                                                                                              \
        type *ptr = (type *) ishmem_ptr(dest, pe);                                                 \
        for (size_t iter = 0; iter < iterations; iter++) {                                         \
            sycl::atomic_ref<type, sycl::memory_order::relaxed, sycl::memory_scope::device,        \
                             sycl::access::address_space::global_space>                            \
                atomic_d(*ptr);                                                                    \
            atomic_d += (type) 1;                                                                  \
        }                                                                                          \
        break;                                                                                     \
    }

ISHMEM_GEN_AMO_STANDARD_FUNCTION(int test_add(SIGNATURE), RESULT, add)

// FETCH AND
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator)                                              \
    {                                                                                              \
        type *ptr = (type *) ishmem_ptr(dest, pe);                                                 \
        volatile type val __attribute__((unused)) = 0;                                             \
        for (size_t iter = 0; iter < iterations; iter++) {                                         \
            sycl::atomic_ref<type, sycl::memory_order::relaxed, sycl::memory_scope::device,        \
                             sycl::access::address_space::global_space>                            \
                atomic_d(*ptr);                                                                    \
            val = atomic_d.fetch_and((type) 1, sycl::memory_order::relaxed,                        \
                                     sycl::memory_scope::device);                                  \
        }                                                                                          \
        break;                                                                                     \
    }

ISHMEM_GEN_AMO_BITWISE_FUNCTION(int test_fetch_and(SIGNATURE), RESULT, fetch_and)

// AND
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator)                                              \
    {                                                                                              \
        type *ptr = (type *) ishmem_ptr(dest, pe);                                                 \
        for (size_t iter = 0; iter < iterations; iter++) {                                         \
            sycl::atomic_ref<type, sycl::memory_order::relaxed, sycl::memory_scope::device,        \
                             sycl::access::address_space::global_space>                            \
                atomic_d(*ptr);                                                                    \
            atomic_d &= (type) 1;                                                                  \
        }                                                                                          \
        break;                                                                                     \
    }

ISHMEM_GEN_AMO_BITWISE_FUNCTION(int test_and(SIGNATURE), RESULT, and)

// FETCH_OR
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator)                                              \
    {                                                                                              \
        type *ptr = (type *) ishmem_ptr(dest, pe);                                                 \
        volatile type val __attribute__((unused)) = 0;                                             \
        for (size_t iter = 0; iter < iterations; iter++) {                                         \
            sycl::atomic_ref<type, sycl::memory_order::relaxed, sycl::memory_scope::device,        \
                             sycl::access::address_space::global_space>                            \
                atomic_d(*ptr);                                                                    \
            val = atomic_d.fetch_or((type) 1, sycl::memory_order::relaxed,                         \
                                    sycl::memory_scope::device);                                   \
        }                                                                                          \
        break;                                                                                     \
    }

ISHMEM_GEN_AMO_BITWISE_FUNCTION(int test_fetch_or(SIGNATURE), RESULT, fetch_or)

// OR
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator)                                              \
    {                                                                                              \
        type *ptr = (type *) ishmem_ptr(dest, pe);                                                 \
        for (size_t iter = 0; iter < iterations; iter++) {                                         \
            sycl::atomic_ref<type, sycl::memory_order::relaxed, sycl::memory_scope::device,        \
                             sycl::access::address_space::global_space>                            \
                atomic_d(*ptr);                                                                    \
            atomic_d |= (type) 1;                                                                  \
        }                                                                                          \
        break;                                                                                     \
    }

ISHMEM_GEN_AMO_BITWISE_FUNCTION(int test_or(SIGNATURE), RESULT, or)

// FETCH_XOR
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator)                                              \
    {                                                                                              \
        type *ptr = (type *) ishmem_ptr(dest, pe);                                                 \
        volatile type val __attribute__((unused)) = 0;                                             \
        for (size_t iter = 0; iter < iterations; iter++) {                                         \
            sycl::atomic_ref<type, sycl::memory_order::relaxed, sycl::memory_scope::device,        \
                             sycl::access::address_space::global_space>                            \
                atomic_d(*ptr);                                                                    \
            val = atomic_d.fetch_or((type) 1, sycl::memory_order::relaxed,                         \
                                    sycl::memory_scope::device);                                   \
        }                                                                                          \
        break;                                                                                     \
    }

ISHMEM_GEN_AMO_BITWISE_FUNCTION(int test_fetch_xor(SIGNATURE), RESULT, fetch_xor)

// XOR
#undef ISHMEM_AMO_BRANCH
#define ISHMEM_AMO_BRANCH(enum, name, type, operator)                                              \
    {                                                                                              \
        type *ptr = (type *) ishmem_ptr(dest, pe);                                                 \
        for (size_t iter = 0; iter < iterations; iter++) {                                         \
            sycl::atomic_ref<type, sycl::memory_order::relaxed, sycl::memory_scope::device,        \
                             sycl::access::address_space::global_space>                            \
                atomic_d(*ptr);                                                                    \
            atomic_d ^= (type) 1;                                                                  \
            ishmem_##name##_atomic_xor((type *) dest, (type) 0, pe);                               \
        }                                                                                          \
        break;                                                                                     \
    }

ISHMEM_GEN_AMO_BITWISE_FUNCTION(int test_xor(SIGNATURE), RESULT, xor)

#define BRANCH(name)                                                                               \
    test_##name(t, op, iterations, dest, pe);                                                      \
    break

SYCL_EXTERNAL int test_amo(ishmemi_type_t t, ishmemi_op_t op, size_t iterations, void *dest, int pe)
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
    return (0);
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
    ishmem_sync_all();
    return (test.finalize_and_report(errors));
}
