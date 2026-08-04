/* Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* Link/build smoke test for the primary
 * ishmemx_<op>_reduce_work_group<T, Group> templates.
 *
 * The existing reduce_*.cpp unit tests exercise only the typed wrappers
 * (ishmemx_<typename>_<op>_reduce_work_group<Group>), so a missing explicit
 * instantiation of the primary template — declared SYCL_EXTERNAL in
 * ishmemx.h — would slip past CI. This test calls representative
 * (op, C++ type, Group) primary-template specialisations and confirms the
 * SYCL JIT/AOT link succeeds.
 *
 * Type coverage is intentionally narrow (one signed + one unsigned per family,
 * plus float/double for arithmetic) — the regression we are guarding against
 * is the all-or-nothing absence of the ISHMEMI_INSTANTIATE_*_PRIMARY macros
 * in src/collectives/reduce_impl.h, not per-type behaviour. Group coverage
 * is wide (sycl::group<1..3> + sycl::sub_group).
 *
 * Correctness of the reduction is covered by reduce_<op>.cpp.
 */

#include <cstdint>
#include <cstdio>
#include <sycl/sycl.hpp>

#include "ishmem.h"
#include "ishmemx.h"

template <typename T>
static T *alloc_dest(T *&src_out)
{
    /* dest + source live in the symmetric heap. nelems = 1 keeps the kernel cheap;
     * we are testing link resolution and dispatch, not reduction throughput. */
    T *dest = static_cast<T *>(ishmem_malloc(sizeof(T)));
    if (!dest) {
        src_out = nullptr;
        return nullptr;
    }
    src_out = static_cast<T *>(ishmem_malloc(sizeof(T)));
    if (!src_out) {
        ishmem_free(dest);
        return nullptr;
    }
    return dest;
}

template <typename T>
static void free_pair(T *dest, T *src)
{
    ishmem_free(dest);
    ishmem_free(src);
}

/* Call both the no-team and the explicit-team primary-template overloads. */
#define DEFINE_OP_INVOKER(opname)                                                                  \
    struct opname##_invoker{template <typename T, typename Group> static void call(                \
        T * dest, const T *src, const Group &grp){                                                 \
        (void) ishmemx_##opname##_reduce_work_group<T, Group>(dest, src, 1, grp);                  \
    (void) ishmemx_##opname##_reduce_work_group<T, Group>(ISHMEM_TEAM_WORLD, dest, src, 1, grp);   \
    }                                                                                              \
    }                                                                                              \
    ;

DEFINE_OP_INVOKER(and)
DEFINE_OP_INVOKER(or)
DEFINE_OP_INVOKER(xor)
DEFINE_OP_INVOKER(max)
DEFINE_OP_INVOKER(min)
DEFINE_OP_INVOKER(sum)
DEFINE_OP_INVOKER(prod)

/* Buffer pair per representative type. */
struct buffers {
    /* Bitwise representatives. */
    uint32_t *d_u32;
    uint32_t *s_u32;
    int32_t *d_i32;
    int32_t *s_i32;
    /* Arithmetic representatives. */
    int *d_int;
    int *s_int;
    unsigned int *d_uint;
    unsigned int *s_uint;
    float *d_f;
    float *s_f;
    double *d_d;
    double *s_d;
};

#define CALL_BITWISE(b, grp)                                                                       \
    do {                                                                                           \
        and_invoker::call((b).d_u32, (b).s_u32, grp);                                              \
        and_invoker::call((b).d_i32, (b).s_i32, grp);                                              \
        or_invoker::call((b).d_u32, (b).s_u32, grp);                                               \
        or_invoker::call((b).d_i32, (b).s_i32, grp);                                               \
        xor_invoker::call((b).d_u32, (b).s_u32, grp);                                              \
        xor_invoker::call((b).d_i32, (b).s_i32, grp);                                              \
    } while (0)

#define CALL_ARITH_FOR(opinv, b, grp)                                                              \
    do {                                                                                           \
        opinv::call((b).d_int, (b).s_int, grp);                                                    \
        opinv::call((b).d_uint, (b).s_uint, grp);                                                  \
        opinv::call((b).d_f, (b).s_f, grp);                                                        \
        opinv::call((b).d_d, (b).s_d, grp);                                                        \
    } while (0)

#define CALL_ARITHMETIC(b, grp)                                                                    \
    do {                                                                                           \
        CALL_ARITH_FOR(sum_invoker, b, grp);                                                       \
        CALL_ARITH_FOR(prod_invoker, b, grp);                                                      \
        CALL_ARITH_FOR(max_invoker, b, grp);                                                       \
        CALL_ARITH_FOR(min_invoker, b, grp);                                                       \
    } while (0)

int main(int /*argc*/, char ** /*argv*/)
{
    ishmem_init();
    sycl::queue q;

    buffers b{};
    b.d_u32 = alloc_dest<uint32_t>(b.s_u32);
    b.d_i32 = alloc_dest<int32_t>(b.s_i32);
    b.d_int = alloc_dest<int>(b.s_int);
    b.d_uint = alloc_dest<unsigned int>(b.s_uint);
    b.d_f = alloc_dest<float>(b.s_f);
    b.d_d = alloc_dest<double>(b.s_d);

    // Ensure all PEs have allocated before we launch kernels that use the buffers
    ishmem_barrier_all();

    if (!b.d_u32 || !b.s_u32 || !b.d_i32 || !b.s_i32 || !b.d_int || !b.s_int || !b.d_uint ||
        !b.s_uint || !b.d_f || !b.s_f || !b.d_d || !b.s_d) {
        fprintf(stderr, "[%d] reduce_primary_work_group: ishmem_malloc failed\n", ishmem_my_pe());
        ishmem_finalize();
        return 1;
    }

    /* Smallest nd_range that exercises each Group shape. Reduction is over 1
     * element; we only need a real work-group / sub-group for the kernel to
     * compile and dispatch. */
    constexpr size_t WG = 16;
    sycl::nd_range<1> r1{sycl::range<1>(WG), sycl::range<1>(WG)};
    sycl::nd_range<2> r2{sycl::range<2>(4, 4), sycl::range<2>(4, 4)};
    sycl::nd_range<3> r3{sycl::range<3>(2, 2, 4), sycl::range<3>(2, 2, 4)};

    int errors = 0;
    try {
        /* sycl::group<1> */
        q.parallel_for(r1, [=](sycl::nd_item<1> it) {
             auto grp = it.get_group();
             CALL_BITWISE(b, grp);
             CALL_ARITHMETIC(b, grp);
         }).wait_and_throw();

        /* sycl::sub_group */
        q.parallel_for(r1, [=](sycl::nd_item<1> it) {
             auto grp = it.get_sub_group();
             CALL_BITWISE(b, grp);
             CALL_ARITHMETIC(b, grp);
         }).wait_and_throw();

        /* sycl::group<2> */
        q.parallel_for(r2, [=](sycl::nd_item<2> it) {
             auto grp = it.get_group();
             CALL_BITWISE(b, grp);
             CALL_ARITHMETIC(b, grp);
         }).wait_and_throw();

        /* sycl::group<3> */
        q.parallel_for(r3, [=](sycl::nd_item<3> it) {
             auto grp = it.get_group();
             CALL_BITWISE(b, grp);
             CALL_ARITHMETIC(b, grp);
         }).wait_and_throw();

        printf("[%d] reduce_primary_work_group: all primary-template kernels built and ran\n",
               ishmem_my_pe());
    } catch (sycl::exception &e) {
        fprintf(stderr, "[%d] reduce_primary_work_group: SYCL exception: %s\n", ishmem_my_pe(),
                e.what());
        errors = 1;
    }

    // Wait for all PEs to finish before we free buffers and finalize
    ishmem_barrier_all();

    free_pair(b.d_u32, b.s_u32);
    free_pair(b.d_i32, b.s_i32);
    free_pair(b.d_int, b.s_int);
    free_pair(b.d_uint, b.s_uint);
    free_pair(b.d_f, b.s_f);
    free_pair(b.d_d, b.s_d);

    ishmem_finalize();
    return errors;
}
