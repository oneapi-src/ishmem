/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "proxy_func.h"
#include "ishmem.h"
#include "impl_proxy.h"  // ISHMEMI_RUNTIME_REQUEST_HELPER

ishmemi_runtime_proxy_func_t **ishmemi_upcall_funcs;

/* these are the table functions which are different for upcalls than
 * for calling the runtime
 */

void ishmemi_proxy_uint8_put_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    ishmem_uint8_put(dest, src, nelems, pe);
}

void ishmemi_proxy_uint8_get_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    ishmem_uint8_get(dest, src, nelems, pe);
}

void ishmemi_proxy_uint8_alltoall_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = ishmem_uint8_alltoall(dest, src, nelems);
}

void ishmemi_proxy_uint8_broadcast_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = ishmem_uint8_broadcast(dest, src, nelems, root);
}

void ishmemi_proxy_uint8_collect_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = ishmem_uint8_collect(dest, src, nelems);
}

void ishmemi_proxy_uint8_fcollect_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = ishmem_uint8_fcollect(dest, src, nelems);
}

/* Reductions */
void ishmemi_proxy_uint8_and_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = ishmem_uint8_and_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint8_or_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = ishmem_uint8_or_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint8_xor_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = ishmem_uint8_xor_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint8_max_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = ishmem_uint8_max_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint8_min_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = ishmem_uint8_min_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint8_sum_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = ishmem_uint8_sum_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint8_prod_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ui8);
    comp->completion.ret.i = ishmem_uint8_prod_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint16_and_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = ishmem_uint16_and_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint16_or_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = ishmem_uint16_or_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint16_xor_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = ishmem_uint16_xor_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint16_max_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = ishmem_uint16_max_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint16_min_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = ishmem_uint16_min_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint16_sum_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = ishmem_uint16_sum_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint16_prod_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint16_t, ui16);
    comp->completion.ret.i = ishmem_uint16_prod_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint32_and_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = ishmem_uint32_and_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint32_or_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = ishmem_uint32_or_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint32_xor_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = ishmem_uint32_xor_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint32_max_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = ishmem_uint32_max_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint32_min_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = ishmem_uint32_min_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint32_sum_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = ishmem_uint32_sum_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint32_prod_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint32_t, ui32);
    comp->completion.ret.i = ishmem_uint32_prod_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint64_and_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = ishmem_uint64_and_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint64_or_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = ishmem_uint64_or_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint64_xor_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = ishmem_uint64_xor_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint64_max_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = ishmem_uint64_max_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint64_min_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = ishmem_uint64_min_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint64_sum_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = ishmem_uint64_sum_reduce(dest, src, nelems);
}

void ishmemi_proxy_uint64_prod_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint64_t, ui64);
    comp->completion.ret.i = ishmem_uint64_prod_reduce(dest, src, nelems);
}

void ishmemi_proxy_ulonglong_and_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = ishmem_ulonglong_and_reduce(dest, src, nelems);
}

void ishmemi_proxy_ulonglong_or_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = ishmem_ulonglong_or_reduce(dest, src, nelems);
}

void ishmemi_proxy_ulonglong_xor_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = ishmem_ulonglong_xor_reduce(dest, src, nelems);
}

void ishmemi_proxy_ulonglong_max_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = ishmem_ulonglong_max_reduce(dest, src, nelems);
}

void ishmemi_proxy_ulonglong_min_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = ishmem_ulonglong_min_reduce(dest, src, nelems);
}

void ishmemi_proxy_ulonglong_sum_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = ishmem_ulonglong_sum_reduce(dest, src, nelems);
}

void ishmemi_proxy_ulonglong_prod_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(unsigned long long, ull);
    comp->completion.ret.i = ishmem_ulonglong_prod_reduce(dest, src, nelems);
}

void ishmemi_proxy_int8_and_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = ishmem_int8_and_reduce(dest, src, nelems);
}

void ishmemi_proxy_int8_or_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = ishmem_int8_or_reduce(dest, src, nelems);
}

void ishmemi_proxy_int8_xor_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = ishmem_int8_xor_reduce(dest, src, nelems);
}

void ishmemi_proxy_int8_max_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = ishmem_int8_max_reduce(dest, src, nelems);
}

void ishmemi_proxy_int8_min_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = ishmem_int8_min_reduce(dest, src, nelems);
}

void ishmemi_proxy_int8_sum_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = ishmem_int8_sum_reduce(dest, src, nelems);
}

void ishmemi_proxy_int8_prod_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int8_t, i8);
    comp->completion.ret.i = ishmem_int8_prod_reduce(dest, src, nelems);
}

void ishmemi_proxy_int16_and_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = ishmem_int16_and_reduce(dest, src, nelems);
}

void ishmemi_proxy_int16_or_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = ishmem_int16_or_reduce(dest, src, nelems);
}

void ishmemi_proxy_int16_xor_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = ishmem_int16_xor_reduce(dest, src, nelems);
}

void ishmemi_proxy_int16_max_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = ishmem_int16_max_reduce(dest, src, nelems);
}

void ishmemi_proxy_int16_min_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = ishmem_int16_min_reduce(dest, src, nelems);
}

void ishmemi_proxy_int16_sum_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = ishmem_int16_sum_reduce(dest, src, nelems);
}

void ishmemi_proxy_int16_prod_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int16_t, i16);
    comp->completion.ret.i = ishmem_int16_prod_reduce(dest, src, nelems);
}

void ishmemi_proxy_int32_and_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = ishmem_int32_and_reduce(dest, src, nelems);
}

void ishmemi_proxy_int32_or_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = ishmem_int32_or_reduce(dest, src, nelems);
}

void ishmemi_proxy_int32_xor_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = ishmem_int32_xor_reduce(dest, src, nelems);
}

void ishmemi_proxy_int32_max_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = ishmem_int32_max_reduce(dest, src, nelems);
}

void ishmemi_proxy_int32_min_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = ishmem_int32_min_reduce(dest, src, nelems);
}

void ishmemi_proxy_int32_sum_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = ishmem_int32_sum_reduce(dest, src, nelems);
}

void ishmemi_proxy_int32_prod_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int32_t, i32);
    comp->completion.ret.i = ishmem_int32_prod_reduce(dest, src, nelems);
}

void ishmemi_proxy_int64_and_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = ishmem_int64_and_reduce(dest, src, nelems);
}

void ishmemi_proxy_int64_or_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = ishmem_int64_or_reduce(dest, src, nelems);
}

void ishmemi_proxy_int64_xor_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = ishmem_int64_xor_reduce(dest, src, nelems);
}

void ishmemi_proxy_int64_max_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = ishmem_int64_max_reduce(dest, src, nelems);
}

void ishmemi_proxy_int64_min_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = ishmem_int64_min_reduce(dest, src, nelems);
}

void ishmemi_proxy_int64_sum_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = ishmem_int64_sum_reduce(dest, src, nelems);
}

void ishmemi_proxy_int64_prod_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(int64_t, i64);
    comp->completion.ret.i = ishmem_int64_prod_reduce(dest, src, nelems);
}

void ishmemi_proxy_longlong_max_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i = shmem_longlong_max_reduce(team, dest, src, nelems);
}

void ishmemi_proxy_longlong_min_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i = shmem_longlong_min_reduce(team, dest, src, nelems);
}

void ishmemi_proxy_longlong_sum_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i = shmem_longlong_sum_reduce(team, dest, src, nelems);
}

void ishmemi_proxy_longlong_prod_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(long long, ll);
    comp->completion.ret.i = shmem_longlong_prod_reduce(team, dest, src, nelems);
}

void ishmemi_proxy_float_max_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    comp->completion.ret.i = ishmem_float_max_reduce(dest, src, nelems);
}

void ishmemi_proxy_float_min_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    comp->completion.ret.i = ishmem_float_min_reduce(dest, src, nelems);
}

void ishmemi_proxy_float_sum_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    comp->completion.ret.i = ishmem_float_sum_reduce(dest, src, nelems);
}

void ishmemi_proxy_float_prod_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(float, f);
    comp->completion.ret.i = ishmem_float_prod_reduce(dest, src, nelems);
}

void ishmemi_proxy_double_max_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    comp->completion.ret.i = ishmem_double_max_reduce(dest, src, nelems);
}

void ishmemi_proxy_double_min_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    comp->completion.ret.i = ishmem_double_min_reduce(dest, src, nelems);
}

void ishmemi_proxy_double_sum_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    comp->completion.ret.i = ishmem_double_sum_reduce(dest, src, nelems);
}

void ishmemi_proxy_double_prod_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(double, ld);
    comp->completion.ret.i = ishmem_double_prod_reduce(dest, src, nelems);
}

/* misc functions */
void ishmemi_runtime_proxy_debug_test(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    fprintf(stderr, "proxy seq %d op %d type %d comp %d\n", msg->sequence, msg->op, msg->type,
            msg->completion);
    fflush(stderr);
}

void ishmemi_runtime_proxy_nop(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp) {}

void ishmemi_runtime_proxy_timestamp(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    unsigned long *p = (unsigned long *) msg->dst;
    *p = rdtsc();
}

void ishmemi_runtime_print(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    switch (msg->dest_pe) {
        case ishmemx_print_msg_type_t::DEBUG:
            ISHMEM_DEBUG_MSG("%s", (char *) msg->src);
            break;
        case ishmemx_print_msg_type_t::WARNING:
            ISHMEM_WARN_MSG("%s", (char *) msg->src);
            break;
        case ishmemx_print_msg_type_t::ERROR:
            RAISE_ERROR_MSG("%s", (char *) msg->src);
            break;
        case ishmemx_print_msg_type_t::STDOUT:
            printf("%s", (char *) msg->src);
            fflush(stdout);
            break;
        case ishmemx_print_msg_type_t::STDERR:
            fprintf(stderr, "%s", (char *) msg->src);
            fflush(stderr);
            break;
    }
}

int ishmemi_proxy_func_init()
{
    ishmemi_upcall_funcs = (ishmemi_runtime_proxy_func_t **) malloc(
        sizeof(ishmemi_runtime_proxy_func_t *) * ISHMEMI_OP_END);
    ISHMEM_CHECK_GOTO_MSG(ishmemi_proxy_funcs == nullptr, fn_exit,
                          "Allocation of ishmemi_upcall_funcs failed\n");
    for (int i = 0; i < ISHMEMI_OP_END; ++i) {
        ishmemi_upcall_funcs[i] = (ishmemi_runtime_proxy_func_t *) malloc(
            sizeof(ishmemi_runtime_proxy_func_t) * ishmemi_runtime_proxy_func_num_types);
        ISHMEM_CHECK_GOTO_MSG(ishmemi_proxy_funcs == nullptr, fn_exit,
                              "Allocation of ishmemi_upcall_funcs row failed\n");
        for (int j = 0; j < ishmemi_runtime_proxy_func_num_types; ++j) {
            ishmemi_upcall_funcs[i][j] = ishmemi_proxy_funcs[i][j];
        }
    }
    /* Debug operations */
    ishmemi_upcall_funcs[DEBUG_TEST][0] = ishmemi_runtime_proxy_debug_test;
    ishmemi_upcall_funcs[NOP][0] = ishmemi_runtime_proxy_nop;
    ishmemi_upcall_funcs[NOP_NO_R][0] = ishmemi_runtime_proxy_nop;
    ishmemi_upcall_funcs[TIMESTAMP][0] = ishmemi_runtime_proxy_timestamp;
    ishmemi_upcall_funcs[PRINT][0] = ishmemi_runtime_print;

    /* RMA upcalls */
    ishmemi_upcall_funcs[PUT][UINT8] = ishmemi_proxy_uint8_put_up;
    ishmemi_upcall_funcs[GET][UINT8] = ishmemi_proxy_uint8_get_up;

    /* collectives and reductions that call the runtime rather than the host implementation */
    /* Collectives */
    ishmemi_upcall_funcs[ALLTOALL][UINT8] = ishmemi_proxy_uint8_alltoall_up;
    ishmemi_upcall_funcs[BCAST][UINT8] = ishmemi_proxy_uint8_broadcast_up;
    ishmemi_upcall_funcs[COLLECT][UINT8] = ishmemi_proxy_uint8_collect_up;
    ishmemi_upcall_funcs[FCOLLECT][UINT8] = ishmemi_proxy_uint8_fcollect_up;

    /* Reductions */
    ishmemi_upcall_funcs[AND_REDUCE][UINT8] = ishmemi_proxy_uint8_and_reduce_up;
    ishmemi_upcall_funcs[OR_REDUCE][UINT8] = ishmemi_proxy_uint8_or_reduce_up;
    ishmemi_upcall_funcs[XOR_REDUCE][UINT8] = ishmemi_proxy_uint8_xor_reduce_up;
    ishmemi_upcall_funcs[MAX_REDUCE][UINT8] = ishmemi_proxy_uint8_max_reduce_up;
    ishmemi_upcall_funcs[MIN_REDUCE][UINT8] = ishmemi_proxy_uint8_min_reduce_up;
    ishmemi_upcall_funcs[SUM_REDUCE][UINT8] = ishmemi_proxy_uint8_sum_reduce_up;
    ishmemi_upcall_funcs[PROD_REDUCE][UINT8] = ishmemi_proxy_uint8_prod_reduce_up;

    ishmemi_upcall_funcs[AND_REDUCE][UINT16] = ishmemi_proxy_uint16_and_reduce_up;
    ishmemi_upcall_funcs[OR_REDUCE][UINT16] = ishmemi_proxy_uint16_or_reduce_up;
    ishmemi_upcall_funcs[XOR_REDUCE][UINT16] = ishmemi_proxy_uint16_xor_reduce_up;
    ishmemi_upcall_funcs[MAX_REDUCE][UINT16] = ishmemi_proxy_uint16_max_reduce_up;
    ishmemi_upcall_funcs[MIN_REDUCE][UINT16] = ishmemi_proxy_uint16_min_reduce_up;
    ishmemi_upcall_funcs[SUM_REDUCE][UINT16] = ishmemi_proxy_uint16_sum_reduce_up;
    ishmemi_upcall_funcs[PROD_REDUCE][UINT16] = ishmemi_proxy_uint16_prod_reduce_up;

    ishmemi_upcall_funcs[AND_REDUCE][UINT32] = ishmemi_proxy_uint32_and_reduce_up;
    ishmemi_upcall_funcs[OR_REDUCE][UINT32] = ishmemi_proxy_uint32_or_reduce_up;
    ishmemi_upcall_funcs[XOR_REDUCE][UINT32] = ishmemi_proxy_uint32_xor_reduce_up;
    ishmemi_upcall_funcs[MAX_REDUCE][UINT32] = ishmemi_proxy_uint32_max_reduce_up;
    ishmemi_upcall_funcs[MIN_REDUCE][UINT32] = ishmemi_proxy_uint32_min_reduce_up;
    ishmemi_upcall_funcs[SUM_REDUCE][UINT32] = ishmemi_proxy_uint32_sum_reduce_up;
    ishmemi_upcall_funcs[PROD_REDUCE][UINT32] = ishmemi_proxy_uint32_prod_reduce_up;

    ishmemi_upcall_funcs[AND_REDUCE][UINT64] = ishmemi_proxy_uint64_and_reduce_up;
    ishmemi_upcall_funcs[OR_REDUCE][UINT64] = ishmemi_proxy_uint64_or_reduce_up;
    ishmemi_upcall_funcs[XOR_REDUCE][UINT64] = ishmemi_proxy_uint64_xor_reduce_up;
    ishmemi_upcall_funcs[MAX_REDUCE][UINT64] = ishmemi_proxy_uint64_max_reduce_up;
    ishmemi_upcall_funcs[MIN_REDUCE][UINT64] = ishmemi_proxy_uint64_min_reduce_up;
    ishmemi_upcall_funcs[SUM_REDUCE][UINT64] = ishmemi_proxy_uint64_sum_reduce_up;
    ishmemi_upcall_funcs[PROD_REDUCE][UINT64] = ishmemi_proxy_uint64_prod_reduce_up;

    ishmemi_upcall_funcs[AND_REDUCE][ULONGLONG] = ishmemi_proxy_ulonglong_and_reduce_up;
    ishmemi_upcall_funcs[OR_REDUCE][ULONGLONG] = ishmemi_proxy_ulonglong_or_reduce_up;
    ishmemi_upcall_funcs[XOR_REDUCE][ULONGLONG] = ishmemi_proxy_ulonglong_xor_reduce_up;
    ishmemi_upcall_funcs[MAX_REDUCE][ULONGLONG] = ishmemi_proxy_ulonglong_max_reduce_up;
    ishmemi_upcall_funcs[MIN_REDUCE][ULONGLONG] = ishmemi_proxy_ulonglong_min_reduce_up;
    ishmemi_upcall_funcs[SUM_REDUCE][ULONGLONG] = ishmemi_proxy_ulonglong_sum_reduce_up;
    ishmemi_upcall_funcs[PROD_REDUCE][ULONGLONG] = ishmemi_proxy_ulonglong_prod_reduce_up;

    ishmemi_upcall_funcs[AND_REDUCE][INT8] = ishmemi_proxy_int8_and_reduce_up;
    ishmemi_upcall_funcs[OR_REDUCE][INT8] = ishmemi_proxy_int8_or_reduce_up;
    ishmemi_upcall_funcs[XOR_REDUCE][INT8] = ishmemi_proxy_int8_xor_reduce_up;
    ishmemi_upcall_funcs[MAX_REDUCE][INT8] = ishmemi_proxy_int8_max_reduce_up;
    ishmemi_upcall_funcs[MIN_REDUCE][INT8] = ishmemi_proxy_int8_min_reduce_up;
    ishmemi_upcall_funcs[SUM_REDUCE][INT8] = ishmemi_proxy_int8_sum_reduce_up;
    ishmemi_upcall_funcs[PROD_REDUCE][INT8] = ishmemi_proxy_int8_prod_reduce_up;

    ishmemi_upcall_funcs[AND_REDUCE][INT16] = ishmemi_proxy_int16_and_reduce_up;
    ishmemi_upcall_funcs[OR_REDUCE][INT16] = ishmemi_proxy_int16_or_reduce_up;
    ishmemi_upcall_funcs[XOR_REDUCE][INT16] = ishmemi_proxy_int16_xor_reduce_up;
    ishmemi_upcall_funcs[MAX_REDUCE][INT16] = ishmemi_proxy_int16_max_reduce_up;
    ishmemi_upcall_funcs[MIN_REDUCE][INT16] = ishmemi_proxy_int16_min_reduce_up;
    ishmemi_upcall_funcs[SUM_REDUCE][INT16] = ishmemi_proxy_int16_sum_reduce_up;
    ishmemi_upcall_funcs[PROD_REDUCE][INT16] = ishmemi_proxy_int16_prod_reduce_up;

    ishmemi_upcall_funcs[AND_REDUCE][INT32] = ishmemi_proxy_int32_and_reduce_up;
    ishmemi_upcall_funcs[OR_REDUCE][INT32] = ishmemi_proxy_int32_or_reduce_up;
    ishmemi_upcall_funcs[XOR_REDUCE][INT32] = ishmemi_proxy_int32_xor_reduce_up;
    ishmemi_upcall_funcs[MAX_REDUCE][INT32] = ishmemi_proxy_int32_max_reduce_up;
    ishmemi_upcall_funcs[MIN_REDUCE][INT32] = ishmemi_proxy_int32_min_reduce_up;
    ishmemi_upcall_funcs[SUM_REDUCE][INT32] = ishmemi_proxy_int32_sum_reduce_up;
    ishmemi_upcall_funcs[PROD_REDUCE][INT32] = ishmemi_proxy_int32_prod_reduce_up;

    ishmemi_upcall_funcs[AND_REDUCE][INT64] = ishmemi_proxy_int64_and_reduce_up;
    ishmemi_upcall_funcs[OR_REDUCE][INT64] = ishmemi_proxy_int64_or_reduce_up;
    ishmemi_upcall_funcs[XOR_REDUCE][INT64] = ishmemi_proxy_int64_xor_reduce_up;
    ishmemi_upcall_funcs[MAX_REDUCE][INT64] = ishmemi_proxy_int64_max_reduce_up;
    ishmemi_upcall_funcs[MIN_REDUCE][INT64] = ishmemi_proxy_int64_min_reduce_up;
    ishmemi_upcall_funcs[SUM_REDUCE][INT64] = ishmemi_proxy_int64_sum_reduce_up;
    ishmemi_upcall_funcs[PROD_REDUCE][INT64] = ishmemi_proxy_int64_prod_reduce_up;

    ishmemi_upcall_funcs[MAX_REDUCE][LONGLONG] = ishmemi_proxy_longlong_max_reduce_up;
    ishmemi_upcall_funcs[MIN_REDUCE][LONGLONG] = ishmemi_proxy_longlong_min_reduce_up;
    ishmemi_upcall_funcs[SUM_REDUCE][LONGLONG] = ishmemi_proxy_longlong_sum_reduce_up;
    ishmemi_upcall_funcs[PROD_REDUCE][LONGLONG] = ishmemi_proxy_longlong_prod_reduce_up;

    ishmemi_upcall_funcs[MAX_REDUCE][FLOAT] = ishmemi_proxy_float_max_reduce_up;
    ishmemi_upcall_funcs[MIN_REDUCE][FLOAT] = ishmemi_proxy_float_min_reduce_up;
    ishmemi_upcall_funcs[SUM_REDUCE][FLOAT] = ishmemi_proxy_float_sum_reduce_up;
    ishmemi_upcall_funcs[PROD_REDUCE][FLOAT] = ishmemi_proxy_float_prod_reduce_up;

    ishmemi_upcall_funcs[MAX_REDUCE][DOUBLE] = ishmemi_proxy_double_max_reduce_up;
    ishmemi_upcall_funcs[MIN_REDUCE][DOUBLE] = ishmemi_proxy_double_min_reduce_up;
    ishmemi_upcall_funcs[SUM_REDUCE][DOUBLE] = ishmemi_proxy_double_sum_reduce_up;
    ishmemi_upcall_funcs[PROD_REDUCE][DOUBLE] = ishmemi_proxy_double_prod_reduce_up;
    return (0);
fn_exit:
    return (1);
}

int ishmemi_proxy_func_fini()
{
    for (int i = 0; i < ISHMEMI_OP_END; ++i) {
        for (int j = 0; j < ishmemi_runtime_proxy_func_num_types; ++j) {
            ishmemi_upcall_funcs[i][j] = nullptr;
        }
        free(ishmemi_upcall_funcs[i]);
        ishmemi_upcall_funcs[i] = nullptr;
    }
    free(ishmemi_upcall_funcs);
    ishmemi_upcall_funcs = nullptr;
    return (0);
}
