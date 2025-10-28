/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "proxy_func.h"
#include "ishmem.h"
#include "proxy_impl.h"
#include "timestamp.h"
#include "collectives/reduce_impl.h"

ishmemi_runtime_proxy_func_t **ishmemi_upcall_funcs;

/* these are the table functions which are different for upcalls than
 * for calling the runtime
 */

int ishmemi_proxy_uint8_put_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, PUT);
    ishmem_uint8_put(dest, src, nelems, pe);
    return 0;
}

int ishmemi_proxy_uint8_get_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, GET);
    ishmem_uint8_get(dest, src, nelems, pe);
    return 0;
}

int ishmemi_proxy_uint8_put_nbi_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, PUT_NBI);
    ishmem_uint8_put_nbi(dest, src, nelems, pe);
    return 0;
}

int ishmemi_proxy_uint8_get_nbi_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, GET_NBI);
    ishmem_uint8_get_nbi(dest, src, nelems, pe);
    return 0;
}

int ishmemi_proxy_uint8_alltoall_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, ALLTOALL);
    comp->completion.ret.i = ishmem_uint8_alltoall(team, dest, src, nelems);
    return 0;
}

int ishmemi_proxy_uint8_broadcast_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, BCAST);
    comp->completion.ret.i = ishmem_uint8_broadcast(team, dest, src, nelems, root);
    return 0;
}

int ishmemi_proxy_uint8_collect_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, COLLECT);
    comp->completion.ret.i = ishmem_uint8_collect(team, dest, src, nelems);
    return 0;
}

int ishmemi_proxy_uint8_fcollect_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(uint8_t, FCOLLECT);
    comp->completion.ret.i = ishmem_uint8_fcollect(team, dest, src, nelems);
    return 0;
}

template <typename T, ishmemi_op_t OP>
int ishmemi_proxy_reduce_up(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ISHMEMI_RUNTIME_REQUEST_HELPER(T, OP);
    comp->completion.ret.i = ishmemi_reduce<T, OP>(team, (T *) dest, (T *) src, nelems);
    return 0;
}

/* misc functions */
int ishmemi_runtime_proxy_debug_test(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    fprintf(stderr, "proxy seq %d op %d type %d comp %d\n", msg->sequence, msg->op, msg->type,
            msg->completion);
    fflush(stderr);
    return 0;
}

int ishmemi_runtime_proxy_nop(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    return 0;
}

int ishmemi_runtime_proxy_timestamp(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    unsigned long *p = (unsigned long *) msg->dst;
    *p = rdtsc();
    return 0;
}

int ishmemi_runtime_print(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    ishmemi_message_t *printmsg = (ishmemi_message_t *) msg->src;
    ishmemx_print(printmsg->file, printmsg->line, printmsg->func, printmsg->message,
                  (ishmemx_print_msg_type_t) msg->dest_pe);
    return 0;
}

int ishmemi_proxy_func_init()
{
    ishmemi_upcall_funcs = (ishmemi_runtime_proxy_func_t **) ::malloc(
        sizeof(ishmemi_runtime_proxy_func_t *) * ISHMEMI_OP_END);
    ISHMEM_CHECK_GOTO_MSG(ishmemi_upcall_funcs == nullptr, fn_exit,
                          "Allocation of ishmemi_upcall_funcs failed\n");
    for (size_t i = 0; i < ISHMEMI_OP_END; ++i) {
        ishmemi_upcall_funcs[i] = (ishmemi_runtime_proxy_func_t *) ::malloc(
            sizeof(ishmemi_runtime_proxy_func_t) * ishmemi_runtime->proxy_func_num_types);
        ISHMEM_CHECK_GOTO_MSG(ishmemi_upcall_funcs[i] == nullptr, fn_exit,
                              "Allocation of ishmemi_upcall_funcs row failed\n");
        for (size_t j = 0; j < ishmemi_runtime->proxy_func_num_types; ++j) {
            ishmemi_upcall_funcs[i][j] = ishmemi_runtime->proxy_funcs[i][j];
        }
    }

    /* Debug operations */
    ishmemi_upcall_funcs[DEBUG_TEST][NONE] = ishmemi_runtime_proxy_debug_test;
    ishmemi_upcall_funcs[NOP][NONE] = ishmemi_runtime_proxy_nop;
    ishmemi_upcall_funcs[NOP_NO_R][NONE] = ishmemi_runtime_proxy_nop;
    ishmemi_upcall_funcs[TIMESTAMP][NONE] = ishmemi_runtime_proxy_timestamp;
    ishmemi_upcall_funcs[PRINT][NONE] = ishmemi_runtime_print;

    /* RMA upcalls */
    ishmemi_upcall_funcs[PUT][UINT8] = ishmemi_proxy_uint8_put_up;
    ishmemi_upcall_funcs[GET][UINT8] = ishmemi_proxy_uint8_get_up;
    ishmemi_upcall_funcs[PUT_NBI][UINT8] = ishmemi_proxy_uint8_put_nbi_up;
    ishmemi_upcall_funcs[GET_NBI][UINT8] = ishmemi_proxy_uint8_get_nbi_up;

    /* collectives and reductions that call the runtime rather than the host implementation */
    /* Collectives */
    ishmemi_upcall_funcs[ALLTOALL][UINT8] = ishmemi_proxy_uint8_alltoall_up;
    ishmemi_upcall_funcs[BCAST][UINT8] = ishmemi_proxy_uint8_broadcast_up;
    ishmemi_upcall_funcs[COLLECT][UINT8] = ishmemi_proxy_uint8_collect_up;
    ishmemi_upcall_funcs[FCOLLECT][UINT8] = ishmemi_proxy_uint8_fcollect_up;

    /* Reductions */
    ishmemi_upcall_funcs[AND_REDUCE][UINT8] = ishmemi_proxy_reduce_up<uint8_t, AND_REDUCE>;
    ishmemi_upcall_funcs[OR_REDUCE][UINT8] = ishmemi_proxy_reduce_up<uint8_t, OR_REDUCE>;
    ishmemi_upcall_funcs[XOR_REDUCE][UINT8] = ishmemi_proxy_reduce_up<uint8_t, XOR_REDUCE>;
    ishmemi_upcall_funcs[MAX_REDUCE][UINT8] = ishmemi_proxy_reduce_up<uint8_t, MAX_REDUCE>;
    ishmemi_upcall_funcs[MIN_REDUCE][UINT8] = ishmemi_proxy_reduce_up<uint8_t, MIN_REDUCE>;
    ishmemi_upcall_funcs[SUM_REDUCE][UINT8] = ishmemi_proxy_reduce_up<uint8_t, SUM_REDUCE>;
    ishmemi_upcall_funcs[PROD_REDUCE][UINT8] = ishmemi_proxy_reduce_up<uint8_t, PROD_REDUCE>;

    ishmemi_upcall_funcs[AND_REDUCE][UINT16] = ishmemi_proxy_reduce_up<uint16_t, AND_REDUCE>;
    ishmemi_upcall_funcs[OR_REDUCE][UINT16] = ishmemi_proxy_reduce_up<uint16_t, OR_REDUCE>;
    ishmemi_upcall_funcs[XOR_REDUCE][UINT16] = ishmemi_proxy_reduce_up<uint16_t, XOR_REDUCE>;
    ishmemi_upcall_funcs[MAX_REDUCE][UINT16] = ishmemi_proxy_reduce_up<uint16_t, MAX_REDUCE>;
    ishmemi_upcall_funcs[MIN_REDUCE][UINT16] = ishmemi_proxy_reduce_up<uint16_t, MIN_REDUCE>;
    ishmemi_upcall_funcs[SUM_REDUCE][UINT16] = ishmemi_proxy_reduce_up<uint16_t, SUM_REDUCE>;
    ishmemi_upcall_funcs[PROD_REDUCE][UINT16] = ishmemi_proxy_reduce_up<uint16_t, PROD_REDUCE>;

    ishmemi_upcall_funcs[AND_REDUCE][UINT32] = ishmemi_proxy_reduce_up<uint32_t, AND_REDUCE>;
    ishmemi_upcall_funcs[OR_REDUCE][UINT32] = ishmemi_proxy_reduce_up<uint32_t, OR_REDUCE>;
    ishmemi_upcall_funcs[XOR_REDUCE][UINT32] = ishmemi_proxy_reduce_up<uint32_t, XOR_REDUCE>;
    ishmemi_upcall_funcs[MAX_REDUCE][UINT32] = ishmemi_proxy_reduce_up<uint32_t, MAX_REDUCE>;
    ishmemi_upcall_funcs[MIN_REDUCE][UINT32] = ishmemi_proxy_reduce_up<uint32_t, MIN_REDUCE>;
    ishmemi_upcall_funcs[SUM_REDUCE][UINT32] = ishmemi_proxy_reduce_up<uint32_t, SUM_REDUCE>;
    ishmemi_upcall_funcs[PROD_REDUCE][UINT32] = ishmemi_proxy_reduce_up<uint32_t, PROD_REDUCE>;

    ishmemi_upcall_funcs[AND_REDUCE][UINT64] = ishmemi_proxy_reduce_up<uint64_t, AND_REDUCE>;
    ishmemi_upcall_funcs[OR_REDUCE][UINT64] = ishmemi_proxy_reduce_up<uint64_t, OR_REDUCE>;
    ishmemi_upcall_funcs[XOR_REDUCE][UINT64] = ishmemi_proxy_reduce_up<uint64_t, XOR_REDUCE>;
    ishmemi_upcall_funcs[MAX_REDUCE][UINT64] = ishmemi_proxy_reduce_up<uint64_t, MAX_REDUCE>;
    ishmemi_upcall_funcs[MIN_REDUCE][UINT64] = ishmemi_proxy_reduce_up<uint64_t, MIN_REDUCE>;
    ishmemi_upcall_funcs[SUM_REDUCE][UINT64] = ishmemi_proxy_reduce_up<uint64_t, SUM_REDUCE>;
    ishmemi_upcall_funcs[PROD_REDUCE][UINT64] = ishmemi_proxy_reduce_up<uint64_t, PROD_REDUCE>;

    ishmemi_upcall_funcs[AND_REDUCE][ULONGLONG] =
        ishmemi_proxy_reduce_up<unsigned long long, AND_REDUCE>;
    ishmemi_upcall_funcs[OR_REDUCE][ULONGLONG] =
        ishmemi_proxy_reduce_up<unsigned long long, OR_REDUCE>;
    ishmemi_upcall_funcs[XOR_REDUCE][ULONGLONG] =
        ishmemi_proxy_reduce_up<unsigned long long, XOR_REDUCE>;
    ishmemi_upcall_funcs[MAX_REDUCE][ULONGLONG] =
        ishmemi_proxy_reduce_up<unsigned long long, MAX_REDUCE>;
    ishmemi_upcall_funcs[MIN_REDUCE][ULONGLONG] =
        ishmemi_proxy_reduce_up<unsigned long long, MIN_REDUCE>;
    ishmemi_upcall_funcs[SUM_REDUCE][ULONGLONG] =
        ishmemi_proxy_reduce_up<unsigned long long, SUM_REDUCE>;
    ishmemi_upcall_funcs[PROD_REDUCE][ULONGLONG] =
        ishmemi_proxy_reduce_up<unsigned long long, PROD_REDUCE>;

    ishmemi_upcall_funcs[AND_REDUCE][INT8] = ishmemi_proxy_reduce_up<int8_t, AND_REDUCE>;
    ishmemi_upcall_funcs[OR_REDUCE][INT8] = ishmemi_proxy_reduce_up<int8_t, OR_REDUCE>;
    ishmemi_upcall_funcs[XOR_REDUCE][INT8] = ishmemi_proxy_reduce_up<int8_t, XOR_REDUCE>;
    ishmemi_upcall_funcs[MAX_REDUCE][INT8] = ishmemi_proxy_reduce_up<int8_t, MAX_REDUCE>;
    ishmemi_upcall_funcs[MIN_REDUCE][INT8] = ishmemi_proxy_reduce_up<int8_t, MIN_REDUCE>;
    ishmemi_upcall_funcs[SUM_REDUCE][INT8] = ishmemi_proxy_reduce_up<int8_t, SUM_REDUCE>;
    ishmemi_upcall_funcs[PROD_REDUCE][INT8] = ishmemi_proxy_reduce_up<int8_t, PROD_REDUCE>;

    ishmemi_upcall_funcs[AND_REDUCE][INT16] = ishmemi_proxy_reduce_up<int16_t, AND_REDUCE>;
    ishmemi_upcall_funcs[OR_REDUCE][INT16] = ishmemi_proxy_reduce_up<int16_t, OR_REDUCE>;
    ishmemi_upcall_funcs[XOR_REDUCE][INT16] = ishmemi_proxy_reduce_up<int16_t, XOR_REDUCE>;
    ishmemi_upcall_funcs[MAX_REDUCE][INT16] = ishmemi_proxy_reduce_up<int16_t, MAX_REDUCE>;
    ishmemi_upcall_funcs[MIN_REDUCE][INT16] = ishmemi_proxy_reduce_up<int16_t, MIN_REDUCE>;
    ishmemi_upcall_funcs[SUM_REDUCE][INT16] = ishmemi_proxy_reduce_up<int16_t, SUM_REDUCE>;
    ishmemi_upcall_funcs[PROD_REDUCE][INT16] = ishmemi_proxy_reduce_up<int16_t, PROD_REDUCE>;

    ishmemi_upcall_funcs[AND_REDUCE][INT32] = ishmemi_proxy_reduce_up<int32_t, AND_REDUCE>;
    ishmemi_upcall_funcs[OR_REDUCE][INT32] = ishmemi_proxy_reduce_up<int32_t, OR_REDUCE>;
    ishmemi_upcall_funcs[XOR_REDUCE][INT32] = ishmemi_proxy_reduce_up<int32_t, XOR_REDUCE>;
    ishmemi_upcall_funcs[MAX_REDUCE][INT32] = ishmemi_proxy_reduce_up<int32_t, MAX_REDUCE>;
    ishmemi_upcall_funcs[MIN_REDUCE][INT32] = ishmemi_proxy_reduce_up<int32_t, MIN_REDUCE>;
    ishmemi_upcall_funcs[SUM_REDUCE][INT32] = ishmemi_proxy_reduce_up<int32_t, SUM_REDUCE>;
    ishmemi_upcall_funcs[PROD_REDUCE][INT32] = ishmemi_proxy_reduce_up<int32_t, PROD_REDUCE>;

    ishmemi_upcall_funcs[AND_REDUCE][INT64] = ishmemi_proxy_reduce_up<int64_t, AND_REDUCE>;
    ishmemi_upcall_funcs[OR_REDUCE][INT64] = ishmemi_proxy_reduce_up<int64_t, OR_REDUCE>;
    ishmemi_upcall_funcs[XOR_REDUCE][INT64] = ishmemi_proxy_reduce_up<int64_t, XOR_REDUCE>;
    ishmemi_upcall_funcs[MAX_REDUCE][INT64] = ishmemi_proxy_reduce_up<int64_t, MAX_REDUCE>;
    ishmemi_upcall_funcs[MIN_REDUCE][INT64] = ishmemi_proxy_reduce_up<int64_t, MIN_REDUCE>;
    ishmemi_upcall_funcs[SUM_REDUCE][INT64] = ishmemi_proxy_reduce_up<int64_t, SUM_REDUCE>;
    ishmemi_upcall_funcs[PROD_REDUCE][INT64] = ishmemi_proxy_reduce_up<int64_t, PROD_REDUCE>;

    ishmemi_upcall_funcs[MAX_REDUCE][LONGLONG] = ishmemi_proxy_reduce_up<long long, MAX_REDUCE>;
    ishmemi_upcall_funcs[MIN_REDUCE][LONGLONG] = ishmemi_proxy_reduce_up<long long, MIN_REDUCE>;
    ishmemi_upcall_funcs[SUM_REDUCE][LONGLONG] = ishmemi_proxy_reduce_up<long long, SUM_REDUCE>;
    ishmemi_upcall_funcs[PROD_REDUCE][LONGLONG] = ishmemi_proxy_reduce_up<long long, PROD_REDUCE>;

    ishmemi_upcall_funcs[MAX_REDUCE][FLOAT] = ishmemi_proxy_reduce_up<float, MAX_REDUCE>;
    ishmemi_upcall_funcs[MIN_REDUCE][FLOAT] = ishmemi_proxy_reduce_up<float, MIN_REDUCE>;
    ishmemi_upcall_funcs[SUM_REDUCE][FLOAT] = ishmemi_proxy_reduce_up<float, SUM_REDUCE>;
    ishmemi_upcall_funcs[PROD_REDUCE][FLOAT] = ishmemi_proxy_reduce_up<float, PROD_REDUCE>;

    ishmemi_upcall_funcs[MAX_REDUCE][DOUBLE] = ishmemi_proxy_reduce_up<double, MAX_REDUCE>;
    ishmemi_upcall_funcs[MIN_REDUCE][DOUBLE] = ishmemi_proxy_reduce_up<double, MIN_REDUCE>;
    ishmemi_upcall_funcs[SUM_REDUCE][DOUBLE] = ishmemi_proxy_reduce_up<double, SUM_REDUCE>;
    ishmemi_upcall_funcs[PROD_REDUCE][DOUBLE] = ishmemi_proxy_reduce_up<double, PROD_REDUCE>;

    return 0;
fn_exit:
    return 1;
}

int ishmemi_proxy_func_fini()
{
    for (size_t i = 0; i < ISHMEMI_OP_END; ++i) {
        for (size_t j = 0; j < ishmemi_runtime->proxy_func_num_types; ++j) {
            ishmemi_upcall_funcs[i][j] = nullptr;
        }
        ::free(ishmemi_upcall_funcs[i]);
        ishmemi_upcall_funcs[i] = nullptr;
    }
    ::free(ishmemi_upcall_funcs);
    ishmemi_upcall_funcs = nullptr;
    return 0;
}
