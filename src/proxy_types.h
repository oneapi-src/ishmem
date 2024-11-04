/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_PROXY_TYPES_H
#define ISHMEM_PROXY_TYPES_H

#include "ishmem/types.h"
#include "teams.h"

/* if completion is 0, it is the same as sequence & (RING_SIZE-1)
 * if completion is not 0, it is as given and should be in the range [RING_SIZE..2*RING_SIZE)
 */
typedef struct {
    /* Destination PE */
    int dest_pe;
    /* Root PE used in broadcast */
    int root;
    /* Source, destination, and number of elements */
    union {
        const void *src;
        /* For non-blocking amos */
        void *fetch;
        /* output parameter for [wait_until,test]_some; array of indices which met comparison
         * criteria */
        size_t *indices;
    };
    void *dst;
    size_t nelems;
    union {
        /* Signal address for put-signal */
        uint64_t *sig_addr;
        /* Optional mask array for [wait_until,test]_[all,any,some] */
        const int *status;
        /* Block size for ibput/ibget */
        size_t bsize;
        /* Index of a team */
        ishmem_team_t team;
    };
    /* Attribute used for condition, comparison types, or signal operation types */
    union {
        /* Condition used for compare_swap */
        ishmemi_union_type cond;
        /* Comparison used for pt-pt synch. */
        int cmp;
        /* Signal operation */
        int sig_op;
        /* Destination stride value for iput/iget*/
        ptrdiff_t dst_stride;
    };
    /* Attribute used for values needed for comparison, signal  */
    union {
        ishmemi_union_type value;
        ishmemi_union_type cmp_value;
        const void *cmp_values;
        uint64_t signal;
        /* Source stride value for iput/iget*/
        ptrdiff_t src_stride;
    };
    /* Operation and data type. */
    ishmemi_op_t op;
    ishmemi_type_t type;
    /* Sequence number and completion for request, This must be last for PCIe ordering */
    uint16_t sequence;
    uint16_t completion;
} ishmemi_request_t;

typedef struct {
    uint64_t padding[6];
    ishmemi_union_type ret;
    int lock;               // 0 if free
    unsigned int sequence;  // set by CPU
} ishmemi_completion_t;

typedef union {
    ulong8 data;
    ishmemi_completion_t completion;
} ishmemi_ringcompletion_t;

typedef enum {
    READY,
    PROCESSING,
    REQUEST,
    EXIT
} ishmemi_proxy_state_t;

#endif  //* ISHMEM_PROXY_TYPES_H */
