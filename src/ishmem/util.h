/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portions derived from Sandia OpenSHMEM (https://github.com/Sandia-OpenSHMEM/SOS)
 * For license and copyright information, see the LICENSE and third-party-programs.txt
 * files in the top level directory of this distribution.
 */

/* internal API and globals */
#ifndef ISHMEM_UTIL_H
#define ISHMEM_UTIL_H

#include "env_utils.h"
#include "ishmem/types.h"

#include <iostream>
#include <cstdlib>
#include <CL/sycl.hpp>

#include "ishmem.h"
#include "ishmemx.h"

#ifdef __SYCL_DEVICE_ONLY__
#define ISHMEMI_LOCAL_PES global_info->local_pes
#define ISHMEMI_N_TEAMS   global_info->n_teams
#else
#define ISHMEMI_LOCAL_PES ishmemi_local_pes
#define ISHMEMI_N_TEAMS   ishmemi_mmap_gpu_info->n_teams
#endif

#define MAX_LOCAL_PES 64

extern int ishmemi_my_pe;
extern int ishmemi_n_pes;

typedef struct ishmemi_info_t ishmemi_info_t;

/* TODO should these be combined into ishmem_host_data_t? */
/* Device parameters for the device copy of the data */
extern void *ishmemi_heap_base;
extern size_t ishmemi_heap_length;
extern uintptr_t ishmemi_heap_last;
extern ishmemi_info_t *ishmemi_gpu_info;
/* this is the device global */
ISHMEM_DEVICE_ATTRIBUTES extern sycl::ext::oneapi::experimental::device_global<ishmemi_info_t *>
    global_info;

/* allocated size for info data structure (variable due to n_pes) */
extern size_t ishmemi_info_size;

/* Host parameters for the device data structures */
extern ishmemi_info_t *ishmemi_mmap_gpu_info;
extern void *ishmemi_mmap_heap_base;

/* Host globals to hold the host version of data */
extern uint8_t *ishmemi_local_pes;
extern void *ishmemi_ipc_buffers[MAX_LOCAL_PES + 1];

/* host global for host address of host memory copy of ipc_buffer_delta */
extern ptrdiff_t ishmemi_ipc_buffer_delta[MAX_LOCAL_PES + 1];
extern bool ishmemi_only_intra_node;

/* Used to reduce reliance on macros in function definitions */
#ifdef __SYCL_DEVICE_ONLY__
constexpr bool ishmemi_is_device = true;
#else
constexpr bool ishmemi_is_device = false;
#endif

/* In cleanup, free an object only if not null, then set it to null */
#define ISHMEMI_FREE(freefn, x)                                                                    \
    if ((x) != nullptr) {                                                                          \
        freefn(x);                                                                                 \
        x = nullptr;                                                                               \
    }

template <typename T>
inline int ishmemi_comparison(T val1, T val2, int cmp)
{
    switch (cmp) {
        case (ISHMEM_CMP_EQ):
            return (int) (val1 == val2);
        case (ISHMEM_CMP_NE):
            return (int) (val1 != val2);
        case (ISHMEM_CMP_GT):
            return (int) (val1 > val2);
        case (ISHMEM_CMP_GE):
            return (int) (val1 >= val2);
        case (ISHMEM_CMP_LT):
            return (int) (val1 < val2);
        case (ISHMEM_CMP_LE):
            return (int) (val1 <= val2);
        default:
            // TODO: Add global exit method callable from GPU
            ishmemx_print("invalid 'cmp' value provided.\n", ishmemx_print_msg_type_t::ERROR);
            return -1;
    }
}

static inline void ishmemi_bit_set(unsigned char *ptr, size_t size, size_t index)
{
    /* TODO: add non-persistent assert? */
    assert(size > 0 && (index < size * CHAR_BIT));

    size_t which_byte = index / CHAR_BIT;
    ptr[which_byte] |= (1 << (index % CHAR_BIT));

    return;
}

static inline void ishmemi_bit_clear(unsigned char *ptr, size_t size, size_t index)
{
    assert(size > 0 && (index < size * CHAR_BIT));

    size_t which_byte = index / CHAR_BIT;
    ptr[which_byte] &= ~(1 << (index % CHAR_BIT));

    return;
}

static inline char ishmemi_bit_fetch(unsigned char *ptr, size_t size, size_t index)
{
    assert(size > 0 && (index < size * CHAR_BIT));

    size_t which_byte = index / CHAR_BIT;
    return (ptr[which_byte] >> (index % CHAR_BIT)) & 1;
}

static inline size_t ishmemi_bit_1st_nonzero(const unsigned char *ptr, const size_t size)
{
    /* The following ignores endianess: */
    for (size_t i = 0; i < size; i++) {
        unsigned char bit_val = ptr[i];
        for (size_t j = 0; bit_val && j < CHAR_BIT; j++) {
            if (bit_val & 1) return i * CHAR_BIT + j;
            bit_val >>= 1;
        }
    }

    return static_cast<size_t>(-1);
}

/* Create a bit string of the format AAAAAAAA.BBBBBBBB into str for the byte
 * array passed via ptr. */
static inline void ishmemi_bit_to_string(char *str, size_t str_size, unsigned char *ptr,
                                         size_t ptr_size)
{
    size_t off = 0;

    for (size_t i = 0; i < ptr_size; i++) {
        for (size_t j = 0; j < CHAR_BIT; j++) {
            off += static_cast<size_t>(snprintf(str + off, str_size - off, "%s",
                                                (ptr[i] & (1 << (CHAR_BIT - 1 - j))) ? "1" : "0"));
            if (off >= str_size) return;
        }
        if (i < ptr_size - 1) {
            off += static_cast<size_t>(snprintf(str + off, str_size - off, "."));
            if (off >= str_size) return;
        }
    }
}

#endif /* ISHMEM_UTIL_H */
