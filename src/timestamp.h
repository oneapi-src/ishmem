/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_TIMESTAMP_H
#define ISHMEM_TIMESTAMP_H

static inline unsigned long rdtsc()
{
    unsigned int hi, lo;

    __asm volatile(
        "xorl %%eax, %%eax\n"
        "cpuid            \n"
        "rdtsc            \n"
        : "=a"(lo), "=d"(hi)
        :
        : "%ebx", "%ecx");

    return ((unsigned long) hi << 32) | lo;
}

#endif
