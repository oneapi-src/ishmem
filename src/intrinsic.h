/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_INTRINSIC_H
#define ISHMEM_INTRINSIC_H

/* This block of code enables the Intel® Graphics Compiler
 * intrinsic instruction for an uncached store
 *
 * if USE_BUILTIN==1, then the intrinsic is used
 * if USE_BUILTIN==0, then the intrinsic is not used
 */

/* This enables use of the graphics compiler intrinsic for Send */
/* TODO figure out whether this is needed */
#define USE_BUILTIN 2

///////////////////////////////////////////////////////////////////////
// LSC Fence support
///////////////////////////////////////////////////////////////////////

// FS - Fence Scope
enum LSC_FS {
    LSC_FS_THREAD_GROUP,
    LSC_FS_LOCAL,
    LSC_FS_TILE,
    LSC_FS_GPU,
    LSC_FS_GPUs,
    LSC_FS_SYSTEM_RELEASE,
    LSC_FS_SYSTEM_ACQUIRE
};

// FT - Fence Type
enum LSC_FT {
    LSC_FT_DEFAULT,
    LSC_FT_EVICT,
    LSC_FT_INVALIDATE,
    LSC_FT_DISCARD,
    LSC_FT_CLEAN,
    LSC_FT_L3
};

SYCL_EXTERNAL extern "C" void __builtin_IB_lsc_fence_global_untyped(
    enum LSC_FS scope, enum LSC_FT flushType);  // Mem Port - UGM
SYCL_EXTERNAL extern "C" void __builtin_IB_lsc_fence_global_untyped_cross_tile(
    enum LSC_FS scope, enum LSC_FT flushType);  // Mem Port - UGML
SYCL_EXTERNAL extern "C" void __builtin_IB_lsc_fence_global_typed(
    enum LSC_FS scope, enum LSC_FT flushType);                           // Mem Port - TGM
SYCL_EXTERNAL extern "C" void __builtin_IB_lsc_fence_local();            // Mem Port - SLM
SYCL_EXTERNAL extern "C" void __builtin_IB_lsc_fence_evict_to_memory();  // Mem Port - UGM

enum LSC_STCC {
    LSC_STCC_DEFAULT = 0,
    LSC_STCC_L1UC_L3UC = 1,  // Override to L1 uncached and L3 uncached
    LSC_STCC_L1UC_L3WB = 2,  // Override to L1 uncached and L3 written back
    LSC_STCC_L1WT_L3UC = 3,  // Override to L1 written through and L3 uncached
    LSC_STCC_L1WT_L3WB = 4,  // Override to L1 written through and L3 written back
    LSC_STCC_L1S_L3UC = 5,   // Override to L1 streaming and L3 uncached
    LSC_STCC_L1S_L3WB = 6,   // Override to L1 streaming and L3 written back
    LSC_STCC_L1WB_L3WB = 7,  // Override to L1 written through and L3 written back
};

#ifdef __SYCL_DEVICE_ONLY__
#define __global
SYCL_EXTERNAL extern "C" void __builtin_IB_lsc_store_global_ulong8(
    __global ulong8 *base, int immElemOff, ulong8 val, enum LSC_STCC cacheOpt);  // D64V8

#endif  // end __SYCL_DEVICE_ONLY__

static inline void ucs_ulong8(ulong8 *base, ulong8 val)
{
#ifdef __SYCL_DEVICE_ONLY__
#if USE_BUILTIN == 1
    __builtin_IB_lsc_store_global_ulong8(base, 0, val, LSC_STCC_DEFAULT);
#elif USE_BUILTIN == 2  // end USE_BUILTIN start !USE_BUILTIN
    base[0][0] = val[0];
    base[0][1] = val[1];
    base[0][2] = val[2];
    base[0][3] = val[3];
    base[0][4] = val[4];
    base[0][5] = val[5];
    base[0][6] = val[6];
    __builtin_IB_lsc_fence_global_untyped(LSC_FS_SYSTEM_RELEASE, LSC_FT_EVICT);
    //    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    base[0][7] = val[7];
    __builtin_IB_lsc_fence_global_untyped(LSC_FS_SYSTEM_RELEASE, LSC_FT_EVICT);
#else
    *base = val;
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
#endif  // end !USE_BUILTIN
#else   // end __SYCL_DEVICE_ONLY__ start !__SYCL_DEVICE_ONLY__
    *base = val;
#endif  // end !__SYCL_DEVICE_ONLY__
}

#endif
