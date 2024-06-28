/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/err.h"
#include "proxy_impl.h"
#include "runtime.h"
#include "runtime_ipc.h"
#include "rma_impl.h"
#include "memory.h"

/* Put */
template <typename T>
void ishmem_put(T *dest, const T *src, size_t nelems, int pe)
{
    ishmem_internal_put(dest, src, nelems, pe);
}

void ishmem_putmem(void *dest, const void *src, size_t nelems, int pe)
{
    ishmem_internal_put((uint8_t *) dest, (uint8_t *) src, nelems, pe);
}

/* Put (work-group) */
template <typename T, typename Group>
void ishmemx_put_work_group(T *dest, const T *src, size_t nelems, int pe, const Group &grp)
{
    ishmemx_internal_put_work_group(dest, src, nelems, pe, grp);
}

/* clang-format off */
template void ishmemx_putmem_work_group<sycl::group<1>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<1> &grp);
template void ishmemx_putmem_work_group<sycl::group<2>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<2> &grp);
template void ishmemx_putmem_work_group<sycl::group<3>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<3> &grp);
template void ishmemx_putmem_work_group<sycl::sub_group>(void *dest, const void *src, size_t nelems, int pe, const sycl::sub_group &grp);
/* clang-format on */

template <typename Group>
void ishmemx_putmem_work_group(void *dest, const void *src, size_t nelems, int pe, const Group &grp)
{
    ishmemx_internal_put_work_group((uint8_t *) dest, (uint8_t *) src, nelems, pe, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_PUT(TYPENAME, TYPE)                                                                                                             \
    void ishmem_##TYPENAME##_put(TYPE *dest, const TYPE *src, size_t nelems, int pe) { ishmem_put(dest, src, nelems, pe); }                              \
    template void ishmemx_##TYPENAME##_put_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::group<1> &grp);    \
    template void ishmemx_##TYPENAME##_put_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::group<2> &grp);    \
    template void ishmemx_##TYPENAME##_put_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::group<3> &grp);    \
    template void ishmemx_##TYPENAME##_put_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::sub_group &grp);  \
    template <typename Group> void ishmemx_##TYPENAME##_put_work_group(TYPE *dest, const TYPE *src, size_t nelems, int pe, const Group &grp) { ishmemx_put_work_group(dest, src, nelems, pe, grp); }

#define ISHMEMI_API_IMPL_PUTSIZE(SIZE, ELEMSIZE)                                                                                                                                        \
    void ishmem_put##SIZE(void *dest, const void *src, size_t nelems, int pe) { ishmem_put((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, nelems * (SIZE / ELEMSIZE), pe); }  \
    template void ishmemx_put##SIZE##_work_group<sycl::group<1>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<1> &grp);                                        \
    template void ishmemx_put##SIZE##_work_group<sycl::group<2>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<2> &grp);                                        \
    template void ishmemx_put##SIZE##_work_group<sycl::group<3>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<3> &grp);                                        \
    template void ishmemx_put##SIZE##_work_group<sycl::sub_group>(void *dest, const void *src, size_t nelems, int pe, const sycl::sub_group &grp);                                      \
    template <typename Group> void ishmemx_put##SIZE##_work_group(void *dest, const void *src, size_t nelems, int pe, const Group &grp) { ishmemx_put_work_group((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, nelems * (SIZE / ELEMSIZE), pe, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_PUT(float, float)
ISHMEMI_API_IMPL_PUT(double, double)
ISHMEMI_API_IMPL_PUT(char, char)
ISHMEMI_API_IMPL_PUT(schar, signed char)
ISHMEMI_API_IMPL_PUT(short, short)
ISHMEMI_API_IMPL_PUT(int, int)
ISHMEMI_API_IMPL_PUT(long, long)
ISHMEMI_API_IMPL_PUT(longlong, long long)
ISHMEMI_API_IMPL_PUT(uchar, unsigned char)
ISHMEMI_API_IMPL_PUT(ushort, unsigned short)
ISHMEMI_API_IMPL_PUT(uint, unsigned int)
ISHMEMI_API_IMPL_PUT(ulong, unsigned long)
ISHMEMI_API_IMPL_PUT(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_PUT(int8, int8_t)
ISHMEMI_API_IMPL_PUT(int16, int16_t)
ISHMEMI_API_IMPL_PUT(int32, int32_t)
ISHMEMI_API_IMPL_PUT(int64, int64_t)
ISHMEMI_API_IMPL_PUT(uint8, uint8_t)
ISHMEMI_API_IMPL_PUT(uint16, uint16_t)
ISHMEMI_API_IMPL_PUT(uint32, uint32_t)
ISHMEMI_API_IMPL_PUT(uint64, uint64_t)
ISHMEMI_API_IMPL_PUT(size, size_t)
ISHMEMI_API_IMPL_PUT(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_PUTSIZE(8, 8)
ISHMEMI_API_IMPL_PUTSIZE(16, 16)
ISHMEMI_API_IMPL_PUTSIZE(32, 32)
ISHMEMI_API_IMPL_PUTSIZE(64, 64)
ISHMEMI_API_IMPL_PUTSIZE(128, 64)

/* IPut */
template <typename T>
void ishmem_iput(T *dest, const T *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)
{
    size_t nbytes = nelems * sizeof(T);

    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, (void *) src, nbytes, dst, sst);
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        if ((local_index != 0) && !ISHMEM_STRIDED_RMA_CUTOVER) {
            stride_copy(ISHMEMI_ADJUST_PTR(T, local_index, dest), src, dst, sst, nelems);
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.src = src;
    req.dst = dest;
    req.nelems = nelems;
    req.dst_stride = dst;
    req.src_stride = sst;
    req.op = IPUT;
    req.type = ishmemi_proxy_get_base_type<T>();

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* IPut (work-group) */
template <typename T, typename Group>
void ishmemx_iput_work_group(T *dest, const T *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems,
                             int pe, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        size_t nbytes = nelems * sizeof(T);
        sycl::group_barrier(grp);
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters(pe, (void *) dest, (void *) src, nbytes, dst, sst);
        }
        uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
        if ((local_index != 0) && !ISHMEM_STRIDED_RMA_GROUP_CUTOVER) {
            stride_copy_work_group(ISHMEMI_ADJUST_PTR(T, local_index, dest), src, dst, sst, nelems,
                                   grp);
        } else {
            if (grp.leader()) {
                ishmemi_request_t req;
                req.dest_pe = pe;
                req.src = src;
                req.dst = dest;
                req.nelems = nelems;
                req.dst_stride = dst;
                req.src_stride = sst;
                req.op = IPUT;
                req.type = ishmemi_proxy_get_base_type<T>();

                ishmemi_proxy_blocking_request(req);
            }
        }
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_iput_work_group not callable from host\n");
    }
}

/* clang-format off */
#define ISHMEMI_API_IMPL_IPUT(TYPENAME, TYPE)                                                                                                                                           \
    void ishmem_##TYPENAME##_iput(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { ishmem_iput(dest, src, dst, sst, nelems, pe); }                   \
    template void ishmemx_##TYPENAME##_iput_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::group<1> &grp);    \
    template void ishmemx_##TYPENAME##_iput_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::group<2> &grp);    \
    template void ishmemx_##TYPENAME##_iput_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::group<3> &grp);    \
    template void ishmemx_##TYPENAME##_iput_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::sub_group &grp);  \
    template <typename Group> void ishmemx_##TYPENAME##_iput_work_group(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const Group &grp) { ishmemx_iput_work_group(dest, src, dst, sst, nelems, pe, grp); }

#define ISHMEMI_API_IMPL_IPUTSIZE(SIZE, ELEMSIZE)                                                                                                                                                                                 \
    void ishmem_iput##SIZE(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { ishmem_iput((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, dst, sst, nelems * (SIZE / ELEMSIZE), pe); }  \
    template void ishmemx_iput##SIZE##_work_group<sycl::group<1>>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::group<1> &grp);                                                   \
    template void ishmemx_iput##SIZE##_work_group<sycl::group<2>>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::group<2> &grp);                                                   \
    template void ishmemx_iput##SIZE##_work_group<sycl::group<3>>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::group<3> &grp);                                                   \
    template void ishmemx_iput##SIZE##_work_group<sycl::sub_group>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::sub_group &grp);                                                 \
    template <typename Group> void ishmemx_iput##SIZE##_work_group(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const Group &grp) { ishmemx_iput_work_group((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, dst, sst, nelems * (SIZE / ELEMSIZE), pe, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_IPUT(float, float)
ISHMEMI_API_IMPL_IPUT(double, double)
ISHMEMI_API_IMPL_IPUT(char, char)
ISHMEMI_API_IMPL_IPUT(schar, signed char)
ISHMEMI_API_IMPL_IPUT(short, short)
ISHMEMI_API_IMPL_IPUT(int, int)
ISHMEMI_API_IMPL_IPUT(long, long)
ISHMEMI_API_IMPL_IPUT(longlong, long long)
ISHMEMI_API_IMPL_IPUT(uchar, unsigned char)
ISHMEMI_API_IMPL_IPUT(ushort, unsigned short)
ISHMEMI_API_IMPL_IPUT(uint, unsigned int)
ISHMEMI_API_IMPL_IPUT(ulong, unsigned long)
ISHMEMI_API_IMPL_IPUT(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_IPUT(int8, int8_t)
ISHMEMI_API_IMPL_IPUT(int16, int16_t)
ISHMEMI_API_IMPL_IPUT(int32, int32_t)
ISHMEMI_API_IMPL_IPUT(int64, int64_t)
ISHMEMI_API_IMPL_IPUT(uint8, uint8_t)
ISHMEMI_API_IMPL_IPUT(uint16, uint16_t)
ISHMEMI_API_IMPL_IPUT(uint32, uint32_t)
ISHMEMI_API_IMPL_IPUT(uint64, uint64_t)
ISHMEMI_API_IMPL_IPUT(size, size_t)
ISHMEMI_API_IMPL_IPUT(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_IPUTSIZE(8, 8)
ISHMEMI_API_IMPL_IPUTSIZE(16, 16)
ISHMEMI_API_IMPL_IPUTSIZE(32, 32)
ISHMEMI_API_IMPL_IPUTSIZE(64, 64)
ISHMEMI_API_IMPL_IPUTSIZE(128, 64)

/* IBPut */
template <typename T>
void ishmemx_ibput(T *dest, const T *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize,
                   size_t nblocks, int pe)
{
    size_t nbytes = nblocks * bsize * sizeof(T);

    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, (void *) src, nbytes, dst, sst, bsize);
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        if ((local_index != 0) && !ISHMEM_STRIDED_RMA_CUTOVER) {
            stride_bcopy_push(ISHMEMI_ADJUST_PTR(T, local_index, dest), src, dst, sst, bsize,
                              nblocks);
            return;
        }
    }

    ishmemi_request_t req;
    req.dest_pe = pe;
    req.src = src;
    req.dst = dest;
    req.nelems = nblocks;
    req.dst_stride = dst;
    req.src_stride = sst;
    req.bsize = bsize;
    req.op = IBPUT;
    req.type = ishmemi_proxy_get_base_type<T>();

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* IBPut (work-group) */
template <typename T, typename Group>
void ishmemx_ibput_work_group(T *dest, const T *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize,
                              size_t nblocks, int pe, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        size_t nbytes = nblocks * bsize * sizeof(T);
        sycl::group_barrier(grp);
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters(pe, (void *) dest, (void *) src, nbytes, dst, sst, bsize);
        }
        uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
        if ((local_index != 0) && !ISHMEM_STRIDED_RMA_GROUP_CUTOVER) {
            stride_bcopy_work_group_push(ISHMEMI_ADJUST_PTR(T, local_index, dest), src, dst, sst,
                                         bsize, nblocks, grp);
        } else {
            if (grp.leader()) {
                ishmemi_request_t req;
                req.dest_pe = pe;
                req.src = src;
                req.dst = dest;
                req.nelems = nblocks;
                req.dst_stride = dst;
                req.src_stride = sst;
                req.bsize = bsize;
                req.op = IBPUT;
                req.type = ishmemi_proxy_get_base_type<T>();

                ishmemi_proxy_blocking_request(req);
            }
        }
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_ibput_work_group not callable from host\n");
    }
}

/* clang-format off */
#define ISHMEMI_API_IMPL_IBPUT(TYPENAME, TYPE)                                                                                                                                                          \
    void ishmemx_##TYPENAME##_ibput(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe) { ishmemx_ibput(dest, src, dst, sst, bsize, nblocks, pe); }        \
    template void ishmemx_##TYPENAME##_ibput_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::group<1> &grp);    \
    template void ishmemx_##TYPENAME##_ibput_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::group<2> &grp);    \
    template void ishmemx_##TYPENAME##_ibput_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::group<3> &grp);    \
    template void ishmemx_##TYPENAME##_ibput_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::sub_group &grp);  \
    template <typename Group> void ishmemx_##TYPENAME##_ibput_work_group(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const Group &grp) { ishmemx_ibput_work_group(dest, src, dst, sst, bsize, nblocks, pe, grp); }

#define ISHMEMI_API_IMPL_IBPUTSIZE(SIZE, ELEMSIZE)                                                                                                                                                                                                                                                   \
    void ishmemx_ibput##SIZE(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe) { ishmemx_ibput((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, dst * (SIZE / ELEMSIZE), sst * (SIZE / ELEMSIZE), bsize * (SIZE / ELEMSIZE), nblocks, pe); }  \
    template void ishmemx_ibput##SIZE##_work_group<sycl::group<1>>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::group<1> &grp);                                                                                                      \
    template void ishmemx_ibput##SIZE##_work_group<sycl::group<2>>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::group<2> &grp);                                                                                                      \
    template void ishmemx_ibput##SIZE##_work_group<sycl::group<3>>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::group<3> &grp);                                                                                                      \
    template void ishmemx_ibput##SIZE##_work_group<sycl::sub_group>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::sub_group &grp);                                                                                                    \
    template <typename Group> void ishmemx_ibput##SIZE##_work_group(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const Group &grp) { ishmemx_ibput_work_group((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, dst * (SIZE / ELEMSIZE), sst * (SIZE / ELEMSIZE), bsize * (SIZE / ELEMSIZE), nblocks, pe, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_IBPUT(float, float)
ISHMEMI_API_IMPL_IBPUT(double, double)
ISHMEMI_API_IMPL_IBPUT(char, char)
ISHMEMI_API_IMPL_IBPUT(schar, signed char)
ISHMEMI_API_IMPL_IBPUT(short, short)
ISHMEMI_API_IMPL_IBPUT(int, int)
ISHMEMI_API_IMPL_IBPUT(long, long)
ISHMEMI_API_IMPL_IBPUT(longlong, long long)
ISHMEMI_API_IMPL_IBPUT(uchar, unsigned char)
ISHMEMI_API_IMPL_IBPUT(ushort, unsigned short)
ISHMEMI_API_IMPL_IBPUT(uint, unsigned int)
ISHMEMI_API_IMPL_IBPUT(ulong, unsigned long)
ISHMEMI_API_IMPL_IBPUT(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_IBPUT(int8, int8_t)
ISHMEMI_API_IMPL_IBPUT(int16, int16_t)
ISHMEMI_API_IMPL_IBPUT(int32, int32_t)
ISHMEMI_API_IMPL_IBPUT(int64, int64_t)
ISHMEMI_API_IMPL_IBPUT(uint8, uint8_t)
ISHMEMI_API_IMPL_IBPUT(uint16, uint16_t)
ISHMEMI_API_IMPL_IBPUT(uint32, uint32_t)
ISHMEMI_API_IMPL_IBPUT(uint64, uint64_t)
ISHMEMI_API_IMPL_IBPUT(size, size_t)
ISHMEMI_API_IMPL_IBPUT(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_IBPUTSIZE(8, 8)
ISHMEMI_API_IMPL_IBPUTSIZE(16, 16)
ISHMEMI_API_IMPL_IBPUTSIZE(32, 32)
ISHMEMI_API_IMPL_IBPUTSIZE(64, 64)
ISHMEMI_API_IMPL_IBPUTSIZE(128, 64)

/* P */
template <typename T>
void ishmem_p(T *dest, T val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, sizeof(T));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        if (local_index != 0) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, dest);
            *p = val;
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = dest;
    req.op = P;
    req.type = ishmemi_proxy_get_base_type<T>();

    ishmemi_proxy_set_field_value<T, true, true>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    int ret = 1;
    if (local_index != 0) ret = ishmemi_ipc_put(dest, &val, 1, pe);
    if (ret != 0) ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* clang-format off */
#define ISHMEMI_API_IMPL_P(TYPENAME, TYPE)  \
    void ishmem_##TYPENAME##_p(TYPE *dest, TYPE val, int pe) { ishmem_p(dest, val, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_P(float, float)
ISHMEMI_API_IMPL_P(double, double)
ISHMEMI_API_IMPL_P(char, char)
ISHMEMI_API_IMPL_P(schar, signed char)
ISHMEMI_API_IMPL_P(short, short)
ISHMEMI_API_IMPL_P(int, int)
ISHMEMI_API_IMPL_P(long, long)
ISHMEMI_API_IMPL_P(longlong, long long)
ISHMEMI_API_IMPL_P(uchar, unsigned char)
ISHMEMI_API_IMPL_P(ushort, unsigned short)
ISHMEMI_API_IMPL_P(uint, unsigned int)
ISHMEMI_API_IMPL_P(ulong, unsigned long)
ISHMEMI_API_IMPL_P(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_P(int8, int8_t)
ISHMEMI_API_IMPL_P(int16, int16_t)
ISHMEMI_API_IMPL_P(int32, int32_t)
ISHMEMI_API_IMPL_P(int64, int64_t)
ISHMEMI_API_IMPL_P(uint8, uint8_t)
ISHMEMI_API_IMPL_P(uint16, uint16_t)
ISHMEMI_API_IMPL_P(uint32, uint32_t)
ISHMEMI_API_IMPL_P(uint64, uint64_t)
ISHMEMI_API_IMPL_P(size, size_t)
ISHMEMI_API_IMPL_P(ptrdiff, ptrdiff_t)

/* Get */
template <typename T>
void ishmem_get(T *dest, const T *src, size_t nelems, int pe)
{
    ishmem_internal_get(dest, src, nelems, pe);
}

void ishmem_getmem(void *dest, const void *src, size_t nelems, int pe)
{
    ishmem_internal_get((uint8_t *) dest, (uint8_t *) src, nelems, pe);
}

/* Get (work-group) */
template <typename T, typename Group>
void ishmemx_get_work_group(T *dest, const T *src, size_t nelems, int pe, const Group &grp)
{
    ishmemx_internal_get_work_group(dest, src, nelems, pe, grp);
}

/* clang-format off */
template void ishmemx_getmem_work_group<sycl::group<1>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<1> &grp);
template void ishmemx_getmem_work_group<sycl::group<2>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<2> &grp);
template void ishmemx_getmem_work_group<sycl::group<3>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<3> &grp);
template void ishmemx_getmem_work_group<sycl::sub_group>(void *dest, const void *src, size_t nelems, int pe, const sycl::sub_group &grp);
/* clang-format on */

template <typename Group>
void ishmemx_getmem_work_group(void *dest, const void *src, size_t nelems, int pe, const Group &grp)
{
    ishmemx_internal_get_work_group((uint8_t *) dest, (uint8_t *) src, nelems, pe, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_GET(TYPENAME, TYPE)                                                                                                             \
    void ishmem_##TYPENAME##_get(TYPE *dest, const TYPE *src, size_t nelems, int pe) { ishmem_get(dest, src, nelems, pe); }                              \
    template void ishmemx_##TYPENAME##_get_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::group<1> &grp);    \
    template void ishmemx_##TYPENAME##_get_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::group<2> &grp);    \
    template void ishmemx_##TYPENAME##_get_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::group<3> &grp);    \
    template void ishmemx_##TYPENAME##_get_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::sub_group &grp);  \
    template <typename Group> void ishmemx_##TYPENAME##_get_work_group(TYPE *dest, const TYPE *src, size_t nelems, int pe, const Group &grp) { ishmemx_get_work_group(dest, src, nelems, pe, grp); }

#define ISHMEMI_API_IMPL_GETSIZE(SIZE, ELEMSIZE)                                                                                                                                    \
    void ishmem_get##SIZE(void *dest, const void *src, size_t nelems, int pe) { ishmem_get((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, nelems * (SIZE / ELEMSIZE), pe); }  \
    template void ishmemx_get##SIZE##_work_group<sycl::group<1>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<1> &grp);                                    \
    template void ishmemx_get##SIZE##_work_group<sycl::group<2>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<2> &grp);                                    \
    template void ishmemx_get##SIZE##_work_group<sycl::group<3>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<3> &grp);                                    \
    template void ishmemx_get##SIZE##_work_group<sycl::sub_group>(void *dest, const void *src, size_t nelems, int pe, const sycl::sub_group &grp);                                  \
    template <typename Group> void ishmemx_get##SIZE##_work_group(void *dest, const void *src, size_t nelems, int pe, const Group &grp) { ishmemx_get_work_group((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, nelems * (SIZE / ELEMSIZE), pe, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_GET(float, float)
ISHMEMI_API_IMPL_GET(double, double)
ISHMEMI_API_IMPL_GET(char, char)
ISHMEMI_API_IMPL_GET(schar, signed char)
ISHMEMI_API_IMPL_GET(short, short)
ISHMEMI_API_IMPL_GET(int, int)
ISHMEMI_API_IMPL_GET(long, long)
ISHMEMI_API_IMPL_GET(longlong, long long)
ISHMEMI_API_IMPL_GET(uchar, unsigned char)
ISHMEMI_API_IMPL_GET(ushort, unsigned short)
ISHMEMI_API_IMPL_GET(uint, unsigned int)
ISHMEMI_API_IMPL_GET(ulong, unsigned long)
ISHMEMI_API_IMPL_GET(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_GET(int8, int8_t)
ISHMEMI_API_IMPL_GET(int16, int16_t)
ISHMEMI_API_IMPL_GET(int32, int32_t)
ISHMEMI_API_IMPL_GET(int64, int64_t)
ISHMEMI_API_IMPL_GET(uint8, uint8_t)
ISHMEMI_API_IMPL_GET(uint16, uint16_t)
ISHMEMI_API_IMPL_GET(uint32, uint32_t)
ISHMEMI_API_IMPL_GET(uint64, uint64_t)
ISHMEMI_API_IMPL_GET(size, size_t)
ISHMEMI_API_IMPL_GET(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_GETSIZE(8, 8)
ISHMEMI_API_IMPL_GETSIZE(16, 16)
ISHMEMI_API_IMPL_GETSIZE(32, 32)
ISHMEMI_API_IMPL_GETSIZE(64, 64)
ISHMEMI_API_IMPL_GETSIZE(128, 64)

/* IGet */
template <typename T>
void ishmem_iget(T *dest, const T *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)
{
    size_t nbytes = nelems * sizeof(T);

    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) src, (void *) dest, nbytes, dst, sst);
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        if ((local_index != 0) && !ISHMEM_STRIDED_RMA_CUTOVER) {
            stride_copy(dest, ISHMEMI_ADJUST_PTR(T, local_index, src), dst, sst, nelems);
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.src = src;
    req.dst = dest;
    req.nelems = nelems;
    req.dst_stride = dst;
    req.src_stride = sst;
    req.op = IGET;
    req.type = ishmemi_proxy_get_base_type<T>();

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* IGet (work-group) */
template <typename T, typename Group>
void ishmemx_iget_work_group(T *dest, const T *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems,
                             int pe, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        size_t nbytes = nelems * sizeof(T);
        sycl::group_barrier(grp);
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters(pe, (void *) src, (void *) dest, nbytes, dst, sst);
        }
        uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
        if ((local_index != 0) && !ISHMEM_STRIDED_RMA_GROUP_CUTOVER) {
            stride_copy_work_group(dest, ISHMEMI_ADJUST_PTR(T, local_index, src), dst, sst, nelems,
                                   grp);
        } else {
            if (grp.leader()) {
                ishmemi_request_t req;
                req.dest_pe = pe;
                req.src = src;
                req.dst = dest;
                req.nelems = nelems;
                req.dst_stride = dst;
                req.src_stride = sst;
                req.op = IGET;
                req.type = ishmemi_proxy_get_base_type<T>();

                ishmemi_proxy_blocking_request(req);
            }
        }
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_iget_work_group not callable from host\n");
    }
}

/* clang-format off */
#define ISHMEMI_API_IMPL_IGET(TYPENAME, TYPE)                                                                                                                                           \
    void ishmem_##TYPENAME##_iget(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { ishmem_iget(dest, src, dst, sst, nelems, pe); }                   \
    template void ishmemx_##TYPENAME##_iget_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::group<1> &grp);    \
    template void ishmemx_##TYPENAME##_iget_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::group<2> &grp);    \
    template void ishmemx_##TYPENAME##_iget_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::group<3> &grp);    \
    template void ishmemx_##TYPENAME##_iget_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::sub_group &grp);  \
    template <typename Group> void ishmemx_##TYPENAME##_iget_work_group(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const Group &grp) { ishmemx_iget_work_group(dest, src, dst, sst, nelems, pe, grp); }

#define ISHMEMI_API_IMPL_IGETSIZE(SIZE, ELEMSIZE)                                                                                                                                                                                 \
    void ishmem_iget##SIZE(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { ishmem_iget((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, dst, sst, nelems * (SIZE / ELEMSIZE), pe); }  \
    template void ishmemx_iget##SIZE##_work_group<sycl::group<1>>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::group<1> &grp);                                                   \
    template void ishmemx_iget##SIZE##_work_group<sycl::group<2>>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::group<2> &grp);                                                   \
    template void ishmemx_iget##SIZE##_work_group<sycl::group<3>>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::group<3> &grp);                                                   \
    template void ishmemx_iget##SIZE##_work_group<sycl::sub_group>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const sycl::sub_group &grp);                                                 \
    template <typename Group> void ishmemx_iget##SIZE##_work_group(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const Group &grp) { ishmemx_iget_work_group((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, dst, sst, nelems * (SIZE / ELEMSIZE), pe, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_IGET(float, float)
ISHMEMI_API_IMPL_IGET(double, double)
ISHMEMI_API_IMPL_IGET(char, char)
ISHMEMI_API_IMPL_IGET(schar, signed char)
ISHMEMI_API_IMPL_IGET(short, short)
ISHMEMI_API_IMPL_IGET(int, int)
ISHMEMI_API_IMPL_IGET(long, long)
ISHMEMI_API_IMPL_IGET(longlong, long long)
ISHMEMI_API_IMPL_IGET(uchar, unsigned char)
ISHMEMI_API_IMPL_IGET(ushort, unsigned short)
ISHMEMI_API_IMPL_IGET(uint, unsigned int)
ISHMEMI_API_IMPL_IGET(ulong, unsigned long)
ISHMEMI_API_IMPL_IGET(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_IGET(int8, int8_t)
ISHMEMI_API_IMPL_IGET(int16, int16_t)
ISHMEMI_API_IMPL_IGET(int32, int32_t)
ISHMEMI_API_IMPL_IGET(int64, int64_t)
ISHMEMI_API_IMPL_IGET(uint8, uint8_t)
ISHMEMI_API_IMPL_IGET(uint16, uint16_t)
ISHMEMI_API_IMPL_IGET(uint32, uint32_t)
ISHMEMI_API_IMPL_IGET(uint64, uint64_t)
ISHMEMI_API_IMPL_IGET(size, size_t)
ISHMEMI_API_IMPL_IGET(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_IGETSIZE(8, 8)
ISHMEMI_API_IMPL_IGETSIZE(16, 16)
ISHMEMI_API_IMPL_IGETSIZE(32, 32)
ISHMEMI_API_IMPL_IGETSIZE(64, 64)
ISHMEMI_API_IMPL_IGETSIZE(128, 64)

/* IBGet */
template <typename T>
void ishmemx_ibget(T *dest, const T *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize,
                   size_t nblocks, int pe)
{
    size_t nbytes = nblocks * bsize * sizeof(T);

    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) src, (void *) dest, nbytes, dst, sst, bsize);
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        if ((local_index != 0) && !ISHMEM_STRIDED_RMA_CUTOVER) {
            stride_bcopy_pull(dest, ISHMEMI_ADJUST_PTR(T, local_index, src), dst, sst, bsize,
                              nblocks);
            return;
        }
    }

    ishmemi_request_t req;
    req.dest_pe = pe;
    req.src = src;
    req.dst = dest;
    req.nelems = nblocks;
    req.dst_stride = dst;
    req.src_stride = sst;
    req.bsize = bsize;
    req.op = IBGET;
    req.type = ishmemi_proxy_get_base_type<T>();

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* IBGet (work-group) */
template <typename T, typename Group>
void ishmemx_ibget_work_group(T *dest, const T *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize,
                              size_t nblocks, int pe, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        size_t nbytes = nblocks * bsize * sizeof(T);
        sycl::group_barrier(grp);
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters(pe, (void *) src, (void *) dest, nbytes, dst, sst, bsize);
        }
        uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
        if ((local_index != 0) && !ISHMEM_STRIDED_RMA_GROUP_CUTOVER) {
            stride_bcopy_work_group_pull(dest, ISHMEMI_ADJUST_PTR(T, local_index, src), dst, sst,
                                         bsize, nblocks, grp);
        } else {
            if (grp.leader()) {
                ishmemi_request_t req;
                req.dest_pe = pe;
                req.src = src;
                req.dst = dest;
                req.nelems = nblocks;
                req.dst_stride = dst;
                req.src_stride = sst;
                req.bsize = bsize;
                req.op = IBGET;
                req.type = ishmemi_proxy_get_base_type<T>();

                ishmemi_proxy_blocking_request(req);
            }
        }
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_ibget_work_group not callable from host\n");
    }
}

/* clang-format off */
#define ISHMEMI_API_IMPL_IBGET(TYPENAME, TYPE)                                                                                                                                                          \
    void ishmemx_##TYPENAME##_ibget(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe) { ishmemx_ibget(dest, src, dst, sst, bsize, nblocks, pe); }        \
    template void ishmemx_##TYPENAME##_ibget_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::group<1> &grp);    \
    template void ishmemx_##TYPENAME##_ibget_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::group<2> &grp);    \
    template void ishmemx_##TYPENAME##_ibget_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::group<3> &grp);    \
    template void ishmemx_##TYPENAME##_ibget_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::sub_group &grp);  \
    template <typename Group> void ishmemx_##TYPENAME##_ibget_work_group(TYPE *dest, const TYPE *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const Group &grp) { ishmemx_ibget_work_group(dest, src, dst, sst, bsize, nblocks, pe, grp); }

#define ISHMEMI_API_IMPL_IBGETSIZE(SIZE, ELEMSIZE)                                                                                                                                                                                                                                                   \
    void ishmemx_ibget##SIZE(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe) { ishmemx_ibget((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, dst * (SIZE / ELEMSIZE), sst * (SIZE / ELEMSIZE), bsize * (SIZE / ELEMSIZE), nblocks, pe); }  \
    template void ishmemx_ibget##SIZE##_work_group<sycl::group<1>>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::group<1> &grp);                                                                                                      \
    template void ishmemx_ibget##SIZE##_work_group<sycl::group<2>>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::group<2> &grp);                                                                                                      \
    template void ishmemx_ibget##SIZE##_work_group<sycl::group<3>>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::group<3> &grp);                                                                                                      \
    template void ishmemx_ibget##SIZE##_work_group<sycl::sub_group>(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const sycl::sub_group &grp);                                                                                                    \
    template <typename Group> void ishmemx_ibget##SIZE##_work_group(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t bsize, size_t nblocks, int pe, const Group &grp) { ishmemx_ibget_work_group((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, dst * (SIZE / ELEMSIZE), sst * (SIZE / ELEMSIZE), bsize * (SIZE / ELEMSIZE), nblocks, pe, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_IBGET(float, float)
ISHMEMI_API_IMPL_IBGET(double, double)
ISHMEMI_API_IMPL_IBGET(char, char)
ISHMEMI_API_IMPL_IBGET(schar, signed char)
ISHMEMI_API_IMPL_IBGET(short, short)
ISHMEMI_API_IMPL_IBGET(int, int)
ISHMEMI_API_IMPL_IBGET(long, long)
ISHMEMI_API_IMPL_IBGET(longlong, long long)
ISHMEMI_API_IMPL_IBGET(uchar, unsigned char)
ISHMEMI_API_IMPL_IBGET(ushort, unsigned short)
ISHMEMI_API_IMPL_IBGET(uint, unsigned int)
ISHMEMI_API_IMPL_IBGET(ulong, unsigned long)
ISHMEMI_API_IMPL_IBGET(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_IBGET(int8, int8_t)
ISHMEMI_API_IMPL_IBGET(int16, int16_t)
ISHMEMI_API_IMPL_IBGET(int32, int32_t)
ISHMEMI_API_IMPL_IBGET(int64, int64_t)
ISHMEMI_API_IMPL_IBGET(uint8, uint8_t)
ISHMEMI_API_IMPL_IBGET(uint16, uint16_t)
ISHMEMI_API_IMPL_IBGET(uint32, uint32_t)
ISHMEMI_API_IMPL_IBGET(uint64, uint64_t)
ISHMEMI_API_IMPL_IBGET(size, size_t)
ISHMEMI_API_IMPL_IBGET(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_IBGETSIZE(8, 8)
ISHMEMI_API_IMPL_IBGETSIZE(16, 16)
ISHMEMI_API_IMPL_IBGETSIZE(32, 32)
ISHMEMI_API_IMPL_IBGETSIZE(64, 64)
ISHMEMI_API_IMPL_IBGETSIZE(128, 64)

/* G */
template <typename T>
T ishmem_g(T *src, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) src, sizeof(T));
    }

    T ret = static_cast<T>(0);
    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        if (local_index != 0) {
            T *p = ISHMEMI_ADJUST_PTR(T, local_index, src);
            ret = *p;
            return ret;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.src = src;
    req.op = G;
    req.type = ishmemi_proxy_get_base_type<T>();

#if __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_return<T>(req);
#else
    int retval = 1;
    if (local_index != 0) retval = ishmemi_ipc_get(src, &ret, 1, pe);
    if (retval != 0) {
        ishmemi_ringcompletion_t comp;
        ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
        ret = ishmemi_proxy_get_field_value<T, true, true>(comp.completion.ret);
    }
#endif
    return ret;
}

/* clang-format off */
#define ISHMEMI_API_IMPL_G(TYPENAME, TYPE)  \
    TYPE ishmem_##TYPENAME##_g(TYPE *dest, int pe) { return ishmem_g(dest, pe); }
/* clang-format on */

ISHMEMI_API_IMPL_G(float, float)
ISHMEMI_API_IMPL_G(double, double)
ISHMEMI_API_IMPL_G(char, char)
ISHMEMI_API_IMPL_G(schar, signed char)
ISHMEMI_API_IMPL_G(short, short)
ISHMEMI_API_IMPL_G(int, int)
ISHMEMI_API_IMPL_G(long, long)
ISHMEMI_API_IMPL_G(longlong, long long)
ISHMEMI_API_IMPL_G(uchar, unsigned char)
ISHMEMI_API_IMPL_G(ushort, unsigned short)
ISHMEMI_API_IMPL_G(uint, unsigned int)
ISHMEMI_API_IMPL_G(ulong, unsigned long)
ISHMEMI_API_IMPL_G(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_G(int8, int8_t)
ISHMEMI_API_IMPL_G(int16, int16_t)
ISHMEMI_API_IMPL_G(int32, int32_t)
ISHMEMI_API_IMPL_G(int64, int64_t)
ISHMEMI_API_IMPL_G(uint8, uint8_t)
ISHMEMI_API_IMPL_G(uint16, uint16_t)
ISHMEMI_API_IMPL_G(uint32, uint32_t)
ISHMEMI_API_IMPL_G(uint64, uint64_t)
ISHMEMI_API_IMPL_G(size, size_t)
ISHMEMI_API_IMPL_G(ptrdiff, ptrdiff_t)
