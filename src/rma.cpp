/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "internal.h"
#include "impl_proxy.h"
#include "runtime.h"
#include "runtime_ipc.h"
#include "rma_impl.h"

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
        if ((local_index != 0) && !ISHMEM_RMA_CUTOVER) {
            stride_copy(ISHMEMI_ADJUST_PTR(T, local_index, dest), src, dst, sst, nelems);
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req = {
        .op = IPUT,
        .type = UINT8,
        .dest_pe = pe,
        .src = src,
        .dst = dest,
        .nelems = nbytes,
        .dst_stride = dst,
        .src_stride = sst,
    };

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(&req);
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
        if ((local_index != 0) && !ISHMEM_RMA_GROUP_CUTOVER) {
            stride_copy_work_group(ISHMEMI_ADJUST_PTR(T, local_index, dest), src, dst, sst, nelems,
                                   grp);
        } else {
            if (grp.leader()) {
                ishmemi_request_t req = {
                    .op = IPUT,
                    .type = UINT8,
                    .dest_pe = pe,
                    .src = src,
                    .dst = dest,
                    .nelems = nbytes,
                    .dst_stride = dst,
                    .src_stride = sst,
                };

                ishmemi_proxy_blocking_request(&req);
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
    ishmemi_request_t req = {
        .op = P,
        .type = ishmemi_proxy_get_base_type<T>(),
        .dest_pe = pe,
        .dst = dest,
    };

    ishmemi_proxy_set_field_value<T, true, true>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(&req);
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
        if ((local_index != 0) && !ISHMEM_RMA_CUTOVER) {
            stride_copy(dest, ISHMEMI_ADJUST_PTR(T, local_index, src), dst, sst, nelems);
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req = {
        .op = IGET,
        .type = UINT8,
        .dest_pe = pe,
        .src = src,
        .dst = dest,
        .nelems = nbytes,
        .dst_stride = dst,
        .src_stride = sst,
    };

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(&req);
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
        if ((local_index != 0) && !ISHMEM_RMA_GROUP_CUTOVER) {
            stride_copy_work_group(dest, ISHMEMI_ADJUST_PTR(T, local_index, src), dst, sst, nelems,
                                   grp);
        } else {
            if (grp.leader()) {
                ishmemi_request_t req = {
                    .op = IGET,
                    .type = UINT8,
                    .dest_pe = pe,
                    .src = src,
                    .dst = dest,
                    .nelems = nbytes,
                    .dst_stride = dst,
                    .src_stride = sst,
                };

                ishmemi_proxy_blocking_request(&req);
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
    ishmemi_request_t req = {
        .op = G,
        .type = ishmemi_proxy_get_base_type<T>(),
        .dest_pe = pe,
        .src = src,
    };

#if __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_return<T>(&req);
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
