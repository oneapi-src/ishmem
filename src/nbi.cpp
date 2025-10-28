/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/err.h"
#include "proxy_impl.h"
#include "runtime.h"
#include "runtime_ipc.h"
#include "nbi_impl.h"
#include "on_queue.h"

/* Non-blocking Put */
template <typename T>
void ishmem_put_nbi(T *dest, const T *src, size_t nelems, int pe)
{
    ishmem_internal_put_nbi(dest, src, nelems, pe);
}

void ishmem_putmem_nbi(void *dest, const void *src, size_t nelems, int pe)
{
    ishmem_internal_put_nbi((uint8_t *) dest, (uint8_t *) src, nelems, pe);
}

template <typename T>
sycl::event ishmemx_put_nbi_on_queue(T *dest, const T *src, size_t nelems, int pe, sycl::queue &q,
                                     const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        cgh.single_task([=]() { ishmem_put_nbi(dest, src, nelems, pe); });
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

sycl::event ishmemx_putmem_nbi_on_queue(void *dest, const void *src, size_t nelems, int pe,
                                        sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return ishmemx_put_nbi_on_queue((uint8_t *) dest, (uint8_t *) src, nelems, pe, q, deps);
}

/* Non-blocking Put (work-group) */
template <typename T, typename Group>
void ishmemx_put_nbi_work_group(T *dest, const T *src, size_t nelems, int pe, const Group &grp)
{
    ishmemx_internal_put_nbi_work_group(dest, src, nelems, pe, grp);
}

/* clang-format off */
template void ishmemx_putmem_nbi_work_group<sycl::group<1>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<1> &grp);
template void ishmemx_putmem_nbi_work_group<sycl::group<2>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<2> &grp);
template void ishmemx_putmem_nbi_work_group<sycl::group<3>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<3> &grp);
template void ishmemx_putmem_nbi_work_group<sycl::sub_group>(void *dest, const void *src, size_t nelems, int pe, const sycl::sub_group &grp);
/* clang-format on */

template <typename Group>
void ishmemx_putmem_nbi_work_group(void *dest, const void *src, size_t nelems, int pe,
                                   const Group &grp)
{
    ishmemx_internal_put_nbi_work_group((uint8_t *) dest, (uint8_t *) src, nelems, pe, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_PUT_NBI(TYPENAME, TYPE)                                                                                                                  \
    ISHMEM_INSTANTIATE_TYPE_##TYPENAME(TYPE);                                                                                                                     \
    void ishmem_##TYPENAME##_put_nbi(TYPE *dest, const TYPE *src, size_t nelems, int pe) { ishmem_put_nbi(dest, src, nelems, pe); }                               \
    sycl::event ishmemx_##TYPENAME##_put_nbi_on_queue(TYPE *dest, const TYPE *src, size_t nelems, int pe, sycl::queue &q, const std::vector<sycl::event> &deps) { \
        return ishmemx_put_nbi_on_queue(dest, src, nelems, pe, q, deps);                                                                                          \
    }                                                                                                                                                             \
    template void ishmemx_##TYPENAME##_put_nbi_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::group<1> &grp);         \
    template void ishmemx_##TYPENAME##_put_nbi_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::group<2> &grp);         \
    template void ishmemx_##TYPENAME##_put_nbi_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::group<3> &grp);         \
    template void ishmemx_##TYPENAME##_put_nbi_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::sub_group &grp);       \
    template <typename Group> void ishmemx_##TYPENAME##_put_nbi_work_group(TYPE *dest, const TYPE *src, size_t nelems, int pe, const Group &grp) { ishmemx_put_nbi_work_group(dest, src, nelems, pe, grp); }

#define ISHMEMI_API_IMPL_PUTSIZE_NBI(SIZE, ELEMSIZE)                                                                                                                                             \
    void ishmem_put##SIZE##_nbi(void *dest, const void *src, size_t nelems, int pe) { ishmem_put_nbi((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, nelems * (SIZE / ELEMSIZE), pe); } \
    sycl::event ishmemx_put##SIZE##_nbi_on_queue(void *dest, const void *src, size_t nelems, int pe, sycl::queue &q, const std::vector<sycl::event> &deps) {                                     \
        return ishmemx_put_nbi_on_queue((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, nelems * (SIZE / ELEMSIZE), pe, q, deps);                                                       \
    }                                                                                                                                                                                            \
    template void ishmemx_put##SIZE##_nbi_work_group<sycl::group<1>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<1> &grp);                                             \
    template void ishmemx_put##SIZE##_nbi_work_group<sycl::group<2>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<2> &grp);                                             \
    template void ishmemx_put##SIZE##_nbi_work_group<sycl::group<3>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<3> &grp);                                             \
    template void ishmemx_put##SIZE##_nbi_work_group<sycl::sub_group>(void *dest, const void *src, size_t nelems, int pe, const sycl::sub_group &grp);                                           \
    template <typename Group> void ishmemx_put##SIZE##_nbi_work_group(void *dest, const void *src, size_t nelems, int pe, const Group &grp) { ishmemx_put_nbi_work_group((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, nelems * (SIZE / ELEMSIZE), pe, grp); }

#define ISHMEM_INSTANTIATE_TYPE(TYPE) template void ishmem_put_nbi(TYPE *, const TYPE *, size_t, int)
ISHMEMI_API_IMPL_PUT_NBI(float, float)
ISHMEMI_API_IMPL_PUT_NBI(double, double)
ISHMEMI_API_IMPL_PUT_NBI(char, char)
ISHMEMI_API_IMPL_PUT_NBI(schar, signed char)
ISHMEMI_API_IMPL_PUT_NBI(short, short)
ISHMEMI_API_IMPL_PUT_NBI(int, int)
ISHMEMI_API_IMPL_PUT_NBI(long, long)
ISHMEMI_API_IMPL_PUT_NBI(longlong, long long)
ISHMEMI_API_IMPL_PUT_NBI(uchar, unsigned char)
ISHMEMI_API_IMPL_PUT_NBI(ushort, unsigned short)
ISHMEMI_API_IMPL_PUT_NBI(uint, unsigned int)
ISHMEMI_API_IMPL_PUT_NBI(ulong, unsigned long)
ISHMEMI_API_IMPL_PUT_NBI(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_PUT_NBI(int8, int8_t)
ISHMEMI_API_IMPL_PUT_NBI(int16, int16_t)
ISHMEMI_API_IMPL_PUT_NBI(int32, int32_t)
ISHMEMI_API_IMPL_PUT_NBI(int64, int64_t)
ISHMEMI_API_IMPL_PUT_NBI(uint8, uint8_t)
ISHMEMI_API_IMPL_PUT_NBI(uint16, uint16_t)
ISHMEMI_API_IMPL_PUT_NBI(uint32, uint32_t)
ISHMEMI_API_IMPL_PUT_NBI(uint64, uint64_t)
ISHMEMI_API_IMPL_PUT_NBI(size, size_t)
ISHMEMI_API_IMPL_PUT_NBI(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_PUTSIZE_NBI(8, 8)
ISHMEMI_API_IMPL_PUTSIZE_NBI(16, 16)
ISHMEMI_API_IMPL_PUTSIZE_NBI(32, 32)
ISHMEMI_API_IMPL_PUTSIZE_NBI(64, 64)
ISHMEMI_API_IMPL_PUTSIZE_NBI(128, 64)
#undef ISHMEM_INSTANTIATE_TYPE
/* clang-format on */

/* Non-blocking Get */
template <typename T>
void ishmem_get_nbi(T *dest, const T *src, size_t nelems, int pe)
{
    ishmem_internal_get_nbi(dest, src, nelems, pe);
}

void ishmem_getmem_nbi(void *dest, const void *src, size_t nelems, int pe)
{
    ishmem_internal_get_nbi((uint8_t *) dest, (uint8_t *) src, nelems, pe);
}

template <typename T>
sycl::event ishmemx_get_nbi_on_queue(T *dest, const T *src, size_t nelems, int pe, sycl::queue &q,
                                     const std::vector<sycl::event> &deps)
{
    bool entry_already_exists = true;
    const std::lock_guard<std::mutex> lock(ishmemi_on_queue_events_map.map_mtx);
    auto iter = ishmemi_on_queue_events_map.get_entry_info(q, entry_already_exists);

    auto e = q.submit([&](sycl::handler &cgh) {
        set_cmd_grp_dependencies(cgh, entry_already_exists, iter->second->event, deps);
        cgh.single_task([=]() { ishmem_get_nbi(dest, src, nelems, pe); });
    });
    ishmemi_on_queue_events_map[&q]->event = e;
    return e;
}

sycl::event ishmemx_getmem_nbi_on_queue(void *dest, const void *src, size_t nelems, int pe,
                                        sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return ishmemx_get_nbi_on_queue((uint8_t *) dest, (uint8_t *) src, nelems, pe, q, deps);
}

/* Non-blocking Get (work-group) */
template <typename T, typename Group>
void ishmemx_get_nbi_work_group(T *dest, const T *src, size_t nelems, int pe, const Group &grp)
{
    ishmemx_internal_get_nbi_work_group(dest, src, nelems, pe, grp);
}

/* clang-format off */
template void ishmemx_getmem_nbi_work_group<sycl::group<1>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<1> &grp);
template void ishmemx_getmem_nbi_work_group<sycl::group<2>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<2> &grp);
template void ishmemx_getmem_nbi_work_group<sycl::group<3>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<3> &grp);
template void ishmemx_getmem_nbi_work_group<sycl::sub_group>(void *dest, const void *src, size_t nelems, int pe, const sycl::sub_group &grp);
/* clang-format on */

template <typename Group>
void ishmemx_getmem_nbi_work_group(void *dest, const void *src, size_t nelems, int pe,
                                   const Group &grp)
{
    ishmemx_internal_get_nbi_work_group((uint8_t *) dest, (uint8_t *) src, nelems, pe, grp);
}

/* clang-format off */
#define ISHMEMI_API_IMPL_GET_NBI(TYPENAME, TYPE)                                                                                                                  \
    ISHMEM_INSTANTIATE_TYPE_##TYPENAME(TYPE);                                                                                                                     \
    void ishmem_##TYPENAME##_get_nbi(TYPE *dest, const TYPE *src, size_t nelems, int pe) { ishmem_get_nbi(dest, src, nelems, pe); }                               \
    sycl::event ishmemx_##TYPENAME##_get_nbi_on_queue(TYPE *dest, const TYPE *src, size_t nelems, int pe, sycl::queue &q, const std::vector<sycl::event> &deps) { \
        return ishmemx_get_nbi_on_queue(dest, src, nelems, pe, q, deps);                                                                                          \
    }                                                                                                                                                             \
    template void ishmemx_##TYPENAME##_get_nbi_work_group<sycl::group<1>>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::group<1> &grp);         \
    template void ishmemx_##TYPENAME##_get_nbi_work_group<sycl::group<2>>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::group<2> &grp);         \
    template void ishmemx_##TYPENAME##_get_nbi_work_group<sycl::group<3>>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::group<3> &grp);         \
    template void ishmemx_##TYPENAME##_get_nbi_work_group<sycl::sub_group>(TYPE *dest, const TYPE *src, size_t nelems, int pe, const sycl::sub_group &grp);       \
    template <typename Group> void ishmemx_##TYPENAME##_get_nbi_work_group(TYPE *dest, const TYPE *src, size_t nelems, int pe, const Group &grp) { ishmemx_get_nbi_work_group(dest, src, nelems, pe, grp); }

#define ISHMEMI_API_IMPL_GETSIZE_NBI(SIZE, ELEMSIZE)                                                                                                                                             \
    void ishmem_get##SIZE##_nbi(void *dest, const void *src, size_t nelems, int pe) { ishmem_get_nbi((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, nelems * (SIZE / ELEMSIZE), pe); } \
    sycl::event ishmemx_get##SIZE##_nbi_on_queue(void *dest, const void *src, size_t nelems, int pe, sycl::queue &q, const std::vector<sycl::event> &deps) {                                     \
        return ishmemx_get_nbi_on_queue((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, nelems * (SIZE / ELEMSIZE), pe, q, deps);                                                       \
    }                                                                                                                                                                                            \
    template void ishmemx_get##SIZE##_nbi_work_group<sycl::group<1>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<1> &grp);                                             \
    template void ishmemx_get##SIZE##_nbi_work_group<sycl::group<2>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<2> &grp);                                             \
    template void ishmemx_get##SIZE##_nbi_work_group<sycl::group<3>>(void *dest, const void *src, size_t nelems, int pe, const sycl::group<3> &grp);                                             \
    template void ishmemx_get##SIZE##_nbi_work_group<sycl::sub_group>(void *dest, const void *src, size_t nelems, int pe, const sycl::sub_group &grp);                                           \
    template <typename Group> void ishmemx_get##SIZE##_nbi_work_group(void *dest, const void *src, size_t nelems, int pe, const Group &grp) { ishmemx_get_nbi_work_group((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, nelems * (SIZE / ELEMSIZE), pe, grp); }

#define ISHMEM_INSTANTIATE_TYPE(TYPE) template void ishmem_get_nbi(TYPE *, const TYPE *, size_t, int)
ISHMEMI_API_IMPL_GET_NBI(float, float)
ISHMEMI_API_IMPL_GET_NBI(double, double)
ISHMEMI_API_IMPL_GET_NBI(char, char)
ISHMEMI_API_IMPL_GET_NBI(schar, signed char)
ISHMEMI_API_IMPL_GET_NBI(short, short)
ISHMEMI_API_IMPL_GET_NBI(int, int)
ISHMEMI_API_IMPL_GET_NBI(long, long)
ISHMEMI_API_IMPL_GET_NBI(longlong, long long)
ISHMEMI_API_IMPL_GET_NBI(uchar, unsigned char)
ISHMEMI_API_IMPL_GET_NBI(ushort, unsigned short)
ISHMEMI_API_IMPL_GET_NBI(uint, unsigned int)
ISHMEMI_API_IMPL_GET_NBI(ulong, unsigned long)
ISHMEMI_API_IMPL_GET_NBI(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_GET_NBI(int8, int8_t)
ISHMEMI_API_IMPL_GET_NBI(int16, int16_t)
ISHMEMI_API_IMPL_GET_NBI(int32, int32_t)
ISHMEMI_API_IMPL_GET_NBI(int64, int64_t)
ISHMEMI_API_IMPL_GET_NBI(uint8, uint8_t)
ISHMEMI_API_IMPL_GET_NBI(uint16, uint16_t)
ISHMEMI_API_IMPL_GET_NBI(uint32, uint32_t)
ISHMEMI_API_IMPL_GET_NBI(uint64, uint64_t)
ISHMEMI_API_IMPL_GET_NBI(size, size_t)
ISHMEMI_API_IMPL_GET_NBI(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_GETSIZE_NBI(8, 8)
ISHMEMI_API_IMPL_GETSIZE_NBI(16, 16)
ISHMEMI_API_IMPL_GETSIZE_NBI(32, 32)
ISHMEMI_API_IMPL_GETSIZE_NBI(64, 64)
ISHMEMI_API_IMPL_GETSIZE_NBI(128, 64)
#undef ISHMEM_INSTANTIATE_TYPE
/* clang-format on */
