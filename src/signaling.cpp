/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ishmem/err.h"
#include "ishmem/types.h"
#include "ishmem/util.h"
#include "proxy_impl.h"
#include "runtime.h"
#include "memory.h"

/* Put with signal */
template <typename T>
void ishmem_put_signal(T *dest, const T *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,
                       int sig_op, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, (void *) src, (void *) sig_addr, sizeof(T) * nelems,
                            sizeof(uint64_t));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        if (local_index != 0) {
            vec_copy_push(ISHMEMI_ADJUST_PTR(T, local_index, dest), src, nelems);
            uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, sig_addr);
            sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            if (sig_op == ISHMEM_SIGNAL_SET) {
                atomic_p.store(signal);
            } else {
                atomic_p += signal;
            }
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.src = src;
    req.dst = dest;
    req.nelems = nelems * sizeof(T);
    req.sig_addr = sig_addr;
    req.sig_op = sig_op;
    req.signal = signal;
    req.op = PUT_SIGNAL;
    req.type = UINT8;

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

void ishmem_putmem_signal(void *dest, const void *src, size_t nelems, uint64_t *sig_addr,
                          uint64_t signal, int sig_op, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, (void *) src, (void *) sig_addr,
                            sizeof(char) * nelems, sizeof(uint64_t));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        if (local_index != 0) {
            vec_copy_push(ISHMEMI_ADJUST_PTR(uint8_t, local_index, (uint8_t *) dest),
                          (uint8_t *) src, nelems);
            uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, sig_addr);
            sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            if (sig_op == ISHMEM_SIGNAL_SET) {
                atomic_p.store(signal);
            } else {
                atomic_p += signal;
            }
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.src = src;
    req.dst = dest;
    req.nelems = nelems;
    req.sig_addr = sig_addr;
    req.sig_op = sig_op;
    req.signal = signal;
    req.op = PUT_SIGNAL;
    req.type = UINT8;

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* Put with signal (work-group) */
template <typename T, typename Group>
void ishmemx_put_signal_work_group(T *dest, const T *src, size_t nelems, uint64_t *sig_addr,
                                   uint64_t signal, int sig_op, int pe, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        sycl::group_barrier(grp);
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters(pe, (void *) dest, (void *) src, (void *) sig_addr,
                                    sizeof(T) * nelems, sizeof(uint64_t));
        }
        uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
        if (local_index != 0) {
            size_t my_nelems_work_item;
            size_t work_item_start_idx;
            ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item,
                                               work_item_start_idx);
            vec_copy_push(ISHMEMI_ADJUST_PTR(T, local_index, dest + work_item_start_idx),
                          src + work_item_start_idx, my_nelems_work_item);
            sycl::group_barrier(grp); /* To make sure all copies are complete */
            if (grp.leader()) {
                uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, sig_addr);
                sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>
                    atomic_p(*p);
                if (sig_op == ISHMEM_SIGNAL_SET) {
                    atomic_p.store(signal);
                } else {
                    atomic_p += signal;
                }
            }
        } else {
            if (grp.leader()) {
                ishmemi_request_t req;
                req.dest_pe = pe;
                req.src = src;
                req.dst = dest;
                req.nelems = nelems * sizeof(T);
                req.sig_addr = sig_addr;
                req.sig_op = sig_op;
                req.signal = signal;
                req.op = PUT_SIGNAL;
                req.type = UINT8;

                ishmemi_proxy_blocking_request(req);
            }
        }
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_put_signal_work_group not callable from host\n");
    }
}

/* clang-format off */
template void ishmemx_putmem_signal_work_group<sycl::group<1>>(void *dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, const sycl::group<1> &grp);
template void ishmemx_putmem_signal_work_group<sycl::group<2>>(void *dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, const sycl::group<2> &grp);
template void ishmemx_putmem_signal_work_group<sycl::group<3>>(void *dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, const sycl::group<3> &grp);
template void ishmemx_putmem_signal_work_group<sycl::sub_group>(void *dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, const sycl::sub_group &grp);
/* clang-format on */

template <typename Group>
void ishmemx_putmem_signal_work_group(void *dest, const void *src, size_t nelems,
                                      uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                      const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        sycl::group_barrier(grp);
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters(pe, (void *) dest, (void *) src, (void *) sig_addr,
                                    sizeof(char) * nelems, sizeof(uint64_t));
        }
        uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
        if (local_index != 0) {
            size_t my_nelems_work_item;
            size_t work_item_start_idx;
            ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item,
                                               work_item_start_idx);
            vec_copy_push(
                ISHMEMI_ADJUST_PTR(uint8_t, local_index, (uint8_t *) dest + work_item_start_idx),
                (uint8_t *) src + work_item_start_idx, my_nelems_work_item);
            sycl::group_barrier(grp); /* To make sure all copies are complete */
            if (grp.leader()) {
                uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, sig_addr);
                sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>
                    atomic_p(*p);
                if (sig_op == ISHMEM_SIGNAL_SET) {
                    atomic_p.store(signal);
                } else {
                    atomic_p += signal;
                }
            }
        } else {
            if (grp.leader()) {
                ishmemi_request_t req;
                req.dest_pe = pe;
                req.src = src;
                req.dst = dest;
                req.nelems = nelems;
                req.sig_addr = sig_addr;
                req.op = PUT_SIGNAL;
                req.type = UINT8;

                /* sig_op and signal are outside the initializer list because of a compiler bug:
                 *
                 * error: field designator (null) does not refer to any field in type
                 * 'ishmemi_request_t'
                 *
                 * https://github.com/llvm/llvm-project/issues/46132
                 */
                req.sig_op = sig_op;
                req.signal = signal;
                ishmemi_proxy_blocking_request(req);
            }
        }
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_putmem_signal_work_group not callable from host\n");
    }
}

/* clang-format off */
#define ISHMEMI_API_IMPL_PUT_SIGNAL(TYPENAME, TYPE)                                                                                                                                                                            \
    void ishmem_##TYPENAME##_put_signal(TYPE *dest, const TYPE *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { return ishmem_put_signal(dest, src, nelems, sig_addr, signal, sig_op, pe); }    \
    template void ishmemx_##TYPENAME##_put_signal_work_group<sycl::group<1>>(TYPE * dest, const TYPE *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::group<1> &grp);                  \
    template void ishmemx_##TYPENAME##_put_signal_work_group<sycl::group<2>>(TYPE * dest, const TYPE *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::group<2> &grp);                  \
    template void ishmemx_##TYPENAME##_put_signal_work_group<sycl::group<3>>(TYPE * dest, const TYPE *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::group<3> &grp);                  \
    template void ishmemx_##TYPENAME##_put_signal_work_group<sycl::sub_group>(TYPE * dest, const TYPE *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::sub_group &grp);                \
    template <typename Group> void ishmemx_##TYPENAME##_put_signal_work_group(TYPE *dest, const TYPE *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, const Group &grp) { ishmemx_put_signal_work_group(dest, src, nelems, sig_addr, signal, sig_op, pe, grp); }

#define ISHMEMI_API_IMPL_PUTSIZE_SIGNAL(SIZE, ELEMSIZE)                                                                                                                                                                                                                                   \
    void ishmem_put##SIZE##_signal(void *dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { return ishmem_put_signal((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, nelems * (SIZE / ELEMSIZE), sig_addr, signal, sig_op, pe); }  \
    template void ishmemx_put##SIZE##_signal_work_group<sycl::group<1>>(void * dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::group<1> &grp);                                                                                  \
    template void ishmemx_put##SIZE##_signal_work_group<sycl::group<2>>(void * dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::group<2> &grp);                                                                                  \
    template void ishmemx_put##SIZE##_signal_work_group<sycl::group<3>>(void * dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::group<3> &grp);                                                                                  \
    template void ishmemx_put##SIZE##_signal_work_group<sycl::sub_group>(void * dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::sub_group &grp);                                                                                \
    template <typename Group> void ishmemx_put##SIZE##_signal_work_group(void *dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, const Group &grp) { ishmemx_put_signal_work_group((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, nelems * (SIZE / ELEMSIZE), sig_addr, signal, sig_op, pe, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_PUT_SIGNAL(float, float)
ISHMEMI_API_IMPL_PUT_SIGNAL(double, double)
ISHMEMI_API_IMPL_PUT_SIGNAL(char, char)
ISHMEMI_API_IMPL_PUT_SIGNAL(schar, signed char)
ISHMEMI_API_IMPL_PUT_SIGNAL(short, short)
ISHMEMI_API_IMPL_PUT_SIGNAL(int, int)
ISHMEMI_API_IMPL_PUT_SIGNAL(long, long)
ISHMEMI_API_IMPL_PUT_SIGNAL(longlong, long long)
ISHMEMI_API_IMPL_PUT_SIGNAL(uchar, unsigned char)
ISHMEMI_API_IMPL_PUT_SIGNAL(ushort, unsigned short)
ISHMEMI_API_IMPL_PUT_SIGNAL(uint, unsigned int)
ISHMEMI_API_IMPL_PUT_SIGNAL(ulong, unsigned long)
ISHMEMI_API_IMPL_PUT_SIGNAL(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_PUT_SIGNAL(int8, int8_t)
ISHMEMI_API_IMPL_PUT_SIGNAL(int16, int16_t)
ISHMEMI_API_IMPL_PUT_SIGNAL(int32, int32_t)
ISHMEMI_API_IMPL_PUT_SIGNAL(int64, int64_t)
ISHMEMI_API_IMPL_PUT_SIGNAL(uint8, uint8_t)
ISHMEMI_API_IMPL_PUT_SIGNAL(uint16, uint16_t)
ISHMEMI_API_IMPL_PUT_SIGNAL(uint32, uint32_t)
ISHMEMI_API_IMPL_PUT_SIGNAL(uint64, uint64_t)
ISHMEMI_API_IMPL_PUT_SIGNAL(size, size_t)
ISHMEMI_API_IMPL_PUT_SIGNAL(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_PUTSIZE_SIGNAL(8, 8)
ISHMEMI_API_IMPL_PUTSIZE_SIGNAL(16, 16)
ISHMEMI_API_IMPL_PUTSIZE_SIGNAL(32, 32)
ISHMEMI_API_IMPL_PUTSIZE_SIGNAL(64, 64)
ISHMEMI_API_IMPL_PUTSIZE_SIGNAL(128, 64)

/* Non-blocking Put with signal */
template <typename T>
void ishmem_put_signal_nbi(T *dest, const T *src, size_t nelems, uint64_t *sig_addr,
                           uint64_t signal, int sig_op, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, (void *) src, (void *) sig_addr, sizeof(T) * nelems,
                            sizeof(uint64_t));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        if (local_index != 0) {
            vec_copy_push(ISHMEMI_ADJUST_PTR(T, local_index, dest), src, nelems);
            uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, sig_addr);
            sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            if (sig_op == ISHMEM_SIGNAL_SET) {
                atomic_p.store(signal);
            } else {
                atomic_p += signal;
            }
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.src = src;
    req.dst = dest;
    req.nelems = nelems * sizeof(T);
    req.sig_addr = sig_addr;
    req.sig_op = sig_op;
    req.signal = signal;
    req.op = PUT_SIGNAL_NBI;
    req.type = UINT8;

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_nonblocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

void ishmem_putmem_signal_nbi(void *dest, const void *src, size_t nelems, uint64_t *sig_addr,
                              uint64_t signal, int sig_op, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) dest, (void *) src, (void *) sig_addr,
                            sizeof(char) * nelems, sizeof(uint64_t));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        if (local_index != 0) {
            vec_copy_push(ISHMEMI_ADJUST_PTR(uint8_t, local_index, (uint8_t *) dest),
                          (uint8_t *) src, nelems);
            uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, sig_addr);
            sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            if (sig_op == ISHMEM_SIGNAL_SET) {
                atomic_p.store(signal);
            } else {
                atomic_p += signal;
            }
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.src = src;
    req.dst = dest;
    req.nelems = nelems;
    req.sig_addr = sig_addr;
    req.sig_op = sig_op;
    req.signal = signal;
    req.op = PUT_SIGNAL_NBI;
    req.type = UINT8;

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_proxy_nonblocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* Non-blocking Put with signal (work-group) */
template <typename T, typename Group>
void ishmemx_put_signal_nbi_work_group(T *dest, const T *src, size_t nelems, uint64_t *sig_addr,
                                       uint64_t signal, int sig_op, int pe, const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        sycl::group_barrier(grp);
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters(pe, (void *) dest, (void *) src, (void *) sig_addr,
                                    sizeof(T) * nelems, sizeof(uint64_t));
        }
        uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
        if (local_index != 0) {
            size_t my_nelems_work_item;
            size_t work_item_start_idx;
            ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item,
                                               work_item_start_idx);
            vec_copy_push(ISHMEMI_ADJUST_PTR(T, local_index, dest + work_item_start_idx),
                          src + work_item_start_idx, my_nelems_work_item);
            sycl::group_barrier(grp); /* To make sure all copies are complete */
            if (grp.leader()) {
                uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, sig_addr);
                sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>
                    atomic_p(*p);
                if (sig_op == ISHMEM_SIGNAL_SET) {
                    atomic_p.store(signal);
                } else {
                    atomic_p += signal;
                }
            }
        } else {
            if (grp.leader()) {
                ishmemi_request_t req;
                req.dest_pe = pe;
                req.src = src;
                req.dst = dest;
                req.nelems = nelems * sizeof(T);
                req.op = PUT_SIGNAL_NBI;
                req.type = UINT8;

                /* sig_op and signal are outside the initializer list because of a compiler bug:
                 *
                 * error: field designator (null) does not refer to any field in type
                 * 'ishmemi_request_t'
                 *
                 * https://github.com/llvm/llvm-project/issues/46132
                 */
                req.sig_op = sig_op;
                req.signal = signal;

                ishmemi_proxy_nonblocking_request(req);
            }
        }
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_put_signal_nbi_work_group not callable from host\n");
    }
}

/* clang-format off */
template void ishmemx_putmem_signal_nbi_work_group<sycl::group<1>>(void *dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, const sycl::group<1> &grp);
template void ishmemx_putmem_signal_nbi_work_group<sycl::group<2>>(void *dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, const sycl::group<2> &grp);
template void ishmemx_putmem_signal_nbi_work_group<sycl::group<3>>(void *dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, const sycl::group<3> &grp);
template void ishmemx_putmem_signal_nbi_work_group<sycl::sub_group>(void *dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, const sycl::sub_group &grp);
/* clang-format on */

template <typename Group>
void ishmemx_putmem_signal_nbi_work_group(void *dest, const void *src, size_t nelems,
                                          uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                          const Group &grp)
{
    if constexpr (ishmemi_is_device) {
        sycl::group_barrier(grp);
        if constexpr (enable_error_checking) {
            if (grp.leader())
                validate_parameters(pe, (void *) dest, (void *) src, (void *) sig_addr,
                                    sizeof(char) * nelems, sizeof(uint64_t));
        }
        uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
        if (local_index != 0) {
            size_t my_nelems_work_item;
            size_t work_item_start_idx;
            ishmemi_work_item_calculate_offset(nelems, grp, my_nelems_work_item,
                                               work_item_start_idx);
            vec_copy_push(
                ISHMEMI_ADJUST_PTR(uint8_t, local_index, (uint8_t *) dest + work_item_start_idx),
                (uint8_t *) src + work_item_start_idx, my_nelems_work_item);
            sycl::group_barrier(grp); /* To make sure all copies are complete */
            if (grp.leader()) {
                uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, sig_addr);
                sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>
                    atomic_p(*p);
                if (sig_op == ISHMEM_SIGNAL_SET) {
                    atomic_p.store(signal);
                } else {
                    atomic_p += signal;
                }
            }
        } else {
            if (grp.leader()) {
                ishmemi_request_t req;
                req.dest_pe = pe;
                req.src = src;
                req.dst = dest;
                req.nelems = nelems;
                req.op = PUT_SIGNAL_NBI;
                req.type = UINT8;

                /* sig_op and signal are outside the initializer list because of a compiler bug:
                 *
                 * error: field designator (null) does not refer to any field in type
                 * 'ishmemi_request_t'
                 *
                 * https://github.com/llvm/llvm-project/issues/46132
                 */
                req.sig_op = sig_op;
                req.signal = signal;

                ishmemi_proxy_nonblocking_request(req);
            }
        }
        sycl::group_barrier(grp);
    } else {
        RAISE_ERROR_MSG("ishmemx_putmem_signal_nbi_work_group not callable from host\n");
    }
}

/* clang-format off */
#define ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(TYPENAME, TYPE)                                                                                                                                                                                \
    void ishmem_##TYPENAME##_put_signal_nbi(TYPE *dest, const TYPE *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { return ishmem_put_signal_nbi(dest, src, nelems, sig_addr, signal, sig_op, pe); }    \
    template void ishmemx_##TYPENAME##_put_signal_nbi_work_group<sycl::group<1>>(TYPE * dest, const TYPE *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::group<1> &grp);                      \
    template void ishmemx_##TYPENAME##_put_signal_nbi_work_group<sycl::group<2>>(TYPE * dest, const TYPE *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::group<2> &grp);                      \
    template void ishmemx_##TYPENAME##_put_signal_nbi_work_group<sycl::group<3>>(TYPE * dest, const TYPE *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::group<3> &grp);                      \
    template void ishmemx_##TYPENAME##_put_signal_nbi_work_group<sycl::sub_group>(TYPE * dest, const TYPE *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::sub_group &grp);                    \
    template <typename Group> void ishmemx_##TYPENAME##_put_signal_nbi_work_group(TYPE *dest, const TYPE *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, const Group &grp) { ishmemx_put_signal_nbi_work_group(dest, src, nelems, sig_addr, signal, sig_op, pe, grp); }

#define ISHMEMI_API_IMPL_PUTSIZE_SIGNAL_NBI(SIZE, ELEMSIZE)                                                                                                                                                                                                                                       \
    void ishmem_put##SIZE##_signal_nbi(void *dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { return ishmem_put_signal_nbi((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, nelems * (SIZE / ELEMSIZE), sig_addr, signal, sig_op, pe); }  \
    template void ishmemx_put##SIZE##_signal_nbi_work_group<sycl::group<1>>(void * dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::group<1> &grp);                                                                                      \
    template void ishmemx_put##SIZE##_signal_nbi_work_group<sycl::group<2>>(void * dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::group<2> &grp);                                                                                      \
    template void ishmemx_put##SIZE##_signal_nbi_work_group<sycl::group<3>>(void * dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::group<3> &grp);                                                                                      \
    template void ishmemx_put##SIZE##_signal_nbi_work_group<sycl::sub_group>(void * dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal,int sig_op, int pe, const sycl::sub_group &grp);                                                                                    \
    template <typename Group> void ishmemx_put##SIZE##_signal_nbi_work_group(void *dest, const void *src, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, const Group &grp) { ishmemx_put_signal_nbi_work_group((uint##ELEMSIZE##_t *) dest, (uint##ELEMSIZE##_t *) src, nelems * (SIZE / ELEMSIZE), sig_addr, signal, sig_op, pe, grp); }
/* clang-format on */

ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(float, float)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(double, double)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(char, char)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(schar, signed char)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(short, short)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(int, int)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(long, long)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(longlong, long long)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(uchar, unsigned char)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(ushort, unsigned short)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(uint, unsigned int)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(ulong, unsigned long)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(ulonglong, unsigned long long)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(int8, int8_t)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(int16, int16_t)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(int32, int32_t)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(int64, int64_t)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(uint8, uint8_t)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(uint16, uint16_t)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(uint32, uint32_t)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(uint64, uint64_t)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(size, size_t)
ISHMEMI_API_IMPL_PUT_SIGNAL_NBI(ptrdiff, ptrdiff_t)
ISHMEMI_API_IMPL_PUTSIZE_SIGNAL_NBI(8, 8)
ISHMEMI_API_IMPL_PUTSIZE_SIGNAL_NBI(16, 16)
ISHMEMI_API_IMPL_PUTSIZE_SIGNAL_NBI(32, 32)
ISHMEMI_API_IMPL_PUTSIZE_SIGNAL_NBI(64, 64)
ISHMEMI_API_IMPL_PUTSIZE_SIGNAL_NBI(128, 64)

/* Signal fetch */
uint64_t ishmem_signal_fetch(uint64_t *sig_addr)
{
    uint64_t ret = static_cast<uint64_t>(0);

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        if (global_info->only_intra_node) {
            sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_sig(*sig_addr);
            ret = atomic_sig.load();
            return ret;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.src = (void *) sig_addr;
    req.op = AMO_FETCH;
    req.type = UINT64;

#if __SYCL_DEVICE_ONLY__
    ret = ishmemi_proxy_blocking_request_return<uint64_t>(req);
#else
    ishmemi_ringcompletion_t comp;
    ishmemi_proxy_funcs[req.op][req.type](&req, &comp);
    ret = ishmemi_proxy_get_field_value<uint64_t>(comp.completion.ret);
#endif
    return ret;
}

/* Signal set */
void ishmemx_signal_set(uint64_t *sig_addr, uint64_t val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) sig_addr, sizeof(uint64_t));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, sig_addr);
            sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            atomic_p = val;
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = sig_addr;
    req.op = SIGNAL_SET;
    req.type = ishmemi_proxy_get_base_type<uint64_t>();

    ishmemi_proxy_set_field_value<uint64_t>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}

/* Signal add */
void ishmemx_signal_add(uint64_t *sig_addr, uint64_t val, int pe)
{
    if constexpr (enable_error_checking) {
        validate_parameters(pe, (void *) sig_addr, sizeof(uint64_t));
    }

    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, sig_addr);
            sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            atomic_p += val;
            return;
        }
    }

    /* Otherwise */
    ishmemi_request_t req;
    req.dest_pe = pe;
    req.dst = sig_addr;
    req.op = SIGNAL_ADD;
    req.type = ishmemi_proxy_get_base_type<uint64_t>();

    ishmemi_proxy_set_field_value<uint64_t>(req.value, val);

#if __SYCL_DEVICE_ONLY__
    ishmemi_proxy_blocking_request(req);
#else
    ishmemi_proxy_funcs[req.op][req.type](&req, nullptr);
#endif
}
