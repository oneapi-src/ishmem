/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef I_SHMEMX_H
#define I_SHMEMX_H

#include <CL/sycl.hpp>
#define ISHMEM_DEVICE_ATTRIBUTES SYCL_EXTERNAL

/* Enumeration of runtimes */
typedef enum {
    ISHMEMX_RUNTIME_MPI,
    ISHMEMX_RUNTIME_OPENSHMEM,
    ISHMEMX_RUNTIME_PMI
} ishmemx_runtime_type_t;

typedef struct ishmemx_attr_t {
    /* By default, the runtime is assumed to be OpenSHMEM */
    ishmemx_runtime_type_t runtime = ISHMEMX_RUNTIME_OPENSHMEM;
    /* By default, runtimes are assumed to be initialized by ISHMEM */
    bool initialize_runtime = true;
    /* By default, gpu is used */
    bool gpu = true;
} ishmemx_attr_t;

/* ISHMEMX APIs */
/* Library setup and exit routines (host) */
void ishmemx_init_attr(ishmemx_attr_t *attr);

/* clang-format off */
/* put_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put_work_group(T *, const T *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_float_put_work_group(float *, const float *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_double_put_work_group(double *, const double *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_char_put_work_group(char *, const char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_schar_put_work_group(signed char *, const signed char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_short_put_work_group(short *, const short *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_put_work_group(int *, const int *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_put_work_group(long *, const long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_put_work_group(long long *, const long long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uchar_put_work_group(unsigned char *, const unsigned char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ushort_put_work_group(unsigned short *, const unsigned short *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_put_work_group(unsigned int *, const unsigned int *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_put_work_group(unsigned long *, const unsigned long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_put_work_group(unsigned long long *, const unsigned long long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int8_put_work_group(int8_t *, const int8_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int16_put_work_group(int16_t *, const int16_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_put_work_group(int32_t *, const int32_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_put_work_group(int64_t *, const int64_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint8_put_work_group(uint8_t *, const uint8_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint16_put_work_group(uint16_t *, const uint16_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_put_work_group(uint32_t *, const uint32_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_put_work_group(uint64_t *, const uint64_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_put_work_group(size_t *, const size_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_put_work_group(ptrdiff_t *, const ptrdiff_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_putmem_work_group(void *, const void *, size_t, int, const Group &);

/* iput_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_iput_work_group(T *, const T *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_float_iput_work_group(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_double_iput_work_group(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_char_iput_work_group(char *, const char *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_schar_iput_work_group(signed char *, const signed char *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_short_iput_work_group(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_iput_work_group(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_iput_work_group(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_iput_work_group(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uchar_iput_work_group(unsigned char *, const unsigned char *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ushort_iput_work_group(unsigned short *, const unsigned short *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_iput_work_group(unsigned int *, const unsigned int *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_iput_work_group(unsigned long *, const unsigned long *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_iput_work_group(unsigned long long *, const unsigned long long *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int8_iput_work_group(int8_t *, const int8_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int16_iput_work_group(int16_t *, const int16_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_iput_work_group(int32_t *, const int32_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_iput_work_group(int64_t *, const int64_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint8_iput_work_group(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint16_iput_work_group(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_iput_work_group(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_iput_work_group(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_iput_work_group(size_t *, const size_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_iput_work_group(ptrdiff_t *, const ptrdiff_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_iputmem_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);

/* get_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_get_work_group(T *, const T *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_float_get_work_group(float *, const float *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_double_get_work_group(double *, const double *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_char_get_work_group(char *, const char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_schar_get_work_group(signed char *, const signed char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_short_get_work_group(short *, const short *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_get_work_group(int *, const int *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_get_work_group(long *, const long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_get_work_group(long long *, const long long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uchar_get_work_group(unsigned char *, const unsigned char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ushort_get_work_group(unsigned short *, const unsigned short *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_get_work_group(unsigned int *, const unsigned int *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_get_work_group(unsigned long *, const unsigned long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_get_work_group(unsigned long long *, const unsigned long long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int8_get_work_group(int8_t *, const int8_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int16_get_work_group(int16_t *, const int16_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_get_work_group(int32_t *, const int32_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_get_work_group(int64_t *, const int64_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint8_get_work_group(uint8_t *, const uint8_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint16_get_work_group(uint16_t *, const uint16_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_get_work_group(uint32_t *, const uint32_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_get_work_group(uint64_t *, const uint64_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_get_work_group(size_t *, const size_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_get_work_group(ptrdiff_t *, const ptrdiff_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_getmem_work_group(void *, const void *, size_t, int, const Group &);

/* iget_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_iget_work_group(T *, const T *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_float_iget_work_group(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_double_iget_work_group(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_char_iget_work_group(char *, const char *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_schar_iget_work_group(signed char *, const signed char *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_short_iget_work_group(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_iget_work_group(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_iget_work_group(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_iget_work_group(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uchar_iget_work_group(unsigned char *, const unsigned char *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ushort_iget_work_group(unsigned short *, const unsigned short *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_iget_work_group(unsigned int *, const unsigned int *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_iget_work_group(unsigned long *, const unsigned long *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_iget_work_group(unsigned long long *, const unsigned long long *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int8_iget_work_group(int8_t *, const int8_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int16_iget_work_group(int16_t *, const int16_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_iget_work_group(int32_t *, const int32_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_iget_work_group(int64_t *, const int64_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint8_iget_work_group(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint16_iget_work_group(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_iget_work_group(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_iget_work_group(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_iget_work_group(size_t *, const size_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_iget_work_group(ptrdiff_t *, const ptrdiff_t *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_igetmem_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);

/* put_nbi_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put_nbi_work_group(T *, const T *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_float_put_nbi_work_group(float *, const float *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_double_put_nbi_work_group(double *, const double *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_char_put_nbi_work_group(char *, const char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_schar_put_nbi_work_group(signed char *, const signed char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_short_put_nbi_work_group(short *, const short *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_put_nbi_work_group(int *, const int *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_put_nbi_work_group(long *, const long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_put_nbi_work_group(long long *, const long long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uchar_put_nbi_work_group(unsigned char *, const unsigned char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ushort_put_nbi_work_group(unsigned short *, const unsigned short *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_put_nbi_work_group(unsigned int *, const unsigned int *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_put_nbi_work_group(unsigned long *, const unsigned long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_put_nbi_work_group(unsigned long long *, const unsigned long long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int8_put_nbi_work_group(int8_t *, const int8_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int16_put_nbi_work_group(int16_t *, const int16_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_put_nbi_work_group(int32_t *, const int32_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_put_nbi_work_group(int64_t *, const int64_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint8_put_nbi_work_group(uint8_t *, const uint8_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint16_put_nbi_work_group(uint16_t *, const uint16_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_put_nbi_work_group(uint32_t *, const uint32_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_put_nbi_work_group(uint64_t *, const uint64_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_put_nbi_work_group(size_t *, const size_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_put_nbi_work_group(ptrdiff_t *, const ptrdiff_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_putmem_nbi_work_group(void *, const void *, size_t, int, const Group &);

/* get_nbi_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_get_nbi_work_group(T *, const T *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_float_get_nbi_work_group(float *, const float *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_double_get_nbi_work_group(double *, const double *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_char_get_nbi_work_group(char *, const char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_schar_get_nbi_work_group(signed char *, const signed char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_short_get_nbi_work_group(short *, const short *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_get_nbi_work_group(int *, const int *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_get_nbi_work_group(long *, const long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_get_nbi_work_group(long long *, const long long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uchar_get_nbi_work_group(unsigned char *, const unsigned char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ushort_get_nbi_work_group(unsigned short *, const unsigned short *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_get_nbi_work_group(unsigned int *, const unsigned int *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_get_nbi_work_group(unsigned long *, const unsigned long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_get_nbi_work_group(unsigned long long *, const unsigned long long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int8_get_nbi_work_group(int8_t *, const int8_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int16_get_nbi_work_group(int16_t *, const int16_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_get_nbi_work_group(int32_t *, const int32_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_get_nbi_work_group(int64_t *, const int64_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint8_get_nbi_work_group(uint8_t *, const uint8_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint16_get_nbi_work_group(uint16_t *, const uint16_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_get_nbi_work_group(uint32_t *, const uint32_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_get_nbi_work_group(uint64_t *, const uint64_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_get_nbi_work_group(size_t *, const size_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_get_nbi_work_group(ptrdiff_t *, const ptrdiff_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_getmem_nbi_work_group(void *, const void *, size_t, int, const Group &);

/* put_signal_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put_signal_work_group(T *, const T *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_float_put_signal_work_group(float *, const float *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_double_put_signal_work_group(double *, const double *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_char_put_signal_work_group(char *, const char *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_schar_put_signal_work_group(signed char *, const signed char *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_short_put_signal_work_group(short *, const short *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_put_signal_work_group(int *, const int *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_put_signal_work_group(long *, const long *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_put_signal_work_group(long long *, const long long *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uchar_put_signal_work_group(unsigned char *, const unsigned char *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ushort_put_signal_work_group(unsigned short *, const unsigned short *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_put_signal_work_group(unsigned int *, const unsigned int *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_put_signal_work_group(unsigned long *, const unsigned long *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_put_signal_work_group(unsigned long long *, const unsigned long long *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int8_put_signal_work_group(int8_t *, const int8_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int16_put_signal_work_group(int16_t *, const int16_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_put_signal_work_group(int32_t *, const int32_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_put_signal_work_group(int64_t *, const int64_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint8_put_signal_work_group(uint8_t *, const uint8_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint16_put_signal_work_group(uint16_t *, const uint16_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_put_signal_work_group(uint32_t *, const uint32_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_put_signal_work_group(uint64_t *, const uint64_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_put_signal_work_group(size_t *, const size_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_put_signal_work_group(ptrdiff_t *, const ptrdiff_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_putmem_signal_work_group(void *, const void *, size_t, uint64_t *, uint64_t, int, int, const Group &);

/* put_signal_nbi_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put_signal_nbi_work_group(T *, const T *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_float_put_signal_nbi_work_group(float *, const float *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_double_put_signal_nbi_work_group(double *, const double *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_char_put_signal_nbi_work_group(char *, const char *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_schar_put_signal_nbi_work_group(signed char *, const signed char *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_short_put_signal_nbi_work_group(short *, const short *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_put_signal_nbi_work_group(int *, const int *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_put_signal_nbi_work_group(long *, const long *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_put_signal_nbi_work_group(long long *, const long long *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uchar_put_signal_nbi_work_group(unsigned char *, const unsigned char *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ushort_put_signal_nbi_work_group(unsigned short *, const unsigned short *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_put_signal_nbi_work_group(unsigned int *, const unsigned int *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_put_signal_nbi_work_group(unsigned long *, const unsigned long *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_put_signal_nbi_work_group(unsigned long long *, const unsigned long long *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int8_put_signal_nbi_work_group(int8_t *, const int8_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int16_put_signal_nbi_work_group(int16_t *, const int16_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_put_signal_nbi_work_group(int32_t *, const int32_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_put_signal_nbi_work_group(int64_t *, const int64_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint8_put_signal_nbi_work_group(uint8_t *, const uint8_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint16_put_signal_nbi_work_group(uint16_t *, const uint16_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_put_signal_nbi_work_group(uint32_t *, const uint32_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_put_signal_nbi_work_group(uint64_t *, const uint64_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_put_signal_nbi_work_group(size_t *, const size_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_put_signal_nbi_work_group(ptrdiff_t *, const ptrdiff_t *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_putmem_signal_nbi_work_group(void *, const void *, size_t, uint64_t *, uint64_t, int, int, const Group &);

/* alltoall_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_alltoall_work_group(T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_alltoall_work_group(float *, const float *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_alltoall_work_group(double *, const double *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_alltoall_work_group(char *, const char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_alltoall_work_group(signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_alltoall_work_group(short *, const short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_alltoall_work_group(int *, const int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_alltoall_work_group(long *, const long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_alltoall_work_group(long long *, const long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_alltoall_work_group(unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_alltoall_work_group(unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_alltoall_work_group(unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_alltoall_work_group(unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_alltoall_work_group(unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_alltoall_work_group(int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_alltoall_work_group(int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_alltoall_work_group(int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_alltoall_work_group(int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_alltoall_work_group(uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_alltoall_work_group(uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_alltoall_work_group(uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_alltoall_work_group(uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_alltoall_work_group(size_t *, const size_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_alltoall_work_group(ptrdiff_t *, const ptrdiff_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_alltoallmem_work_group(void *, const void *, size_t, const Group &);

/* broadcast_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_broadcast_work_group(T *, const T *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_broadcast_work_group(float *, const float *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_broadcast_work_group(double *, const double *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_broadcast_work_group(char *, const char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_broadcast_work_group(signed char *, const signed char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_broadcast_work_group(short *, const short *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_broadcast_work_group(int *, const int *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_broadcast_work_group(long *, const long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_broadcast_work_group(long long *, const long long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_broadcast_work_group(unsigned char *, const unsigned char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_broadcast_work_group(unsigned short *, const unsigned short *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_broadcast_work_group(unsigned int *, const unsigned int *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_broadcast_work_group(unsigned long *, const unsigned long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_broadcast_work_group(unsigned long long *, const unsigned long long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_broadcast_work_group(int8_t *, const int8_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_broadcast_work_group(int16_t *, const int16_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_broadcast_work_group(int32_t *, const int32_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_broadcast_work_group(int64_t *, const int64_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_broadcast_work_group(uint8_t *, const uint8_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_broadcast_work_group(uint16_t *, const uint16_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_broadcast_work_group(uint32_t *, const uint32_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_broadcast_work_group(uint64_t *, const uint64_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_broadcast_work_group(size_t *, const size_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_broadcast_work_group(ptrdiff_t *, const ptrdiff_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_broadcastmem_work_group(void *, const void *, size_t, int, const Group &);

/* collect_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_collect_work_group(T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_collect_work_group(float *, const float *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_collect_work_group(double *, const double *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_collect_work_group(char *, const char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_collect_work_group(signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_collect_work_group(short *, const short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_collect_work_group(int *, const int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_collect_work_group(long *, const long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_collect_work_group(long long *, const long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_collect_work_group(unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_collect_work_group(unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_collect_work_group(unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_collect_work_group(unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_collect_work_group(unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_collect_work_group(int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_collect_work_group(int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_collect_work_group(int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_collect_work_group(int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_collect_work_group(uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_collect_work_group(uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_collect_work_group(uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_collect_work_group(uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_collect_work_group(size_t *, const size_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_collect_work_group(ptrdiff_t *, const ptrdiff_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_collectmem_work_group(void *, const void *, size_t, const Group &);

/* fcollect_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_fcollect_work_group(T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_fcollect_work_group(float *, const float *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_fcollect_work_group(double *, const double *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_fcollect_work_group(char *, const char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_fcollect_work_group(signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_fcollect_work_group(short *, const short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_fcollect_work_group(int *, const int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_fcollect_work_group(long *, const long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_fcollect_work_group(long long *, const long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_fcollect_work_group(unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_fcollect_work_group(unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_fcollect_work_group(unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_fcollect_work_group(unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_fcollect_work_group(unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_fcollect_work_group(int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_fcollect_work_group(int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_fcollect_work_group(int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_fcollect_work_group(int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_fcollect_work_group(uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_fcollect_work_group(uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_fcollect_work_group(uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_fcollect_work_group(uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_fcollect_work_group(size_t *, const size_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_fcollect_work_group(ptrdiff_t *, const ptrdiff_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_fcollectmem_work_group(void *, const void *, size_t, const Group &);

/* and_reduce_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_and_reduce_work_group(T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_and_reduce_work_group(signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_and_reduce_work_group(unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_and_reduce_work_group(unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_and_reduce_work_group(unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_and_reduce_work_group(unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_and_reduce_work_group(unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_and_reduce_work_group(int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_and_reduce_work_group(int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_and_reduce_work_group(int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_and_reduce_work_group(int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_and_reduce_work_group(uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_and_reduce_work_group(uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_and_reduce_work_group(uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_and_reduce_work_group(uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_and_reduce_work_group(size_t *, const size_t *, size_t, const Group &);

/* or_reduce_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_or_reduce_work_group(T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_or_reduce_work_group(signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_or_reduce_work_group(unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_or_reduce_work_group(unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_or_reduce_work_group(unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_or_reduce_work_group(unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_or_reduce_work_group(unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_or_reduce_work_group(int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_or_reduce_work_group(int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_or_reduce_work_group(int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_or_reduce_work_group(int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_or_reduce_work_group(uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_or_reduce_work_group(uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_or_reduce_work_group(uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_or_reduce_work_group(uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_or_reduce_work_group(size_t *, const size_t *, size_t, const Group &);

/* xor_reduce_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_xor_reduce_work_group(T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_xor_reduce_work_group(signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_xor_reduce_work_group(unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_xor_reduce_work_group(unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_xor_reduce_work_group(unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_xor_reduce_work_group(unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_xor_reduce_work_group(unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_xor_reduce_work_group(int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_xor_reduce_work_group(int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_xor_reduce_work_group(int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_xor_reduce_work_group(int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_xor_reduce_work_group(uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_xor_reduce_work_group(uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_xor_reduce_work_group(uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_xor_reduce_work_group(uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_xor_reduce_work_group(size_t *, const size_t *, size_t, const Group &);

/* max_reduce_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_max_reduce_work_group(T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_max_reduce_work_group(float *, const float *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_max_reduce_work_group(double *, const double *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_max_reduce_work_group(char *, const char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_max_reduce_work_group(signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_max_reduce_work_group(short *, const short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_max_reduce_work_group(int *, const int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_max_reduce_work_group(long *, const long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_max_reduce_work_group(long long *, const long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_max_reduce_work_group(unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_max_reduce_work_group(unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_max_reduce_work_group(unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_max_reduce_work_group(unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_max_reduce_work_group(unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_max_reduce_work_group(int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_max_reduce_work_group(int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_max_reduce_work_group(int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_max_reduce_work_group(int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_max_reduce_work_group(uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_max_reduce_work_group(uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_max_reduce_work_group(uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_max_reduce_work_group(uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_max_reduce_work_group(size_t *, const size_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_max_reduce_work_group(ptrdiff_t *, const ptrdiff_t *, size_t, const Group &);

/* min_reduce_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_min_reduce_work_group(T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_min_reduce_work_group(float *, const float *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_min_reduce_work_group(double *, const double *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_min_reduce_work_group(char *, const char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_min_reduce_work_group(signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_min_reduce_work_group(short *, const short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_min_reduce_work_group(int *, const int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_min_reduce_work_group(long *, const long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_min_reduce_work_group(long long *, const long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_min_reduce_work_group(unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_min_reduce_work_group(unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_min_reduce_work_group(unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_min_reduce_work_group(unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_min_reduce_work_group(unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_min_reduce_work_group(int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_min_reduce_work_group(int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_min_reduce_work_group(int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_min_reduce_work_group(int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_min_reduce_work_group(uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_min_reduce_work_group(uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_min_reduce_work_group(uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_min_reduce_work_group(uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_min_reduce_work_group(size_t *, const size_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_min_reduce_work_group(ptrdiff_t *, const ptrdiff_t *, size_t, const Group &);

/* sum_reduce_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_sum_reduce_work_group(T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_sum_reduce_work_group(float *, const float *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_sum_reduce_work_group(double *, const double *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_sum_reduce_work_group(char *, const char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_sum_reduce_work_group(signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_sum_reduce_work_group(short *, const short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_sum_reduce_work_group(int *, const int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_sum_reduce_work_group(long *, const long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_sum_reduce_work_group(long long *, const long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_sum_reduce_work_group(unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_sum_reduce_work_group(unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_sum_reduce_work_group(unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_sum_reduce_work_group(unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_sum_reduce_work_group(unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_sum_reduce_work_group(int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_sum_reduce_work_group(int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_sum_reduce_work_group(int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_sum_reduce_work_group(int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_sum_reduce_work_group(uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_sum_reduce_work_group(uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_sum_reduce_work_group(uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_sum_reduce_work_group(uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_sum_reduce_work_group(size_t *, const size_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_sum_reduce_work_group(ptrdiff_t *, const ptrdiff_t *, size_t, const Group &);

/* prod_reduce_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_prod_reduce_work_group(T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_prod_reduce_work_group(float *, const float *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_prod_reduce_work_group(double *, const double *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_prod_reduce_work_group(char *, const char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_prod_reduce_work_group(signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_prod_reduce_work_group(short *, const short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_prod_reduce_work_group(int *, const int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_prod_reduce_work_group(long *, const long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_prod_reduce_work_group(long long *, const long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_prod_reduce_work_group(unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_prod_reduce_work_group(unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_prod_reduce_work_group(unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_prod_reduce_work_group(unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_prod_reduce_work_group(unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_prod_reduce_work_group(int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_prod_reduce_work_group(int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_prod_reduce_work_group(int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_prod_reduce_work_group(int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_prod_reduce_work_group(uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_prod_reduce_work_group(uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_prod_reduce_work_group(uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_prod_reduce_work_group(uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_prod_reduce_work_group(size_t *, const size_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_prod_reduce_work_group(ptrdiff_t *, const ptrdiff_t *, size_t, const Group &);

/* test_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_test_work_group(T *, int, T, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_test_work_group(signed char *, int, signed char, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_test_work_group(int *, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_test_work_group(long *, int, long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_test_work_group(long long *, int, long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_test_work_group(unsigned int *, int, unsigned int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_test_work_group(unsigned long *, int, unsigned long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_test_work_group(unsigned long long *, int, unsigned long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_test_work_group(int32_t *, int, int32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_test_work_group(int64_t *, int, int64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_test_work_group(uint32_t *, int, uint32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_test_work_group(uint64_t *, int, uint64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_test_work_group(size_t *, int, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_test_work_group(ptrdiff_t *, int, ptrdiff_t, const Group &);

/* wait_until_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_wait_until_work_group(T *, int, T, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_schar_wait_until_work_group(signed char *, int, signed char, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_wait_until_work_group(int *, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_wait_until_work_group(long *, int, long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_wait_until_work_group(long long *, int, long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_wait_until_work_group(unsigned int *, int, unsigned int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_wait_until_work_group(unsigned long *, int, unsigned long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_wait_until_work_group(unsigned long long *, int, unsigned long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_wait_until_work_group(int32_t *, int, int32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_wait_until_work_group(int64_t *, int, int64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_wait_until_work_group(uint32_t *, int, uint32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_wait_until_work_group(uint64_t *, int, uint64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_wait_until_work_group(size_t *, int, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_wait_until_work_group(ptrdiff_t *, int, ptrdiff_t, const Group &);

/* barrier_all_work_group */
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_barrier_all_work_group(const Group &);

/* sync_all_work_group */
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_sync_all_work_group(const Group &);

/* fence_work_group */
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_fence_work_group(const Group &);

/* quiet_work_group */
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_quiet_work_group(const Group &);

/* clang-format on */
/* Debugging APIs */
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_nop(void);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_nop_no_r(void);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_debug_test(void);

/* GPU print message extension */
typedef enum {
    DEBUG,
    WARNING,
    ERROR,
    STDOUT,
    STDERR,
} ishmemx_print_msg_type_t;

/* this version uses DEBUG */
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_print(const char *out);
/* this version you specify */
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_print(const char *out, ishmemx_print_msg_type_t msg_type);

/* Timestamp extension */
typedef uintptr_t ishmemx_ts_handle_t;
static inline ishmemx_ts_handle_t ishmemx_ts_handle(unsigned long *pointer)
{
    return (ishmemx_ts_handle_t) pointer;
}

/* call these on device, with pointer to cell in host memory to be filled in with rdtsc value */
/* timestamp waits for a completion signal from the proxy, timestamp_nbi does not */
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_timestamp(ishmemx_ts_handle_t dst);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_timestamp_nbi(ishmemx_ts_handle_t dst);

#endif /* I_SHMEMX_H */
