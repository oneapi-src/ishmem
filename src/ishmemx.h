/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef I_SHMEMX_H
#define I_SHMEMX_H

#include <ishmem.h>
#define ISHMEM_DEVICE_ATTRIBUTES SYCL_EXTERNAL

#define ISHMEMX_TEAM_NODE 2

/* Enumeration of runtimes */
typedef enum : uint8_t {
    ISHMEMX_RUNTIME_MPI,
    ISHMEMX_RUNTIME_OPENSHMEM,
    ISHMEMX_RUNTIME_PMI,
    ISHMEMX_RUNTIME_INVALID,
} ishmemx_runtime_type_t;

typedef struct ishmemx_attr_t {
    /* By default, the runtime is assumed to be MPI */
    ishmemx_runtime_type_t runtime = ISHMEM_DEFAULT_RUNTIME;
    /* By default, runtimes are assumed to be initialized by ISHMEM */
    bool initialize_runtime = true;
    /* By default, gpu is used */
    bool gpu = true;
    /* By default, the base team/comm is uninitialized, representing the default global team/comm */
    union {
        /* TODO: add support for user-provided shmem_team as global team */
        void *mpi_comm = nullptr;
    };
} ishmemx_attr_t;

/* ISHMEMX APIs */
/* Library setup and query routines (host) */
void ishmemx_init_attr(ishmemx_attr_t *attr);
ishmemx_runtime_type_t ishmemx_runtime_get_type();
void ishmemx_query_initialized(int *initialized);

/* clang-format off */
/* put_on_queue */
template <typename T> sycl::event ishmemx_put_on_queue(T *, const T *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_put_on_queue(float *, const float *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_put_on_queue(double *, const double *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_put_on_queue(char *, const char *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_put_on_queue(signed char *, const signed char *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_put_on_queue(short *, const short *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_put_on_queue(int *, const int *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_put_on_queue(long *, const long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_put_on_queue(long long *, const long long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_put_on_queue(unsigned char *, const unsigned char *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_put_on_queue(unsigned short *, const unsigned short *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_put_on_queue(unsigned int *, const unsigned int *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_put_on_queue(unsigned long *, const unsigned long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_put_on_queue(unsigned long long *, const unsigned long long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_put_on_queue(int8_t *, const int8_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_put_on_queue(int16_t *, const int16_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_put_on_queue(int32_t *, const int32_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_put_on_queue(int64_t *, const int64_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_put_on_queue(uint8_t *, const uint8_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_put_on_queue(uint16_t *, const uint16_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_put_on_queue(uint32_t *, const uint32_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_put_on_queue(uint64_t *, const uint64_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_put_on_queue(size_t *, const size_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_put_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put8_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put16_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put32_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put64_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put128_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_putmem_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});

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
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put8_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put16_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put32_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put64_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put128_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_putmem_work_group(void *, const void *, size_t, int, const Group &);

/* iput_on_queue */
template <typename T> sycl::event ishmemx_iput_on_queue(T *, const T *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_iput_on_queue(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_iput_on_queue(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_iput_on_queue(char *, const char *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_iput_on_queue(signed char *, const signed char *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_iput_on_queue(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_iput_on_queue(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_iput_on_queue(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_iput_on_queue(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_iput_on_queue(unsigned char *, const unsigned char *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_iput_on_queue(unsigned short *, const unsigned short *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_iput_on_queue(unsigned int *, const unsigned int *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_iput_on_queue(unsigned long *, const unsigned long *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_iput_on_queue(unsigned long long *, const unsigned long long *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_iput_on_queue(int8_t *, const int8_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_iput_on_queue(int16_t *, const int16_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_iput_on_queue(int32_t *, const int32_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_iput_on_queue(int64_t *, const int64_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_iput_on_queue(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_iput_on_queue(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_iput_on_queue(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_iput_on_queue(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_iput_on_queue(size_t *, const size_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_iput_on_queue(ptrdiff_t *, const ptrdiff_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_iput8_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_iput16_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_iput32_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_iput64_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_iput128_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});

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
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_iput8_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_iput16_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_iput32_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_iput64_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_iput128_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);

/* ibput */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibput(T *, const T *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_float_ibput(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_double_ibput(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_char_ibput(char *, const char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_schar_ibput(signed char *, const signed char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_short_ibput(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_ibput(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_ibput(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_ibput(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uchar_ibput(unsigned char *, const unsigned char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ushort_ibput(unsigned short *, const unsigned short *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_ibput(unsigned int *, const unsigned int *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_ibput(unsigned long *, const unsigned long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_ibput(unsigned long long *, const unsigned long long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int8_ibput(int8_t *, const int8_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int16_ibput(int16_t *, const int16_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_ibput(int32_t *, const int32_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_ibput(int64_t *, const int64_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint8_ibput(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint16_ibput(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_ibput(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_ibput(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_ibput(size_t *, const size_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_ibput(ptrdiff_t *, const ptrdiff_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibput8(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibput16(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibput32(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibput64(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibput128(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);

/* ibput_on_queue */
template <typename T> sycl::event ishmemx_ibput_on_queue(T *, const T *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_ibput_on_queue(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_ibput_on_queue(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_ibput_on_queue(char *, const char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_ibput_on_queue(signed char *, const signed char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_ibput_on_queue(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_ibput_on_queue(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_ibput_on_queue(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_ibput_on_queue(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_ibput_on_queue(unsigned char *, const unsigned char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_ibput_on_queue(unsigned short *, const unsigned short *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_ibput_on_queue(unsigned int *, const unsigned int *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_ibput_on_queue(unsigned long *, const unsigned long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_ibput_on_queue(unsigned long long *, const unsigned long long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_ibput_on_queue(int8_t *, const int8_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_ibput_on_queue(int16_t *, const int16_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_ibput_on_queue(int32_t *, const int32_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_ibput_on_queue(int64_t *, const int64_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_ibput_on_queue(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_ibput_on_queue(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_ibput_on_queue(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_ibput_on_queue(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_ibput_on_queue(size_t *, const size_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_ibput_on_queue(ptrdiff_t *, const ptrdiff_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ibput8_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ibput16_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ibput32_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ibput64_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ibput128_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});

/* ibput_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibput_work_group(T *, const T *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_float_ibput_work_group(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_double_ibput_work_group(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_char_ibput_work_group(char *, const char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_schar_ibput_work_group(signed char *, const signed char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_short_ibput_work_group(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_ibput_work_group(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_ibput_work_group(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_ibput_work_group(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uchar_ibput_work_group(unsigned char *, const unsigned char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ushort_ibput_work_group(unsigned short *, const unsigned short *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_ibput_work_group(unsigned int *, const unsigned int *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_ibput_work_group(unsigned long *, const unsigned long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_ibput_work_group(unsigned long long *, const unsigned long long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int8_ibput_work_group(int8_t *, const int8_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int16_ibput_work_group(int16_t *, const int16_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_ibput_work_group(int32_t *, const int32_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_ibput_work_group(int64_t *, const int64_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint8_ibput_work_group(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint16_ibput_work_group(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_ibput_work_group(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_ibput_work_group(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_ibput_work_group(size_t *, const size_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_ibput_work_group(ptrdiff_t *, const ptrdiff_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibput8_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibput16_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibput32_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibput64_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibput128_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);

/* get_on_queue */
template <typename T> sycl::event ishmemx_get_on_queue(T *, const T *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_get_on_queue(float *, const float *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_get_on_queue(double *, const double *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_get_on_queue(char *, const char *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_get_on_queue(signed char *, const signed char *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_get_on_queue(short *, const short *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_get_on_queue(int *, const int *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_get_on_queue(long *, const long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_get_on_queue(long long *, const long long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_get_on_queue(unsigned char *, const unsigned char *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_get_on_queue(unsigned short *, const unsigned short *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_get_on_queue(unsigned int *, const unsigned int *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_get_on_queue(unsigned long *, const unsigned long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_get_on_queue(unsigned long long *, const unsigned long long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_get_on_queue(int8_t *, const int8_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_get_on_queue(int16_t *, const int16_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_get_on_queue(int32_t *, const int32_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_get_on_queue(int64_t *, const int64_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_get_on_queue(uint8_t *, const uint8_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_get_on_queue(uint16_t *, const uint16_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_get_on_queue(uint32_t *, const uint32_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_get_on_queue(uint64_t *, const uint64_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_get_on_queue(size_t *, const size_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_get_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_get8_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_get16_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_get32_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_get64_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_get128_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_getmem_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});

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
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_get8_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_get16_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_get32_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_get64_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_get128_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_getmem_work_group(void *, const void *, size_t, int, const Group &);

/* iget_on_queue */
template <typename T> sycl::event ishmemx_iget_on_queue(T *, const T *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_iget_on_queue(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_iget_on_queue(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_iget_on_queue(char *, const char *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_iget_on_queue(signed char *, const signed char *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_iget_on_queue(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_iget_on_queue(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_iget_on_queue(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_iget_on_queue(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_iget_on_queue(unsigned char *, const unsigned char *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_iget_on_queue(unsigned short *, const unsigned short *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_iget_on_queue(unsigned int *, const unsigned int *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_iget_on_queue(unsigned long *, const unsigned long *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_iget_on_queue(unsigned long long *, const unsigned long long *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_iget_on_queue(int8_t *, const int8_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_iget_on_queue(int16_t *, const int16_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_iget_on_queue(int32_t *, const int32_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_iget_on_queue(int64_t *, const int64_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_iget_on_queue(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_iget_on_queue(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_iget_on_queue(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_iget_on_queue(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_iget_on_queue(size_t *, const size_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_iget_on_queue(ptrdiff_t *, const ptrdiff_t *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_iget8_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_iget16_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_iget32_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_iget64_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_iget128_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});

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
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_iget8_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_iget16_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_iget32_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_iget64_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_iget128_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, const Group &);

/* ibget */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibget(T *, const T *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_float_ibget(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_double_ibget(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_char_ibget(char *, const char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_schar_ibget(signed char *, const signed char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_short_ibget(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_ibget(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_ibget(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_ibget(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uchar_ibget(unsigned char *, const unsigned char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ushort_ibget(unsigned short *, const unsigned short *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_ibget(unsigned int *, const unsigned int *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_ibget(unsigned long *, const unsigned long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_ibget(unsigned long long *, const unsigned long long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int8_ibget(int8_t *, const int8_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int16_ibget(int16_t *, const int16_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_ibget(int32_t *, const int32_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_ibget(int64_t *, const int64_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint8_ibget(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint16_ibget(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_ibget(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_ibget(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_ibget(size_t *, const size_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_ibget(ptrdiff_t *, const ptrdiff_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibget8(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibget16(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibget32(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibget64(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibget128(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int);

/* ibget_on_queue */
template <typename T> sycl::event ishmemx_ibget_on_queue(T *, const T *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_ibget_on_queue(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_ibget_on_queue(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_ibget_on_queue(char *, const char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_ibget_on_queue(signed char *, const signed char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_ibget_on_queue(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_ibget_on_queue(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_ibget_on_queue(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_ibget_on_queue(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_ibget_on_queue(unsigned char *, const unsigned char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_ibget_on_queue(unsigned short *, const unsigned short *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_ibget_on_queue(unsigned int *, const unsigned int *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_ibget_on_queue(unsigned long *, const unsigned long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_ibget_on_queue(unsigned long long *, const unsigned long long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_ibget_on_queue(int8_t *, const int8_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_ibget_on_queue(int16_t *, const int16_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_ibget_on_queue(int32_t *, const int32_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_ibget_on_queue(int64_t *, const int64_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_ibget_on_queue(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_ibget_on_queue(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_ibget_on_queue(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_ibget_on_queue(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_ibget_on_queue(size_t *, const size_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_ibget_on_queue(ptrdiff_t *, const ptrdiff_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ibget8_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ibget16_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ibget32_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ibget64_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ibget128_on_queue(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});

/* ibget_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibget_work_group(T *, const T *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_float_ibget_work_group(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_double_ibget_work_group(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_char_ibget_work_group(char *, const char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_schar_ibget_work_group(signed char *, const signed char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_short_ibget_work_group(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_ibget_work_group(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_ibget_work_group(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_ibget_work_group(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uchar_ibget_work_group(unsigned char *, const unsigned char *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ushort_ibget_work_group(unsigned short *, const unsigned short *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_ibget_work_group(unsigned int *, const unsigned int *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_ibget_work_group(unsigned long *, const unsigned long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_ibget_work_group(unsigned long long *, const unsigned long long *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int8_ibget_work_group(int8_t *, const int8_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int16_ibget_work_group(int16_t *, const int16_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_ibget_work_group(int32_t *, const int32_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_ibget_work_group(int64_t *, const int64_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint8_ibget_work_group(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint16_ibget_work_group(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_ibget_work_group(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_ibget_work_group(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_ibget_work_group(size_t *, const size_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_ibget_work_group(ptrdiff_t *, const ptrdiff_t *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibget8_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibget16_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibget32_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibget64_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ibget128_work_group(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, size_t, int, const Group &);

/* put_nbi_on_queue */
template <typename T> sycl::event ishmemx_put_nbi_on_queue(T *, const T *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_put_nbi_on_queue(float *, const float *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_put_nbi_on_queue(double *, const double *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_put_nbi_on_queue(char *, const char *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_put_nbi_on_queue(signed char *, const signed char *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_put_nbi_on_queue(short *, const short *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_put_nbi_on_queue(int *, const int *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_put_nbi_on_queue(long *, const long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_put_nbi_on_queue(long long *, const long long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_put_nbi_on_queue(unsigned char *, const unsigned char *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_put_nbi_on_queue(unsigned short *, const unsigned short *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_put_nbi_on_queue(unsigned int *, const unsigned int *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_put_nbi_on_queue(unsigned long *, const unsigned long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_put_nbi_on_queue(unsigned long long *, const unsigned long long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_put_nbi_on_queue(int8_t *, const int8_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_put_nbi_on_queue(int16_t *, const int16_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_put_nbi_on_queue(int32_t *, const int32_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_put_nbi_on_queue(int64_t *, const int64_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_put_nbi_on_queue(uint8_t *, const uint8_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_put_nbi_on_queue(uint16_t *, const uint16_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_put_nbi_on_queue(uint32_t *, const uint32_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_put_nbi_on_queue(uint64_t *, const uint64_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_put_nbi_on_queue(size_t *, const size_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_put_nbi_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put8_nbi_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put16_nbi_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put32_nbi_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put64_nbi_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put128_nbi_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_putmem_nbi_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});

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
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put8_nbi_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put16_nbi_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put32_nbi_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put64_nbi_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put128_nbi_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_putmem_nbi_work_group(void *, const void *, size_t, int, const Group &);

/* get_nbi_on_queue */
template <typename T> sycl::event ishmemx_get_nbi_on_queue(T *, const T *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_get_nbi_on_queue(float *, const float *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_get_nbi_on_queue(double *, const double *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_get_nbi_on_queue(char *, const char *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_get_nbi_on_queue(signed char *, const signed char *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_get_nbi_on_queue(short *, const short *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_get_nbi_on_queue(int *, const int *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_get_nbi_on_queue(long *, const long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_get_nbi_on_queue(long long *, const long long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_get_nbi_on_queue(unsigned char *, const unsigned char *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_get_nbi_on_queue(unsigned short *, const unsigned short *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_get_nbi_on_queue(unsigned int *, const unsigned int *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_get_nbi_on_queue(unsigned long *, const unsigned long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_get_nbi_on_queue(unsigned long long *, const unsigned long long *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_get_nbi_on_queue(int8_t *, const int8_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_get_nbi_on_queue(int16_t *, const int16_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_get_nbi_on_queue(int32_t *, const int32_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_get_nbi_on_queue(int64_t *, const int64_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_get_nbi_on_queue(uint8_t *, const uint8_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_get_nbi_on_queue(uint16_t *, const uint16_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_get_nbi_on_queue(uint32_t *, const uint32_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_get_nbi_on_queue(uint64_t *, const uint64_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_get_nbi_on_queue(size_t *, const size_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_get_nbi_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_get8_nbi_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_get16_nbi_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_get32_nbi_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_get64_nbi_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_get128_nbi_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_getmem_nbi_on_queue(void *, const void *, size_t, int, sycl::queue &, const std::vector<sycl::event> & = {});

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
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_get8_nbi_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_get16_nbi_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_get32_nbi_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_get64_nbi_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_get128_nbi_work_group(void *, const void *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_getmem_nbi_work_group(void *, const void *, size_t, int, const Group &);

/* put_signal_on_queue */
template <typename T> sycl::event ishmemx_put_signal_on_queue(T *, const T *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_put_signal_on_queue(float *, const float *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_put_signal_on_queue(double *, const double *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_put_signal_on_queue(char *, const char *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_put_signal_on_queue(signed char *, const signed char *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_put_signal_on_queue(short *, const short *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_put_signal_on_queue(int *, const int *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_put_signal_on_queue(long *, const long *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_put_signal_on_queue(long long *, const long long *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_put_signal_on_queue(unsigned char *, const unsigned char *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_put_signal_on_queue(unsigned short *, const unsigned short *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_put_signal_on_queue(unsigned int *, const unsigned int *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_put_signal_on_queue(unsigned long *, const unsigned long *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_put_signal_on_queue(unsigned long long *, const unsigned long long *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_put_signal_on_queue(int8_t *, const int8_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_put_signal_on_queue(int16_t *, const int16_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_put_signal_on_queue(int32_t *, const int32_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_put_signal_on_queue(int64_t *, const int64_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_put_signal_on_queue(uint8_t *, const uint8_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_put_signal_on_queue(uint16_t *, const uint16_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_put_signal_on_queue(uint32_t *, const uint32_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_put_signal_on_queue(uint64_t *, const uint64_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_put_signal_on_queue(size_t *, const size_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_put_signal_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put8_signal_on_queue(void *, const void *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put16_signal_on_queue(void *, const void *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put32_signal_on_queue(void *, const void *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put64_signal_on_queue(void *, const void *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put128_signal_on_queue(void *, const void *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_putmem_signal_on_queue(void *, const void *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});

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
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put8_signal_work_group(void *, const void *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put16_signal_work_group(void *, const void *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put32_signal_work_group(void *, const void *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put64_signal_work_group(void *, const void *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put128_signal_work_group(void *, const void *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_putmem_signal_work_group(void *, const void *, size_t, uint64_t *, uint64_t, int, int, const Group &);

/* put_signal_nbi_on_queue */
template <typename T> sycl::event ishmemx_put_signal_nbi_on_queue(T *, const T *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_put_signal_nbi_on_queue(float *, const float *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_put_signal_nbi_on_queue(double *, const double *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_put_signal_nbi_on_queue(char *, const char *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_put_signal_nbi_on_queue(signed char *, const signed char *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_put_signal_nbi_on_queue(short *, const short *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_put_signal_nbi_on_queue(int *, const int *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_put_signal_nbi_on_queue(long *, const long *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_put_signal_nbi_on_queue(long long *, const long long *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_put_signal_nbi_on_queue(unsigned char *, const unsigned char *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_put_signal_nbi_on_queue(unsigned short *, const unsigned short *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_put_signal_nbi_on_queue(unsigned int *, const unsigned int *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_put_signal_nbi_on_queue(unsigned long *, const unsigned long *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_put_signal_nbi_on_queue(unsigned long long *, const unsigned long long *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_put_signal_nbi_on_queue(int8_t *, const int8_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_put_signal_nbi_on_queue(int16_t *, const int16_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_put_signal_nbi_on_queue(int32_t *, const int32_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_put_signal_nbi_on_queue(int64_t *, const int64_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_put_signal_nbi_on_queue(uint8_t *, const uint8_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_put_signal_nbi_on_queue(uint16_t *, const uint16_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_put_signal_nbi_on_queue(uint32_t *, const uint32_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_put_signal_nbi_on_queue(uint64_t *, const uint64_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_put_signal_nbi_on_queue(size_t *, const size_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_put_signal_nbi_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put8_signal_nbi_on_queue(void *, const void *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put16_signal_nbi_on_queue(void *, const void *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put32_signal_nbi_on_queue(void *, const void *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put64_signal_nbi_on_queue(void *, const void *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_put128_signal_nbi_on_queue(void *, const void *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_putmem_signal_nbi_on_queue(void *, const void *, size_t, uint64_t *, uint64_t, int, int, sycl::queue &, const std::vector<sycl::event> & = {});

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
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put8_signal_nbi_work_group(void *, const void *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put16_signal_nbi_work_group(void *, const void *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put32_signal_nbi_work_group(void *, const void *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put64_signal_nbi_work_group(void *, const void *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_put128_signal_nbi_work_group(void *, const void *, size_t, uint64_t *, uint64_t, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_putmem_signal_nbi_work_group(void *, const void *, size_t, uint64_t *, uint64_t, int, int, const Group &);

/* Signal OPs */
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_signal_add(uint64_t *, uint64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_signal_set(uint64_t *, uint64_t, int);

/* alltoall_on_queue */
template <typename T> sycl::event ishmemx_alltoall_on_queue(T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_alltoall_on_queue(float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_alltoall_on_queue(double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_alltoall_on_queue(char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_alltoall_on_queue(signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_alltoall_on_queue(short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_alltoall_on_queue(int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_alltoall_on_queue(long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_alltoall_on_queue(long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_alltoall_on_queue(unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_alltoall_on_queue(unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_alltoall_on_queue(unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_alltoall_on_queue(unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_alltoall_on_queue(unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_alltoall_on_queue(int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_alltoall_on_queue(int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_alltoall_on_queue(int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_alltoall_on_queue(int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_alltoall_on_queue(uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_alltoall_on_queue(uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_alltoall_on_queue(uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_alltoall_on_queue(uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_alltoall_on_queue(size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_alltoall_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_alltoallmem_on_queue(void *, const void *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

/* alltoall_on_queue on a team */
template <typename T> sycl::event ishmemx_alltoall_on_queue(ishmem_team_t, T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_alltoall_on_queue(ishmem_team_t, float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_alltoall_on_queue(ishmem_team_t, double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_alltoall_on_queue(ishmem_team_t, char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_alltoall_on_queue(ishmem_team_t, signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_alltoall_on_queue(ishmem_team_t, short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_alltoall_on_queue(ishmem_team_t, int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_alltoall_on_queue(ishmem_team_t, long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_alltoall_on_queue(ishmem_team_t, long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_alltoall_on_queue(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_alltoall_on_queue(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_alltoall_on_queue(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_alltoall_on_queue(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_alltoall_on_queue(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_alltoall_on_queue(ishmem_team_t, int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_alltoall_on_queue(ishmem_team_t, int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_alltoall_on_queue(ishmem_team_t, int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_alltoall_on_queue(ishmem_team_t, int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_alltoall_on_queue(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_alltoall_on_queue(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_alltoall_on_queue(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_alltoall_on_queue(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_alltoall_on_queue(ishmem_team_t, size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_alltoall_on_queue(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_alltoallmem_on_queue(ishmem_team_t, void *, const void *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

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

/* alltoall_work_group on a team */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_alltoall_work_group(ishmem_team_t, T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_alltoall_work_group(ishmem_team_t, float *, const float *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_alltoall_work_group(ishmem_team_t, double *, const double *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_alltoall_work_group(ishmem_team_t, char *, const char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_alltoall_work_group(ishmem_team_t, signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_alltoall_work_group(ishmem_team_t, short *, const short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_alltoall_work_group(ishmem_team_t, int *, const int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_alltoall_work_group(ishmem_team_t, long *, const long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_alltoall_work_group(ishmem_team_t, long long *, const long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_alltoall_work_group(ishmem_team_t, unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_alltoall_work_group(ishmem_team_t, unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_alltoall_work_group(ishmem_team_t, unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_alltoall_work_group(ishmem_team_t, unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_alltoall_work_group(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_alltoall_work_group(ishmem_team_t, int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_alltoall_work_group(ishmem_team_t, int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_alltoall_work_group(ishmem_team_t, int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_alltoall_work_group(ishmem_team_t, int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_alltoall_work_group(ishmem_team_t, uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_alltoall_work_group(ishmem_team_t, uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_alltoall_work_group(ishmem_team_t, uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_alltoall_work_group(ishmem_team_t, uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_alltoall_work_group(ishmem_team_t, size_t *, const size_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_alltoall_work_group(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_alltoallmem_work_group(ishmem_team_t, void *, const void *, size_t, const Group &);

/* broadcast_on_queue */
template <typename T> sycl::event ishmemx_broadcast_on_queue(T *, const T *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_broadcast_on_queue(float *, const float *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_broadcast_on_queue(double *, const double *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_broadcast_on_queue(char *, const char *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_broadcast_on_queue(signed char *, const signed char *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_broadcast_on_queue(short *, const short *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_broadcast_on_queue(int *, const int *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_broadcast_on_queue(long *, const long *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_broadcast_on_queue(long long *, const long long *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_broadcast_on_queue(unsigned char *, const unsigned char *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_broadcast_on_queue(unsigned short *, const unsigned short *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_broadcast_on_queue(unsigned int *, const unsigned int *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_broadcast_on_queue(unsigned long *, const unsigned long *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_broadcast_on_queue(unsigned long long *, const unsigned long long *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_broadcast_on_queue(int8_t *, const int8_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_broadcast_on_queue(int16_t *, const int16_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_broadcast_on_queue(int32_t *, const int32_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_broadcast_on_queue(int64_t *, const int64_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_broadcast_on_queue(uint8_t *, const uint8_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_broadcast_on_queue(uint16_t *, const uint16_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_broadcast_on_queue(uint32_t *, const uint32_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_broadcast_on_queue(uint64_t *, const uint64_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_broadcast_on_queue(size_t *, const size_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_broadcast_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_broadcastmem_on_queue(void *, const void *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});

/* broadcast_on_queue on a team */
template <typename T> sycl::event ishmemx_broadcast_on_queue(ishmem_team_t, T *, const T *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_broadcast_on_queue(ishmem_team_t, float *, const float *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_broadcast_on_queue(ishmem_team_t, double *, const double *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_broadcast_on_queue(ishmem_team_t, char *, const char *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_broadcast_on_queue(ishmem_team_t, signed char *, const signed char *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_broadcast_on_queue(ishmem_team_t, short *, const short *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_broadcast_on_queue(ishmem_team_t, int *, const int *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_broadcast_on_queue(ishmem_team_t, long *, const long *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_broadcast_on_queue(ishmem_team_t, long long *, const long long *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_broadcast_on_queue(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_broadcast_on_queue(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_broadcast_on_queue(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_broadcast_on_queue(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_broadcast_on_queue(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_broadcast_on_queue(ishmem_team_t, int8_t *, const int8_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_broadcast_on_queue(ishmem_team_t, int16_t *, const int16_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_broadcast_on_queue(ishmem_team_t, int32_t *, const int32_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_broadcast_on_queue(ishmem_team_t, int64_t *, const int64_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_broadcast_on_queue(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_broadcast_on_queue(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_broadcast_on_queue(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_broadcast_on_queue(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_broadcast_on_queue(ishmem_team_t, size_t *, const size_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_broadcast_on_queue(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_broadcastmem_on_queue(ishmem_team_t, void *, const void *, size_t, int, int *, sycl::queue &, const std::vector<sycl::event> & = {});

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

/* broadcast_work_group on a team */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_broadcast_work_group(ishmem_team_t, T *, const T *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_broadcast_work_group(ishmem_team_t, float *, const float *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_broadcast_work_group(ishmem_team_t, double *, const double *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_broadcast_work_group(ishmem_team_t, char *, const char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_broadcast_work_group(ishmem_team_t, signed char *, const signed char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_broadcast_work_group(ishmem_team_t, short *, const short *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_broadcast_work_group(ishmem_team_t, int *, const int *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_broadcast_work_group(ishmem_team_t, long *, const long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_broadcast_work_group(ishmem_team_t, long long *, const long long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_broadcast_work_group(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_broadcast_work_group(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_broadcast_work_group(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_broadcast_work_group(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_broadcast_work_group(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_broadcast_work_group(ishmem_team_t, int8_t *, const int8_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_broadcast_work_group(ishmem_team_t, int16_t *, const int16_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_broadcast_work_group(ishmem_team_t, int32_t *, const int32_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_broadcast_work_group(ishmem_team_t, int64_t *, const int64_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_broadcast_work_group(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_broadcast_work_group(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_broadcast_work_group(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_broadcast_work_group(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_broadcast_work_group(ishmem_team_t, size_t *, const size_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_broadcast_work_group(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_broadcastmem_work_group(ishmem_team_t, void *, const void *, size_t, int, const Group &);

/* collect_on_queue */
template <typename T> sycl::event ishmemx_collect_on_queue(T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_collect_on_queue(float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_collect_on_queue(double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_collect_on_queue(char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_collect_on_queue(signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_collect_on_queue(short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_collect_on_queue(int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_collect_on_queue(long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_collect_on_queue(long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_collect_on_queue(unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_collect_on_queue(unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_collect_on_queue(unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_collect_on_queue(unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_collect_on_queue(unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_collect_on_queue(int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_collect_on_queue(int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_collect_on_queue(int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_collect_on_queue(int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_collect_on_queue(uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_collect_on_queue(uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_collect_on_queue(uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_collect_on_queue(uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_collect_on_queue(size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_collect_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_collectmem_on_queue(void *, const void *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

/* collect_on_queue on a team */
template <typename T> sycl::event ishmemx_collect_on_queue(ishmem_team_t, T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_collect_on_queue(ishmem_team_t, float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_collect_on_queue(ishmem_team_t, double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_collect_on_queue(ishmem_team_t, char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_collect_on_queue(ishmem_team_t, signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_collect_on_queue(ishmem_team_t, short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_collect_on_queue(ishmem_team_t, int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_collect_on_queue(ishmem_team_t, long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_collect_on_queue(ishmem_team_t, long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_collect_on_queue(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_collect_on_queue(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_collect_on_queue(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_collect_on_queue(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_collect_on_queue(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_collect_on_queue(ishmem_team_t, int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_collect_on_queue(ishmem_team_t, int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_collect_on_queue(ishmem_team_t, int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_collect_on_queue(ishmem_team_t, int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_collect_on_queue(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_collect_on_queue(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_collect_on_queue(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_collect_on_queue(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_collect_on_queue(ishmem_team_t, size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_collect_on_queue(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_collectmem_on_queue(ishmem_team_t, void *, const void *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

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

/* collect_work_group on a team */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_collect_work_group(ishmem_team_t, T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_collect_work_group(ishmem_team_t, float *, const float *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_collect_work_group(ishmem_team_t, double *, const double *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_collect_work_group(ishmem_team_t, char *, const char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_collect_work_group(ishmem_team_t, signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_collect_work_group(ishmem_team_t, short *, const short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_collect_work_group(ishmem_team_t, int *, const int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_collect_work_group(ishmem_team_t, long *, const long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_collect_work_group(ishmem_team_t, long long *, const long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_collect_work_group(ishmem_team_t, unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_collect_work_group(ishmem_team_t, unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_collect_work_group(ishmem_team_t, unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_collect_work_group(ishmem_team_t, unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_collect_work_group(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_collect_work_group(ishmem_team_t, int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_collect_work_group(ishmem_team_t, int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_collect_work_group(ishmem_team_t, int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_collect_work_group(ishmem_team_t, int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_collect_work_group(ishmem_team_t, uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_collect_work_group(ishmem_team_t, uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_collect_work_group(ishmem_team_t, uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_collect_work_group(ishmem_team_t, uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_collect_work_group(ishmem_team_t, size_t *, const size_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_collect_work_group(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_collectmem_work_group(ishmem_team_t, void *, const void *, size_t, const Group &);

/* fcollect_on_queue */
template <typename T> sycl::event ishmemx_fcollect_on_queue(T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_fcollect_on_queue(float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_fcollect_on_queue(double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_fcollect_on_queue(char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_fcollect_on_queue(signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_fcollect_on_queue(short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_fcollect_on_queue(int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_fcollect_on_queue(long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_fcollect_on_queue(long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_fcollect_on_queue(unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_fcollect_on_queue(unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_fcollect_on_queue(unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_fcollect_on_queue(unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_fcollect_on_queue(unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_fcollect_on_queue(int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_fcollect_on_queue(int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_fcollect_on_queue(int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_fcollect_on_queue(int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_fcollect_on_queue(uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_fcollect_on_queue(uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_fcollect_on_queue(uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_fcollect_on_queue(uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_fcollect_on_queue(size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_fcollect_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_fcollectmem_on_queue(void *, const void *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

/* fcollect_on_queue on a team */
template <typename T> sycl::event ishmemx_fcollect_on_queue(ishmem_team_t, T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_fcollect_on_queue(ishmem_team_t, float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_fcollect_on_queue(ishmem_team_t, double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_fcollect_on_queue(ishmem_team_t, char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_fcollect_on_queue(ishmem_team_t, signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_fcollect_on_queue(ishmem_team_t, short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_fcollect_on_queue(ishmem_team_t, int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_fcollect_on_queue(ishmem_team_t, long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_fcollect_on_queue(ishmem_team_t, long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_fcollect_on_queue(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_fcollect_on_queue(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_fcollect_on_queue(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_fcollect_on_queue(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_fcollect_on_queue(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_fcollect_on_queue(ishmem_team_t, int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_fcollect_on_queue(ishmem_team_t, int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_fcollect_on_queue(ishmem_team_t, int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_fcollect_on_queue(ishmem_team_t, int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_fcollect_on_queue(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_fcollect_on_queue(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_fcollect_on_queue(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_fcollect_on_queue(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_fcollect_on_queue(ishmem_team_t, size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_fcollect_on_queue(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_fcollectmem_on_queue(ishmem_team_t, void *, const void *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

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

/* fcollect_work_group on a team */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_fcollect_work_group(ishmem_team_t, T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_fcollect_work_group(ishmem_team_t, float *, const float *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_fcollect_work_group(ishmem_team_t, double *, const double *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_fcollect_work_group(ishmem_team_t, char *, const char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_fcollect_work_group(ishmem_team_t, signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_fcollect_work_group(ishmem_team_t, short *, const short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_fcollect_work_group(ishmem_team_t, int *, const int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_fcollect_work_group(ishmem_team_t, long *, const long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_fcollect_work_group(ishmem_team_t, long long *, const long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_fcollect_work_group(ishmem_team_t, unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_fcollect_work_group(ishmem_team_t, unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_fcollect_work_group(ishmem_team_t, unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_fcollect_work_group(ishmem_team_t, unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_fcollect_work_group(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_fcollect_work_group(ishmem_team_t, int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_fcollect_work_group(ishmem_team_t, int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_fcollect_work_group(ishmem_team_t, int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_fcollect_work_group(ishmem_team_t, int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_fcollect_work_group(ishmem_team_t, uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_fcollect_work_group(ishmem_team_t, uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_fcollect_work_group(ishmem_team_t, uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_fcollect_work_group(ishmem_team_t, uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_fcollect_work_group(ishmem_team_t, size_t *, const size_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_fcollect_work_group(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_fcollectmem_work_group(ishmem_team_t, void *, const void *, size_t, const Group &);

/* and_reduce_on_queue */
template <typename T> sycl::event ishmemx_and_reduce_on_queue(T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_and_reduce_on_queue(signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_and_reduce_on_queue(unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_and_reduce_on_queue(unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_and_reduce_on_queue(unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_and_reduce_on_queue(unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_and_reduce_on_queue(unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_and_reduce_on_queue(int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_and_reduce_on_queue(int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_and_reduce_on_queue(int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_and_reduce_on_queue(int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_and_reduce_on_queue(uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_and_reduce_on_queue(uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_and_reduce_on_queue(uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_and_reduce_on_queue(uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_and_reduce_on_queue(size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

/* and_reduce_on_queue on a team */
template <typename T> sycl::event ishmemx_and_reduce_on_queue(ishmem_team_t, T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_and_reduce_on_queue(ishmem_team_t, signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_and_reduce_on_queue(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_and_reduce_on_queue(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_and_reduce_on_queue(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_and_reduce_on_queue(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_and_reduce_on_queue(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_and_reduce_on_queue(ishmem_team_t, int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_and_reduce_on_queue(ishmem_team_t, int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_and_reduce_on_queue(ishmem_team_t, int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_and_reduce_on_queue(ishmem_team_t, int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_and_reduce_on_queue(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_and_reduce_on_queue(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_and_reduce_on_queue(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_and_reduce_on_queue(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_and_reduce_on_queue(ishmem_team_t, size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

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

/* and_reduce_work_group on a team */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_and_reduce_work_group(ishmem_team_t, T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_and_reduce_work_group(ishmem_team_t, signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_and_reduce_work_group(ishmem_team_t, unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_and_reduce_work_group(ishmem_team_t, unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_and_reduce_work_group(ishmem_team_t, unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_and_reduce_work_group(ishmem_team_t, unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_and_reduce_work_group(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_and_reduce_work_group(ishmem_team_t, int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_and_reduce_work_group(ishmem_team_t, int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_and_reduce_work_group(ishmem_team_t, int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_and_reduce_work_group(ishmem_team_t, int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_and_reduce_work_group(ishmem_team_t, uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_and_reduce_work_group(ishmem_team_t, uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_and_reduce_work_group(ishmem_team_t, uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_and_reduce_work_group(ishmem_team_t, uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_and_reduce_work_group(ishmem_team_t, size_t *, const size_t *, size_t, const Group &);

/* or_reduce_on_queue */
template <typename T> sycl::event ishmemx_or_reduce_on_queue(T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_or_reduce_on_queue(signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_or_reduce_on_queue(unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_or_reduce_on_queue(unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_or_reduce_on_queue(unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_or_reduce_on_queue(unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_or_reduce_on_queue(unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_or_reduce_on_queue(int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_or_reduce_on_queue(int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_or_reduce_on_queue(int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_or_reduce_on_queue(int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_or_reduce_on_queue(uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_or_reduce_on_queue(uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_or_reduce_on_queue(uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_or_reduce_on_queue(uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_or_reduce_on_queue(size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

/* or_reduce_on_queue on a team */
template <typename T> sycl::event ishmemx_or_reduce_on_queue(ishmem_team_t, T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_or_reduce_on_queue(ishmem_team_t, signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_or_reduce_on_queue(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_or_reduce_on_queue(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_or_reduce_on_queue(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_or_reduce_on_queue(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_or_reduce_on_queue(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_or_reduce_on_queue(ishmem_team_t, int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_or_reduce_on_queue(ishmem_team_t, int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_or_reduce_on_queue(ishmem_team_t, int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_or_reduce_on_queue(ishmem_team_t, int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_or_reduce_on_queue(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_or_reduce_on_queue(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_or_reduce_on_queue(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_or_reduce_on_queue(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_or_reduce_on_queue(ishmem_team_t, size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

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

/* or_reduce_work_group on a team */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_or_reduce_work_group(ishmem_team_t, T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_or_reduce_work_group(ishmem_team_t, signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_or_reduce_work_group(ishmem_team_t, unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_or_reduce_work_group(ishmem_team_t, unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_or_reduce_work_group(ishmem_team_t, unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_or_reduce_work_group(ishmem_team_t, unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_or_reduce_work_group(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_or_reduce_work_group(ishmem_team_t, int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_or_reduce_work_group(ishmem_team_t, int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_or_reduce_work_group(ishmem_team_t, int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_or_reduce_work_group(ishmem_team_t, int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_or_reduce_work_group(ishmem_team_t, uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_or_reduce_work_group(ishmem_team_t, uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_or_reduce_work_group(ishmem_team_t, uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_or_reduce_work_group(ishmem_team_t, uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_or_reduce_work_group(ishmem_team_t, size_t *, const size_t *, size_t, const Group &);

/* xor_reduce_on_queue */
template <typename T> sycl::event ishmemx_xor_reduce_on_queue(T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_xor_reduce_on_queue(signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_xor_reduce_on_queue(unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_xor_reduce_on_queue(unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_xor_reduce_on_queue(unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_xor_reduce_on_queue(unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_xor_reduce_on_queue(unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_xor_reduce_on_queue(int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_xor_reduce_on_queue(int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_xor_reduce_on_queue(int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_xor_reduce_on_queue(int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_xor_reduce_on_queue(uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_xor_reduce_on_queue(uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_xor_reduce_on_queue(uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_xor_reduce_on_queue(uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_xor_reduce_on_queue(size_t *, const size_t *, size_t, int *, sycl::queue&, const std::vector<sycl::event> & = {});

/* xor_reduce_on_queue on a team */
template <typename T> sycl::event ishmemx_xor_reduce_on_queue(ishmem_team_t, T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_xor_reduce_on_queue(ishmem_team_t, signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_xor_reduce_on_queue(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_xor_reduce_on_queue(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_xor_reduce_on_queue(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_xor_reduce_on_queue(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_xor_reduce_on_queue(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_xor_reduce_on_queue(ishmem_team_t, int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_xor_reduce_on_queue(ishmem_team_t, int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_xor_reduce_on_queue(ishmem_team_t, int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_xor_reduce_on_queue(ishmem_team_t, int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_xor_reduce_on_queue(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_xor_reduce_on_queue(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_xor_reduce_on_queue(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_xor_reduce_on_queue(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_xor_reduce_on_queue(ishmem_team_t, size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

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

/* xor_reduce_work_group on a team */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_xor_reduce_work_group(ishmem_team_t, T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_xor_reduce_work_group(ishmem_team_t, signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_xor_reduce_work_group(ishmem_team_t, unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_xor_reduce_work_group(ishmem_team_t, unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_xor_reduce_work_group(ishmem_team_t, unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_xor_reduce_work_group(ishmem_team_t, unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_xor_reduce_work_group(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_xor_reduce_work_group(ishmem_team_t, int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_xor_reduce_work_group(ishmem_team_t, int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_xor_reduce_work_group(ishmem_team_t, int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_xor_reduce_work_group(ishmem_team_t, int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_xor_reduce_work_group(ishmem_team_t, uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_xor_reduce_work_group(ishmem_team_t, uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_xor_reduce_work_group(ishmem_team_t, uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_xor_reduce_work_group(ishmem_team_t, uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_xor_reduce_work_group(ishmem_team_t, size_t *, const size_t *, size_t, const Group &);

/* max_reduce_on_queue */
template <typename T> sycl::event ishmemx_max_reduce_on_queue(T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_max_reduce_on_queue(float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_max_reduce_on_queue(double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_max_reduce_on_queue(char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_max_reduce_on_queue(signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_max_reduce_on_queue(short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_max_reduce_on_queue(int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_max_reduce_on_queue(long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_max_reduce_on_queue(long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_max_reduce_on_queue(unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_max_reduce_on_queue(unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_max_reduce_on_queue(unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_max_reduce_on_queue(unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_max_reduce_on_queue(unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_max_reduce_on_queue(int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_max_reduce_on_queue(int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_max_reduce_on_queue(int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_max_reduce_on_queue(int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_max_reduce_on_queue(uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_max_reduce_on_queue(uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_max_reduce_on_queue(uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_max_reduce_on_queue(uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_max_reduce_on_queue(size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_max_reduce_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

/* max_reduce_on_queue on a team */
template <typename T> sycl::event ishmemx_max_reduce_on_queue(ishmem_team_t, T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_max_reduce_on_queue(ishmem_team_t, float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_max_reduce_on_queue(ishmem_team_t, double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_max_reduce_on_queue(ishmem_team_t, char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_max_reduce_on_queue(ishmem_team_t, signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_max_reduce_on_queue(ishmem_team_t, short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_max_reduce_on_queue(ishmem_team_t, int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_max_reduce_on_queue(ishmem_team_t, long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_max_reduce_on_queue(ishmem_team_t, long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_max_reduce_on_queue(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_max_reduce_on_queue(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_max_reduce_on_queue(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_max_reduce_on_queue(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_max_reduce_on_queue(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_max_reduce_on_queue(ishmem_team_t, int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_max_reduce_on_queue(ishmem_team_t, int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_max_reduce_on_queue(ishmem_team_t, int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_max_reduce_on_queue(ishmem_team_t, int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_max_reduce_on_queue(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_max_reduce_on_queue(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_max_reduce_on_queue(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_max_reduce_on_queue(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_max_reduce_on_queue(ishmem_team_t, size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_max_reduce_on_queue(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

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

/* max_reduce_work_group on a team */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_max_reduce_work_group(ishmem_team_t, T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_max_reduce_work_group(ishmem_team_t, float *, const float *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_max_reduce_work_group(ishmem_team_t, double *, const double *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_max_reduce_work_group(ishmem_team_t, char *, const char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_max_reduce_work_group(ishmem_team_t, signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_max_reduce_work_group(ishmem_team_t, short *, const short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_max_reduce_work_group(ishmem_team_t, int *, const int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_max_reduce_work_group(ishmem_team_t, long *, const long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_max_reduce_work_group(ishmem_team_t, long long *, const long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_max_reduce_work_group(ishmem_team_t, unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_max_reduce_work_group(ishmem_team_t, unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_max_reduce_work_group(ishmem_team_t, unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_max_reduce_work_group(ishmem_team_t, unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_max_reduce_work_group(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_max_reduce_work_group(ishmem_team_t, int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_max_reduce_work_group(ishmem_team_t, int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_max_reduce_work_group(ishmem_team_t, int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_max_reduce_work_group(ishmem_team_t, int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_max_reduce_work_group(ishmem_team_t, uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_max_reduce_work_group(ishmem_team_t, uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_max_reduce_work_group(ishmem_team_t, uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_max_reduce_work_group(ishmem_team_t, uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_max_reduce_work_group(ishmem_team_t, size_t *, const size_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_max_reduce_work_group(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, const Group &);

/* min_reduce_on_queue */
template <typename T> sycl::event ishmemx_min_reduce_on_queue(T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_min_reduce_on_queue(float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_min_reduce_on_queue(double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_min_reduce_on_queue(char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_min_reduce_on_queue(signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_min_reduce_on_queue(short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_min_reduce_on_queue(int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_min_reduce_on_queue(long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_min_reduce_on_queue(long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_min_reduce_on_queue(unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_min_reduce_on_queue(unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_min_reduce_on_queue(unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_min_reduce_on_queue(unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_min_reduce_on_queue(unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_min_reduce_on_queue(int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_min_reduce_on_queue(int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_min_reduce_on_queue(int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_min_reduce_on_queue(int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_min_reduce_on_queue(uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_min_reduce_on_queue(uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_min_reduce_on_queue(uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_min_reduce_on_queue(uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_min_reduce_on_queue(size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_min_reduce_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

/* min_reduce_on_queue on a team */
template <typename T> sycl::event ishmemx_min_reduce_on_queue(ishmem_team_t, T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_min_reduce_on_queue(ishmem_team_t, float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_min_reduce_on_queue(ishmem_team_t, double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_min_reduce_on_queue(ishmem_team_t, char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_min_reduce_on_queue(ishmem_team_t, signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_min_reduce_on_queue(ishmem_team_t, short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_min_reduce_on_queue(ishmem_team_t, int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_min_reduce_on_queue(ishmem_team_t, long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_min_reduce_on_queue(ishmem_team_t, long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_min_reduce_on_queue(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_min_reduce_on_queue(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_min_reduce_on_queue(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_min_reduce_on_queue(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_min_reduce_on_queue(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_min_reduce_on_queue(ishmem_team_t, int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_min_reduce_on_queue(ishmem_team_t, int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_min_reduce_on_queue(ishmem_team_t, int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_min_reduce_on_queue(ishmem_team_t, int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_min_reduce_on_queue(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_min_reduce_on_queue(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_min_reduce_on_queue(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_min_reduce_on_queue(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_min_reduce_on_queue(ishmem_team_t, size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_min_reduce_on_queue(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

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

/* min_reduce_work_group on a team */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_min_reduce_work_group(ishmem_team_t, T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_min_reduce_work_group(ishmem_team_t, float *, const float *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_min_reduce_work_group(ishmem_team_t, double *, const double *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_min_reduce_work_group(ishmem_team_t, char *, const char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_min_reduce_work_group(ishmem_team_t, signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_min_reduce_work_group(ishmem_team_t, short *, const short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_min_reduce_work_group(ishmem_team_t, int *, const int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_min_reduce_work_group(ishmem_team_t, long *, const long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_min_reduce_work_group(ishmem_team_t, long long *, const long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_min_reduce_work_group(ishmem_team_t, unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_min_reduce_work_group(ishmem_team_t, unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_min_reduce_work_group(ishmem_team_t, unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_min_reduce_work_group(ishmem_team_t, unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_min_reduce_work_group(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_min_reduce_work_group(ishmem_team_t, int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_min_reduce_work_group(ishmem_team_t, int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_min_reduce_work_group(ishmem_team_t, int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_min_reduce_work_group(ishmem_team_t, int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_min_reduce_work_group(ishmem_team_t, uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_min_reduce_work_group(ishmem_team_t, uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_min_reduce_work_group(ishmem_team_t, uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_min_reduce_work_group(ishmem_team_t, uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_min_reduce_work_group(ishmem_team_t, size_t *, const size_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_min_reduce_work_group(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, const Group &);

/* sum_reduce_on_queue */
template <typename T> sycl::event ishmemx_sum_reduce_on_queue(T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_sum_reduce_on_queue(float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_sum_reduce_on_queue(double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_sum_reduce_on_queue(char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_sum_reduce_on_queue(signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_sum_reduce_on_queue(short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_sum_reduce_on_queue(int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_sum_reduce_on_queue(long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_sum_reduce_on_queue(long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_sum_reduce_on_queue(unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_sum_reduce_on_queue(unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_sum_reduce_on_queue(unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_sum_reduce_on_queue(unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_sum_reduce_on_queue(unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_sum_reduce_on_queue(int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_sum_reduce_on_queue(int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_sum_reduce_on_queue(int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_sum_reduce_on_queue(int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_sum_reduce_on_queue(uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_sum_reduce_on_queue(uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_sum_reduce_on_queue(uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_sum_reduce_on_queue(uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_sum_reduce_on_queue(size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_sum_reduce_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

/* sum_reduce_on_queue on a team */
template <typename T> sycl::event ishmemx_sum_reduce_on_queue(ishmem_team_t, T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_sum_reduce_on_queue(ishmem_team_t, float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_sum_reduce_on_queue(ishmem_team_t, double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_sum_reduce_on_queue(ishmem_team_t, char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_sum_reduce_on_queue(ishmem_team_t, signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_sum_reduce_on_queue(ishmem_team_t, short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_sum_reduce_on_queue(ishmem_team_t, int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_sum_reduce_on_queue(ishmem_team_t, long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_sum_reduce_on_queue(ishmem_team_t, long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_sum_reduce_on_queue(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_sum_reduce_on_queue(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_sum_reduce_on_queue(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_sum_reduce_on_queue(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_sum_reduce_on_queue(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_sum_reduce_on_queue(ishmem_team_t, int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_sum_reduce_on_queue(ishmem_team_t, int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_sum_reduce_on_queue(ishmem_team_t, int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_sum_reduce_on_queue(ishmem_team_t, int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_sum_reduce_on_queue(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_sum_reduce_on_queue(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_sum_reduce_on_queue(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_sum_reduce_on_queue(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_sum_reduce_on_queue(ishmem_team_t, size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_sum_reduce_on_queue(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

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

/* sum_reduce_work_group on a team */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_sum_reduce_work_group(ishmem_team_t, T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_sum_reduce_work_group(ishmem_team_t, float *, const float *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_sum_reduce_work_group(ishmem_team_t, double *, const double *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_sum_reduce_work_group(ishmem_team_t, char *, const char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_sum_reduce_work_group(ishmem_team_t, signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_sum_reduce_work_group(ishmem_team_t, short *, const short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_sum_reduce_work_group(ishmem_team_t, int *, const int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_sum_reduce_work_group(ishmem_team_t, long *, const long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_sum_reduce_work_group(ishmem_team_t, long long *, const long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_sum_reduce_work_group(ishmem_team_t, unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_sum_reduce_work_group(ishmem_team_t, unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_sum_reduce_work_group(ishmem_team_t, unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_sum_reduce_work_group(ishmem_team_t, unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_sum_reduce_work_group(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_sum_reduce_work_group(ishmem_team_t, int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_sum_reduce_work_group(ishmem_team_t, int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_sum_reduce_work_group(ishmem_team_t, int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_sum_reduce_work_group(ishmem_team_t, int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_sum_reduce_work_group(ishmem_team_t, uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_sum_reduce_work_group(ishmem_team_t, uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_sum_reduce_work_group(ishmem_team_t, uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_sum_reduce_work_group(ishmem_team_t, uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_sum_reduce_work_group(ishmem_team_t, size_t *, const size_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_sum_reduce_work_group(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, const Group &);

/* prod_reduce_on_queue */
template <typename T> sycl::event ishmemx_prod_reduce_on_queue(T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_prod_reduce_on_queue(float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_prod_reduce_on_queue(double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_prod_reduce_on_queue(char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_prod_reduce_on_queue(signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_prod_reduce_on_queue(short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_prod_reduce_on_queue(int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_prod_reduce_on_queue(long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_prod_reduce_on_queue(long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_prod_reduce_on_queue(unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_prod_reduce_on_queue(unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_prod_reduce_on_queue(unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_prod_reduce_on_queue(unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_prod_reduce_on_queue(unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_prod_reduce_on_queue(int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_prod_reduce_on_queue(int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_prod_reduce_on_queue(int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_prod_reduce_on_queue(int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_prod_reduce_on_queue(uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_prod_reduce_on_queue(uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_prod_reduce_on_queue(uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_prod_reduce_on_queue(uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_prod_reduce_on_queue(size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_prod_reduce_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

/* prod_reduce_on_queue on a team */
template <typename T> sycl::event ishmemx_prod_reduce_on_queue(ishmem_team_t, T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_prod_reduce_on_queue(ishmem_team_t, float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_prod_reduce_on_queue(ishmem_team_t, double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_prod_reduce_on_queue(ishmem_team_t, char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_prod_reduce_on_queue(ishmem_team_t, signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_prod_reduce_on_queue(ishmem_team_t, short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_prod_reduce_on_queue(ishmem_team_t, int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_prod_reduce_on_queue(ishmem_team_t, long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_prod_reduce_on_queue(ishmem_team_t, long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_prod_reduce_on_queue(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_prod_reduce_on_queue(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_prod_reduce_on_queue(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_prod_reduce_on_queue(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_prod_reduce_on_queue(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_prod_reduce_on_queue(ishmem_team_t, int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_prod_reduce_on_queue(ishmem_team_t, int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_prod_reduce_on_queue(ishmem_team_t, int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_prod_reduce_on_queue(ishmem_team_t, int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_prod_reduce_on_queue(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_prod_reduce_on_queue(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_prod_reduce_on_queue(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_prod_reduce_on_queue(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_prod_reduce_on_queue(ishmem_team_t, size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_prod_reduce_on_queue(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

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

/* prod_reduce_work_group on a team */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_prod_reduce_work_group(ishmem_team_t, T *, const T *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_float_prod_reduce_work_group(ishmem_team_t, float *, const float *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_double_prod_reduce_work_group(ishmem_team_t, double *, const double *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_char_prod_reduce_work_group(ishmem_team_t, char *, const char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_schar_prod_reduce_work_group(ishmem_team_t, signed char *, const signed char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_short_prod_reduce_work_group(ishmem_team_t, short *, const short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_prod_reduce_work_group(ishmem_team_t, int *, const int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_prod_reduce_work_group(ishmem_team_t, long *, const long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_prod_reduce_work_group(ishmem_team_t, long long *, const long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uchar_prod_reduce_work_group(ishmem_team_t, unsigned char *, const unsigned char *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ushort_prod_reduce_work_group(ishmem_team_t, unsigned short *, const unsigned short *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_prod_reduce_work_group(ishmem_team_t, unsigned int *, const unsigned int *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_prod_reduce_work_group(ishmem_team_t, unsigned long *, const unsigned long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_prod_reduce_work_group(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int8_prod_reduce_work_group(ishmem_team_t, int8_t *, const int8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int16_prod_reduce_work_group(ishmem_team_t, int16_t *, const int16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_prod_reduce_work_group(ishmem_team_t, int32_t *, const int32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_prod_reduce_work_group(ishmem_team_t, int64_t *, const int64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint8_prod_reduce_work_group(ishmem_team_t, uint8_t *, const uint8_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint16_prod_reduce_work_group(ishmem_team_t, uint16_t *, const uint16_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_prod_reduce_work_group(ishmem_team_t, uint32_t *, const uint32_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_prod_reduce_work_group(ishmem_team_t, uint64_t *, const uint64_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_prod_reduce_work_group(ishmem_team_t, size_t *, const size_t *, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_prod_reduce_work_group(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, const Group &);

/* scan_on_queue (prefix sum) */
template <typename T> sycl::event ishmemx_sum_inscan_on_queue(T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_sum_inscan_on_queue(float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_sum_inscan_on_queue(double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_sum_inscan_on_queue(char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_sum_inscan_on_queue(signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_sum_inscan_on_queue(short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_sum_inscan_on_queue(int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_sum_inscan_on_queue(long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_sum_inscan_on_queue(long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_sum_inscan_on_queue(unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_sum_inscan_on_queue(unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_sum_inscan_on_queue(unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_sum_inscan_on_queue(unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_sum_inscan_on_queue(unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_sum_inscan_on_queue(int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_sum_inscan_on_queue(int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_sum_inscan_on_queue(int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_sum_inscan_on_queue(int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_sum_inscan_on_queue(uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_sum_inscan_on_queue(uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_sum_inscan_on_queue(uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_sum_inscan_on_queue(uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_sum_inscan_on_queue(size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_sum_inscan_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

template <typename T> sycl::event ishmemx_sum_exscan_on_queue(T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_sum_exscan_on_queue(float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_sum_exscan_on_queue(double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_sum_exscan_on_queue(char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_sum_exscan_on_queue(signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_sum_exscan_on_queue(short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_sum_exscan_on_queue(int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_sum_exscan_on_queue(long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_sum_exscan_on_queue(long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_sum_exscan_on_queue(unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_sum_exscan_on_queue(unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_sum_exscan_on_queue(unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_sum_exscan_on_queue(unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_sum_exscan_on_queue(unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_sum_exscan_on_queue(int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_sum_exscan_on_queue(int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_sum_exscan_on_queue(int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_sum_exscan_on_queue(int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_sum_exscan_on_queue(uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_sum_exscan_on_queue(uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_sum_exscan_on_queue(uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_sum_exscan_on_queue(uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_sum_exscan_on_queue(size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_sum_exscan_on_queue(ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

/* scan_on_queue (prefix sum) on a team */
template <typename T> sycl::event ishmemx_sum_inscan_on_queue(ishmem_team_t, T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_sum_inscan_on_queue(ishmem_team_t, float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_sum_inscan_on_queue(ishmem_team_t, double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_sum_inscan_on_queue(ishmem_team_t, char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_sum_inscan_on_queue(ishmem_team_t, signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_sum_inscan_on_queue(ishmem_team_t, short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_sum_inscan_on_queue(ishmem_team_t, int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_sum_inscan_on_queue(ishmem_team_t, long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_sum_inscan_on_queue(ishmem_team_t, long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_sum_inscan_on_queue(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_sum_inscan_on_queue(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_sum_inscan_on_queue(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_sum_inscan_on_queue(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_sum_inscan_on_queue(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_sum_inscan_on_queue(ishmem_team_t, int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_sum_inscan_on_queue(ishmem_team_t, int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_sum_inscan_on_queue(ishmem_team_t, int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_sum_inscan_on_queue(ishmem_team_t, int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_sum_inscan_on_queue(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_sum_inscan_on_queue(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_sum_inscan_on_queue(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_sum_inscan_on_queue(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_sum_inscan_on_queue(ishmem_team_t, size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_sum_inscan_on_queue(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

template <typename T> sycl::event ishmemx_sum_exscan_on_queue(ishmem_team_t, T *, const T *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_float_sum_exscan_on_queue(ishmem_team_t, float *, const float *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_double_sum_exscan_on_queue(ishmem_team_t, double *, const double *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_char_sum_exscan_on_queue(ishmem_team_t, char *, const char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_schar_sum_exscan_on_queue(ishmem_team_t, signed char *, const signed char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_short_sum_exscan_on_queue(ishmem_team_t, short *, const short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_sum_exscan_on_queue(ishmem_team_t, int *, const int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_sum_exscan_on_queue(ishmem_team_t, long *, const long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_sum_exscan_on_queue(ishmem_team_t, long long *, const long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uchar_sum_exscan_on_queue(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ushort_sum_exscan_on_queue(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_sum_exscan_on_queue(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_sum_exscan_on_queue(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_sum_exscan_on_queue(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int8_sum_exscan_on_queue(ishmem_team_t, int8_t *, const int8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int16_sum_exscan_on_queue(ishmem_team_t, int16_t *, const int16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_sum_exscan_on_queue(ishmem_team_t, int32_t *, const int32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_sum_exscan_on_queue(ishmem_team_t, int64_t *, const int64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint8_sum_exscan_on_queue(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint16_sum_exscan_on_queue(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_sum_exscan_on_queue(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_sum_exscan_on_queue(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_sum_exscan_on_queue(ishmem_team_t, size_t *, const size_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_sum_exscan_on_queue(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

/* test_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_test_work_group(T *, int, T, const Group &);
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

/* test_all_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_test_all_work_group(T *, size_t, const int *, int, T, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_test_all_work_group(int *, size_t, const int *, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_test_all_work_group(long *, size_t, const int *, int, long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_test_all_work_group(long long *, size_t, const int *, int, long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_test_all_work_group(unsigned int *, size_t, const int *, int, unsigned int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_test_all_work_group(unsigned long *, size_t, const int *, int, unsigned long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_test_all_work_group(unsigned long long *, size_t, const int *, int, unsigned long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_test_all_work_group(int32_t *, size_t, const int *, int, int32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_test_all_work_group(int64_t *, size_t, const int *, int, int64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_test_all_work_group(uint32_t *, size_t, const int *, int, uint32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_test_all_work_group(uint64_t *, size_t, const int *, int, uint64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_test_all_work_group(size_t *, size_t, const int *, int, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_test_all_work_group(ptrdiff_t *, size_t, const int *, int, ptrdiff_t, const Group &);

/* test_any_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_test_any_work_group(T *, size_t, const int *, int, T, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int_test_any_work_group(int *, size_t, const int *, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_long_test_any_work_group(long *, size_t, const int *, int, long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_longlong_test_any_work_group(long long *, size_t, const int *, int, long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint_test_any_work_group(unsigned int *, size_t, const int *, int, unsigned int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulong_test_any_work_group(unsigned long *, size_t, const int *, int, unsigned long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulonglong_test_any_work_group(unsigned long long *, size_t, const int *, int, unsigned long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int32_test_any_work_group(int32_t *, size_t, const int *, int, int32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int64_test_any_work_group(int64_t *, size_t, const int *, int, int64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint32_test_any_work_group(uint32_t *, size_t, const int *, int, uint32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint64_test_any_work_group(uint64_t *, size_t, const int *, int, uint64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_size_test_any_work_group(size_t *, size_t, const int *, int, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ptrdiff_test_any_work_group(ptrdiff_t *, size_t, const int *, int, ptrdiff_t, const Group &);

/* test_some_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_test_some_work_group(T *, size_t, size_t *, const int *, int, T, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int_test_some_work_group(int *, size_t, size_t *, const int *, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_long_test_some_work_group(long *, size_t, size_t *, const int *, int, long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_longlong_test_some_work_group(long long *, size_t, size_t *, const int *, int, long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint_test_some_work_group(unsigned int *, size_t, size_t *, const int *, int, unsigned int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulong_test_some_work_group(unsigned long *, size_t, size_t *, const int *, int, unsigned long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulonglong_test_some_work_group(unsigned long long *, size_t, size_t *, const int *, int, unsigned long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int32_test_some_work_group(int32_t *, size_t, size_t *, const int *, int, int32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int64_test_some_work_group(int64_t *, size_t, size_t *, const int *, int, int64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint32_test_some_work_group(uint32_t *, size_t, size_t *, const int *, int, uint32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint64_test_some_work_group(uint64_t *, size_t, size_t *, const int *, int, uint64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_size_test_some_work_group(size_t *, size_t, size_t *, const int *, int, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ptrdiff_test_some_work_group(ptrdiff_t *, size_t, size_t *, const int *, int, ptrdiff_t, const Group &);

/* test_all_vector_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_test_all_vector_work_group(T *, size_t, const int *, int, const T *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int_test_all_vector_work_group(int *, size_t, const int *, int, const int *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_long_test_all_vector_work_group(long *, size_t, const int *, int, const long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_longlong_test_all_vector_work_group(long long *, size_t, const int *, int, const long long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint_test_all_vector_work_group(unsigned int *, size_t, const int *, int, const unsigned int *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulong_test_all_vector_work_group(unsigned long *, size_t, const int *, int, const unsigned long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ulonglong_test_all_vector_work_group(unsigned long long *, size_t, const int *, int, const unsigned long long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int32_test_all_vector_work_group(int32_t *, size_t, const int *, int, const int32_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_int64_test_all_vector_work_group(int64_t *, size_t, const int *, int, const int64_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint32_test_all_vector_work_group(uint32_t *, size_t, const int *, int, const uint32_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_uint64_test_all_vector_work_group(uint64_t *, size_t, const int *, int, const uint64_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_size_test_all_vector_work_group(size_t *, size_t, const int *, int, const size_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES int ishmemx_ptrdiff_test_all_vector_work_group(ptrdiff_t *, size_t, const int *, int, const ptrdiff_t *, const Group &);

/* test_any_vector_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_test_any_vector_work_group(T *, size_t, const int *, int, const T *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int_test_any_vector_work_group(int *, size_t, const int *, int, const int *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_long_test_any_vector_work_group(long *, size_t, const int *, int, const long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_longlong_test_any_vector_work_group(long long *, size_t, const int *, int, const long long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint_test_any_vector_work_group(unsigned int *, size_t, const int *, int, const unsigned int *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulong_test_any_vector_work_group(unsigned long *, size_t, const int *, int, const unsigned long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulonglong_test_any_vector_work_group(unsigned long long *, size_t, const int *, int, const unsigned long long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int32_test_any_vector_work_group(int32_t *, size_t, const int *, int, const int32_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int64_test_any_vector_work_group(int64_t *, size_t, const int *, int, const int64_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint32_test_any_vector_work_group(uint32_t *, size_t, const int *, int, const uint32_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint64_test_any_vector_work_group(uint64_t *, size_t, const int *, int, const uint64_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_size_test_any_vector_work_group(size_t *, size_t, const int *, int, const size_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ptrdiff_test_any_vector_work_group(ptrdiff_t *, size_t, const int *, int, const ptrdiff_t *, const Group &);

/* test_some_vector_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_test_some_vector_work_group(T *, size_t, size_t *, const int *, int, const T *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int_test_some_vector_work_group(int *, size_t, size_t *, const int *, int, const int *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_long_test_some_vector_work_group(long *, size_t, size_t *, const int *, int, const long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_longlong_test_some_vector_work_group(long long *, size_t, size_t *, const int *, int, const long long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint_test_some_vector_work_group(unsigned int *, size_t, size_t *, const int *, int, const unsigned int *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulong_test_some_vector_work_group(unsigned long *, size_t, size_t *, const int *, int, const unsigned long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulonglong_test_some_vector_work_group(unsigned long long *, size_t, size_t *, const int *, int, const unsigned long long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int32_test_some_vector_work_group(int32_t *, size_t, size_t *, const int *, int, const int32_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int64_test_some_vector_work_group(int64_t *, size_t, size_t *, const int *, int, const int64_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint32_test_some_vector_work_group(uint32_t *, size_t, size_t *, const int *, int, const uint32_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint64_test_some_vector_work_group(uint64_t *, size_t, size_t *, const int *, int, const uint64_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_size_test_some_vector_work_group(size_t *, size_t, size_t *, const int *, int, const size_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ptrdiff_test_some_vector_work_group(ptrdiff_t *, size_t, size_t *, const int *, int, const ptrdiff_t *, const Group &);

/* wait_until_on_queue */
template <typename T> sycl::event ishmemx_wait_until_on_queue(T *, int, T, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_wait_until_on_queue(int *, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_wait_until_on_queue(long *, int, long, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_wait_until_on_queue(long long *, int, long long, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_wait_until_on_queue(unsigned int *, int, unsigned int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_wait_until_on_queue(unsigned long *, int, unsigned long, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_wait_until_on_queue(unsigned long long *, int, unsigned long long, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_wait_until_on_queue(int32_t *, int, int32_t, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_wait_until_on_queue(int64_t *, int, int64_t, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_wait_until_on_queue(uint32_t *, int, uint32_t, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_wait_until_on_queue(uint64_t *, int, uint64_t, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_wait_until_on_queue(size_t *, int, size_t, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_wait_until_on_queue(ptrdiff_t *, int, ptrdiff_t, sycl::queue &, const std::vector<sycl::event> & = {});

/* wait_until_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_wait_until_work_group(T *, int, T, const Group &);
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

/* wait_until_all_on_queue */
template <typename T> sycl::event ishmemx_wait_until_all_on_queue(T *, size_t, const int *, int, T, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_wait_until_all_on_queue(int *, size_t, const int *, int, int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_wait_until_all_on_queue(long *, size_t, const int *, int, long, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_wait_until_all_on_queue(long long *, size_t, const int *, int, long long, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_wait_until_all_on_queue(unsigned int *, size_t, const int *, int, unsigned int, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_wait_until_all_on_queue(unsigned long *, size_t, const int *, int, unsigned long, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_wait_until_all_on_queue(unsigned long long *, size_t, const int *, int, unsigned long long, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_wait_until_all_on_queue(int32_t *, size_t, const int *, int, int32_t, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_wait_until_all_on_queue(int64_t *, size_t, const int *, int, int64_t, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_wait_until_all_on_queue(uint32_t *, size_t, const int *, int, uint32_t, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_wait_until_all_on_queue(uint64_t *, size_t, const int *, int, uint64_t, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_wait_until_all_on_queue(size_t *, size_t, const int *, int, size_t, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_wait_until_all_on_queue(ptrdiff_t *, size_t, const int *, int, ptrdiff_t, sycl::queue &, const std::vector<sycl::event> & = {});

/* wait_until_all_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_wait_until_all_work_group(T *, size_t, const int *, int, T, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_wait_until_all_work_group(int *, size_t, const int *, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_wait_until_all_work_group(long *, size_t, const int *, int, long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_wait_until_all_work_group(long long *, size_t, const int *, int, long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_wait_until_all_work_group(unsigned int *, size_t, const int *, int, unsigned int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_wait_until_all_work_group(unsigned long *, size_t, const int *, int, unsigned long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_wait_until_all_work_group(unsigned long long *, size_t, const int *, int, unsigned long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_wait_until_all_work_group(int32_t *, size_t, const int *, int, int32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_wait_until_all_work_group(int64_t *, size_t, const int *, int, int64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_wait_until_all_work_group(uint32_t *, size_t, const int *, int, uint32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_wait_until_all_work_group(uint64_t *, size_t, const int *, int, uint64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_wait_until_all_work_group(size_t *, size_t, const int *, int, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_wait_until_all_work_group(ptrdiff_t *, size_t, const int *, int, ptrdiff_t, const Group &);

/* wait_until_any_on_queue */
template <typename T> sycl::event ishmemx_wait_until_any_on_queue(T *, size_t, const int *, int, T, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_wait_until_any_on_queue(int *, size_t, const int *, int, int, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_wait_until_any_on_queue(long *, size_t, const int *, int, long, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_wait_until_any_on_queue(long long *, size_t, const int *, int, long long, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_wait_until_any_on_queue(unsigned int *, size_t, const int *, int, unsigned int, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_wait_until_any_on_queue(unsigned long *, size_t, const int *, int, unsigned long, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_wait_until_any_on_queue(unsigned long long *, size_t, const int *, int, unsigned long long, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_wait_until_any_on_queue(int32_t *, size_t, const int *, int, int32_t, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_wait_until_any_on_queue(int64_t *, size_t, const int *, int, int64_t, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_wait_until_any_on_queue(uint32_t *, size_t, const int *, int, uint32_t, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_wait_until_any_on_queue(uint64_t *, size_t, const int *, int, uint64_t, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_wait_until_any_on_queue(size_t *, size_t, const int *, int, size_t, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_wait_until_any_on_queue(ptrdiff_t *, size_t, const int *, int, ptrdiff_t, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});

/* wait_until_any_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_wait_until_any_work_group(T *, size_t, const int *, int, T, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int_wait_until_any_work_group(int *, size_t, const int *, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_long_wait_until_any_work_group(long *, size_t, const int *, int, long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_longlong_wait_until_any_work_group(long long *, size_t, const int *, int, long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint_wait_until_any_work_group(unsigned int *, size_t, const int *, int, unsigned int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulong_wait_until_any_work_group(unsigned long *, size_t, const int *, int, unsigned long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulonglong_wait_until_any_work_group(unsigned long long *, size_t, const int *, int, unsigned long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int32_wait_until_any_work_group(int32_t *, size_t, const int *, int, int32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int64_wait_until_any_work_group(int64_t *, size_t, const int *, int, int64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint32_wait_until_any_work_group(uint32_t *, size_t, const int *, int, uint32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint64_wait_until_any_work_group(uint64_t *, size_t, const int *, int, uint64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_size_wait_until_any_work_group(size_t *, size_t, const int *, int, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ptrdiff_wait_until_any_work_group(ptrdiff_t *, size_t, const int *, int, ptrdiff_t, const Group &);

/* wait_until_some_on_queue */
template <typename T> sycl::event ishmemx_wait_until_some_on_queue(T *, size_t, size_t *, const int *, int, T, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_wait_until_some_on_queue(int *, size_t, size_t *, const int *, int, int, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_wait_until_some_on_queue(long *, size_t, size_t *, const int *, int, long, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_wait_until_some_on_queue(long long *, size_t, size_t *, const int *, int, long long, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_wait_until_some_on_queue(unsigned int *, size_t, size_t *, const int *, int, unsigned int, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_wait_until_some_on_queue(unsigned long *, size_t, size_t *, const int *, int, unsigned long, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_wait_until_some_on_queue(unsigned long long *, size_t, size_t *, const int *, int, unsigned long long, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_wait_until_some_on_queue(int32_t *, size_t, size_t *, const int *, int, int32_t, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_wait_until_some_on_queue(int64_t *, size_t, size_t *, const int *, int, int64_t, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_wait_until_some_on_queue(uint32_t *, size_t, size_t *, const int *, int, uint32_t, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_wait_until_some_on_queue(uint64_t *, size_t, size_t *, const int *, int, uint64_t, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_wait_until_some_on_queue(size_t *, size_t, size_t *, const int *, int, size_t, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_wait_until_some_on_queue(ptrdiff_t *, size_t, size_t *, const int *, int, ptrdiff_t, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});

/* wait_until_some_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_wait_until_some_work_group(T *, size_t, size_t *, const int *, int, T, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int_wait_until_some_work_group(int *, size_t, size_t *, const int *, int, int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_long_wait_until_some_work_group(long *, size_t, size_t *, const int *, int, long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_longlong_wait_until_some_work_group(long long *, size_t, size_t *, const int *, int, long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint_wait_until_some_work_group(unsigned int *, size_t, size_t *, const int *, int, unsigned int, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulong_wait_until_some_work_group(unsigned long *, size_t, size_t *, const int *, int, unsigned long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulonglong_wait_until_some_work_group(unsigned long long *, size_t, size_t *, const int *, int, unsigned long long, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int32_wait_until_some_work_group(int32_t *, size_t, size_t *, const int *, int, int32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int64_wait_until_some_work_group(int64_t *, size_t, size_t *, const int *, int, int64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint32_wait_until_some_work_group(uint32_t *, size_t, size_t *, const int *, int, uint32_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint64_wait_until_some_work_group(uint64_t *, size_t, size_t *, const int *, int, uint64_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_size_wait_until_some_work_group(size_t *, size_t, size_t *, const int *, int, size_t, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ptrdiff_wait_until_some_work_group(ptrdiff_t *, size_t, size_t *, const int *, int, ptrdiff_t, const Group &);

/* wait_until_all_vector_on_queue */
template <typename T> sycl::event ishmemx_wait_until_all_vector_on_queue(T *, size_t, const int *, int, const T *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_wait_until_all_vector_on_queue(int *, size_t, const int *, int, const int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_wait_until_all_vector_on_queue(long *, size_t, const int *, int, const long *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_wait_until_all_vector_on_queue(long long *, size_t, const int *, int, const long long *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_wait_until_all_vector_on_queue(unsigned int *, size_t, const int *, int, const unsigned int *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_wait_until_all_vector_on_queue(unsigned long *, size_t, const int *, int, const unsigned long *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_wait_until_all_vector_on_queue(unsigned long long *, size_t, const int *, int, const unsigned long long *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_wait_until_all_vector_on_queue(int32_t *, size_t, const int *, int, const int32_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_wait_until_all_vector_on_queue(int64_t *, size_t, const int *, int, const int64_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_wait_until_all_vector_on_queue(uint32_t *, size_t, const int *, int, const uint32_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_wait_until_all_vector_on_queue(uint64_t *, size_t, const int *, int, const uint64_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_wait_until_all_vector_on_queue(size_t *, size_t, const int *, int, const size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_wait_until_all_vector_on_queue(ptrdiff_t *, size_t, const int *, int, const ptrdiff_t *, sycl::queue &, const std::vector<sycl::event> & = {});

/* wait_until_all_vector_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_wait_until_all_vector_work_group(T *, size_t, const int *, int, const T *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int_wait_until_all_vector_work_group(int *, size_t, const int *, int, const int *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_long_wait_until_all_vector_work_group(long *, size_t, const int *, int, const long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_longlong_wait_until_all_vector_work_group(long long *, size_t, const int *, int, const long long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint_wait_until_all_vector_work_group(unsigned int *, size_t, const int *, int, const unsigned int *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulong_wait_until_all_vector_work_group(unsigned long *, size_t, const int *, int, const unsigned long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ulonglong_wait_until_all_vector_work_group(unsigned long long *, size_t, const int *, int, const unsigned long long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int32_wait_until_all_vector_work_group(int32_t *, size_t, const int *, int, const int32_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_int64_wait_until_all_vector_work_group(int64_t *, size_t, const int *, int, const int64_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint32_wait_until_all_vector_work_group(uint32_t *, size_t, const int *, int, const uint32_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_uint64_wait_until_all_vector_work_group(uint64_t *, size_t, const int *, int, const uint64_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_size_wait_until_all_vector_work_group(size_t *, size_t, const int *, int, const size_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_ptrdiff_wait_until_all_vector_work_group(ptrdiff_t *, size_t, const int *, int, const ptrdiff_t *, const Group &);

/* wait_until_any_vector_on_queue */
template <typename T> sycl::event ishmemx_wait_until_any_vector_on_queue(T *, size_t, const int *, int, const T *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_wait_until_any_vector_on_queue(int *, size_t, const int *, int, const int *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_wait_until_any_vector_on_queue(long *, size_t, const int *, int, const long *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_wait_until_any_vector_on_queue(long long *, size_t, const int *, int, const long long *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_wait_until_any_vector_on_queue(unsigned int *, size_t, const int *, int, const unsigned int *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_wait_until_any_vector_on_queue(unsigned long *, size_t, const int *, int, const unsigned long *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_wait_until_any_vector_on_queue(unsigned long long *, size_t, const int *, int, const unsigned long long *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_wait_until_any_vector_on_queue(int32_t *, size_t, const int *, int, const int32_t *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_wait_until_any_vector_on_queue(int64_t *, size_t, const int *, int, const int64_t *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_wait_until_any_vector_on_queue(uint32_t *, size_t, const int *, int, const uint32_t *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_wait_until_any_vector_on_queue(uint64_t *, size_t, const int *, int, const uint64_t *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_wait_until_any_vector_on_queue(size_t *, size_t, const int *, int, const size_t *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_wait_until_any_vector_on_queue(ptrdiff_t *, size_t, const int *, int, const ptrdiff_t *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});

/* wait_until_any_vector_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_wait_until_any_vector_work_group(T *, size_t, const int *, int, const T *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int_wait_until_any_vector_work_group(int *, size_t, const int *, int, const int *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_long_wait_until_any_vector_work_group(long *, size_t, const int *, int, const long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_longlong_wait_until_any_vector_work_group(long long *, size_t, const int *, int, const long long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint_wait_until_any_vector_work_group(unsigned int *, size_t, const int *, int, const unsigned int *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulong_wait_until_any_vector_work_group(unsigned long *, size_t, const int *, int, const unsigned long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulonglong_wait_until_any_vector_work_group(unsigned long long *, size_t, const int *, int, const unsigned long long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int32_wait_until_any_vector_work_group(int32_t *, size_t, const int *, int, const int32_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int64_wait_until_any_vector_work_group(int64_t *, size_t, const int *, int, const int64_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint32_wait_until_any_vector_work_group(uint32_t *, size_t, const int *, int, const uint32_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint64_wait_until_any_vector_work_group(uint64_t *, size_t, const int *, int, const uint64_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_size_wait_until_any_vector_work_group(size_t *, size_t, const int *, int, const size_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ptrdiff_wait_until_any_vector_work_group(ptrdiff_t *, size_t, const int *, int, const ptrdiff_t *, const Group &);

/* wait_until_some_vector_on_queue */
template <typename T> sycl::event ishmemx_wait_until_some_vector_on_queue(T *, size_t, size_t *, const int *, int, const T *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int_wait_until_some_vector_on_queue(int *, size_t, size_t *, const int *, int, const int *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_long_wait_until_some_vector_on_queue(long *, size_t, size_t *, const int *, int, const long *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_longlong_wait_until_some_vector_on_queue(long long *, size_t, size_t *, const int *, int, const long long *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint_wait_until_some_vector_on_queue(unsigned int *, size_t, size_t *, const int *, int, const unsigned int *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulong_wait_until_some_vector_on_queue(unsigned long *, size_t, size_t *, const int *, int, const unsigned long *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ulonglong_wait_until_some_vector_on_queue(unsigned long long *, size_t, size_t *, const int *, int, const unsigned long long *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int32_wait_until_some_vector_on_queue(int32_t *, size_t, size_t *, const int *, int, const int32_t *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_int64_wait_until_some_vector_on_queue(int64_t *, size_t, size_t *, const int *, int, const int64_t *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint32_wait_until_some_vector_on_queue(uint32_t *, size_t, size_t *, const int *, int, const uint32_t *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_uint64_wait_until_some_vector_on_queue(uint64_t *, size_t, size_t *, const int *, int, const uint64_t *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_size_wait_until_some_vector_on_queue(size_t *, size_t, size_t *, const int *, int, const size_t *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_ptrdiff_wait_until_some_vector_on_queue(ptrdiff_t *, size_t, size_t *, const int *, int, const ptrdiff_t *, size_t *, sycl::queue &, const std::vector<sycl::event> & = {});

/* wait_until_some_vector_work_group */
template <typename T, typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_wait_until_some_vector_work_group(T *, size_t, size_t *, const int *, int, const T *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int_wait_until_some_vector_work_group(int *, size_t, size_t *, const int *, int, const int *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_long_wait_until_some_vector_work_group(long *, size_t, size_t *, const int *, int, const long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_longlong_wait_until_some_vector_work_group(long long *, size_t, size_t *, const int *, int, const long long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint_wait_until_some_vector_work_group(unsigned int *, size_t, size_t *, const int *, int, const unsigned int *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulong_wait_until_some_vector_work_group(unsigned long *, size_t, size_t *, const int *, int, const unsigned long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ulonglong_wait_until_some_vector_work_group(unsigned long long *, size_t, size_t *, const int *, int, const unsigned long long *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int32_wait_until_some_vector_work_group(int32_t *, size_t, size_t *, const int *, int, const int32_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_int64_wait_until_some_vector_work_group(int64_t *, size_t, size_t *, const int *, int, const int64_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint32_wait_until_some_vector_work_group(uint32_t *, size_t, size_t *, const int *, int, const uint32_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_uint64_wait_until_some_vector_work_group(uint64_t *, size_t, size_t *, const int *, int, const uint64_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_size_wait_until_some_vector_work_group(size_t *, size_t, size_t *, const int *, int, const size_t *, const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES size_t ishmemx_ptrdiff_wait_until_some_vector_work_group(ptrdiff_t *, size_t, size_t *, const int *, int, const ptrdiff_t *, const Group &);

/* signal_wait_until_on_queue */
sycl::event ishmemx_signal_wait_until_on_queue(uint64_t *, int, uint64_t, uint64_t *, sycl::queue &, const std::vector<sycl::event> & = {});

/* signal_wait_until_work_group */
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES uint64_t ishmemx_signal_wait_until_work_group(uint64_t *, int, uint64_t, const Group &);

/* barrier_all_on_queue */
sycl::event ishmemx_barrier_all_on_queue(sycl::queue &, const std::vector<sycl::event> & = {});

/* barrier_all_work_group */
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_barrier_all_work_group(const Group &);

/* sync_on_queue */
sycl::event ishmemx_sync_all_on_queue(sycl::queue &, const std::vector<sycl::event> & = {});
sycl::event ishmemx_team_sync_on_queue(ishmem_team_t, int *, sycl::queue &, const std::vector<sycl::event> & = {});

/* sync_work_group */
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_sync_all_work_group(const Group &);
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_team_sync_work_group(ishmem_team_t, const Group &);

/* fence_work_group */
template <typename Group> ISHMEM_DEVICE_ATTRIBUTES void ishmemx_fence_work_group(const Group &);

/* quiet_on_queue */
sycl::event ishmemx_quiet_on_queue(sycl::queue &, const std::vector<sycl::event> & = {});

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
/* this version you can include __FILE__, __LINE__, __func__ */
ISHMEM_DEVICE_ATTRIBUTES void ishmemx_print(const char *file, long int line, const char *func,
                                            const char *out, ishmemx_print_msg_type_t msg_type);

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
