/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef I_SHMEM_H
#define I_SHMEM_H

#if __INTEL_CLANG_COMPILER >= 20250000
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <ishmem/config.h>

#define ISHMEM_DEVICE_ATTRIBUTES SYCL_EXTERNAL

#define ISHMEM_MAJOR_VERSION 1
#define ISHMEM_MINOR_VERSION 5
#define ISHMEM_PATCH_VERSION 0
#define ISHMEM_MAX_NAME_LEN  256
#define ISHMEM_VENDOR_STRING "Intel® SHMEM"

#define ISHMEM_THREAD_SINGLE     0
#define ISHMEM_THREAD_FUNNELED   1
#define ISHMEM_THREAD_SERIALIZED 2
#define ISHMEM_THREAD_MULTIPLE   3

#define ISHMEM_CMP_EQ 1
#define ISHMEM_CMP_NE 2
#define ISHMEM_CMP_GT 3
#define ISHMEM_CMP_GE 4
#define ISHMEM_CMP_LT 5
#define ISHMEM_CMP_LE 6

#define ISHMEM_SIGNAL_SET 0
#define ISHMEM_SIGNAL_ADD 1

/* ISHMEM APIs */
/* Library setup and exit routines (host) */
void ishmem_init(void);
void ishmem_finalize(void);

/* Thread support */
int ishmem_init_thread(int requested, int *provided);
void ishmem_query_thread(int *provided);

/* Memory management (host) */
void *ishmem_malloc(size_t size);
void *ishmem_align(size_t alignment, size_t size);
void *ishmem_calloc(size_t count, size_t size);
void ishmem_free(void *ptr);

/* Library query routines (host and device) */
ISHMEM_DEVICE_ATTRIBUTES int ishmem_my_pe(void);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_n_pes(void);
ISHMEM_DEVICE_ATTRIBUTES void *ishmem_ptr(const void *dest, int pe);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_info_get_version(int *major, int *minor);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_info_get_name(char *name);

/* Team management routines */
typedef int ishmem_team_t;

typedef struct {
    int num_contexts;
} ishmem_team_config_t;

#define ISHMEM_TEAM_NUM_CONTEXTS 1L

#define ISHMEM_TEAM_INVALID -1
#define ISHMEM_TEAM_WORLD   0
#define ISHMEM_TEAM_SHARED  1
/* ISHMEMX_TEAM_NODE defined (2) in ishmemx.h */

ISHMEM_DEVICE_ATTRIBUTES int ishmem_team_my_pe(ishmem_team_t team);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_team_n_pes(ishmem_team_t team);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_team_translate_pe(ishmem_team_t src_team, int src_pe,
                                                      ishmem_team_t dest_team);
int ishmem_team_get_config(ishmem_team_t team, long config_mask, ishmem_team_config_t *config);
int ishmem_team_split_strided(ishmem_team_t parent_team, int start, int stride, int size,
                              const ishmem_team_config_t *config, long config_mask,
                              ishmem_team_t *new_team);
int ishmem_team_split_2d(ishmem_team_t parent_team, int xrange,
                         const ishmem_team_config_t *xaxis_config, long xaxis_mask,
                         ishmem_team_t *xaxis_team, const ishmem_team_config_t *yaxis_config,
                         long yaxis_mask, ishmem_team_t *yaxis_team);
void ishmem_team_destroy(ishmem_team_t team);

/* clang-format off */
/* put */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_put(T *, const T *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_float_put(float *, const float *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_double_put(double *, const double *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_char_put(char *, const char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_put(signed char *, const signed char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_short_put(short *, const short *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_put(int *, const int *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_put(long *, const long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_put(long long *, const long long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uchar_put(unsigned char *, const unsigned char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ushort_put(unsigned short *, const unsigned short *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_put(unsigned int *, const unsigned int *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_put(unsigned long *, const unsigned long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_put(unsigned long long *, const unsigned long long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int8_put(int8_t *, const int8_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int16_put(int16_t *, const int16_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_put(int32_t *, const int32_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_put(int64_t *, const int64_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint8_put(uint8_t *, const uint8_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint16_put(uint16_t *, const uint16_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_put(uint32_t *, const uint32_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_put(uint64_t *, const uint64_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_put(size_t *, const size_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_put(ptrdiff_t *, const ptrdiff_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put8(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put16(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put32(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put64(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put128(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_putmem(void *, const void *, size_t, int);

/* iput */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_iput(T *, const T *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_float_iput(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_double_iput(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_char_iput(char *, const char *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_iput(signed char *, const signed char *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_short_iput(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_iput(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_iput(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_iput(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uchar_iput(unsigned char *, const unsigned char *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ushort_iput(unsigned short *, const unsigned short *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_iput(unsigned int *, const unsigned int *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_iput(unsigned long *, const unsigned long *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_iput(unsigned long long *, const unsigned long long *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int8_iput(int8_t *, const int8_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int16_iput(int16_t *, const int16_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_iput(int32_t *, const int32_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_iput(int64_t *, const int64_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint8_iput(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint16_iput(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_iput(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_iput(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_iput(size_t *, const size_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_iput(ptrdiff_t *, const ptrdiff_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_iput8(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_iput16(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_iput32(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_iput64(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_iput128(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int);

/* p */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_p(T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_float_p(float *, float, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_double_p(double *, double, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_char_p(char *, char, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_p(signed char *, signed char, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_short_p(short *, short, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_p(int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_p(long *, long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_p(long long *, long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uchar_p(unsigned char *, unsigned char, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ushort_p(unsigned short *, unsigned short, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_p(unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_p(unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_p(unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int8_p(int8_t *, int8_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int16_p(int16_t *, int16_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_p(int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_p(int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint8_p(uint8_t *, uint8_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint16_p(uint16_t *, uint16_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_p(uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_p(uint64_t *, uint64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_p(size_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_p(ptrdiff_t *, ptrdiff_t, int);

/* get */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_get(T *, const T *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_float_get(float *, const float *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_double_get(double *, const double *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_char_get(char *, const char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_get(signed char *, const signed char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_short_get(short *, const short *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_get(int *, const int *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_get(long *, const long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_get(long long *, const long long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uchar_get(unsigned char *, const unsigned char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ushort_get(unsigned short *, const unsigned short *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_get(unsigned int *, const unsigned int *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_get(unsigned long *, const unsigned long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_get(unsigned long long *, const unsigned long long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int8_get(int8_t *, const int8_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int16_get(int16_t *, const int16_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_get(int32_t *, const int32_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_get(int64_t *, const int64_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint8_get(uint8_t *, const uint8_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint16_get(uint16_t *, const uint16_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_get(uint32_t *, const uint32_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_get(uint64_t *, const uint64_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_get(size_t *, const size_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_get(ptrdiff_t *, const ptrdiff_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_get8(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_get16(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_get32(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_get64(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_get128(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_getmem(void *, const void *, size_t, int);

/* iget */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_iget(T *, const T *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_float_iget(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_double_iget(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_char_iget(char *, const char *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_iget(signed char *, const signed char *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_short_iget(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_iget(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_iget(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_iget(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uchar_iget(unsigned char *, const unsigned char *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ushort_iget(unsigned short *, const unsigned short *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_iget(unsigned int *, const unsigned int *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_iget(unsigned long *, const unsigned long *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_iget(unsigned long long *, const unsigned long long *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int8_iget(int8_t *, const int8_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int16_iget(int16_t *, const int16_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_iget(int32_t *, const int32_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_iget(int64_t *, const int64_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint8_iget(uint8_t *, const uint8_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint16_iget(uint16_t *, const uint16_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_iget(uint32_t *, const uint32_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_iget(uint64_t *, const uint64_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_iget(size_t *, const size_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_iget(ptrdiff_t *, const ptrdiff_t *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_iget8(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_iget16(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_iget32(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_iget64(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_iget128(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int);

/* g */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES T ishmem_g(const T *, int);
ISHMEM_DEVICE_ATTRIBUTES float ishmem_float_g(const float *, int);
ISHMEM_DEVICE_ATTRIBUTES double ishmem_double_g(const double *, int);
ISHMEM_DEVICE_ATTRIBUTES char ishmem_char_g(const char *, int);
ISHMEM_DEVICE_ATTRIBUTES signed char ishmem_schar_g(const signed char *, int);
ISHMEM_DEVICE_ATTRIBUTES short ishmem_short_g(const short *, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_g(const int *, int);
ISHMEM_DEVICE_ATTRIBUTES long ishmem_long_g(const long *, int);
ISHMEM_DEVICE_ATTRIBUTES long long ishmem_longlong_g(const long long *, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned char ishmem_uchar_g(const unsigned char *, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned short ishmem_ushort_g(const unsigned short *, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned int ishmem_uint_g(const unsigned int *, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long ishmem_ulong_g(const unsigned long *, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long long ishmem_ulonglong_g(const unsigned long long *, int);
ISHMEM_DEVICE_ATTRIBUTES int8_t ishmem_int8_g(const int8_t *, int);
ISHMEM_DEVICE_ATTRIBUTES int16_t ishmem_int16_g(const int16_t *, int);
ISHMEM_DEVICE_ATTRIBUTES int32_t ishmem_int32_g(const int32_t *, int);
ISHMEM_DEVICE_ATTRIBUTES int64_t ishmem_int64_g(const int64_t *, int);
ISHMEM_DEVICE_ATTRIBUTES uint8_t ishmem_uint8_g(const uint8_t *, int);
ISHMEM_DEVICE_ATTRIBUTES uint16_t ishmem_uint16_g(const uint16_t *, int);
ISHMEM_DEVICE_ATTRIBUTES uint32_t ishmem_uint32_g(const uint32_t *, int);
ISHMEM_DEVICE_ATTRIBUTES uint64_t ishmem_uint64_g(const uint64_t *, int);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_size_g(const size_t *, int);
ISHMEM_DEVICE_ATTRIBUTES ptrdiff_t ishmem_ptrdiff_g(const ptrdiff_t *, int);

/* put_nbi */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_put_nbi(T *, const T *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_float_put_nbi(float *, const float *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_double_put_nbi(double *, const double *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_char_put_nbi(char *, const char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_put_nbi(signed char *, const signed char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_short_put_nbi(short *, const short *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_put_nbi(int *, const int *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_put_nbi(long *, const long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_put_nbi(long long *, const long long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uchar_put_nbi(unsigned char *, const unsigned char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ushort_put_nbi(unsigned short *, const unsigned short *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_put_nbi(unsigned int *, const unsigned int *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_put_nbi(unsigned long *, const unsigned long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_put_nbi(unsigned long long *, const unsigned long long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int8_put_nbi(int8_t *, const int8_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int16_put_nbi(int16_t *, const int16_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_put_nbi(int32_t *, const int32_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_put_nbi(int64_t *, const int64_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint8_put_nbi(uint8_t *, const uint8_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint16_put_nbi(uint16_t *, const uint16_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_put_nbi(uint32_t *, const uint32_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_put_nbi(uint64_t *, const uint64_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_put_nbi(size_t *, const size_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_put_nbi(ptrdiff_t *, const ptrdiff_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put8_nbi(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put16_nbi(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put32_nbi(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put64_nbi(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put128_nbi(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_putmem_nbi(void *, const void *, size_t, int);

/* get_nbi */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_get_nbi(T *, const T *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_float_get_nbi(float *, const float *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_double_get_nbi(double *, const double *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_char_get_nbi(char *, const char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_get_nbi(signed char *, const signed char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_short_get_nbi(short *, const short *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_get_nbi(int *, const int *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_get_nbi(long *, const long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_get_nbi(long long *, const long long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uchar_get_nbi(unsigned char *, const unsigned char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ushort_get_nbi(unsigned short *, const unsigned short *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_get_nbi(unsigned int *, const unsigned int *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_get_nbi(unsigned long *, const unsigned long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_get_nbi(unsigned long long *, const unsigned long long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int8_get_nbi(int8_t *, const int8_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int16_get_nbi(int16_t *, const int16_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_get_nbi(int32_t *, const int32_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_get_nbi(int64_t *, const int64_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint8_get_nbi(uint8_t *, const uint8_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint16_get_nbi(uint16_t *, const uint16_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_get_nbi(uint32_t *, const uint32_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_get_nbi(uint64_t *, const uint64_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_get_nbi(size_t *, const size_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_get_nbi(ptrdiff_t *, const ptrdiff_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_get8_nbi(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_get16_nbi(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_get32_nbi(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_get64_nbi(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_get128_nbi(void *, const void *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_getmem_nbi(void *, const void *, size_t, int);

/* atomic_fetch */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES T ishmem_atomic_fetch(T *, int);
ISHMEM_DEVICE_ATTRIBUTES float ishmem_float_atomic_fetch(float *, int);
ISHMEM_DEVICE_ATTRIBUTES double ishmem_double_atomic_fetch(double *, int);
ISHMEM_DEVICE_ATTRIBUTES signed char ishmem_schar_atomic_fetch(signed char *, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_atomic_fetch(int *, int);
ISHMEM_DEVICE_ATTRIBUTES long ishmem_long_atomic_fetch(long *, int);
ISHMEM_DEVICE_ATTRIBUTES long long ishmem_longlong_atomic_fetch(long long *, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned int ishmem_uint_atomic_fetch(unsigned int *, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long ishmem_ulong_atomic_fetch(unsigned long *, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long long ishmem_ulonglong_atomic_fetch(unsigned long long *, int);
ISHMEM_DEVICE_ATTRIBUTES int32_t ishmem_int32_atomic_fetch(int32_t *, int);
ISHMEM_DEVICE_ATTRIBUTES int64_t ishmem_int64_atomic_fetch(int64_t *, int);
ISHMEM_DEVICE_ATTRIBUTES uint32_t ishmem_uint32_atomic_fetch(uint32_t *, int);
ISHMEM_DEVICE_ATTRIBUTES uint64_t ishmem_uint64_atomic_fetch(uint64_t *, int);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_size_atomic_fetch(size_t *, int);
ISHMEM_DEVICE_ATTRIBUTES ptrdiff_t ishmem_ptrdiff_atomic_fetch(ptrdiff_t *, int);

/* atomic_set */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_atomic_set(T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_float_atomic_set(float *, float, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_double_atomic_set(double *, double, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_atomic_set(signed char *, signed char, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_atomic_set(int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_atomic_set(long *, long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_atomic_set(long long *, long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_atomic_set(unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_atomic_set(unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_atomic_set(unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_atomic_set(int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_atomic_set(int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_atomic_set(uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_atomic_set(uint64_t *, uint64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_atomic_set(size_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_atomic_set(ptrdiff_t *, ptrdiff_t, int);

/* atomic_swap */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES T ishmem_atomic_swap(T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES float ishmem_float_atomic_swap(float *, float, int);
ISHMEM_DEVICE_ATTRIBUTES double ishmem_double_atomic_swap(double *, double, int);
ISHMEM_DEVICE_ATTRIBUTES signed char ishmem_schar_atomic_swap(signed char *, signed char, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_atomic_swap(int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES long ishmem_long_atomic_swap(long *, long, int);
ISHMEM_DEVICE_ATTRIBUTES long long ishmem_longlong_atomic_swap(long long *, long long, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned int ishmem_uint_atomic_swap(unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long ishmem_ulong_atomic_swap(unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long long ishmem_ulonglong_atomic_swap(unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES int32_t ishmem_int32_atomic_swap(int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES int64_t ishmem_int64_atomic_swap(int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES uint32_t ishmem_uint32_atomic_swap(uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES uint64_t ishmem_uint64_atomic_swap(uint64_t *, uint64_t, int);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_size_atomic_swap(size_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES ptrdiff_t ishmem_ptrdiff_atomic_swap(ptrdiff_t *, ptrdiff_t, int);

/* atomic_compare_swap */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES T ishmem_atomic_compare_swap(T *, T, T, int);
ISHMEM_DEVICE_ATTRIBUTES signed char ishmem_schar_atomic_compare_swap(signed char *, signed char, signed char, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_atomic_compare_swap(int *, int, int, int);
ISHMEM_DEVICE_ATTRIBUTES long ishmem_long_atomic_compare_swap(long *, long, long, int);
ISHMEM_DEVICE_ATTRIBUTES long long ishmem_longlong_atomic_compare_swap(long long *, long long, long long, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned int ishmem_uint_atomic_compare_swap(unsigned int *, unsigned int, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long ishmem_ulong_atomic_compare_swap(unsigned long *, unsigned long, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long long ishmem_ulonglong_atomic_compare_swap(unsigned long long *, unsigned long long, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES int32_t ishmem_int32_atomic_compare_swap(int32_t *, int32_t, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES int64_t ishmem_int64_atomic_compare_swap(int64_t *, int64_t, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES uint32_t ishmem_uint32_atomic_compare_swap(uint32_t *, uint32_t, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES uint64_t ishmem_uint64_atomic_compare_swap(uint64_t *, uint64_t, uint64_t, int);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_size_atomic_compare_swap(size_t *, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES ptrdiff_t ishmem_ptrdiff_atomic_compare_swap(ptrdiff_t *, ptrdiff_t, ptrdiff_t, int);

/* atomic_fetch_inc */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES T ishmem_atomic_fetch_inc(T *, int);
ISHMEM_DEVICE_ATTRIBUTES signed char ishmem_schar_atomic_fetch_inc(signed char *, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_atomic_fetch_inc(int *, int);
ISHMEM_DEVICE_ATTRIBUTES long ishmem_long_atomic_fetch_inc(long *, int);
ISHMEM_DEVICE_ATTRIBUTES long long ishmem_longlong_atomic_fetch_inc(long long *, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned int ishmem_uint_atomic_fetch_inc(unsigned int *, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long ishmem_ulong_atomic_fetch_inc(unsigned long *, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long long ishmem_ulonglong_atomic_fetch_inc(unsigned long long *, int);
ISHMEM_DEVICE_ATTRIBUTES int32_t ishmem_int32_atomic_fetch_inc(int32_t *, int);
ISHMEM_DEVICE_ATTRIBUTES int64_t ishmem_int64_atomic_fetch_inc(int64_t *, int);
ISHMEM_DEVICE_ATTRIBUTES uint32_t ishmem_uint32_atomic_fetch_inc(uint32_t *, int);
ISHMEM_DEVICE_ATTRIBUTES uint64_t ishmem_uint64_atomic_fetch_inc(uint64_t *, int);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_size_atomic_fetch_inc(size_t *, int);
ISHMEM_DEVICE_ATTRIBUTES ptrdiff_t ishmem_ptrdiff_atomic_fetch_inc(ptrdiff_t *, int);

/* atomic_inc */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_atomic_inc(T *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_atomic_inc(signed char *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_atomic_inc(int *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_atomic_inc(long *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_atomic_inc(long long *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_atomic_inc(unsigned int *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_atomic_inc(unsigned long *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_atomic_inc(unsigned long long *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_atomic_inc(int32_t *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_atomic_inc(int64_t *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_atomic_inc(uint32_t *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_atomic_inc(uint64_t *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_atomic_inc(size_t *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_atomic_inc(ptrdiff_t *, int);

/* atomic_fetch_add */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES T ishmem_atomic_fetch_add(T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES signed char ishmem_schar_atomic_fetch_add(signed char *, signed char, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_atomic_fetch_add(int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES long ishmem_long_atomic_fetch_add(long *, long, int);
ISHMEM_DEVICE_ATTRIBUTES long long ishmem_longlong_atomic_fetch_add(long long *, long long, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned int ishmem_uint_atomic_fetch_add(unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long ishmem_ulong_atomic_fetch_add(unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long long ishmem_ulonglong_atomic_fetch_add(unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES int32_t ishmem_int32_atomic_fetch_add(int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES int64_t ishmem_int64_atomic_fetch_add(int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES uint32_t ishmem_uint32_atomic_fetch_add(uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES uint64_t ishmem_uint64_atomic_fetch_add(uint64_t *, uint64_t, int);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_size_atomic_fetch_add(size_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES ptrdiff_t ishmem_ptrdiff_atomic_fetch_add(ptrdiff_t *, ptrdiff_t, int);

/* atomic_add */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_atomic_add(T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_atomic_add(signed char *, signed char, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_atomic_add(int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_atomic_add(long *, long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_atomic_add(long long *, long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_atomic_add(unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_atomic_add(unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_atomic_add(unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_atomic_add(int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_atomic_add(int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_atomic_add(uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_atomic_add(uint64_t *, uint64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_atomic_add(size_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_atomic_add(ptrdiff_t *, ptrdiff_t, int);

/* atomic_fetch_and */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES T ishmem_atomic_fetch_and(T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES signed char ishmem_schar_atomic_fetch_and(signed char *, signed char, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned int ishmem_uint_atomic_fetch_and(unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long ishmem_ulong_atomic_fetch_and(unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long long ishmem_ulonglong_atomic_fetch_and(unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES int32_t ishmem_int32_atomic_fetch_and(int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES int64_t ishmem_int64_atomic_fetch_and(int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES uint32_t ishmem_uint32_atomic_fetch_and(uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES uint64_t ishmem_uint64_atomic_fetch_and(uint64_t *, uint64_t, int);

/* atomic_and */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_atomic_and(T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_atomic_and(signed char *, signed char, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_atomic_and(unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_atomic_and(unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_atomic_and(unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_atomic_and(int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_atomic_and(int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_atomic_and(uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_atomic_and(uint64_t *, uint64_t, int);

/* atomic_fetch_or */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES T ishmem_atomic_fetch_or(T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES signed char ishmem_schar_atomic_fetch_or(signed char *, signed char, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned int ishmem_uint_atomic_fetch_or(unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long ishmem_ulong_atomic_fetch_or(unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long long ishmem_ulonglong_atomic_fetch_or(unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES int32_t ishmem_int32_atomic_fetch_or(int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES int64_t ishmem_int64_atomic_fetch_or(int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES uint32_t ishmem_uint32_atomic_fetch_or(uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES uint64_t ishmem_uint64_atomic_fetch_or(uint64_t *, uint64_t, int);

/* atomic_or */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_atomic_or(T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_atomic_or(signed char *, signed char, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_atomic_or(unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_atomic_or(unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_atomic_or(unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_atomic_or(int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_atomic_or(int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_atomic_or(uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_atomic_or(uint64_t *, uint64_t, int);

/* atomic_fetch_xor */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES T ishmem_atomic_fetch_xor(T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES signed char ishmem_schar_atomic_fetch_xor(signed char *, signed char, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned int ishmem_uint_atomic_fetch_xor(unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long ishmem_ulong_atomic_fetch_xor(unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES unsigned long long ishmem_ulonglong_atomic_fetch_xor(unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES int32_t ishmem_int32_atomic_fetch_xor(int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES int64_t ishmem_int64_atomic_fetch_xor(int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES uint32_t ishmem_uint32_atomic_fetch_xor(uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES uint64_t ishmem_uint64_atomic_fetch_xor(uint64_t *, uint64_t, int);

/* atomic_xor */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_atomic_xor(T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_atomic_xor(signed char *, signed char, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_atomic_xor(unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_atomic_xor(unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_atomic_xor(unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_atomic_xor(int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_atomic_xor(int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_atomic_xor(uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_atomic_xor(uint64_t *, uint64_t, int);

/* atomic_fetch_nbi */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_atomic_fetch_nbi(T *, T *,  int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_float_atomic_fetch_nbi(float *,float *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_double_atomic_fetch_nbi(double *,double *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_atomic_fetch_nbi(int *, int *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_atomic_fetch_nbi(long *,long * , int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_atomic_fetch_nbi(long long *,long long *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_atomic_fetch_nbi(unsigned int *, unsigned int *,int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_atomic_fetch_nbi(unsigned long *, unsigned long *,  int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_atomic_fetch_nbi(unsigned long long *, unsigned long long *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_atomic_fetch_nbi(int32_t *, int32_t *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_atomic_fetch_nbi(int64_t *, int64_t *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_atomic_fetch_nbi(uint32_t *, uint32_t *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_atomic_fetch_nbi(uint64_t *, uint64_t *,int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_atomic_fetch_nbi(size_t *, size_t *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_atomic_fetch_nbi(ptrdiff_t *, ptrdiff_t *, int);

/* atomic_compare_swap_nbi */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_atomic_compare_swap_nbi(T *, T *, T, T, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_atomic_compare_swap_nbi(int *, int *, int, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_atomic_compare_swap_nbi(long *, long *, long, long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_atomic_compare_swap_nbi(long long *,long long *, long long, long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_atomic_compare_swap_nbi(unsigned int *,unsigned int *, unsigned int, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_atomic_compare_swap_nbi(unsigned long *,unsigned long *, unsigned long, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_atomic_compare_swap_nbi(unsigned long long *,unsigned long long *, unsigned long long, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_atomic_compare_swap_nbi(int32_t *,int32_t *, int32_t, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_atomic_compare_swap_nbi(int64_t *,int64_t *, int64_t, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_atomic_compare_swap_nbi(uint32_t *,uint32_t *, uint32_t, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_atomic_compare_swap_nbi(uint64_t *,uint64_t *, uint64_t, uint64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_atomic_compare_swap_nbi(size_t *,size_t *, size_t, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_atomic_compare_swap_nbi(ptrdiff_t *,ptrdiff_t *, ptrdiff_t, ptrdiff_t, int);

/* atomic_swap_nbi */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_atomic_swap_nbi(T *, T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_float_atomic_swap_nbi(float *, float *, float, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_double_atomic_swap_nbi(double *, double *, double, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_atomic_swap_nbi(int *, int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_atomic_swap_nbi(long *, long *, long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_atomic_swap_nbi(long long *, long long *, long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_atomic_swap_nbi(unsigned int *, unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_atomic_swap_nbi(unsigned long *, unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_atomic_swap_nbi(unsigned long long *, unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_atomic_swap_nbi(int32_t *, int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_atomic_swap_nbi(int64_t *, int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_atomic_swap_nbi(uint32_t *, uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_atomic_swap_nbi(uint64_t *, uint64_t *, uint64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_atomic_swap_nbi(size_t *, size_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_atomic_swap_nbi(ptrdiff_t *, ptrdiff_t *, ptrdiff_t, int);

/* atomic_fetch_inc_nbi */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_atomic_fetch_inc_nbi(T *, T *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_atomic_fetch_inc_nbi(int *, int *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_atomic_fetch_inc_nbi(long *, long *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_atomic_fetch_inc_nbi(long long *, long long *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_atomic_fetch_inc_nbi(unsigned int *, unsigned int *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_atomic_fetch_inc_nbi(unsigned long *, unsigned long *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_atomic_fetch_inc_nbi(unsigned long long *, unsigned long long *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_atomic_fetch_inc_nbi(int32_t *, int32_t *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_atomic_fetch_inc_nbi(int64_t *, int64_t *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_atomic_fetch_inc_nbi(uint32_t *, uint32_t *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_atomic_fetch_inc_nbi(uint64_t *, uint64_t *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_atomic_fetch_inc_nbi(size_t *, size_t *, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_atomic_fetch_inc_nbi(ptrdiff_t *, ptrdiff_t *, int);

/* atomic_fetch_add_nbi */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_atomic_fetch_add_nbi(T *, T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_atomic_fetch_add_nbi(int *, int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_atomic_fetch_add_nbi(long *, long *, long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_atomic_fetch_add_nbi(long long *, long long *, long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_atomic_fetch_add_nbi(unsigned int *, unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_atomic_fetch_add_nbi(unsigned long *, unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_atomic_fetch_add_nbi(unsigned long long *, unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_atomic_fetch_add_nbi(int32_t *, int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_atomic_fetch_add_nbi(int64_t *, int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_atomic_fetch_add_nbi(uint32_t *, uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_atomic_fetch_add_nbi(uint64_t *, uint64_t *, uint64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_atomic_fetch_add_nbi(size_t *, size_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_atomic_fetch_add_nbi(ptrdiff_t *, ptrdiff_t *, ptrdiff_t, int);

/* atomic_fetch_and_nbi */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_atomic_fetch_and_nbi(T *, T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_atomic_fetch_and_nbi(unsigned int *, unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_atomic_fetch_and_nbi(unsigned long *, unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_atomic_fetch_and_nbi(unsigned long long *, unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_atomic_fetch_and_nbi(int32_t *, int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_atomic_fetch_and_nbi(int64_t *, int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_atomic_fetch_and_nbi(uint32_t *, uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_atomic_fetch_and_nbi(uint64_t *, uint64_t *, uint64_t, int);

/* atomic_fetch_or_nbi */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_atomic_fetch_or_nbi(T *, T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_atomic_fetch_or_nbi(unsigned int *, unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_atomic_fetch_or_nbi(unsigned long *, unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_atomic_fetch_or_nbi(unsigned long long *, unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_atomic_fetch_or_nbi(int32_t *, int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_atomic_fetch_or_nbi(int64_t *, int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_atomic_fetch_or_nbi(uint32_t *, uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_atomic_fetch_or_nbi(uint64_t *, uint64_t *, uint64_t, int);

/* atomic_fetch_xor_nbi */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_atomic_fetch_xor_nbi(T *, T *, T, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_atomic_fetch_xor_nbi(unsigned int *, unsigned int *, unsigned int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_atomic_fetch_xor_nbi(unsigned long *, unsigned long *, unsigned long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_atomic_fetch_xor_nbi(unsigned long long *, unsigned long long *, unsigned long long, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_atomic_fetch_xor_nbi(int32_t *, int32_t *, int32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_atomic_fetch_xor_nbi(int64_t *, int64_t *, int64_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_atomic_fetch_xor_nbi(uint32_t *, uint32_t *, uint32_t, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_atomic_fetch_xor_nbi(uint64_t *, uint64_t *, uint64_t, int);

/* put_signal */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_put_signal(T *, const T *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_float_put_signal(float *, const float *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_double_put_signal(double *, const double *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_char_put_signal(char *, const char *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_put_signal(signed char *, const signed char *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_short_put_signal(short *, const short *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_put_signal(int *, const int *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_put_signal(long *, const long *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_put_signal(long long *, const long long *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uchar_put_signal(unsigned char *, const unsigned char *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ushort_put_signal(unsigned short *, const unsigned short *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_put_signal(unsigned int *, const unsigned int *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_put_signal(unsigned long *, const unsigned long *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_put_signal(unsigned long long *, const unsigned long long *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int8_put_signal(int8_t *, const int8_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int16_put_signal(int16_t *, const int16_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_put_signal(int32_t *, const int32_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_put_signal(int64_t *, const int64_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint8_put_signal(uint8_t *, const uint8_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint16_put_signal(uint16_t *, const uint16_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_put_signal(uint32_t *, const uint32_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_put_signal(uint64_t *, const uint64_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_put_signal(size_t *, const size_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_put_signal(ptrdiff_t *, const ptrdiff_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put8_signal(void *, const void *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put16_signal(void *, const void *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put32_signal(void *, const void *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put64_signal(void *, const void *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put128_signal(void *, const void *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_putmem_signal(void *, const void *, size_t, uint64_t *, uint64_t, int, int);

/* put_signal_nbi */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_put_signal_nbi(T *, const T *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_float_put_signal_nbi(float *, const float *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_double_put_signal_nbi(double *, const double *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_char_put_signal_nbi(char *, const char *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_schar_put_signal_nbi(signed char *, const signed char *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_short_put_signal_nbi(short *, const short *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_put_signal_nbi(int *, const int *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_put_signal_nbi(long *, const long *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_put_signal_nbi(long long *, const long long *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uchar_put_signal_nbi(unsigned char *, const unsigned char *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ushort_put_signal_nbi(unsigned short *, const unsigned short *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_put_signal_nbi(unsigned int *, const unsigned int *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_put_signal_nbi(unsigned long *, const unsigned long *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_put_signal_nbi(unsigned long long *, const unsigned long long *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int8_put_signal_nbi(int8_t *, const int8_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int16_put_signal_nbi(int16_t *, const int16_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_put_signal_nbi(int32_t *, const int32_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_put_signal_nbi(int64_t *, const int64_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint8_put_signal_nbi(uint8_t *, const uint8_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint16_put_signal_nbi(uint16_t *, const uint16_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_put_signal_nbi(uint32_t *, const uint32_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_put_signal_nbi(uint64_t *, const uint64_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_put_signal_nbi(size_t *, const size_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_put_signal_nbi(ptrdiff_t *, const ptrdiff_t *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put8_signal_nbi(void *, const void *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put16_signal_nbi(void *, const void *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put32_signal_nbi(void *, const void *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put64_signal_nbi(void *, const void *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_put128_signal_nbi(void *, const void *, size_t, uint64_t *, uint64_t, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_putmem_signal_nbi(void *, const void *, size_t, uint64_t *, uint64_t, int, int);

/* signal_fetch */
ISHMEM_DEVICE_ATTRIBUTES uint64_t ishmem_signal_fetch(uint64_t *);

/* alltoall */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_alltoall(T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_alltoall(float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_alltoall(double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_alltoall(char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_alltoall(signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_alltoall(short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_alltoall(int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_alltoall(long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_alltoall(long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_alltoall(unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_alltoall(unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_alltoall(unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_alltoall(unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_alltoall(unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_alltoall(int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_alltoall(int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_alltoall(int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_alltoall(int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_alltoall(uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_alltoall(uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_alltoall(uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_alltoall(uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_alltoall(size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_alltoall(ptrdiff_t *, const ptrdiff_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_alltoallmem(void *, const void *, size_t);

/* alltoall on a team */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_alltoall(ishmem_team_t, T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_alltoall(ishmem_team_t, float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_alltoall(ishmem_team_t, double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_alltoall(ishmem_team_t, char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_alltoall(ishmem_team_t, signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_alltoall(ishmem_team_t, short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_alltoall(ishmem_team_t, int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_alltoall(ishmem_team_t, long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_alltoall(ishmem_team_t, long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_alltoall(ishmem_team_t, unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_alltoall(ishmem_team_t, unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_alltoall(ishmem_team_t, unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_alltoall(ishmem_team_t, unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_alltoall(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_alltoall(ishmem_team_t, int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_alltoall(ishmem_team_t, int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_alltoall(ishmem_team_t, int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_alltoall(ishmem_team_t, int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_alltoall(ishmem_team_t, uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_alltoall(ishmem_team_t, uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_alltoall(ishmem_team_t, uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_alltoall(ishmem_team_t, uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_alltoall(ishmem_team_t, size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_alltoall(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_alltoallmem(ishmem_team_t, void *, const void *, size_t);

/* broadcast */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_broadcast(T *, const T *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_broadcast(float *, const float *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_broadcast(double *, const double *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_broadcast(char *, const char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_broadcast(signed char *, const signed char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_broadcast(short *, const short *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_broadcast(int *, const int *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_broadcast(long *, const long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_broadcast(long long *, const long long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_broadcast(unsigned char *, const unsigned char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_broadcast(unsigned short *, const unsigned short *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_broadcast(unsigned int *, const unsigned int *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_broadcast(unsigned long *, const unsigned long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_broadcast(unsigned long long *, const unsigned long long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_broadcast(int8_t *, const int8_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_broadcast(int16_t *, const int16_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_broadcast(int32_t *, const int32_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_broadcast(int64_t *, const int64_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_broadcast(uint8_t *, const uint8_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_broadcast(uint16_t *, const uint16_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_broadcast(uint32_t *, const uint32_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_broadcast(uint64_t *, const uint64_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_broadcast(size_t *, const size_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_broadcast(ptrdiff_t *, const ptrdiff_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_broadcastmem(void *, const void *, size_t, int);

/* broadcast on a team */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_broadcast(ishmem_team_t, T *, const T *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_broadcast(ishmem_team_t, float *, const float *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_broadcast(ishmem_team_t, double *, const double *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_broadcast(ishmem_team_t, char *, const char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_broadcast(ishmem_team_t, signed char *, const signed char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_broadcast(ishmem_team_t, short *, const short *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_broadcast(ishmem_team_t, int *, const int *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_broadcast(ishmem_team_t, long *, const long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_broadcast(ishmem_team_t, long long *, const long long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_broadcast(ishmem_team_t, unsigned char *, const unsigned char *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_broadcast(ishmem_team_t, unsigned short *, const unsigned short *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_broadcast(ishmem_team_t, unsigned int *, const unsigned int *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_broadcast(ishmem_team_t, unsigned long *, const unsigned long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_broadcast(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_broadcast(ishmem_team_t, int8_t *, const int8_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_broadcast(ishmem_team_t, int16_t *, const int16_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_broadcast(ishmem_team_t, int32_t *, const int32_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_broadcast(ishmem_team_t, int64_t *, const int64_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_broadcast(ishmem_team_t, uint8_t *, const uint8_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_broadcast(ishmem_team_t, uint16_t *, const uint16_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_broadcast(ishmem_team_t, uint32_t *, const uint32_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_broadcast(ishmem_team_t, uint64_t *, const uint64_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_broadcast(ishmem_team_t, size_t *, const size_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_broadcast(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_broadcastmem(ishmem_team_t, void *, const void *, size_t, int);

/* collect */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_collect(T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_collect(float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_collect(double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_collect(char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_collect(signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_collect(short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_collect(int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_collect(long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_collect(long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_collect(unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_collect(unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_collect(unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_collect(unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_collect(unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_collect(int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_collect(int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_collect(int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_collect(int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_collect(uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_collect(uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_collect(uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_collect(uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_collect(size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_collect(ptrdiff_t *, const ptrdiff_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_collectmem(void *, const void *, size_t);

/* collect on a team */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_collect(ishmem_team_t, T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_collect(ishmem_team_t, float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_collect(ishmem_team_t, double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_collect(ishmem_team_t, char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_collect(ishmem_team_t, signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_collect(ishmem_team_t, short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_collect(ishmem_team_t, int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_collect(ishmem_team_t, long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_collect(ishmem_team_t, long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_collect(ishmem_team_t, unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_collect(ishmem_team_t, unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_collect(ishmem_team_t, unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_collect(ishmem_team_t, unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_collect(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_collect(ishmem_team_t, int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_collect(ishmem_team_t, int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_collect(ishmem_team_t, int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_collect(ishmem_team_t, int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_collect(ishmem_team_t, uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_collect(ishmem_team_t, uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_collect(ishmem_team_t, uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_collect(ishmem_team_t, uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_collect(ishmem_team_t, size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_collect(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_collectmem(ishmem_team_t, void *, const void *, size_t);

/* fcollect */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_fcollect(T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_fcollect(float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_fcollect(double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_fcollect(char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_fcollect(signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_fcollect(short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_fcollect(int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_fcollect(long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_fcollect(long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_fcollect(unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_fcollect(unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_fcollect(unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_fcollect(unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_fcollect(unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_fcollect(int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_fcollect(int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_fcollect(int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_fcollect(int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_fcollect(uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_fcollect(uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_fcollect(uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_fcollect(uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_fcollect(size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_fcollect(ptrdiff_t *, const ptrdiff_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_fcollectmem(void *, const void *, size_t);

/* fcollect on a team */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_fcollect(ishmem_team_t, T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_fcollect(ishmem_team_t, float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_fcollect(ishmem_team_t, double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_fcollect(ishmem_team_t, char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_fcollect(ishmem_team_t, signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_fcollect(ishmem_team_t, short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_fcollect(ishmem_team_t, int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_fcollect(ishmem_team_t, long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_fcollect(ishmem_team_t, long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_fcollect(ishmem_team_t, unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_fcollect(ishmem_team_t, unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_fcollect(ishmem_team_t, unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_fcollect(ishmem_team_t, unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_fcollect(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_fcollect(ishmem_team_t, int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_fcollect(ishmem_team_t, int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_fcollect(ishmem_team_t, int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_fcollect(ishmem_team_t, int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_fcollect(ishmem_team_t, uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_fcollect(ishmem_team_t, uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_fcollect(ishmem_team_t, uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_fcollect(ishmem_team_t, uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_fcollect(ishmem_team_t, size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_fcollect(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_fcollectmem(ishmem_team_t, void *, const void *, size_t);

/* and_reduce */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_and_reduce(T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_and_reduce(signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_and_reduce(unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_and_reduce(unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_and_reduce(unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_and_reduce(unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_and_reduce(unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_and_reduce(int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_and_reduce(int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_and_reduce(int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_and_reduce(int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_and_reduce(uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_and_reduce(uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_and_reduce(uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_and_reduce(uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_and_reduce(size_t *, const size_t *, size_t);

/* and_reduce on a team */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_and_reduce(ishmem_team_t, T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_and_reduce(ishmem_team_t, signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_and_reduce(ishmem_team_t, unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_and_reduce(ishmem_team_t, unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_and_reduce(ishmem_team_t, unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_and_reduce(ishmem_team_t, unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_and_reduce(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_and_reduce(ishmem_team_t, int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_and_reduce(ishmem_team_t, int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_and_reduce(ishmem_team_t, int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_and_reduce(ishmem_team_t, int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_and_reduce(ishmem_team_t, uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_and_reduce(ishmem_team_t, uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_and_reduce(ishmem_team_t, uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_and_reduce(ishmem_team_t, uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_and_reduce(ishmem_team_t, size_t *, const size_t *, size_t);

/* or_reduce */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_or_reduce(T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_or_reduce(signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_or_reduce(unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_or_reduce(unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_or_reduce(unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_or_reduce(unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_or_reduce(unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_or_reduce(int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_or_reduce(int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_or_reduce(int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_or_reduce(int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_or_reduce(uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_or_reduce(uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_or_reduce(uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_or_reduce(uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_or_reduce(size_t *, const size_t *, size_t);

/* or_reduce on a team */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_or_reduce(ishmem_team_t, T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_or_reduce(ishmem_team_t, signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_or_reduce(ishmem_team_t, unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_or_reduce(ishmem_team_t, unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_or_reduce(ishmem_team_t, unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_or_reduce(ishmem_team_t, unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_or_reduce(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_or_reduce(ishmem_team_t, int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_or_reduce(ishmem_team_t, int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_or_reduce(ishmem_team_t, int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_or_reduce(ishmem_team_t, int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_or_reduce(ishmem_team_t, uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_or_reduce(ishmem_team_t, uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_or_reduce(ishmem_team_t, uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_or_reduce(ishmem_team_t, uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_or_reduce(ishmem_team_t, size_t *, const size_t *, size_t);

/* xor_reduce */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_xor_reduce(T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_xor_reduce(signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_xor_reduce(unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_xor_reduce(unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_xor_reduce(unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_xor_reduce(unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_xor_reduce(unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_xor_reduce(int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_xor_reduce(int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_xor_reduce(int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_xor_reduce(int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_xor_reduce(uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_xor_reduce(uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_xor_reduce(uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_xor_reduce(uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_xor_reduce(size_t *, const size_t *, size_t);

/* xor_reduce on a team */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_xor_reduce(ishmem_team_t, T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_xor_reduce(ishmem_team_t, signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_xor_reduce(ishmem_team_t, unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_xor_reduce(ishmem_team_t, unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_xor_reduce(ishmem_team_t, unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_xor_reduce(ishmem_team_t, unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_xor_reduce(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_xor_reduce(ishmem_team_t, int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_xor_reduce(ishmem_team_t, int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_xor_reduce(ishmem_team_t, int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_xor_reduce(ishmem_team_t, int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_xor_reduce(ishmem_team_t, uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_xor_reduce(ishmem_team_t, uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_xor_reduce(ishmem_team_t, uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_xor_reduce(ishmem_team_t, uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_xor_reduce(ishmem_team_t, size_t *, const size_t *, size_t);

/* max_reduce */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_max_reduce(T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_max_reduce(float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_max_reduce(double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_max_reduce(char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_max_reduce(signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_max_reduce(short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_max_reduce(int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_max_reduce(long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_max_reduce(long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_max_reduce(unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_max_reduce(unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_max_reduce(unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_max_reduce(unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_max_reduce(unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_max_reduce(int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_max_reduce(int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_max_reduce(int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_max_reduce(int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_max_reduce(uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_max_reduce(uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_max_reduce(uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_max_reduce(uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_max_reduce(size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_max_reduce(ptrdiff_t *, const ptrdiff_t *, size_t);

/* max_reduce on a team */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_max_reduce(ishmem_team_t, T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_max_reduce(ishmem_team_t, float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_max_reduce(ishmem_team_t, double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_max_reduce(ishmem_team_t, char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_max_reduce(ishmem_team_t, signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_max_reduce(ishmem_team_t, short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_max_reduce(ishmem_team_t, int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_max_reduce(ishmem_team_t, long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_max_reduce(ishmem_team_t, long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_max_reduce(ishmem_team_t, unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_max_reduce(ishmem_team_t, unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_max_reduce(ishmem_team_t, unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_max_reduce(ishmem_team_t, unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_max_reduce(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_max_reduce(ishmem_team_t, int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_max_reduce(ishmem_team_t, int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_max_reduce(ishmem_team_t, int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_max_reduce(ishmem_team_t, int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_max_reduce(ishmem_team_t, uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_max_reduce(ishmem_team_t, uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_max_reduce(ishmem_team_t, uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_max_reduce(ishmem_team_t, uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_max_reduce(ishmem_team_t, size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_max_reduce(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t);

/* min_reduce */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_min_reduce(T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_min_reduce(float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_min_reduce(double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_min_reduce(char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_min_reduce(signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_min_reduce(short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_min_reduce(int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_min_reduce(long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_min_reduce(long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_min_reduce(unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_min_reduce(unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_min_reduce(unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_min_reduce(unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_min_reduce(unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_min_reduce(int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_min_reduce(int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_min_reduce(int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_min_reduce(int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_min_reduce(uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_min_reduce(uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_min_reduce(uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_min_reduce(uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_min_reduce(size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_min_reduce(ptrdiff_t *, const ptrdiff_t *, size_t);

/* min_reduce on a team */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_min_reduce(ishmem_team_t, T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_min_reduce(ishmem_team_t, float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_min_reduce(ishmem_team_t, double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_min_reduce(ishmem_team_t, char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_min_reduce(ishmem_team_t, signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_min_reduce(ishmem_team_t, short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_min_reduce(ishmem_team_t, int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_min_reduce(ishmem_team_t, long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_min_reduce(ishmem_team_t, long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_min_reduce(ishmem_team_t, unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_min_reduce(ishmem_team_t, unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_min_reduce(ishmem_team_t, unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_min_reduce(ishmem_team_t, unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_min_reduce(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_min_reduce(ishmem_team_t, int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_min_reduce(ishmem_team_t, int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_min_reduce(ishmem_team_t, int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_min_reduce(ishmem_team_t, int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_min_reduce(ishmem_team_t, uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_min_reduce(ishmem_team_t, uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_min_reduce(ishmem_team_t, uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_min_reduce(ishmem_team_t, uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_min_reduce(ishmem_team_t, size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_min_reduce(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t);

/* sum_reduce */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_sum_reduce(T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_sum_reduce(float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_sum_reduce(double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_sum_reduce(char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_sum_reduce(signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_sum_reduce(short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_sum_reduce(int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_sum_reduce(long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_sum_reduce(long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_sum_reduce(unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_sum_reduce(unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_sum_reduce(unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_sum_reduce(unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_sum_reduce(unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_sum_reduce(int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_sum_reduce(int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_sum_reduce(int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_sum_reduce(int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_sum_reduce(uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_sum_reduce(uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_sum_reduce(uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_sum_reduce(uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_sum_reduce(size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_sum_reduce(ptrdiff_t *, const ptrdiff_t *, size_t);

/* sum_reduce on a team */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_sum_reduce(ishmem_team_t, T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_sum_reduce(ishmem_team_t, float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_sum_reduce(ishmem_team_t, double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_sum_reduce(ishmem_team_t, char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_sum_reduce(ishmem_team_t, signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_sum_reduce(ishmem_team_t, short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_sum_reduce(ishmem_team_t, int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_sum_reduce(ishmem_team_t, long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_sum_reduce(ishmem_team_t, long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_sum_reduce(ishmem_team_t, unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_sum_reduce(ishmem_team_t, unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_sum_reduce(ishmem_team_t, unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_sum_reduce(ishmem_team_t, unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_sum_reduce(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_sum_reduce(ishmem_team_t, int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_sum_reduce(ishmem_team_t, int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_sum_reduce(ishmem_team_t, int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_sum_reduce(ishmem_team_t, int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_sum_reduce(ishmem_team_t, uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_sum_reduce(ishmem_team_t, uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_sum_reduce(ishmem_team_t, uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_sum_reduce(ishmem_team_t, uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_sum_reduce(ishmem_team_t, size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_sum_reduce(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t);

/* prod_reduce */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_prod_reduce(T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_prod_reduce(float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_prod_reduce(double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_prod_reduce(char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_prod_reduce(signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_prod_reduce(short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_prod_reduce(int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_prod_reduce(long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_prod_reduce(long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_prod_reduce(unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_prod_reduce(unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_prod_reduce(unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_prod_reduce(unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_prod_reduce(unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_prod_reduce(int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_prod_reduce(int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_prod_reduce(int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_prod_reduce(int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_prod_reduce(uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_prod_reduce(uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_prod_reduce(uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_prod_reduce(uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_prod_reduce(size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_prod_reduce(ptrdiff_t *, const ptrdiff_t *, size_t);

/* prod_reduce on a team */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_prod_reduce(ishmem_team_t, T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_prod_reduce(ishmem_team_t, float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_prod_reduce(ishmem_team_t, double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_prod_reduce(ishmem_team_t, char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_prod_reduce(ishmem_team_t, signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_prod_reduce(ishmem_team_t, short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_prod_reduce(ishmem_team_t, int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_prod_reduce(ishmem_team_t, long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_prod_reduce(ishmem_team_t, long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_prod_reduce(ishmem_team_t, unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_prod_reduce(ishmem_team_t, unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_prod_reduce(ishmem_team_t, unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_prod_reduce(ishmem_team_t, unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_prod_reduce(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_prod_reduce(ishmem_team_t, int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_prod_reduce(ishmem_team_t, int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_prod_reduce(ishmem_team_t, int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_prod_reduce(ishmem_team_t, int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_prod_reduce(ishmem_team_t, uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_prod_reduce(ishmem_team_t, uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_prod_reduce(ishmem_team_t, uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_prod_reduce(ishmem_team_t, uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_prod_reduce(ishmem_team_t, size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_prod_reduce(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t);

/* scan (prefix sum) */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_sum_inscan(T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_sum_inscan(float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_sum_inscan(double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_sum_inscan(char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_sum_inscan(signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_sum_inscan(short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_sum_inscan(int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_sum_inscan(long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_sum_inscan(long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_sum_inscan(unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_sum_inscan(unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_sum_inscan(unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_sum_inscan(unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_sum_inscan(unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_sum_inscan(int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_sum_inscan(int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_sum_inscan(int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_sum_inscan(int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_sum_inscan(uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_sum_inscan(uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_sum_inscan(uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_sum_inscan(uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_sum_inscan(size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_sum_inscan(ptrdiff_t *, const ptrdiff_t *, size_t);

template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_sum_exscan(T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_sum_exscan(float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_sum_exscan(double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_sum_exscan(char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_sum_exscan(signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_sum_exscan(short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_sum_exscan(int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_sum_exscan(long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_sum_exscan(long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_sum_exscan(unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_sum_exscan(unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_sum_exscan(unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_sum_exscan(unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_sum_exscan(unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_sum_exscan(int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_sum_exscan(int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_sum_exscan(int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_sum_exscan(int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_sum_exscan(uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_sum_exscan(uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_sum_exscan(uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_sum_exscan(uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_sum_exscan(size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_sum_exscan(ptrdiff_t *, const ptrdiff_t *, size_t);

/* scan (prefix sum) on a team */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_sum_inscan(ishmem_team_t, T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_sum_inscan(ishmem_team_t, float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_sum_inscan(ishmem_team_t, double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_sum_inscan(ishmem_team_t, char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_sum_inscan(ishmem_team_t, signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_sum_inscan(ishmem_team_t, short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_sum_inscan(ishmem_team_t, int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_sum_inscan(ishmem_team_t, long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_sum_inscan(ishmem_team_t, long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_sum_inscan(ishmem_team_t, unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_sum_inscan(ishmem_team_t, unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_sum_inscan(ishmem_team_t, unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_sum_inscan(ishmem_team_t, unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_sum_inscan(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_sum_inscan(ishmem_team_t, int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_sum_inscan(ishmem_team_t, int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_sum_inscan(ishmem_team_t, int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_sum_inscan(ishmem_team_t, int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_sum_inscan(ishmem_team_t, uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_sum_inscan(ishmem_team_t, uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_sum_inscan(ishmem_team_t, uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_sum_inscan(ishmem_team_t, uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_sum_inscan(ishmem_team_t, size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_sum_inscan(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t);

template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_sum_exscan(ishmem_team_t, T *, const T *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_float_sum_exscan(ishmem_team_t, float *, const float *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_double_sum_exscan(ishmem_team_t, double *, const double *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_char_sum_exscan(ishmem_team_t, char *, const char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_schar_sum_exscan(ishmem_team_t, signed char *, const signed char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_short_sum_exscan(ishmem_team_t, short *, const short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_sum_exscan(ishmem_team_t, int *, const int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_sum_exscan(ishmem_team_t, long *, const long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_sum_exscan(ishmem_team_t, long long *, const long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uchar_sum_exscan(ishmem_team_t, unsigned char *, const unsigned char *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ushort_sum_exscan(ishmem_team_t, unsigned short *, const unsigned short *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_sum_exscan(ishmem_team_t, unsigned int *, const unsigned int *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_sum_exscan(ishmem_team_t, unsigned long *, const unsigned long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_sum_exscan(ishmem_team_t, unsigned long long *, const unsigned long long *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int8_sum_exscan(ishmem_team_t, int8_t *, const int8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int16_sum_exscan(ishmem_team_t, int16_t *, const int16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_sum_exscan(ishmem_team_t, int32_t *, const int32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_sum_exscan(ishmem_team_t, int64_t *, const int64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint8_sum_exscan(ishmem_team_t, uint8_t *, const uint8_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint16_sum_exscan(ishmem_team_t, uint16_t *, const uint16_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_sum_exscan(ishmem_team_t, uint32_t *, const uint32_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_sum_exscan(ishmem_team_t, uint64_t *, const uint64_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_sum_exscan(ishmem_team_t, size_t *, const size_t *, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_sum_exscan(ishmem_team_t, ptrdiff_t *, const ptrdiff_t *, size_t);

/* test */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_test(T *, int, T);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_test(int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_test(long *, int, long);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_test(long long *, int, long long);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_test(unsigned int *, int, unsigned int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_test(unsigned long *, int, unsigned long);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_test(unsigned long long *, int, unsigned long long);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_test(int32_t *, int, int32_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_test(int64_t *, int, int64_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_test(uint32_t *, int, uint32_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_test(uint64_t *, int, uint64_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_test(size_t *, int, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_test(ptrdiff_t *, int, ptrdiff_t);

/* test_all */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_test_all(T *, size_t, const int *, int, T);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_test_all(int *, size_t, const int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_test_all(long *, size_t, const int *, int, long);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_test_all(long long *, size_t, const int *, int, long long);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_test_all(unsigned int *, size_t, const int *, int, unsigned int);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_test_all(unsigned long *, size_t, const int *, int, unsigned long);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_test_all(unsigned long long *, size_t, const int *, int, unsigned long long);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_test_all(int32_t *, size_t, const int *, int, int32_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_test_all(int64_t *, size_t, const int *, int, int64_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_test_all(uint32_t *, size_t, const int *, int, uint32_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_test_all(uint64_t *, size_t, const int *, int, uint64_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_test_all(size_t *, size_t, const int *, int, size_t);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_test_all(ptrdiff_t *, size_t, const int *, int, ptrdiff_t);

/* test_any */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_test_any(T *, size_t, const int *, int, T);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int_test_any(int *, size_t, const int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_long_test_any(long *, size_t, const int *, int, long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_longlong_test_any(long long *, size_t, const int *, int, long long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint_test_any(unsigned int *, size_t, const int *, int, unsigned int);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulong_test_any(unsigned long *, size_t, const int *, int, unsigned long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulonglong_test_any(unsigned long long *, size_t, const int *, int, unsigned long long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int32_test_any(int32_t *, size_t, const int *, int, int32_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int64_test_any(int64_t *, size_t, const int *, int, int64_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint32_test_any(uint32_t *, size_t, const int *, int, uint32_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint64_test_any(uint64_t *, size_t, const int *, int, uint64_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_size_test_any(size_t *, size_t, const int *, int, size_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ptrdiff_test_any(ptrdiff_t *, size_t, const int *, int, ptrdiff_t);

/* test_some */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_test_some(T *, size_t, size_t *, const int *, int, T);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int_test_some(int *, size_t, size_t *, const int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_long_test_some(long *, size_t, size_t *, const int *, int, long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_longlong_test_some(long long *, size_t, size_t *, const int *, int, long long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint_test_some(unsigned int *, size_t, size_t *, const int *, int, unsigned int);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulong_test_some(unsigned long *, size_t, size_t *, const int *, int, unsigned long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulonglong_test_some(unsigned long long *, size_t, size_t *, const int *, int, unsigned long long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int32_test_some(int32_t *, size_t, size_t *, const int *, int, int32_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int64_test_some(int64_t *, size_t, size_t *, const int *, int, int64_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint32_test_some(uint32_t *, size_t, size_t *, const int *, int, uint32_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint64_test_some(uint64_t *, size_t, size_t *, const int *, int, uint64_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_size_test_some(size_t *, size_t, size_t *, const int *, int, size_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ptrdiff_test_some(ptrdiff_t *, size_t, size_t *, const int *, int, ptrdiff_t);

/* test_all_vector */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES int ishmem_test_all_vector(T *, size_t, const int *, int, const T *);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int_test_all_vector(int *, size_t, const int *, int, const int *);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_long_test_all_vector(long *, size_t, const int *, int, const long *);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_longlong_test_all_vector(long long *, size_t, const int *, int, const long long *);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint_test_all_vector(unsigned int *, size_t, const int *, int, const unsigned int *);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulong_test_all_vector(unsigned long *, size_t, const int *, int, const unsigned long *);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ulonglong_test_all_vector(unsigned long long *, size_t, const int *, int, const unsigned long long *);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int32_test_all_vector(int32_t *, size_t, const int *, int, const int32_t *);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_int64_test_all_vector(int64_t *, size_t, const int *, int, const int64_t *);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint32_test_all_vector(uint32_t *, size_t, const int *, int, const uint32_t *);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_uint64_test_all_vector(uint64_t *, size_t, const int *, int, const uint64_t *);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_size_test_all_vector(size_t *, size_t, const int *, int, const size_t *);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_ptrdiff_test_all_vector(ptrdiff_t *, size_t, const int *, int, const ptrdiff_t *);

/* test_any_vector */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_test_any_vector(T *, size_t, const int *, int, const T *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int_test_any_vector(int *, size_t, const int *, int, const int *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_long_test_any_vector(long *, size_t, const int *, int, const long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_longlong_test_any_vector(long long *, size_t, const int *, int, const long long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint_test_any_vector(unsigned int *, size_t, const int *, int, const unsigned int *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulong_test_any_vector(unsigned long *, size_t, const int *, int, const unsigned long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulonglong_test_any_vector(unsigned long long *, size_t, const int *, int, const unsigned long long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int32_test_any_vector(int32_t *, size_t, const int *, int, const int32_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int64_test_any_vector(int64_t *, size_t, const int *, int, const int64_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint32_test_any_vector(uint32_t *, size_t, const int *, int, const uint32_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint64_test_any_vector(uint64_t *, size_t, const int *, int, const uint64_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_size_test_any_vector(size_t *, size_t, const int *, int, const size_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ptrdiff_test_any_vector(ptrdiff_t *, size_t, const int *, int, const ptrdiff_t *);

/* test_some_vector */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_test_some_vector(T *, size_t, size_t *, const int *, int, const T *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int_test_some_vector(int *, size_t, size_t *, const int *, int, const int *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_long_test_some_vector(long *, size_t, size_t *, const int *, int, const long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_longlong_test_some_vector(long long *, size_t, size_t *, const int *, int, const long long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint_test_some_vector(unsigned int *, size_t, size_t *, const int *, int, const unsigned int *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulong_test_some_vector(unsigned long *, size_t, size_t *, const int *, int, const unsigned long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulonglong_test_some_vector(unsigned long long *, size_t, size_t *, const int *, int, const unsigned long long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int32_test_some_vector(int32_t *, size_t, size_t *, const int *, int, const int32_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int64_test_some_vector(int64_t *, size_t, size_t *, const int *, int, const int64_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint32_test_some_vector(uint32_t *, size_t, size_t *, const int *, int, const uint32_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint64_test_some_vector(uint64_t *, size_t, size_t *, const int *, int, const uint64_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_size_test_some_vector(size_t *, size_t, size_t *, const int *, int, const size_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ptrdiff_test_some_vector(ptrdiff_t *, size_t, size_t *, const int *, int, const ptrdiff_t *);

/* wait_until */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_wait_until(T *, int, T);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_wait_until(int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_wait_until(long *, int, long);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_wait_until(long long *, int, long long);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_wait_until(unsigned int *, int, unsigned int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_wait_until(unsigned long *, int, unsigned long);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_wait_until(unsigned long long *, int, unsigned long long);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_wait_until(int32_t *, int, int32_t);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_wait_until(int64_t *, int, int64_t);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_wait_until(uint32_t *, int, uint32_t);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_wait_until(uint64_t *, int, uint64_t);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_wait_until(size_t *, int, size_t);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_wait_until(ptrdiff_t *, int, ptrdiff_t);

/* wait_until_all */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_wait_until_all(T *, size_t, const int *, int, T);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_wait_until_all(int *, size_t, const int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_wait_until_all(long *, size_t, const int *, int, long);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_wait_until_all(long long *, size_t, const int *, int, long long);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_wait_until_all(unsigned int *, size_t, const int *, int, unsigned int);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_wait_until_all(unsigned long *, size_t, const int *, int, unsigned long);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_wait_until_all(unsigned long long *, size_t, const int *, int, unsigned long long);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_wait_until_all(int32_t *, size_t, const int *, int, int32_t);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_wait_until_all(int64_t *, size_t, const int *, int, int64_t);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_wait_until_all(uint32_t *, size_t, const int *, int, uint32_t);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_wait_until_all(uint64_t *, size_t, const int *, int, uint64_t);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_wait_until_all(size_t *, size_t, const int *, int, size_t);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_wait_until_all(ptrdiff_t *, size_t, const int *, int, ptrdiff_t);

/* wait_until_any */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_wait_until_any(T *, size_t, const int *, int, T);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int_wait_until_any(int *, size_t, const int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_long_wait_until_any(long *, size_t, const int *, int, long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_longlong_wait_until_any(long long *, size_t, const int *, int, long long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint_wait_until_any(unsigned int *, size_t, const int *, int, unsigned int);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulong_wait_until_any(unsigned long *, size_t, const int *, int, unsigned long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulonglong_wait_until_any(unsigned long long *, size_t, const int *, int, unsigned long long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int32_wait_until_any(int32_t *, size_t, const int *, int, int32_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int64_wait_until_any(int64_t *, size_t, const int *, int, int64_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint32_wait_until_any(uint32_t *, size_t, const int *, int, uint32_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint64_wait_until_any(uint64_t *, size_t, const int *, int, uint64_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_size_wait_until_any(size_t *, size_t, const int *, int, size_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ptrdiff_wait_until_any(ptrdiff_t *, size_t, const int *, int, ptrdiff_t);

/* wait_until_some */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_wait_until_some(T *, size_t, size_t *, const int *, int, T);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int_wait_until_some(int *, size_t, size_t *, const int *, int, int);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_long_wait_until_some(long *, size_t, size_t *, const int *, int, long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_longlong_wait_until_some(long long *, size_t, size_t *, const int *, int, long long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint_wait_until_some(unsigned int *, size_t, size_t *, const int *, int, unsigned int);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulong_wait_until_some(unsigned long *, size_t, size_t *, const int *, int, unsigned long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulonglong_wait_until_some(unsigned long long *, size_t, size_t *, const int *, int, unsigned long long);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int32_wait_until_some(int32_t *, size_t, size_t *, const int *, int, int32_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int64_wait_until_some(int64_t *, size_t, size_t *, const int *, int, int64_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint32_wait_until_some(uint32_t *, size_t, size_t *, const int *, int, uint32_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint64_wait_until_some(uint64_t *, size_t, size_t *, const int *, int, uint64_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_size_wait_until_some(size_t *, size_t, size_t *, const int *, int, size_t);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ptrdiff_wait_until_some(ptrdiff_t *, size_t, size_t *, const int *, int, ptrdiff_t);

/* wait_until_all_vector */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES void ishmem_wait_until_all_vector(T *, size_t, const int *, int, const T *);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int_wait_until_all_vector(int *, size_t, const int *, int, const int *);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_long_wait_until_all_vector(long *, size_t, const int *, int, const long *);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_longlong_wait_until_all_vector(long long *, size_t, const int *, int, const long long *);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint_wait_until_all_vector(unsigned int *, size_t, const int *, int, const unsigned int *);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulong_wait_until_all_vector(unsigned long *, size_t, const int *, int, const unsigned long *);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ulonglong_wait_until_all_vector(unsigned long long *, size_t, const int *, int, const unsigned long long *);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int32_wait_until_all_vector(int32_t *, size_t, const int *, int, const int32_t *);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_int64_wait_until_all_vector(int64_t *, size_t, const int *, int, const int64_t *);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint32_wait_until_all_vector(uint32_t *, size_t, const int *, int, const uint32_t *);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_uint64_wait_until_all_vector(uint64_t *, size_t, const int *, int, const uint64_t *);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_size_wait_until_all_vector(size_t *, size_t, const int *, int, const size_t *);
ISHMEM_DEVICE_ATTRIBUTES void ishmem_ptrdiff_wait_until_all_vector(ptrdiff_t *, size_t, const int *, int, const ptrdiff_t *);

/* wait_until_any_vector */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_wait_until_any_vector(T *, size_t, const int *, int, const T *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int_wait_until_any_vector(int *, size_t, const int *, int, const int *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_long_wait_until_any_vector(long *, size_t, const int *, int, const long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_longlong_wait_until_any_vector(long long *, size_t, const int *, int, const long long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint_wait_until_any_vector(unsigned int *, size_t, const int *, int, const unsigned int *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulong_wait_until_any_vector(unsigned long *, size_t, const int *, int, const unsigned long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulonglong_wait_until_any_vector(unsigned long long *, size_t, const int *, int, const unsigned long long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int32_wait_until_any_vector(int32_t *, size_t, const int *, int, const int32_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int64_wait_until_any_vector(int64_t *, size_t, const int *, int, const int64_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint32_wait_until_any_vector(uint32_t *, size_t, const int *, int, const uint32_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint64_wait_until_any_vector(uint64_t *, size_t, const int *, int, const uint64_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_size_wait_until_any_vector(size_t *, size_t, const int *, int, const size_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ptrdiff_wait_until_any_vector(ptrdiff_t *, size_t, const int *, int, const ptrdiff_t *);

/* wait_until_some_vector */
template <typename T> ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_wait_until_some_vector(T *, size_t, size_t *, const int *, int, const T *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int_wait_until_some_vector(int *, size_t, size_t *, const int *, int, const int *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_long_wait_until_some_vector(long *, size_t, size_t *, const int *, int, const long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_longlong_wait_until_some_vector(long long *, size_t, size_t *, const int *, int, const long long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint_wait_until_some_vector(unsigned int *, size_t, size_t *, const int *, int, const unsigned int *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulong_wait_until_some_vector(unsigned long *, size_t, size_t *, const int *, int, const unsigned long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ulonglong_wait_until_some_vector(unsigned long long *, size_t, size_t *, const int *, int, const unsigned long long *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int32_wait_until_some_vector(int32_t *, size_t, size_t *, const int *, int, const int32_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_int64_wait_until_some_vector(int64_t *, size_t, size_t *, const int *, int, const int64_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint32_wait_until_some_vector(uint32_t *, size_t, size_t *, const int *, int, const uint32_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_uint64_wait_until_some_vector(uint64_t *, size_t, size_t *, const int *, int, const uint64_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_size_wait_until_some_vector(size_t *, size_t, size_t *, const int *, int, const size_t *);
ISHMEM_DEVICE_ATTRIBUTES size_t ishmem_ptrdiff_wait_until_some_vector(ptrdiff_t *, size_t, size_t *, const int *, int, const ptrdiff_t *);

/* signal_wait_until */
ISHMEM_DEVICE_ATTRIBUTES uint64_t ishmem_signal_wait_until(uint64_t *, int, uint64_t);

/* barrier_all */
ISHMEM_DEVICE_ATTRIBUTES void ishmem_barrier_all(void);

/* sync */
ISHMEM_DEVICE_ATTRIBUTES void ishmem_sync_all(void);
ISHMEM_DEVICE_ATTRIBUTES int ishmem_team_sync(ishmem_team_t team);

/* fence */
ISHMEM_DEVICE_ATTRIBUTES void ishmem_fence(void);

/* quiet */
ISHMEM_DEVICE_ATTRIBUTES void ishmem_quiet(void);

/* clang-format on */
#endif /* I_SHMEM_H */
