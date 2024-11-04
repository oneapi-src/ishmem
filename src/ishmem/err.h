/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_ERR_H
#define ISHMEM_ERR_H

#include <cstddef>
#include "ishmem.h"
#include "ishmem/util.h"
#include "ishmem/env_utils.h"

#define ISHMEMI_ERROR_MPI  1
#define ISHMEMI_ERROR_ZE   2
#define ISHMEMI_ERROR_SOCK 3

#define RAISE_PE_PREFIX     "[%04d]        "
#define ISHMEMI_DIAG_STRLEN 1024

void ishmemi_print_trace();

#define ZE_ERR_NAME_EXPANSION(name)                                                                \
    case name:                                                                                     \
        err_name = #name;                                                                          \
        break;

/* Level Zero API doesn't provide an err-to-str function, so make our own */
#define ZE_ERR_GET_NAME(err, err_name)                                                             \
    switch (err) {                                                                                 \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_DEVICE_LOST);                                        \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_UNINITIALIZED);                                      \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_UNSUPPORTED_VERSION);                                \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);                                \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_UNSUPPORTED_SIZE);                                   \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_INVALID_ARGUMENT);                                   \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_INVALID_ENUMERATION);                                \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_INVALID_NULL_HANDLE);                                \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_INVALID_NULL_POINTER);                               \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_INVALID_SIZE);                                       \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT);                     \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY);                                 \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_OVERLAPPING_REGIONS);                                \
        ZE_ERR_NAME_EXPANSION(ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE);                               \
        default:                                                                                   \
            err_name = "Unknown";                                                                  \
            break;                                                                                 \
    }

#define ISHMEM_COMMON_MSG_INTERNAL(typestring, file, line, func, ...)                              \
    do {                                                                                           \
        char str[ISHMEMI_DIAG_STRLEN];                                                             \
        size_t off;                                                                                \
        if (ishmemi_params.ENABLE_VERBOSE_PRINT) {                                                 \
            off = (size_t) snprintf(str, sizeof(str), "[%04d] %s:  %s:%ld: %s\n", ishmemi_my_pe,   \
                                    typestring, file, line, func);                                 \
        } else {                                                                                   \
            off = (size_t) snprintf(str, sizeof(str), "[%04d] %s: %s\n", ishmemi_my_pe,            \
                                    typestring, func);                                             \
        }                                                                                          \
        off += (size_t) snprintf(str + off, sizeof(str) - off, RAISE_PE_PREFIX, ishmemi_my_pe);    \
        off += (size_t) snprintf(str + off, sizeof(str) - off, __VA_ARGS__);                       \
        fprintf(stderr, "%s", str);                                                                \
    } while (0)

#define ISHMEM_COMMON_MSG(typestring, ...)                                                         \
    ISHMEM_COMMON_MSG_INTERNAL(typestring, __FILE__, (long int) __LINE__, __func__, __VA_ARGS__)

#define ISHMEM_WARN_MSG(...) ISHMEM_COMMON_MSG("WARN", __VA_ARGS__)

#define ISHMEM_DEBUG_MSG(...)                                                                      \
    do {                                                                                           \
        if (ishmemi_params.DEBUG) {                                                                \
            ISHMEM_COMMON_MSG("DEBUG", __VA_ARGS__);                                               \
        }                                                                                          \
    } while (0)

#ifdef __SYCL_DEVICE_ONLY__
#define ISHMEM_ERROR_MSG(...)
#else
#define ISHMEM_ERROR_MSG(...) ISHMEM_COMMON_MSG("ERROR", __VA_ARGS__)
#endif

#define ISHMEM_CHECK_GOTO_MSG(ret, lbl, ...)                                                       \
    do {                                                                                           \
        if (ret) {                                                                                 \
            ISHMEM_WARN_MSG(__VA_ARGS__);                                                          \
            goto lbl;                                                                              \
        }                                                                                          \
    } while (0)

#define ISHMEM_CHECK_RETURN_MSG(ret, ...)                                                          \
    do {                                                                                           \
        if (ret) {                                                                                 \
            ISHMEM_WARN_MSG(__VA_ARGS__);                                                          \
            return ret;                                                                            \
        }                                                                                          \
    } while (0)

#ifdef __SYCL_DEVICE_ONLY__
#define RAISE_ERROR_MSG(format, ...)                                                               \
    do {                                                                                           \
        ishmemx_print(__FILE__, __LINE__, __FUNCTION__, format, ishmemx_print_msg_type_t::ERROR);  \
    } while (0)
#else
#define RAISE_ERROR_MSG(...)                                                                       \
    do {                                                                                           \
        ISHMEM_ERROR_MSG(__VA_ARGS__);                                                             \
        if (ishmemi_params.DEBUG) ishmemi_print_trace();                                           \
        exit(1);                                                                                   \
    } while (0)
#endif

/* TODO recommend changing this assign to ret with something returning a value */
#define ZE_CHECK(call)                                                                             \
    do {                                                                                           \
        ze_result_t status = call;                                                                 \
        std::string err_name;                                                                      \
        if (status != ZE_RESULT_SUCCESS) {                                                         \
            ZE_ERR_GET_NAME(status, err_name);                                                     \
            ISHMEM_ERROR_MSG("ZE FAIL: call = '%s' result = '0x%x' (%s)\n", #call, status,         \
                             err_name.c_str());                                                    \
            ret = ISHMEMI_ERROR_ZE;                                                                \
        }                                                                                          \
    } while (0)

#define ISHMEMI_CHECK_RESULT(status, pass, label)                                                  \
    do {                                                                                           \
        if (status != pass) {                                                                      \
            ret = status;                                                                          \
            goto label;                                                                            \
        }                                                                                          \
    } while (0)

#define validate_init() validate_init_internal(__FILE__, __LINE__, __func__)
/* Parameter validation function */
#define validate_parameters(...)                                                                   \
    validate_parameters_internal(__FILE__, __LINE__, __func__, __VA_ARGS__)
/* Internal validation helper functions */
ISHMEM_DEVICE_ATTRIBUTES void validate_init_internal(const char *file, long int line,
                                                     const char *func);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters_internal(const char *file, long int line,
                                                           const char *func, int pe);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters_internal(const char *file, long int line,
                                                           const char *func, int pe, void *ptr,
                                                           size_t src);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters_internal(const char *file, long int line,
                                                           const char *func, int pe, void *ptr1,
                                                           void *ptr2, size_t size);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters_internal(const char *file, long int line,
                                                           const char *func, int pe, void *ptr1,
                                                           void *ptr2, size_t size, ptrdiff_t dst,
                                                           ptrdiff_t sst, size_t bsize = 1);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters_internal(const char *file, long int line,
                                                           const char *func, int pe_root,
                                                           void *dest, void *src, size_t dest_size,
                                                           size_t src_size);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters_internal(const char *file, long int line,
                                                           const char *func, int pe, void *ptr1,
                                                           void *ptr2, void *sig_addr, size_t size,
                                                           size_t sig_addr_size);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters_internal(const char *file, long int line,
                                                           const char *func, void *ivar,
                                                           size_t size);

ISHMEM_DEVICE_ATTRIBUTES void validate_parameters_internal(const char *file, long int line,
                                                           const char *func, void *dest, void *src,
                                                           size_t size);

ISHMEM_DEVICE_ATTRIBUTES void validate_parameters_internal(const char *file, long int line,
                                                           const char *func, void *ptrA, void *ptrB,
                                                           size_t sizeA, size_t sizeB,
                                                           ishmemi_op_t type);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters_internal(const char *file, long int line,
                                                           const char *func, void *dest, void *src,
                                                           size_t dest_size, size_t src_size);
ISHMEM_DEVICE_ATTRIBUTES void validate_parameters_internal(const char *file, long int line,
                                                           const char *func, void *ivars,
                                                           void *indices, void *status,
                                                           size_t ivars_size, size_t indices_size,
                                                           size_t status_size);

#if defined(ENABLE_ERROR_CHECKING)
constexpr bool enable_error_checking = true;
#else
constexpr bool enable_error_checking = false;
#endif

#endif /* ISHMEM_ERR_H */
