# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.17)

file(GLOB TEST_SOURCE_FILES *.cpp)

set(PROJECT_NAME "ishmem internal tests")
set(PROJECT_FULL_NAME "Intel(R) SHMEM internal tests")

project(${PROJECT_NAME})

# -------------------------------------------------------------------
# Perform common test setup

set(ISHMEM_ROOT_DIR "${PROJECT_SOURCE_DIR}/../..")
set(ISHMEM_TEST_ROOT_DIR "${PROJECT_SOURCE_DIR}/..")

include(${ISHMEM_TEST_ROOT_DIR}/cmake/common.cmake)

# -------------------------------------------------------------------
# Setup the common source files

add_library(ishmem-test-common-internal STATIC ${TEST_COMMON_SOURCE_FILES})
target_include_directories(ishmem-test-common-internal PUBLIC ${ISHMEM_TEST_INCLUDE_DIRS})
target_link_libraries(ishmem-test-common-internal PUBLIC ${ISHMEM_TEST_LINK_LIBS})

# -------------------------------------------------------------------
# Set build properties for all tests

foreach(TEST_SOURCE_FILE ${TEST_SOURCE_FILES})
    get_filename_component(EXE ${TEST_SOURCE_FILE} NAME_WE)
    add_executable(${EXE} ${TEST_SOURCE_FILE})
    set_target_properties(${EXE} PROPERTIES ENABLE_EXPORTS TRUE)
    target_link_libraries(${EXE} PRIVATE ishmem-test-common-internal)
endforeach()
