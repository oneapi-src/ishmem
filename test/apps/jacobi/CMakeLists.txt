# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.17)

file(GLOB TEST_SOURCE_FILES jacobi.cpp)

set(PROJECT_NAME "ishmem jacobi app")
set(PROJECT_FULL_NAME "Intel® SHMEM jacobi app")

project(${PROJECT_NAME})

# -------------------------------------------------------------------
# Perform common test setup

set(ISHMEM_ROOT_DIR "${PROJECT_SOURCE_DIR}/../../..")
set(ISHMEM_TEST_ROOT_DIR "${PROJECT_SOURCE_DIR}/../..")

include(${ISHMEM_TEST_ROOT_DIR}/cmake/common.cmake)

# -------------------------------------------------------------------
# Setup the common source files

add_library(ishmem-test-common-jacobi STATIC ${TEST_COMMON_SOURCE_FILES})
target_include_directories(ishmem-test-common-jacobi PUBLIC ${ISHMEM_TEST_INCLUDE_DIRS})
target_link_libraries(ishmem-test-common-jacobi PUBLIC ${ISHMEM_TEST_LINK_LIBS})

# -------------------------------------------------------------------
# Setup default test values

set(TARGET "jacobi")
set(ISHMEM_PE_COUNTS_JACOBI "2" CACHE STRING "Number of PEs to use for each test")
set(ISHMEM_JACOBI_COMMON_ARGS "-niter 6200 -nx 128 -ny 128")

enable_testing()

# -------------------------------------------------------------------
# Show build summary
message(STATUS "\n===== Apps (Jacobi) summary =====")
message(STATUS "Build type:                  ${CMAKE_BUILD_TYPE_CASE_INSENSITIVE}")
message(STATUS "PE count(s):                 ${ISHMEM_PE_COUNTS_JACOBI}")
message(STATUS "CTest launcher:              ${CTEST_LAUNCHER}")
message(STATUS "Enable AOT compilation:      ${ENABLE_AOT_COMPILATION}")
if (ENABLE_AOT_COMPILATION)
    message(STATUS "    AOT Device type(s): ${ISHMEM_AOT_DEVICE_TYPES}")
endif()
message(STATUS "Enable OpenSHMEM support:    ${ENABLE_OPENSHMEM}")
message(STATUS "Enable MPI support:          ${ENABLE_MPI}")
message(STATUS "=================================\n")

# -------------------------------------------------------------------
# Set build properties for test

add_executable(${TARGET} ${TEST_SOURCE_FILES})
set_target_properties(${TARGET} PROPERTIES ENABLE_EXPORTS TRUE)
target_link_libraries(${TARGET} PRIVATE ishmem-test-common-jacobi)

# -------------------------------------------------------------------
# Add ctests

foreach (N ${ISHMEM_PE_COUNTS_JACOBI})
    add_test(NAME ${TARGET}-${N} COMMAND ${CTEST_WRAPPER} ${N} ${ISHMEM_RUN_SCRIPT}
        ./${TARGET} ${ISHMEM_JACOBI_COMMON_ARGS} COMMAND_EXPAND_LISTS)
endforeach()
