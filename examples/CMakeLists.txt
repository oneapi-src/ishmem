# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.17)

file(GLOB TEST_SOURCE_FILES *.cpp)

set(PROJECT_NAME "ishmem examples")
set(PROJECT_FULL_NAME "Intel® SHMEM examples")

project(${PROJECT_NAME})

# -------------------------------------------------------------------
# Configuration Options
option(ENABLE_AOT_COMPILATION "Enables AOT compilation for GPU kernels" TRUE)

# Set default device type(s) for AOT compilation
if (NOT ISHMEM_AOT_DEVICE_TYPES)
    set(ISHMEM_AOT_DEVICE_TYPES "pvc")
endif()

# -------------------------------------------------------------------
# Perform common test setup

set(ISHMEM_ROOT_DIR "${PROJECT_SOURCE_DIR}/..")
set(ISHMEM_TEST_ROOT_DIR "${PROJECT_SOURCE_DIR}/../test")

include(${ISHMEM_TEST_ROOT_DIR}/cmake/common.cmake)

# -------------------------------------------------------------------
# Setup the common source files

add_library(ishmem-test-common-examples STATIC ${TEST_COMMON_SOURCE_FILES})
target_include_directories(ishmem-test-common-examples PUBLIC ${ISHMEM_TEST_INCLUDE_DIRS})
target_link_libraries(ishmem-test-common-examples PUBLIC ${ISHMEM_TEST_LINK_LIBS})

# -------------------------------------------------------------------
# Setup default test values

set(ISHMEM_PE_COUNTS_EXAMPLES "1;2" CACHE STRING "Number of PEs to use for each test")

enable_testing()

# -------------------------------------------------------------------
# Show build summary
message(STATUS "\n======== Examples summary ========")
message(STATUS "Build type:                  ${CMAKE_BUILD_TYPE_CASE_INSENSITIVE}")
message(STATUS "PE count(s):                 ${ISHMEM_PE_COUNTS_EXAMPLES}")
message(STATUS "CTest launcher:              ${CTEST_LAUNCHER}")
message(STATUS "Enable AOT compilation:      ${ENABLE_AOT_COMPILATION}")
if (ENABLE_AOT_COMPILATION)
    message(STATUS "    AOT Device type(s): ${ISHMEM_AOT_DEVICE_TYPES}")
endif()
message(STATUS "Enable OpenSHMEM support:    ${ENABLE_OPENSHMEM}")
message(STATUS "Enable MPI support:          ${ENABLE_MPI}")
message(STATUS "==================================\n")

# -------------------------------------------------------------------
# Set build properties for all tests

foreach(TEST_SOURCE_FILE ${TEST_SOURCE_FILES})
    get_filename_component(EXE ${TEST_SOURCE_FILE} NAME_WE)
    add_executable(${EXE} ${TEST_SOURCE_FILE})
    set_target_properties(${EXE} PROPERTIES ENABLE_EXPORTS TRUE)
    target_link_libraries(${EXE} PRIVATE ishmem-test-common-examples)
endforeach()

# -------------------------------------------------------------------
# Add ctests

foreach(TEST_SOURCE_FILE ${TEST_SOURCE_FILES})
    get_filename_component(EXE ${TEST_SOURCE_FILE} NAME_WE)

    # These are standard tests
    foreach (N ${ISHMEM_PE_COUNTS_EXAMPLES})
        add_test(NAME ${EXE}-${N} COMMAND ${CTEST_WRAPPER} ${N} ${ISHMEM_RUN_SCRIPT}
            ./${EXE} COMMAND_EXPAND_LISTS)
    endforeach()
endforeach()
