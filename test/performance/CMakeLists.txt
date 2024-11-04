# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.17)

file(GLOB TEST_SOURCE_FILES *.cpp)

set(PROJECT_NAME "ishmem performance tests")
set(PROJECT_FULL_NAME "Intel® SHMEM performance tests")

project(${PROJECT_NAME})

# -------------------------------------------------------------------
# Perform common test setup

set(ISHMEM_ROOT_DIR "${PROJECT_SOURCE_DIR}/../..")
set(ISHMEM_TEST_ROOT_DIR "${PROJECT_SOURCE_DIR}/..")

include(${ISHMEM_TEST_ROOT_DIR}/cmake/common.cmake)

# -------------------------------------------------------------------
# Setup the common source files

add_library(ishmem-test-common-performance STATIC ${TEST_COMMON_SOURCE_FILES})
target_include_directories(ishmem-test-common-performance PUBLIC ${ISHMEM_TEST_INCLUDE_DIRS})
target_link_libraries(ishmem-test-common-performance PUBLIC ${ISHMEM_TEST_LINK_LIBS})

# -------------------------------------------------------------------
# Setup default test values

set(ISHMEM_PE_COUNTS_PERF_TESTS "2" CACHE STRING "Number of PEs to use for each test")
set(ISHMEM_PERF_TEST_COMMON_ARGS "--csv -m 512 --test_modes device")

enable_testing()

# -------------------------------------------------------------------
# Show build summary
message(STATUS "\n=== Performance tests summary ===")
message(STATUS "Build type:                  ${CMAKE_BUILD_TYPE_CASE_INSENSITIVE}")
message(STATUS "PE count(s):                 ${ISHMEM_PE_COUNTS_PERF_TESTS}")
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
    target_link_libraries(${EXE} PRIVATE ishmem-test-common-performance)
endforeach()

# -------------------------------------------------------------------
# Add ctests

foreach(TEST_SOURCE_FILE ${TEST_SOURCE_FILES})
    get_filename_component(EXE ${TEST_SOURCE_FILE} NAME_WE)

    # These are standard tests
    foreach (N ${ISHMEM_PE_COUNTS_PERF_TESTS})
        add_test(NAME ${EXE}-${N} COMMAND ${CTEST_WRAPPER} ${N} ${ISHMEM_RUN_SCRIPT}
            ./${EXE} ${ISHMEM_PERF_TEST_COMMON_ARGS} COMMAND_EXPAND_LISTS)
    endforeach()
endforeach()
