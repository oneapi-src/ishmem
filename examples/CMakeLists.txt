# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.17)

file(GLOB TEST_SOURCE_FILES *.cpp)

set(PROJECT_NAME "ishmem examples")
set(PROJECT_FULL_NAME "Intel(R) SHMEM examples")

project(${PROJECT_NAME})

# -------------------------------------------------------------------
# Configuration Options
option(ENABLE_AOT_COMPILATION "Enables AOT compilation for GPU kernels" TRUE)

# Set default device type(s) for AOT compilation
if (NOT ISHMEM_AOT_DEVICE_TYPES)
    # xe-hpc: Intel(R) Data Center GPU Max Series
    # xe2: Intel(R) Arc(TM) B-Series GPU Family
    set(ISHMEM_AOT_DEVICE_TYPES "xe-hpc,xe2")
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

set(ISHMEM_PE_COUNTS_EXAMPLES "2" CACHE STRING "Number of PEs to use for each test")
set(ISHMEM_NON_STANDARD_CTEST_FILES
    7_uid_mpcp)

set(ISHMEM_STANDALONE_RUN_SCRIPT "${SCRIPTS_DIR}/ctest/standalone_wrapper")

enable_testing()

# Skip 7_uid_mpcp unless Intel(R) MPI Library >= 2021.18 is available.
set(ISHMEM_BUILD_7_UID_MPCP FALSE)
set(ISHMEM_7_UID_MPCP_STATUS "skipped (MPI support is disabled)")

if (ENABLE_MPI)
    set(ISHMEM_SKIP_7_UID_MPCP_REASON "")

    find_package(MPI QUIET COMPONENTS CXX)
    if (NOT MPI_CXX_FOUND)
        set(ISHMEM_SKIP_7_UID_MPCP_REASON "MPI CXX component not found")
    elseif (NOT MPIEXEC_EXECUTABLE)
        set(ISHMEM_SKIP_7_UID_MPCP_REASON "MPIEXEC_EXECUTABLE was not found")
    else()
        # 7_uid_mpcp requires Intel(R) MPI Library >= 2021.18.
        execute_process(
            COMMAND ${MPIEXEC_EXECUTABLE} --version
            OUTPUT_VARIABLE ISHMEM_MPI_OUT
            ERROR_VARIABLE ISHMEM_MPI_ERR
            RESULT_VARIABLE ISHMEM_MPI_RC
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE)

        if (NOT ISHMEM_MPI_RC EQUAL 0)
            execute_process(
                COMMAND ${MPIEXEC_EXECUTABLE} -V
                OUTPUT_VARIABLE ISHMEM_MPI_OUT
                ERROR_VARIABLE ISHMEM_MPI_ERR
                RESULT_VARIABLE ISHMEM_MPI_RC
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_STRIP_TRAILING_WHITESPACE)
        endif()

        set(ISHMEM_MPI_VERSION_TEXT "${ISHMEM_MPI_OUT}\n${ISHMEM_MPI_ERR}")

        string(REGEX MATCH "Intel\\(R\\).*MPI" ISHMEM_IS_INTEL_MPI "${ISHMEM_MPI_VERSION_TEXT}")
        if (NOT ISHMEM_IS_INTEL_MPI)
            set(ISHMEM_SKIP_7_UID_MPCP_REASON
                "detected MPI does not appear to be Intel(R) MPI Library")
        else()
            string(REGEX MATCH "Version[ ]*([0-9]+\\.[0-9]+)" ISHMEM_INTEL_MPI_VERSION_MATCH
                "${ISHMEM_MPI_VERSION_TEXT}")
            if (NOT ISHMEM_INTEL_MPI_VERSION_MATCH)
                set(ISHMEM_SKIP_7_UID_MPCP_REASON
                    "could not parse Intel(R) MPI Library release from mpiexec output")
            else()
                set(ISHMEM_INTEL_MPI_VERSION "${CMAKE_MATCH_1}")
                if (ISHMEM_INTEL_MPI_VERSION VERSION_LESS "2021.18")
                    set(ISHMEM_SKIP_7_UID_MPCP_REASON
                        "requires Intel(R) MPI Library release >= 2021.18, found ${ISHMEM_INTEL_MPI_VERSION}")
                else()
                    set(ISHMEM_BUILD_7_UID_MPCP TRUE)
                    set(ISHMEM_7_UID_MPCP_STATUS
                        "enabled (Intel(R) MPI Library ${ISHMEM_INTEL_MPI_VERSION})")
                endif()
            endif()
        endif()
    endif()

    if (NOT ISHMEM_BUILD_7_UID_MPCP)
        set(ISHMEM_7_UID_MPCP_STATUS "skipped (${ISHMEM_SKIP_7_UID_MPCP_REASON})")
    endif()
endif()

# -------------------------------------------------------------------
# Show build summary
message(STATUS "\n======== Examples summary ========")
message(STATUS "Build type:                  ${CMAKE_BUILD_TYPE_CASE_INSENSITIVE}")
message(STATUS "PE count(s):                 ${ISHMEM_PE_COUNTS_EXAMPLES}")
message(STATUS "CTest launcher:              ${CTEST_LAUNCHER}")
message(STATUS "Host link type:              ${ISHMEM_HOST_LINK_TYPE}")
message(STATUS "Enable AOT compilation:      ${ENABLE_AOT_COMPILATION}")
if (ENABLE_AOT_COMPILATION)
    message(STATUS "    AOT Device type(s): ${ISHMEM_AOT_DEVICE_TYPES}")
endif()
message(STATUS "Enable OpenSHMEM support:    ${ENABLE_OPENSHMEM}")
message(STATUS "Enable MPI support:          ${ENABLE_MPI}")
message(STATUS "Example 7_uid_mpcp support:  ${ISHMEM_7_UID_MPCP_STATUS}")
message(STATUS "==================================\n")

# -------------------------------------------------------------------
# Set build properties for all tests

foreach(TEST_SOURCE_FILE ${TEST_SOURCE_FILES})
    get_filename_component(EXE ${TEST_SOURCE_FILE} NAME_WE)

    # Skip 7_uid_mpcp unless Intel(R) MPI Library >= 2021.18 is available.
    if (EXE STREQUAL "7_uid_mpcp" AND NOT ISHMEM_BUILD_7_UID_MPCP)
        continue()
    endif()

    add_executable(${EXE} ${TEST_SOURCE_FILE})
    set_target_properties(${EXE} PROPERTIES ENABLE_EXPORTS TRUE)
    target_link_libraries(${EXE} PRIVATE ishmem-test-common-examples)

    # Special handling for examples that directly use MPI
    if (EXE STREQUAL "7_uid_mpcp")
        if (ENABLE_MPI)
            find_package(MPI REQUIRED)
            target_link_libraries(${EXE} PRIVATE MPI::MPI_CXX)
        endif()
    endif()
endforeach()

# -------------------------------------------------------------------
# Add ctests

foreach(TEST_SOURCE_FILE ${TEST_SOURCE_FILES})
    get_filename_component(EXE ${TEST_SOURCE_FILE} NAME_WE)

    if (${EXE} IN_LIST ISHMEM_NON_STANDARD_CTEST_FILES)
        continue()
    else()
        # These are standard tests
        foreach (N ${ISHMEM_PE_COUNTS_EXAMPLES})
            add_test(NAME ${EXE}-${N} COMMAND ${CTEST_WRAPPER} ${N} ${ISHMEM_RUN_SCRIPT}
                ./${EXE} COMMAND_EXPAND_LISTS)
        endforeach()
    endif()
endforeach()

if (ISHMEM_BUILD_7_UID_MPCP)
    foreach (N ${ISHMEM_PE_COUNTS_EXAMPLES})
        add_test(NAME 7_uid_mpcp-${N} COMMAND
            ${ISHMEM_STANDALONE_RUN_SCRIPT} ${N} ./7_uid_mpcp${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
endif()
