# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# -------------------------------------------------------------------
# Check for ishmem build

set(ISHMEM_SEARCH_PATHS "")
if (${CMAKE_PROJECT_NAME} STREQUAL "ishmem")
    # If the tests are getting configured alongside the library
    if (NOT DEFINED ISHMEM_ROOT)
        # Set the CMake variable ISHMEM_ROOT if it is not provided by the user
        # This ensures the local ishmem is favored over a system installed one
        set(ISHMEM_ROOT ${CMAKE_BINARY_DIR})
    endif()
else()
    if (DEFINED ISHMEM_DIR)
        # <PackageName>_DIR is typically reserved for the CMake config file location. However,
        # previous Intel SHMEM versions used this naming scheme to define the root install
        # directory. So we add both ${ISHMEM_DIR} and ${ISHMEM_DIR}/lib/cmake/ishmem as search
        # paths for find_package
        list(APPEND ISHMEM_SEARCH_PATHS ${ISHMEM_DIR} ${ISHMEM_DIR}/lib/cmake/ishmem)
    endif()
endif()
find_package(ISHMEM REQUIRED PATHS ${ISHMEM_SEARCH_PATHS})

# -------------------------------------------------------------------
# Configure the ctest launcher

set(CTEST_LAUNCHER srun CACHE STRING "Job scheduler used for ctest")
set(VALID_SCHEDULERS srun qsub mpi)

list(FIND VALID_SCHEDULERS "${CTEST_LAUNCHER}" SCHEDULER_FOUND)
if (SCHEDULER_FOUND EQUAL -1)
    string(REPLACE ";" ", " VALID_SCHEDULERS_CSV "${VALID_SCHEDULERS}")
    message(FATAL_ERROR
        "Invalid valid value for CTEST_LAUNCHER provided: ${CTEST_LAUNCHER}\n"
        "Supported launchers: ${VALID_SCHEDULERS_CSV}")
endif()

file(RELATIVE_PATH SCRIPTS_DIR ${CMAKE_CURRENT_BINARY_DIR} "${ISHMEM_ROOT_DIR}/scripts")

set(CTEST_WRAPPER "${SCRIPTS_DIR}/ctest/${CTEST_LAUNCHER}_wrapper")
set(ISHMEM_RUN_SCRIPT "${SCRIPTS_DIR}/ishmrun")

# -------------------------------------------------------------------
# Setup compiler

# Run compiler setup if this using a standalone build
if (NOT ${CMAKE_PROJECT_NAME} STREQUAL "ishmem")
    include(${ISHMEM_ROOT_DIR}/cmake/utils.cmake)
    setup_compiler_options()
endif()

# Set default build type
# Options are: Debug, Release, RelWithDebInfo, and MinSizeRel
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_CASE_INSENSITIVE)

# -------------------------------------------------------------------
# Setup dependencies and runtimes

# Set default paths
if (NOT ${CMAKE_PROJECT_NAME} STREQUAL "ishmem")
    setup_dependencies()
    setup_runtime_backends()
endif()

# -------------------------------------------------------------------
# Setup source files, include directories, and compiler settings

configure_file(${ISHMEM_TEST_ROOT_DIR}/include/ishmem_test_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/ishmem_test_config.h)

set(TEST_COMMON_SOURCE_FILES)
set(ISHMEM_TEST_LINK_LIBS)

list(APPEND ISHMEM_TEST_LINK_LIBS
    ${LEVEL_ZERO_LIBRARIES}
    pthread)

if (NOT ${CMAKE_PROJECT_NAME} STREQUAL "ishmem")
    list(APPEND ISHMEM_TEST_LINK_LIBS ${ISHMEM_LIBRARY})
else()
    list(APPEND ISHMEM_TEST_LINK_LIBS ishmem-static)
endif()

list(APPEND ISHMEM_TEST_INCLUDE_DIRS
    "${ISHMEM_INCLUDE}"
    "${ISHMEM_TEST_ROOT_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}/include")

if (ENABLE_OPENSHMEM)
    list(APPEND ISHMEM_TEST_INCLUDE_DIRS "${OPENSHMEM_INCLUDE_DIRS}")
    list(APPEND TEST_COMMON_SOURCE_FILES ${ISHMEM_TEST_ROOT_DIR}/common/runtime_openshmem.cpp)
endif()

if (ENABLE_MPI)
    list(APPEND ISHMEM_TEST_INCLUDE_DIRS "${MPI_CXX_INCLUDE_DIRS}")
    list(APPEND TEST_COMMON_SOURCE_FILES ${ISHMEM_TEST_ROOT_DIR}/common/runtime_mpi.cpp)
endif()

if (ENABLE_PMI)
    list(APPEND ISHMEM_TEST_LINK_LIBS pmi-simple)
endif()
