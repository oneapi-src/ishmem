# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# -------------------------------------------------------------------
# Check for ishmem build

if (NOT EXISTS "${ISHMEM_DIR}/include/ishmem.h" OR
    NOT EXISTS "${ISHMEM_DIR}/include/ishmemx.h")
    message(FATAL_ERROR
            " Cannot find Intel SHMEM headers!\n"
            " Provided (ISHMEM_DIR): ${ISHMEM_DIR}\n"
            " Required headers:\n"
            "     ishmem.h\n"
            "     ishmemx.h")
endif()

# Only check for libishmem.a if this building the tests stand-alone
if (NOT ${CMAKE_PROJECT_NAME} STREQUAL "ishmem")
    if (NOT EXISTS "${ISHMEM_DIR}/lib/libishmem.a")
        message(FATAL_ERROR
                " Cannot find Intel SHMEM library!\n"
                " Provided (ISHMEM_DIR): ${ISHMEM_DIR}\n"
                " Required library:\n"
                "     libishmem.a")
    endif()
endif()

set(ISHMEM_INC_DIR "${ISHMEM_DIR}/include")
set(ISHMEM_LIB_DIR "${ISHMEM_DIR}/lib")

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
    set(LEVEL_ZERO_DIR "/usr" CACHE PATH "Path to oneAPI Level Zero installation")
    set(SHMEM_DIR "/usr" CACHE PATH "Path to OpenSHMEM installation")
    set(MPI_DIR "/usr" CACHE PATH "Path to MPI installation")

    setup_dependencies()
    setup_runtime_backends()
endif()

# -------------------------------------------------------------------
# Setup source files, include directories, and compiler settings

set(TEST_COMMON_SOURCE_FILES)
set(ISHMEM_TEST_LINK_LIBS)

list(APPEND ISHMEM_TEST_LINK_LIBS
    ze_loader
    pthread)

if (NOT ${CMAKE_PROJECT_NAME} STREQUAL "ishmem")
    list(APPEND ISHMEM_TEST_LINK_LIBS ${ISHMEM_LIB_DIR}/libishmem.a)
else()
    list(APPEND ISHMEM_TEST_LINK_LIBS ishmem-static)
endif()

list(APPEND ISHMEM_TEST_INCLUDE_DIRS
    "${ISHMEM_INC_DIR}"
    "${ISHMEM_TEST_ROOT_DIR}/include")

if (ENABLE_OPENSHMEM)
    list(APPEND ISHMEM_TEST_INCLUDE_DIRS "${SHMEM_INC_DIR}")
    list(APPEND TEST_COMMON_SOURCE_FILES ${ISHMEM_TEST_ROOT_DIR}/common/runtime_openshmem.cpp)
endif()

if (ENABLE_MPI)
    list(APPEND ISHMEM_TEST_INCLUDE_DIRS "${MPI_INC_DIR}")
    list(APPEND TEST_COMMON_SOURCE_FILES ${ISHMEM_TEST_ROOT_DIR}/common/runtime_mpi.cpp)
endif()

if (ENABLE_PMI)
    list(APPEND ISHMEM_TEST_LINK_LIBS pmi-simple)
endif()
