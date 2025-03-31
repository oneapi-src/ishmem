# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#
# Common CMakeLists.txt to build ishmem, tests, examples

cmake_minimum_required(VERSION 3.17)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(PROJECT_NAME "ishmem")
set(PROJECT_FULL_NAME "Intel® SHMEM")

set(ISHMEM_MAJOR_VERSION "1")
set(ISHMEM_MINOR_VERSION "3")
set(ISHMEM_PATCH_VERSION "0")
set(PROJECT_VERSION "${ISHMEM_MAJOR_VERSION}.${ISHMEM_MINOR_VERSION}.${ISHMEM_PATCH_VERSION}")

project(${PROJECT_NAME} VERSION ${PROJECT_VERSION})

set(COMMON_CMAKE_DIR "${PROJECT_SOURCE_DIR}/cmake")
include(${COMMON_CMAKE_DIR}/utils.cmake)

# -------------------------------------------------------------------
# Configuration Options
option(BUILD_UNIT_TESTS "Build unit tests" FALSE)
option(BUILD_PERF_TESTS "Build performance tests" FALSE)
option(BUILD_EXAMPLES "Build examples" FALSE)
option(BUILD_APPS "Build apps" FALSE)
option(BUILD_CMAKE_CONFIG "Build CMake config files" TRUE)
option(ENABLE_ERROR_CHECKING "Validate API inputs" FALSE)
option(ENABLE_DLMALLOC "Enable dlmalloc for shared heap" TRUE)
option(ENABLE_REDUCED_LINK_ENGINES "Enable reduced link engines (i.e. for single tile devices)" FALSE)
option(ENABLE_AOT_COMPILATION "Enables AOT compilation for GPU kernels" TRUE)

# Other options defined elsewhere
# - SKIP_COMPILER_CHECK - default: FALSE  (cmake/utils.cmake)
# - ENABLE_OPENSHMEM    - default: TRUE   (cmake/utils.cmake)
# - ENABLE_MPI          - default: FALSE  (cmake/utils.cmake)

# Set default device type(s) for AOT compilation
if (NOT ISHMEM_AOT_DEVICE_TYPES)
    set(ISHMEM_AOT_DEVICE_TYPES "pvc")
endif()

# -------------------------------------------------------------------
# Set default install path
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/_install" CACHE PATH "Default install path" FORCE)
endif()

# -------------------------------------------------------------------
# Setup compiler
setup_compiler_options()

# Set default build type
# Options are: Debug, Release, RelWithDebInfo, and MinSizeRel
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_CASE_INSENSITIVE)

# -------------------------------------------------------------------
# Show build summary
message(STATUS "\n========== Build summary ==========")
message(STATUS "Installation directory:      ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CXX compiler:                ${CMAKE_CXX_COMPILER}")
message(STATUS "Build type:                  ${CMAKE_BUILD_TYPE_CASE_INSENSITIVE}")
message(STATUS "Build unit tests:            ${BUILD_UNIT_TESTS}")
message(STATUS "Build performance tests:     ${BUILD_PERF_TESTS}")
message(STATUS "Build examples:              ${BUILD_EXAMPLES}")
message(STATUS "Build apps:                  ${BUILD_APPS}")
message(STATUS "Build CMake configs:         ${BUILD_CMAKE_CONFIG}")
message(STATUS "Enable input validation:     ${ENABLE_ERROR_CHECKING}")
message(STATUS "Enable dlmalloc:             ${ENABLE_DLMALLOC}")
message(STATUS "Enable reduced link engines: ${ENABLE_REDUCED_LINK_ENGINES}")
message(STATUS "Enable AOT compilation:      ${ENABLE_AOT_COMPILATION}")
if (ENABLE_AOT_COMPILATION)
    message(STATUS "    AOT Device type(s): ${ISHMEM_AOT_DEVICE_TYPES}")
endif()
message(STATUS "Enable OpenSHMEM support:    ${ENABLE_OPENSHMEM}")
message(STATUS "Enable MPI support:          ${ENABLE_MPI}")
message(STATUS "===================================\n")

# -------------------------------------------------------------------
# Set installation options

include(GNUInstallDirs)

set(CMAKE_INSTALL_LIBDIR "lib")
set(ISHMEM_INSTALL_LIB "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(ISHMEM_INSTALL_INC "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
set(ISHMEM_INSTALL_DOC "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DOCDIR}")
set(ISHMEM_INSTALL_BIN "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
set(ISHMEM_INSTALL_ENV "${CMAKE_INSTALL_PREFIX}/env")
set(ISHMEM_INSTALL_ETC "${CMAKE_INSTALL_PREFIX}/etc")
set(ISHMEM_INSTALL_SHARE "${CMAKE_INSTALL_PREFIX}/share")
set(ISHMEM_INSTALL_LICENSE "${ISHMEM_INSTALL_SHARE}/doc/ishmem/licensing")
set(ISHMEM_INSTALL_TESTS "${CMAKE_INSTALL_PREFIX}/tests")
set(ISHMEM_INSTALL_EXAMPLES "${ISHMEM_INSTALL_SHARE}/doc/ishmem/examples")
set(ISHMEM_INSTALL_MODULE "${ISHMEM_INSTALL_ETC}/modulefiles")

# -------------------------------------------------------------------
# Setup dependencies and runtimes

# Set default paths
set(LEVEL_ZERO_DIR "/usr" CACHE PATH "Path to oneAPI Level Zero installation")
set(SHMEM_DIR "/usr" CACHE PATH "Path to OpenSHMEM installation")
set(MPI_DIR "/usr" CACHE PATH "Path to MPI installation")

setup_dependencies()
setup_runtime_backends()

# -------------------------------------------------------------------
# Misc project settings

set(ISHMEM_BUILD_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
set(ISHMEM_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")

# This is set to cover building tests with the source. It can be set by users in case the tests are built stand-alone
set(ISHMEM_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "Path to Intel SHMEM installation")

# -------------------------------------------------------------------
# Generate and install files
configure_file(${PROJECT_SOURCE_DIR}/src/ishmem_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/ishmem/config.h)
configure_file(${PROJECT_SOURCE_DIR}/cmake/ishmem ${CMAKE_CURRENT_BINARY_DIR}/ishmem @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/cmake/vars.sh.in ${CMAKE_CURRENT_BINARY_DIR}/vars.sh @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/pkgconfig/ishmem.pc.in ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig/ishmem.pc @ONLY)

install(PROGRAMS scripts/ishmrun DESTINATION bin)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/ishmem DESTINATION ${ISHMEM_INSTALL_MODULE})
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/vars.sh DESTINATION ${ISHMEM_INSTALL_ENV})
install(FILES ${PROJECT_SOURCE_DIR}/third-party-programs.txt DESTINATION ${ISHMEM_INSTALL_LICENSE})
install(FILES ${PROJECT_SOURCE_DIR}/LICENSE DESTINATION ${ISHMEM_INSTALL_LICENSE} RENAME license.txt)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig/ishmem.pc DESTINATION ${ISHMEM_INSTALL_LIB}/pkgconfig)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/examples/ DESTINATION ${ISHMEM_INSTALL_EXAMPLES}
        FILES_MATCHING PATTERN "*.cpp"
                       PATTERN "build" EXCLUDE)

if (BUILD_CMAKE_CONFIG)
    # TODO
endif()

# -------------------------------------------------------------------
# Include other CMakeLists.txt

add_subdirectory(pmi-simple)
add_subdirectory(src)

# Build options for tests are handled directly in test/CMakeLists.txt
add_subdirectory(test)

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
