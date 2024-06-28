# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.17)

project(ishmem VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(ISHMRUN_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/ishmrun)

# Configurable paths
set(L0_INSTALL_PREFIX "/usr" CACHE PATH "Path to L0 installation")
set(SHMEM_INSTALL_PREFIX "/usr" CACHE PATH "Path to OpenSHMEM installation")

option(BUILD_SHARED_LIBS ON)

# Build flags for choosing back-ends
option(ENABLE_OPENSHMEM "Enable OpenSHMEM runtime support" ON)
option(ENABLE_MPI "Enable MPI runtime support" OFF)
#option(ENABLE_PMI "Enable PMI runtime support" OFF)
#option(ENABLE_OSHMPI "Enable OSHMPI support for MPI runtimes" OFF)
#option(ENABLE_ONECCL "Enable ONECCL support for collectives" OFF)

# Build flags for selecting unit and performance tests
option(ENABLE_ERROR_CHECKING "Verify correctness of API arguments" OFF)
option(BUILD_TEST "Build Test" ON)
option(BUILD_PERF_TEST "Build Performance Test" ON)

# Build flags for choosing different configurations
#option(ENABLE_GPU_RDMA "Enable GPU RDMA support" ON)

# Other build flags
option(USE_REDUCED_LINK_ENGINE_SET "Reduced link engines for single tile device" OFF)
option(USE_DLMALLOC "DLMALLOC library for allocation and free" ON)

option(CTEST_SCHEDULER "Job scheduler used for ctest" OFF)
if(NOT CTEST_SCHEDULER)
    set(CTEST_SCHEDULER srun CACHE STRING "Job scheduler used for ctest" FORCE)
endif()
set(valid_schedulers srun qsub mpi)
list(FIND valid_schedulers "${CTEST_SCHEDULER}" scheduler_found)
if (scheduler_found EQUAL -1)
        string(REPLACE ";" ", " valid_schedulers_csv "${valid_schedulers}")
        message(FATAL_ERROR
                "Invalid valid value for CTEST_SCHEDULER provided: ${CTEST_SCHEDULER}\n"
                "Supported schedulers: ${valid_schedulers_csv}")
endif()
set(CTEST_WRAPPER ${CMAKE_CURRENT_SOURCE_DIR}/scripts/ctest/${CTEST_SCHEDULER}_wrapper)

enable_testing()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

configure_file(ishmem_config.h.in ishmem_config.h)

# adopt icpx -fsycl for all compilation, replacing dpcpp
set(CMAKE_CXX_COMPILER icpx)
add_compile_options(-fsycl)
add_link_options(-fsycl)

if (ISHMEM_USE_OFFLINE_COMPILER)
        message("will build using ahead of time compiler")
        add_compile_options(-fsycl-targets=spir64_gen)
        set(devtype "-device pvc")
        add_compile_options(--start-no-unused-arguments -Xs ${devtype} --end-no-unused-arguments)
        add_link_options(-fsycl-targets=spir64_gen)
        add_link_options(--start-no-unused-arguments -Xs ${devtype} --end-no-unused-arguments)
else()
        message("will build using incremental compiler")
endif()

add_compile_options(-Werror -Wuninitialized -Wunused-variable)

add_compile_options(-Rno-debug-disables-optimization)
add_link_options(-Rno-debug-disables-optimization)

# Make sure at least one of the backends is enabled
if (ENABLE_MPI STREQUAL "OFF" AND ENABLE_OPENSHMEM STREQUAL "OFF" AND ENABLE_PMI STREQUAL "OFF")
    message(FATAL_ERROR "At least one of 'ENABLE_MPI', 'ENABLE_OPENSHMEM' or 'ENABLE_PMI' must be on")
endif()

# Check for valid L0 path
if (EXISTS "${L0_INSTALL_PREFIX}/include/level_zero/ze_api.h" AND
    EXISTS "${L0_INSTALL_PREFIX}/include/level_zero/zet_api.h")
    list(APPEND EXTRA_INCS "${L0_INSTALL_PREFIX}/include")
else()
    message(FATAL_ERROR
            "Cannot find level zero headers!\n"
            "Provided: ${L0_INSTALL_PREFIX}\n"
            "Required Headers:\n"
            "    level_zero/ze_api.h\n"
            "    level_zero/zet_api.h")
endif()

if (EXISTS "${L0_INSTALL_PREFIX}/lib64/libze_loader.so")
    list(APPEND EXTRA_LIBS "-L${L0_INSTALL_PREFIX}/lib64 -lze_loader")
elseif (EXISTS "${L0_INSTALL_PREFIX}/lib/libze_loader.so")
    list(APPEND EXTRA_LIBS "-L${L0_INSTALL_PREFIX}/lib -lze_loader")
elseif (EXISTS "${L0_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libze_loader.so")
    list(APPEND EXTRA_LIBS "-L${L0_INSTALL_PREFIX}/lib/x86_64-linux-gnu -lze_loader")
else()
    message(FATAL_ERROR
            "Cannot find level zero library!\n"
            "Provided: ${L0_INSTALL_PREFIX}\n"
            "Required Headers:\n"
            "    libze_loader.so")
endif()

if (ENABLE_MPI)
    find_package(MPI REQUIRED)
    string(REGEX REPLACE "[^/]+/?$" "" MPI_CXX_LIBRARY_DIR "${MPI_mpicxx_LIBRARY}")
    list(APPEND EXTRA_INCS ${MPI_CXX_INCLUDE_PATH})
    list(APPEND EXTRA_LIBS -L${MPI_CXX_LIBRARY_DIR})
endif()

if (ENABLE_OPENSHMEM)
    # Check for valid SOS path
    if (EXISTS "${SHMEM_INSTALL_PREFIX}/include/shmem.h" AND
        EXISTS "${SHMEM_INSTALL_PREFIX}/include/shmemx.h")
        list(APPEND EXTRA_INCS "${SHMEM_INSTALL_PREFIX}/include")
    else()
        message(FATAL_ERROR
                "Cannot find OpenSHMEM headers!\n"
                "Provided: ${SHMEM_INSTALL_PREFIX}\n"
                "Required Headers:\n"
                "    shmem.h\n"
                "    shmemx.h")
    endif()

    if (EXISTS "${SHMEM_INSTALL_PREFIX}/bin/oshc++")
        execute_process(COMMAND ${SHMEM_INSTALL_PREFIX}/bin/oshc++ -showlibs OUTPUT_VARIABLE SHMEM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
        list(APPEND EXTRA_LIBS ${SHMEM_LIBS})
    else()
        message(FATAL_ERROR
                "Cannot find OpenSHMEM library!\n"
                "Provided: ${SHMEM_INSTALL_PREFIX}\n"
                "Required files:\n"
                "    oshc++")
    endif()
endif()

add_subdirectory(pmi-simple)
add_subdirectory(src)
add_subdirectory(examples)
if (BUILD_TEST OR BUILD_PERF_TEST)
    add_subdirectory(test)
endif()

# setup installer
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${ISHMEM_MAJOR_VERSION}")
set(CPACK_PACKAGE_VERSION_MINOR "${ISHMEM_MINOR_VERSION}")
include(CPack)

install(PROGRAMS scripts/ishmrun DESTINATION bin)
