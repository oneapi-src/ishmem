# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Setup source compiler options
add_compile_options(-mmovdir64b)
add_compile_options(-mavx512f)
add_compile_options(-mwaitpkg)

# -------------------------------------------------------------------
# Setup source files, include directories, and compiler settings

set(ISHMEM_SRC_EXTRAS)

list(APPEND ISHMEM_SRC_LINK_LIBS
    ${CMAKE_DL_LIBS}
    ze_loader
    pthread)

list(APPEND ISHMEM_SRC_INCLUDE_DIRS
    "${ISHMEM_BUILD_ROOT}/include"
    "${PROJECT_SOURCE_DIR}/src"
    "${LEVEL_ZERO_INC_DIR}")

if (ENABLE_OPENSHMEM)
    list(APPEND ISHMEM_SRC_EXTRAS
        runtime/runtime_openshmem.cpp
        runtime/wrapper_openshmem.cpp)
    list(APPEND ISHMEM_SRC_INCLUDE_DIRS "${SHMEM_INC_DIR}")
endif()

if (ENABLE_MPI)
    list(APPEND ISHMEM_SRC_EXTRAS
        runtime/runtime_mpi.cpp
        runtime/wrapper_mpi.cpp)
    list(APPEND ISHMEM_SRC_INCLUDE_DIRS "${MPI_INC_DIR}")
endif()

if (ENABLE_PMI)
    list(APPEND ISHMEM_SRC_EXTRAS
        runtime/runtime_pmi.cpp
        runtime/wrapper_pmi.cpp)
    list(APPEND ISHMEM_SRC_LINK_LIBS pmi-simple)
endif()

set(ISHMEM_SRC
    ishmem.cpp
    rma.cpp
    ipc.cpp
    runtime_ipc.cpp
    env_utils.cpp
    accelerator.cpp
    amo.cpp
    malloc.cpp
    memory.cpp
    proxy.cpp
    proxy_func.cpp
    runtime.cpp
    collectives.cpp
    nbi.cpp
    signaling.cpp
    synchronization.cpp
    teams.cpp
    collectives/alltoall.cpp
    collectives/barrier.cpp
    collectives/broadcast.cpp
    collectives/collect.cpp
    collectives/reduce.cpp
    collectives/sync.cpp
    err.cpp
    timestamp.cpp
    gpuprint.cpp
    memory_ordering.cpp
    ${ISHMEM_SRC_EXTRAS})

# -------------------------------------------------------------------
# Disable warnings for dlmalloc
set_source_files_properties(malloc.cpp PROPERTIES COMPILE_FLAGS -w)

# -------------------------------------------------------------------
# Handle library dependencies

set(ISHMEM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_custom_target(copy-install-headers ALL)
add_custom_command(TARGET copy-install-headers
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}
    COMMAND ${CMAKE_COMMAND}
        -DISHMEM_BUILD_ROOT=${ISHMEM_BUILD_ROOT}
        -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
        -DISHMEM_SOURCE_DIR=${ISHMEM_SOURCE_DIR}
        -P ${COMMON_CMAKE_DIR}/headers.cmake)

# Need to handle first-time copy for new build directory in case tests are configured to be built
include(${COMMON_CMAKE_DIR}/headers.cmake)

# -------------------------------------------------------------------
# Build library for object files

add_library(ishmem-objects OBJECT ${ISHMEM_SRC})
add_dependencies(ishmem-objects copy-install-headers)
set_target_properties(ishmem-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
set_target_properties(ishmem-objects PROPERTIES ENABLE_EXPORTS TRUE)
target_include_directories(ishmem-objects PRIVATE ${ISHMEM_SRC_INCLUDE_DIRS})

# -------------------------------------------------------------------
# Build static library

add_library(ishmem-static STATIC $<TARGET_OBJECTS:ishmem-objects>)
target_link_libraries(ishmem-static PUBLIC ${ISHMEM_SRC_LINK_LIBS})
set_target_properties(ishmem-static PROPERTIES OUTPUT_NAME ishmem)
set_target_properties(ishmem-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${ISHMEM_BUILD_DIR})
set_target_properties(ishmem-static PROPERTIES ENABLE_EXPORTS TRUE)
set_target_properties(ishmem-static PROPERTIES PUBLIC_HEADER
                      "${ISHMEM_SOURCE_DIR}/ishmem.h;${ISHMEM_SOURCE_DIR}/ishmemx.h")

install(TARGETS ishmem-static
    ARCHIVE DESTINATION ${ISHMEM_INSTALL_LIB}
    LIBRARY DESTINATION ${ISHMEM_INSTALL_LIB}
    RUNTIME DESTINATION ${ISHMEM_INSTALL_BIN}
    PUBLIC_HEADER DESTINATION ${ISHMEM_INSTALL_INC})
install(DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/ishmem DESTINATION ${ISHMEM_INSTALL_INC})

# -------------------------------------------------------------------
# Build shared library
# Note: SYCL does not currently support dynamic library creation with device kernels
# https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-2/example-gpu-commands.html
# TODO: When supported
