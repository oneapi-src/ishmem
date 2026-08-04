# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# -------------------------------------------------------------------
# Setup source files, include directories, and compiler settings

set(ISHMEM_SRC_EXTRAS)

list(APPEND ISHMEM_SRC_LINK_LIBS
    ${CMAKE_DL_LIBS}
    ${LEVEL_ZERO_LIBRARIES}
    pthread)

list(APPEND ISHMEM_SRC_INCLUDE_DIRS
    "${SYCL_INCLUDE_SYCL_DIR}"
    "${LEVEL_ZERO_INCLUDEDIR}"
    "${ISHMEM_BUILD_ROOT}/include"
    "${PROJECT_SOURCE_DIR}/src")

if (ENABLE_OPENSHMEM)
    list(APPEND ISHMEM_SRC_EXTRAS
        runtime/runtime_openshmem.cpp
        runtime/wrapper_openshmem.cpp)
    list(APPEND ISHMEM_SRC_INCLUDE_DIRS "${OPENSHMEM_INCLUDE_DIRS}")
endif()

if (ENABLE_MPI)
    list(APPEND ISHMEM_SRC_EXTRAS
        runtime/runtime_mpi.cpp
        runtime/wrapper_mpi.cpp)
    list(APPEND ISHMEM_SRC_INCLUDE_DIRS "${MPI_CXX_INCLUDE_DIRS}")
endif()

if (ENABLE_PMI)
    list(APPEND ISHMEM_SRC_EXTRAS
        runtime/runtime_pmi.cpp
        runtime/wrapper_pmi.cpp)
    list(APPEND ISHMEM_SRC_LINK_LIBS pmi-simple)
endif()

set(ISHMEM_DEVICE_SRC
    ishmem.cpp
    rma.cpp
    amo.cpp
    nbi.cpp
    signaling.cpp
    synchronization.cpp
    teams.cpp
    collectives/alltoall.cpp
    collectives/barrier.cpp
    collectives/broadcast.cpp
    collectives/collect.cpp
    collectives/reduce.cpp
    collectives/scan.cpp
    collectives/sync.cpp
    err.cpp
    timestamp.cpp
    gpuprint.cpp
    memory_ordering.cpp
    # memory.cpp defines ishmem_ptr(), which is SYCL_EXTERNAL (device-callable). It must be in
    # ISHMEM_DEVICE_SRC so its device IR is bundled into libishmem_device.a. Otherwise, in shared
    # host-linking mode -- where device IR can only come from the device archive, not from the
    # .so -- the device-side ishmem_ptr is missing and tests that call it from a kernel (e.g.
    # ptr_device) fail at Level Zero program-build time with "Unresolved Symbol ishmem_ptr".
    memory.cpp)

set(ISHMEM_HOST_ONLY_SRC
    ipc.cpp
    runtime_ipc.cpp
    env_utils.cpp
    accelerator.cpp
    malloc.cpp
    proxy.cpp
    proxy_func.cpp
    runtime.cpp
    uid.cpp
    ${ISHMEM_SRC_EXTRAS})

set(ISHMEM_SRC
    ${ISHMEM_DEVICE_SRC}
    ${ISHMEM_HOST_ONLY_SRC})

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
# Build libraries for object files

add_library(ishmem-host-objects OBJECT ${ISHMEM_SRC})
add_dependencies(ishmem-host-objects copy-install-headers)
set_target_properties(ishmem-host-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
set_target_properties(ishmem-host-objects PROPERTIES ENABLE_EXPORTS TRUE)
target_compile_options(ishmem-host-objects PRIVATE
    ${ISHMEM_CPU_COMPILE_OPTIONS}
    ${ISHMEM_WARN_COMPILE_OPTIONS}
    ${ISHMEM_DEFAULT_COMPILE_OPTIONS}
    ${ISHMEM_SYCL_COMPILE_OPTIONS}
    ${ISHMEM_SYCL_HOST_ONLY_FLAG}
    -fvisibility=default
    "$<$<CONFIG:Debug>:${ISHMEM_DEBUG_COMPILE_OPTIONS}>"
    "$<$<CONFIG:Release>:${ISHMEM_RELEASE_COMPILE_OPTIONS}>"
    "$<$<CONFIG:RelWithDebInfo>:${ISHMEM_RELWITHDEBINFO_COMPILE_OPTIONS}>")
target_compile_definitions(ishmem-host-objects PRIVATE ISHMEM_GIT_HASH="${ISHMEM_GIT_HASH}")
target_include_directories(ishmem-host-objects PRIVATE ${ISHMEM_SRC_INCLUDE_DIRS})

execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-prog-name=clang-offload-bundler
    OUTPUT_VARIABLE ISHMEM_CLANG_OFFLOAD_BUNDLER
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)

if (NOT IS_ABSOLUTE "${ISHMEM_CLANG_OFFLOAD_BUNDLER}")
    find_program(ISHMEM_CLANG_OFFLOAD_BUNDLER NAMES "${ISHMEM_CLANG_OFFLOAD_BUNDLER}")
endif()

if (NOT ISHMEM_CLANG_OFFLOAD_BUNDLER)
    message(FATAL_ERROR "clang-offload-bundler is required to build the ishmem device library")
endif()

execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -dumpmachine
    OUTPUT_VARIABLE ISHMEM_HOST_TARGET_TRIPLE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if (ENABLE_AOT_COMPILATION)
    set(ISHMEM_DEVICE_TARGET_TRIPLE sycl-spir64_gen-unknown-unknown)
else()
    set(ISHMEM_DEVICE_TARGET_TRIPLE sycl-spir64-unknown-unknown)
endif()
set(ISHMEM_HOST_TARGET_TRIPLE host-${ISHMEM_HOST_TARGET_TRIPLE})

string(TOUPPER "${CMAKE_BUILD_TYPE}" ISHMEM_BUILD_TYPE_UPPER)
set(ISHMEM_BUILD_TYPE_COMPILE_OPTIONS)
if (ISHMEM_BUILD_TYPE_UPPER STREQUAL "DEBUG")
    list(APPEND ISHMEM_BUILD_TYPE_COMPILE_OPTIONS ${ISHMEM_DEBUG_COMPILE_OPTIONS})
elseif (ISHMEM_BUILD_TYPE_UPPER STREQUAL "RELEASE")
    list(APPEND ISHMEM_BUILD_TYPE_COMPILE_OPTIONS ${ISHMEM_RELEASE_COMPILE_OPTIONS})
elseif (ISHMEM_BUILD_TYPE_UPPER STREQUAL "RELWITHDEBINFO")
    list(APPEND ISHMEM_BUILD_TYPE_COMPILE_OPTIONS ${ISHMEM_RELWITHDEBINFO_COMPILE_OPTIONS})
endif()

set(ISHMEM_DEVICE_COMPILE_FLAGS
    ${ISHMEM_SYCL_COMPILE_OPTIONS}
    ${ISHMEM_WARN_COMPILE_OPTIONS}
    ${ISHMEM_DEFAULT_COMPILE_OPTIONS}
    ${ISHMEM_AOT_COMPILE_OPTIONS}
    ${ISHMEM_BUILD_TYPE_COMPILE_OPTIONS})
list(APPEND ISHMEM_DEVICE_COMPILE_FLAGS
    -fPIC
    -fsycl-device-only
    "-DISHMEM_GIT_HASH=\"${ISHMEM_GIT_HASH}\"")

foreach(include_dir IN LISTS ISHMEM_SRC_INCLUDE_DIRS)
    list(APPEND ISHMEM_DEVICE_COMPILE_FLAGS "-I${include_dir}")
endforeach()

set(ISHMEM_EMPTY_HOST_SRC ${CMAKE_CURRENT_BINARY_DIR}/ishmem_empty_host.cpp)
set(ISHMEM_EMPTY_HOST_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ishmem-device-bundles.dir/ishmem_empty_host.cpp.o)
file(WRITE ${ISHMEM_EMPTY_HOST_SRC} "")
add_custom_command(OUTPUT ${ISHMEM_EMPTY_HOST_OBJ}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ishmem-device-bundles.dir
    COMMAND ${CMAKE_CXX_COMPILER} -fPIC -c ${ISHMEM_EMPTY_HOST_SRC} -o ${ISHMEM_EMPTY_HOST_OBJ}
    DEPENDS ${ISHMEM_EMPTY_HOST_SRC}
    COMMENT "Building empty host object for ishmem device bundles"
    VERBATIM)

set(ISHMEM_DEVICE_BUNDLE_OBJECTS)
foreach(src IN LISTS ISHMEM_DEVICE_SRC)
    string(REPLACE "/" "_" src_id "${src}")
    set(device_ir ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ishmem-device-bundles.dir/${src_id}.bc)
    set(device_obj ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ishmem-device-bundles.dir/${src_id}.o)

    add_custom_command(OUTPUT ${device_ir}
        COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ishmem-device-bundles.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${ISHMEM_DEVICE_COMPILE_FLAGS}
                -c ${CMAKE_CURRENT_SOURCE_DIR}/${src} -o ${device_ir}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src}
        COMMENT "Building ishmem device IR ${src}"
        VERBATIM)

    add_custom_command(OUTPUT ${device_obj}
        COMMAND ${ISHMEM_CLANG_OFFLOAD_BUNDLER} -type=o
                -targets=${ISHMEM_DEVICE_TARGET_TRIPLE},${ISHMEM_HOST_TARGET_TRIPLE}
                -input=${device_ir}
                -input=${ISHMEM_EMPTY_HOST_OBJ}
                -output=${device_obj}
        DEPENDS ${device_ir} ${ISHMEM_EMPTY_HOST_OBJ}
        COMMENT "Bundling ishmem device object ${src}"
        VERBATIM)

    list(APPEND ISHMEM_DEVICE_BUNDLE_OBJECTS ${device_obj})
endforeach()

add_custom_target(ishmem-device-bundle-objects DEPENDS ${ISHMEM_DEVICE_BUNDLE_OBJECTS})
add_dependencies(ishmem-device-bundle-objects copy-install-headers)

# -------------------------------------------------------------------
# Build static libraries

add_library(ishmem-host-static STATIC $<TARGET_OBJECTS:ishmem-host-objects>)
target_link_libraries(ishmem-host-static PUBLIC ${ISHMEM_SRC_LINK_LIBS})
set_target_properties(ishmem-host-static PROPERTIES OUTPUT_NAME ishmem_host)
set_target_properties(ishmem-host-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${ISHMEM_BUILD_DIR})
set_target_properties(ishmem-host-static PROPERTIES ENABLE_EXPORTS TRUE)
set_target_properties(ishmem-host-static PROPERTIES PUBLIC_HEADER
                      "${ISHMEM_SOURCE_DIR}/ishmem.h;${ISHMEM_SOURCE_DIR}/ishmemx.h")

add_library(ishmem-host-shared SHARED $<TARGET_OBJECTS:ishmem-host-objects>)
target_link_libraries(ishmem-host-shared PUBLIC ${ISHMEM_SRC_LINK_LIBS})
target_link_options(ishmem-host-shared PRIVATE ${ISHMEM_SYCL_LINK_OPTIONS})
set_target_properties(ishmem-host-shared PROPERTIES OUTPUT_NAME ishmem_host)
set_target_properties(ishmem-host-shared PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${ISHMEM_BUILD_DIR})
set_target_properties(ishmem-host-shared PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ISHMEM_BUILD_DIR})
set_target_properties(ishmem-host-shared PROPERTIES ENABLE_EXPORTS TRUE)
if (UNIX AND NOT APPLE)
    target_link_options(ishmem-host-shared PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/ishmem_host.map")
    # Relink the shared library when the export map changes.
    set_target_properties(ishmem-host-shared PROPERTIES
        LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ishmem_host.map")
endif()

add_library(ishmem-device-static STATIC ${ISHMEM_DEVICE_BUNDLE_OBJECTS})
add_dependencies(ishmem-device-static ishmem-device-bundle-objects)
set_target_properties(ishmem-device-static PROPERTIES OUTPUT_NAME ishmem_device)
set_target_properties(ishmem-device-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${ISHMEM_BUILD_DIR})
set_target_properties(ishmem-device-static PROPERTIES ENABLE_EXPORTS TRUE)
set_target_properties(ishmem-device-static PROPERTIES LINKER_LANGUAGE CXX)

# The host library does not provide usable device IR to the SYCL device-link step:
# host-static is compiled host-only, and host-shared is a .so. Link libishmem_device.a
# twice so device-library dependencies introduced by the first archive scan can be
# resolved by a second scan.
# This keeps extraction demand-driven, unlike --whole-archive, while still resolving
# device-side symbols used by applications.
# Spell the second path differently so CMake does not de-duplicate the archive entries.
add_library(ishmem-static INTERFACE)
target_link_libraries(ishmem-static INTERFACE
    ishmem-host-static
    "$<TARGET_FILE:ishmem-device-static>"
    "$<TARGET_FILE_DIR:ishmem-device-static>/./$<TARGET_FILE_NAME:ishmem-device-static>")
target_compile_options(ishmem-static INTERFACE
    ${ISHMEM_SYCL_COMPILE_OPTIONS}
    ${ISHMEM_AOT_INTERFACE_OPTIONS})
target_link_options(ishmem-static INTERFACE
    ${ISHMEM_SYCL_LINK_OPTIONS}
    ${ISHMEM_AOT_INTERFACE_OPTIONS}
    "$<$<CONFIG:Release>:${ISHMEM_RELEASE_LINK_OPTIONS}>")
add_dependencies(ishmem-static ishmem-device-static)

add_library(ishmem-shared INTERFACE)
target_link_libraries(ishmem-shared INTERFACE
    ishmem-host-shared
    "$<TARGET_FILE:ishmem-device-static>"
    "$<TARGET_FILE_DIR:ishmem-device-static>/./$<TARGET_FILE_NAME:ishmem-device-static>")
target_compile_options(ishmem-shared INTERFACE
    ${ISHMEM_SYCL_COMPILE_OPTIONS}
    ${ISHMEM_AOT_INTERFACE_OPTIONS})
target_link_options(ishmem-shared INTERFACE
    ${ISHMEM_SYCL_LINK_OPTIONS}
    ${ISHMEM_AOT_INTERFACE_OPTIONS}
    "$<$<CONFIG:Release>:${ISHMEM_RELEASE_LINK_OPTIONS}>")
add_dependencies(ishmem-shared ishmem-device-static)

install(TARGETS ishmem-host-static ishmem-host-shared
    ARCHIVE DESTINATION ${ISHMEM_INSTALL_LIB}
    LIBRARY DESTINATION ${ISHMEM_INSTALL_LIB}
    RUNTIME DESTINATION ${ISHMEM_INSTALL_BIN}
    PUBLIC_HEADER DESTINATION ${ISHMEM_INSTALL_INC})
install(TARGETS ishmem-device-static
    ARCHIVE DESTINATION ${ISHMEM_INSTALL_LIB}
    LIBRARY DESTINATION ${ISHMEM_INSTALL_LIB}
    RUNTIME DESTINATION ${ISHMEM_INSTALL_BIN})
install(DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/ishmem DESTINATION ${ISHMEM_INSTALL_INC})

# -------------------------------------------------------------------
# Build shared library
# Note: SYCL does not currently support dynamic library creation with device kernels
# https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-2/example-gpu-commands.html
# TODO: When supported
