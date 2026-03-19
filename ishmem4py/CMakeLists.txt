# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.17)

set(PROJECT_NAME "ishmem4py")
set(PROJECT_FULL_NAME "Intel(R) SHMEM Python bindings")

project(${PROJECT_NAME})

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(ISHMEM4PY_BUILD_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
set(ISHMEM4PY_PYTHON_BUILD_ROOT "${ISHMEM4PY_BUILD_ROOT}/python")
set(ISHMEM4PY_PKG_BUILD_DIR "${ISHMEM4PY_PYTHON_BUILD_ROOT}/ishmem4py")
set(ISHMEM4PY_SOURCE_PYTHON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/python")
set(ISHMEM4PY_SOURCE_PKG_DIR "${ISHMEM4PY_SOURCE_PYTHON_ROOT}/ishmem4py")

add_custom_target(ishmem4py-package-files
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ISHMEM4PY_PYTHON_BUILD_ROOT}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ISHMEM4PY_SOURCE_PYTHON_ROOT}/pyproject.toml"
        "${ISHMEM4PY_PYTHON_BUILD_ROOT}/pyproject.toml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ISHMEM4PY_SOURCE_PYTHON_ROOT}/README.md"
        "${ISHMEM4PY_PYTHON_BUILD_ROOT}/README.md"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ISHMEM4PY_SOURCE_PYTHON_ROOT}/setup.py"
        "${ISHMEM4PY_PYTHON_BUILD_ROOT}/setup.py"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/test"
        "${ISHMEM4PY_BUILD_ROOT}/test"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/examples"
        "${ISHMEM4PY_BUILD_ROOT}/examples"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${ISHMEM4PY_SOURCE_PKG_DIR}"
        "${ISHMEM4PY_PKG_BUILD_DIR}"
    VERBATIM)

add_library(ishmem4py-runtime SHARED
    csrc/ishmem4py_runtime.cpp)

target_include_directories(ishmem4py-runtime PRIVATE
    "${SYCL_INCLUDE_SYCL_DIR}"
    "${LEVEL_ZERO_INCLUDEDIR}"
    "${ISHMEM_BUILD_ROOT}/include"
    "${PROJECT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc")

target_link_libraries(ishmem4py-runtime PRIVATE ishmem-static)
set_target_properties(ishmem4py-runtime PROPERTIES
    OUTPUT_NAME "_ishmem4py_runtime"
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${ISHMEM4PY_PKG_BUILD_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${ISHMEM4PY_PKG_BUILD_DIR}"
    POSITION_INDEPENDENT_CODE 1)

if (ISHMEM4PY_BUILD_TORCH_INTEROP)
    find_package(Python3 COMPONENTS Development.Module REQUIRED)

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_PREFIX_RESULT)
    if (NOT TORCH_PREFIX_RESULT EQUAL 0)
        message(FATAL_ERROR
            "ISHMEM4PY_BUILD_TORCH_INTEROP=ON requires PyTorch to be importable from "
            "Python3_EXECUTABLE (${Python3_EXECUTABLE})")
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
                "import torch.utils.cpp_extension as ce; print(ce.library_paths()[0])"
        OUTPUT_VARIABLE TORCH_LIBRARY_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_LIBRARY_RESULT)
    if (NOT TORCH_LIBRARY_RESULT EQUAL 0)
        message(FATAL_ERROR "failed to determine the PyTorch library directory")
    endif()

    list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
    find_package(Torch REQUIRED)

    add_library(ishmem4py-torch MODULE
        csrc/ishmem4py_torch.cpp)

    target_include_directories(ishmem4py-torch PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
    target_link_directories(ishmem4py-torch PRIVATE "${TORCH_LIBRARY_DIR}")
    target_link_libraries(ishmem4py-torch PRIVATE
        Python3::Module
        ishmem4py-runtime
        torch_python
        ${TORCH_LIBRARIES})
    target_compile_features(ishmem4py-torch PRIVATE cxx_std_17)
    target_compile_options(ishmem4py-torch PRIVATE ${TORCH_CXX_FLAGS})
    set_target_properties(ishmem4py-torch PROPERTIES
        OUTPUT_NAME "_ishmem4py_torch"
        PREFIX ""
        BUILD_RPATH "${TORCH_LIBRARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${ISHMEM4PY_PKG_BUILD_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${ISHMEM4PY_PKG_BUILD_DIR}"
        POSITION_INDEPENDENT_CODE 1)
endif()

add_custom_target(ishmem4py ALL
    DEPENDS ishmem4py-package-files ishmem4py-runtime)

if (ISHMEM4PY_BUILD_TORCH_INTEROP)
    add_dependencies(ishmem4py ishmem4py-torch)
endif()

message(STATUS "\n======= ishmem4py summary ========")
message(STATUS "Python interpreter:          ${Python3_EXECUTABLE}")
message(STATUS "Python package root:         ${ISHMEM4PY_PYTHON_BUILD_ROOT}")
message(STATUS "Runtime library output:      ${ISHMEM4PY_PKG_BUILD_DIR}")
message(STATUS "Torch/XPU interop enabled:   ${ISHMEM4PY_BUILD_TORCH_INTEROP}")
message(STATUS "==================================\n")
