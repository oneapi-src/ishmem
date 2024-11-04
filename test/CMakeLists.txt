# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#
# Common CMakeLists.txt to build ishmem, tests, examples

cmake_minimum_required(VERSION 3.17)

set(PROJECT_NAME "ishmem tests")
set(PROJECT_FULL_NAME "Intel® SHMEM tests")

project(${PROJECT_NAME})

# -------------------------------------------------------------------
# Configuration Options
option(BUILD_UNIT_TESTS "Build unit tests" FALSE)
option(BUILD_PERF_TESTS "Build performance tests" FALSE)
option(BUILD_APPS "Build apps" FALSE)
option(ENABLE_AOT_COMPILATION "Enables AOT compilation for GPU kernels" TRUE)

# Set default device type(s) for AOT compilation
if (NOT ISHMEM_AOT_DEVICE_TYPES)
    set(ISHMEM_AOT_DEVICE_TYPES "pvc")
endif()

# -------------------------------------------------------------------
# Include other CMakeLists.txt

if (BUILD_UNIT_TESTS)
    add_subdirectory(unit)
endif()

if (BUILD_PERF_TESTS)
    add_subdirectory(performance)
endif()

if (BUILD_APPS)
    add_subdirectory(apps)
endif()
