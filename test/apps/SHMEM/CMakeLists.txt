# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

set(target jacobi)

add_executable(${target} ${target}.cpp)
target_include_directories(${target}
               PUBLIC "${CMAKE_SOURCE_DIR}/src"
               PUBLIC "${EXTRA_INCS}")
target_link_libraries(${target} ${EXTRA_LIBS} ishmem-static)

add_test(NAME ${target}-gpu COMMAND ${CTEST_WRAPPER} 4 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/${target} -niter 100 COMMAND_EXPAND_LISTS)
set_tests_properties(${target}-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)

