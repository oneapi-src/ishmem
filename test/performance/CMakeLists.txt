# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

file(GLOB TEST_SOURCE_FILES *.cpp)

foreach(TEST_SOURCE_FILE ${TEST_SOURCE_FILES})
    get_filename_component(TEST_EXE_FILE ${TEST_SOURCE_FILE} NAME_WE)
    add_executable(${TEST_EXE_FILE} ${TEST_SOURCE_FILE})
    set_target_properties(${TEST_EXE_FILE} PROPERTIES OUTPUT_NAME ${TEST_EXE_FILE})
    target_include_directories(${TEST_EXE_FILE}
               PUBLIC "${CMAKE_SOURCE_DIR}/src"
               PUBLIC ${EXTRA_INCS})
    target_link_libraries(${TEST_EXE_FILE} ${EXTRA_LIBS} ishmem-static)
endforeach()

add_test(NAME ishmem_get_bw-3-gpu COMMAND ${CTEST_WRAPPER} 3 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/ishmem_get_bw COMMAND_EXPAND_LISTS)
set_tests_properties(ishmem_get_bw-3-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)

add_test(NAME ishmem_get_work_group_bw-3-gpu COMMAND ${CTEST_WRAPPER} 3 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/ishmem_get_work_group_bw COMMAND_EXPAND_LISTS)
set_tests_properties(ishmem_get_work_group_bw-3-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)

add_test(NAME ishmem_put_bw_ts-2-gpu COMMAND ${CTEST_WRAPPER} 2 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/ishmem_put_bw_ts COMMAND_EXPAND_LISTS)
set_tests_properties(ishmem_put_bw_ts-2-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)

add_test(NAME ishmem_put_bw-3-gpu COMMAND ${CTEST_WRAPPER} 3 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/ishmem_put_bw COMMAND_EXPAND_LISTS)
set_tests_properties(ishmem_put_bw-3-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)

add_test(NAME ishmem_put_work_group_bw-3-gpu COMMAND ${CTEST_WRAPPER} 3 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/ishmem_put_work_group_bw COMMAND_EXPAND_LISTS)
set_tests_properties(ishmem_put_work_group_bw-3-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)
