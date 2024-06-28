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

set(BWARGS "--csv -m 512 --test_modes device ")
add_test(NAME alltoall_bw-2-gpu COMMAND ${CTEST_WRAPPER} 2 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/alltoall_bw ${BWARGS} COMMAND_EXPAND_LISTS)
set_tests_properties(alltoall_bw-2-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)

add_test(NAME amo_bw-2-gpu COMMAND ${CTEST_WRAPPER} 2 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/amo_bw ${BWARGS} COMMAND_EXPAND_LISTS)
set_tests_properties(amo_bw-2-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)

add_test(NAME broadcast_bw-2-gpu COMMAND ${CTEST_WRAPPER} 2 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/broadcast_bw ${BWARGS} COMMAND_EXPAND_LISTS)
set_tests_properties(broadcast_bw-2-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)

add_test(NAME collect_bw-2-gpu COMMAND ${CTEST_WRAPPER} 2 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/collect_bw ${BWARGS} COMMAND_EXPAND_LISTS)
set_tests_properties(collect_bw-2-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)

add_test(NAME fcollect_bw-2-gpu COMMAND ${CTEST_WRAPPER} 2 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/fcollect_bw ${BWARGS} COMMAND_EXPAND_LISTS)
set_tests_properties(fcollect_bw-2-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)

add_test(NAME get_bw-2-gpu COMMAND ${CTEST_WRAPPER} 2 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/get_bw ${BWARGS} COMMAND_EXPAND_LISTS)
set_tests_properties(get_bw-2-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)

add_test(NAME put_bw-2-gpu COMMAND ${CTEST_WRAPPER} 2 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/put_bw ${BWARGS} COMMAND_EXPAND_LISTS)
set_tests_properties(put_bw-2-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)

add_test(NAME reduce_bw-2-gpu COMMAND ${CTEST_WRAPPER} 2 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/reduce_bw ${BWARGS} COMMAND_EXPAND_LISTS)
set_tests_properties(reduce_bw-2-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)
