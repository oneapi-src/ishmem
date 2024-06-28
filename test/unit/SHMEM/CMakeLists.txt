# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

file(GLOB TEST_SOURCE_FILES *.cpp)

set(nonstandard_ctest_files "")
set(skip_ctest_files long_reduce)

# these are run several times in different modes
set(ishmem_tester_ctest_files
    get put ibget ibput alltoall broadcast collect fcollect reduce_sum reduce_prod reduce_max reduce_min reduce_and reduce_or reduce_xor
    test test_all test_any test_some wait_until wait_until_all wait_until_any wait_until_some signal_wait_until)

foreach(TEST_SOURCE_FILE ${TEST_SOURCE_FILES})
    get_filename_component(TEST_EXE_FILE ${TEST_SOURCE_FILE} NAME_WE)
    add_executable(${TEST_EXE_FILE}-shmem ${TEST_SOURCE_FILE})
    set_target_properties(${TEST_EXE_FILE}-shmem PROPERTIES OUTPUT_NAME ${TEST_EXE_FILE})
    target_include_directories(${TEST_EXE_FILE}-shmem
               PUBLIC "${CMAKE_SOURCE_DIR}/src"
               PUBLIC ${EXTRA_INCS})
    target_link_libraries(${TEST_EXE_FILE}-shmem ${EXTRA_LIBS} ishmem-static)

    if(${TEST_EXE_FILE} IN_LIST skip_ctest_files)
        continue()
    endif()
    if(${TEST_EXE_FILE} IN_LIST nonstandard_ctest_files)
        continue()
    endif()
    if(${TEST_EXE_FILE} IN_LIST ishmem_tester_ctest_files)
        continue()
    endif()
    set(pe_count 2)
    foreach(n IN LISTS pe_count)
	    add_test(NAME ${TEST_EXE_FILE}-${n}-gpu COMMAND ${CTEST_WRAPPER} ${n} ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_EXE_FILE} COMMAND_EXPAND_LISTS)
        set_tests_properties(${TEST_EXE_FILE}-${n}-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)
    endforeach()
endforeach()


add_test(NAME int_wait_until_device-4-gpu COMMAND ${CTEST_WRAPPER} 4 ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/int_wait_until_device COMMAND_EXPAND_LISTS)
set_tests_properties(int_wait_until_device-4-gpu PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=:gpu)

set(ishmem_tester_modes
    device )
foreach(testfile IN LISTS ishmem_tester_ctest_files)
    foreach(mode IN LISTS ishmem_tester_modes)
        set(pe_count 2)
        foreach(n IN LISTS pe_count)
                  add_test(NAME ${testfile}-${mode}-${n}-gpu COMMAND ${CTEST_WRAPPER} ${n} ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/${testfile} --test_modes ${mode} --csv COMMAND_EXPAND_LISTS)
          endforeach()
    endforeach()
endforeach()

set(ishmem_tester_modes
    device_grp1 )
foreach(testfile IN LISTS put get)
    foreach(mode IN LISTS ishmem_tester_modes)
        set(pe_count 2)
        foreach(n IN LISTS pe_count)
                  add_test(NAME ${testfile}-${mode}-${n}-gpu COMMAND ${CTEST_WRAPPER} ${n} ${ISHMRUN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/${testfile} --test_modes ${mode} --csv COMMAND_EXPAND_LISTS)
          endforeach()
    endforeach()
endforeach()
