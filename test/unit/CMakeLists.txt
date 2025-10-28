# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.17)

file(GLOB TEST_SOURCE_FILES *.cpp)

set(PROJECT_NAME "ishmem unit tests")
set(PROJECT_FULL_NAME "Intel® SHMEM unit tests")

project(${PROJECT_NAME})

# -------------------------------------------------------------------
# Perform common test setup

set(ISHMEM_ROOT_DIR "${PROJECT_SOURCE_DIR}/../..")
set(ISHMEM_TEST_ROOT_DIR "${PROJECT_SOURCE_DIR}/..")

include(${ISHMEM_TEST_ROOT_DIR}/cmake/common.cmake)

# -------------------------------------------------------------------
# Setup the common source files

add_library(ishmem-test-common-unit STATIC ${TEST_COMMON_SOURCE_FILES})
target_include_directories(ishmem-test-common-unit PUBLIC ${ISHMEM_TEST_INCLUDE_DIRS})
target_link_libraries(ishmem-test-common-unit PUBLIC ${ISHMEM_TEST_LINK_LIBS})

# -------------------------------------------------------------------
# Setup default test values

set(ISHMEM_PE_COUNTS_UNIT_TESTS "2" CACHE STRING "Number of PEs to use for each test")
set(ISHMEM_NON_STANDARD_CTEST_FILES "")
set(ISHMEM_SKIP_CTEST_FILES "")

set(ISHMEM_TESTER_MODES host_device_device device)
set(ISHMEM_TESTER_CTEST_FILES
    alltoall
    broadcast
    collect
    exscan
    fcollect
    get
    get_nbi
    ibget
    ibput
    inscan
    put
    put_nbi
    reduce_and
    reduce_max
    reduce_min
    reduce_or
    reduce_prod
    reduce_sum
    reduce_xor
    signal_wait_until
    test_all
    test_all_vector
    test_any
    test_any_vector
    test_single
    test_some
    test_some_vector
    wait_until
    wait_until_all
    wait_until_all_vector
    wait_until_any
    wait_until_any_vector
    wait_until_some
    wait_until_some_vector)

set(ISHMEM_TESTER_GROUP_MODES device_grp1)
set(ISHMEM_TESTER_GROUP_TESTS
    put
    get)

set(ISHMEM_TESTER_ON_QUEUE_MODES on_queue)
set(ISHMEM_TESTER_ON_QUEUE_TESTS
    alltoall
    broadcast
    collect
    exscan
    fcollect
    get
    get_nbi
    ibget
    ibput
    inscan
    put
    put_nbi
    reduce_and
    reduce_max
    reduce_min
    reduce_or
    reduce_prod
    reduce_sum
    reduce_xor
    signal_wait_until
    wait_until
    wait_until_all
    wait_until_all_vector
    wait_until_any
    wait_until_any_vector
    wait_until_some
    wait_until_some_vector)

enable_testing()

# -------------------------------------------------------------------
# Show build summary
message(STATUS "\n======= Unit tests summary =======")
message(STATUS "Build type:                  ${CMAKE_BUILD_TYPE_CASE_INSENSITIVE}")
message(STATUS "PE count(s):                 ${ISHMEM_PE_COUNTS_UNIT_TESTS}")
message(STATUS "CTest launcher:              ${CTEST_LAUNCHER}")
message(STATUS "Enable AOT compilation:      ${ENABLE_AOT_COMPILATION}")
if (ENABLE_AOT_COMPILATION)
    message(STATUS "    AOT Device type(s): ${ISHMEM_AOT_DEVICE_TYPES}")
endif()
message(STATUS "Enable OpenSHMEM support:    ${ENABLE_OPENSHMEM}")
message(STATUS "Enable MPI support:          ${ENABLE_MPI}")
message(STATUS "==================================\n")

# -------------------------------------------------------------------
# Set build properties for all tests

foreach(TEST_SOURCE_FILE ${TEST_SOURCE_FILES})
    get_filename_component(EXE ${TEST_SOURCE_FILE} NAME_WE)
    add_executable(${EXE} ${TEST_SOURCE_FILE})
    set_target_properties(${EXE} PROPERTIES ENABLE_EXPORTS TRUE)
    target_link_libraries(${EXE} PRIVATE ishmem-test-common-unit)
endforeach()

# -------------------------------------------------------------------
# Add ctests

foreach(TEST_SOURCE_FILE ${TEST_SOURCE_FILES})
    get_filename_component(EXE ${TEST_SOURCE_FILE} NAME_WE)

    if (${EXE} IN_LIST ISHMEM_SKIP_CTEST_FILES)
        continue()
    elseif (${EXE} IN_LIST ISHMEM_NON_STANDARD_CTEST_FILES)
        continue()
    elseif (${EXE} IN_LIST ISHMEM_TESTER_CTEST_FILES)
        # These are all tests that use the 'ishmem_tester' framework
        # Default test settings
        foreach (MODE IN LISTS ISHMEM_TESTER_MODES)
            foreach (N ${ISHMEM_PE_COUNTS_UNIT_TESTS})
                add_test(NAME ${EXE}-${MODE}-${N} COMMAND ${CTEST_WRAPPER} ${N} ${ISHMEM_RUN_SCRIPT}
                    ./${EXE} --test_modes ${MODE} --csv COMMAND_EXPAND_LISTS)
            endforeach()
        endforeach()
        # Special test settings
        if (${EXE} IN_LIST ISHMEM_TESTER_GROUP_TESTS)
            foreach (MODE IN LISTS ISHMEM_TESTER_GROUP_MODES)
                foreach (N ${ISHMEM_PE_COUNTS_UNIT_TESTS})
                    add_test(NAME ${EXE}-${MODE}-${N} COMMAND ${CTEST_WRAPPER} ${N} ${ISHMEM_RUN_SCRIPT}
                        ./${EXE} --test_modes ${MODE} --csv COMMAND_EXPAND_LISTS)
                endforeach()
            endforeach()
        endif()
        if (${EXE} IN_LIST ISHMEM_TESTER_ON_QUEUE_TESTS)
            foreach (MODE IN LISTS ISHMEM_TESTER_ON_QUEUE_MODES)
                foreach (N ${ISHMEM_PE_COUNTS_UNIT_TESTS})
                    add_test(NAME ${EXE}-${MODE}-${N} COMMAND ${CTEST_WRAPPER} ${N} ${ISHMEM_RUN_SCRIPT}
                        ./${EXE} --test_modes ${MODE} --csv COMMAND_EXPAND_LISTS)
                endforeach()
            endforeach()
        endif()
    else ()
        # These are standard tests
        foreach (N ${ISHMEM_PE_COUNTS_UNIT_TESTS})
            add_test(NAME ${EXE}-${N} COMMAND ${CTEST_WRAPPER} ${N} ${ISHMEM_RUN_SCRIPT}
                ./${EXE} COMMAND_EXPAND_LISTS)
        endforeach()
    endif()
endforeach()
