# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#
# CMake utility functions

include(CheckCXXCompilerFlag)

function(check_compiler_flag flag var_name required)
    check_cxx_compiler_flag(${flag} ${var_name})

    if (${required} AND NOT ${${var_name}})
        message(FATAL_ERROR
            "Compiler flag '${flag}' is required, but not supported by your compiler")
    endif()
endfunction(check_compiler_flag)

function(setup_compiler_options)
    # Set compiler requirements
    set(CMAKE_CXX_STANDARD 17 PARENT_SCOPE)
    set(CMAKE_CXX_STANDARD_REQUIRED ON PARENT_SCOPE)

    # This option is mostly for forward-compatibility (i.e. icpx name changes from IntelLLVM in future CMake versions)
    option(SKIP_COMPILER_CHECK "Skips compiler validation (NOT RECOMMENDED)" FALSE)

    # Check the provided compiler (requires icpx with SYCL enabled)
    if (NOT ${SKIP_COMPILER_CHECK})
        if (NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM")
            message(FATAL_ERROR
                "Unsupported compiler!\n"
                "Only the Intel LLVM-Based compiler (icpx) is supported\n"
                "If you believe you are receiving this message in error, proceed with '-DSKIP_COMPILER_CHECK=ON'")
        elseif (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "2024.0")
            message(FATAL_ERROR
                "Unsupported compiler!\n"
                "Only the Intel LLVM-Based compiler (icpx) version 2024.0 or newer is supported\n"
                "If you believe you are receiving this message in error, proceed with '-DSKIP_COMPILER_CHECK=ON'")
        endif()
    endif()

    # Check compiler support for required flags
    check_compiler_flag(-fsycl COMPILER_FLAG_SYCL TRUE)
    check_compiler_flag(-mmovdir64b COMPILER_FLAG_MOVDIR64B TRUE)
    check_compiler_flag(-mavx512f COMPILER_FLAG_AVX512F TRUE)
    check_compiler_flag(-mwaitpkg COMPILER_FLAG_WAITPKG TRUE)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

    # Check the compiler version
    set(ICPX_MIN_SUPPORTED "")

    # Set compiler settings
    set(COMPILER_WARN_FLAGS "-Werror -Wuninitialized -Wunused-variable")

    set(COMPILER_DEFAULT_FLAGS "-D_GNU_SOURCE -fvisibility=internal")
    set(COMPILER_DEBUG_FLAGS "-g -DENABLE_DEBUG -Rno-debug-disables-optimization")
    set(COMPILER_RELEASE_FLAGS "-O3")
    set(COMPILER_RELWITH_DEBINFO_FLAGS "-O2 -g")

    if (ENABLE_AOT_COMPILATION)
        set(COMPILER_DEFAULT_FLAGS "${COMPILER_DEFAULT_FLAGS} -fsycl-targets=spir64_gen")
        set(COMPILER_DEFAULT_FLAGS "${COMPILER_DEFAULT_FLAGS} --start-no-unused-arguments -Xs \"-device ${ISHMEM_AOT_DEVICE_TYPES}\" --end-no-unused-arguments")
        # Suppress the `Compilation from IR - skipping loading of FCL` outputs:
        set(COMPILER_DEFAULT_FLAGS "${COMPILER_DEFAULT_FLAGS} --start-no-unused-arguments -Xsycl-target-backend \"-q\" --end-no-unused-arguments")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPILER_WARN_FLAGS} ${COMPILER_DEFAULT_FLAGS}" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${COMPILER_DEBUG_FLAGS}" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${COMPILER_RELEASE_FLAGS}" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${COMPILER_RELWITHDEBINFO_FLAGS}" PARENT_SCOPE)
endfunction(setup_compiler_options)

function(setup_dependencies)
    # Check for valid Level Zero installation
    if (NOT EXISTS "${LEVEL_ZERO_DIR}/include/level_zero/ze_api.h" OR
        NOT EXISTS "${LEVEL_ZERO_DIR}/include/level_zero/zet_api.h")
        message(FATAL_ERROR
                "Cannot find level zero Headers!\n"
                "Provided (LEVEL_ZERO_DIR): ${LEVEL_ZERO_DIR}\n"
                "Required headers:\n"
                "    level_zero/ze_api.h\n"
                "    level_zero/zet_api.h")
    endif()

    set(LEVEL_ZERO_INC_DIR "${LEVEL_ZERO_DIR}/include" PARENT_SCOPE)

    if (EXISTS "${LEVEL_ZERO_DIR}/lib64/libze_loader.so")
        list(APPEND EXTRA_LIBS "-L${LEVEL_ZERO_DIR}/lib64 -lze_loader")
    elseif (EXISTS "${LEVEL_ZERO_DIR}/lib/libze_loader.so")
        list(APPEND EXTRA_LIBS "-L${LEVEL_ZERO_DIR}/lib -lze_loader")
    elseif (EXISTS "${LEVEL_ZERO_DIR}/lib/x86_64-linux-gnu/libze_loader.so")
        list(APPEND EXTRA_LIBS "-L${LEVEL_ZERO_DIR}/lib/x86_64-linux-gnu -lze_loader")
    else()
        message(FATAL_ERROR
                "Cannot find level zero library!\n"
                "Provided (LEVEL_ZERO_DIR): ${LEVEL_ZERO_DIR}\n"
                "Required Headers:\n"
                "    libze_loader.so")
endif()

endfunction(setup_dependencies)

function(setup_runtime_backends)
    option(ENABLE_OPENSHMEM "Enable OpenSHMEM support" TRUE)
    option(ENABLE_MPI "Enable MPI support" FALSE)

    # At least one of the runtimes must be enabled
    if (NOT ENABLE_MPI AND NOT ENABLE_OPENSHMEM)
        message(FATAL_ERROR "At least one of 'ENABLE_OPENSHMEM' and 'ENABLE_MPI' must be enabled")
    endif()

    if (ENABLE_OPENSHMEM)
        if (NOT EXISTS "${SHMEM_DIR}/include/shmem.h" OR
            NOT EXISTS "${SHMEM_DIR}/include/shmemx.h")
            message(FATAL_ERROR
                    "Cannot find OpenSHMEM headers!\n"
                    "Provided (SHMEM_DIR): ${SHMEM_DIR}\n"
                    "Required headers:\n"
                    "    shmem.h\n"
                    "    shmemx.h")
        endif()

        if (EXISTS "${SHMEM_DIR}/bin/oshc++")
            execute_process(COMMAND ${SHMEM_DIR}/bin/oshc++ -showlibs OUTPUT_VARIABLE SHMEM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
            set(SHMEM_LIBS ${SHMEM_LIBS} PARENT_SCOPE)
        else()
            message(FATAL_ERROR
                    "Cannot find OpenSHMEM library!\n"
                    "Provided (SHMEM_DIR): ${SHMEM_DIR}\n"
                    "Required files:\n"
                    "    oshc++")
        endif()

        set(SHMEM_INC_DIR "${SHMEM_DIR}/include" PARENT_SCOPE)
    endif()

    if (ENABLE_MPI)
        if (NOT EXISTS "${MPI_DIR}/include/mpi.h")
            message(FATAL_ERROR
                    "Cannot find MPI header!\n"
                    "Provided (MPI_DIR): ${MPI_DIR}\n"
                    "Required header:\n"
                    "    mpi.h")
        endif()

        set(MPI_INC_DIR "${MPI_DIR}/include" PARENT_SCOPE)
    endif()
endfunction(setup_runtime_backends)
