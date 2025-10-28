# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#
# CMake utility functions

# Require pkgconfig for finding Level Zero and OpenSHMEM libraries
find_package(PkgConfig REQUIRED)

# Require compiler with SYCL library installed
find_package(IntelSYCL REQUIRED)

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

    # Check compiler support for required flags
    check_compiler_flag(-mmovdir64b COMPILER_FLAG_MOVDIR64B TRUE)
    check_compiler_flag(-mavx512f COMPILER_FLAG_AVX512F TRUE)
    check_compiler_flag(-mwaitpkg COMPILER_FLAG_WAITPKG TRUE)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS}")

    # Set compiler settings
    set(COMPILER_WARN_FLAGS "-Wall -Wextra -Wconversion -Wno-unused-parameter -Wformat -Wformat-security")

    set(COMPILER_DEFAULT_FLAGS "-D_GNU_SOURCE -fvisibility=internal")
    set(COMPILER_DEBUG_FLAGS "-g -DENABLE_DEBUG -Rno-debug-disables-optimization")
    set(COMPILER_RELEASE_FLAGS "-O3")
    set(COMPILER_RELWITH_DEBINFO_FLAGS "-O2 -g")

    set(LINKER_RELEASE_FLAGS "-Wl,-z,noexecstack -Wl,-z,nodlopen")

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

    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${LINKER_RELEASE_FLAGS}" PARENT_SCOPE)
endfunction(setup_compiler_options)

function(setup_dependencies)
    # Keep support for LEVEL_ZERO_DIR for backward-compatibility
    if (EXISTS ${LEVEL_ZERO_DIR})
        set(ENV{PKG_CONFIG_PATH} ${LEVEL_ZERO_DIR}/lib64/pkgconfig:$ENV{PKG_CONFIG_PATH})
    endif()

    # Search for Level Zero using pkgconfig
    pkg_check_modules(LEVEL_ZERO REQUIRED IMPORTED_TARGET level-zero)
    pkg_get_variable(_LEVEL_ZERO_PCFILEDIR level-zero pcfiledir)

    if (LEVEL_ZERO_VERSION LESS 1.20)
        # Incorrect pkgconfig prefix - need to fix defined variables
        get_filename_component(_LEVEL_ZERO_PCFILEDIR_PARENT ${_LEVEL_ZERO_PCFILEDIR} DIRECTORY)
        get_filename_component(LEVEL_ZERO_DIR ${_LEVEL_ZERO_PCFILEDIR_PARENT} DIRECTORY)

        string(REPLACE "/usr" "${LEVEL_ZERO_DIR}" LEVEL_ZERO_INCLUDEDIR "${LEVEL_ZERO_INCLUDEDIR}")
        string(REPLACE "/usr" "${LEVEL_ZERO_DIR}" LEVEL_ZERO_LIBDIR "${LEVEL_ZERO_LIBDIR}")
    endif()

    list(APPEND EXTRA_LIBS "-L${LEVEL_ZERO_LIBDIR}")
endfunction(setup_dependencies)

function(setup_runtime_backends)
    option(ENABLE_OPENSHMEM "Enable OpenSHMEM support" FALSE)
    option(ENABLE_MPI "Enable MPI support" FALSE)

    # At least one of the runtimes must be enabled
    if (NOT ENABLE_MPI AND NOT ENABLE_OPENSHMEM)
        message(FATAL_ERROR "At least one of 'ENABLE_OPENSHMEM' and 'ENABLE_MPI' must be enabled")
    endif()

    if (DEFINED ISHMEM_DEFAULT_RUNTIME)
        string(TOUPPER "${ISHMEM_DEFAULT_RUNTIME}" ISHMEM_DEFAULT_RUNTIME_UPPER)
    endif()

    if (ENABLE_OPENSHMEM)
        # Keep support for SHMEM_DIR for backward-compatibility
        if (EXISTS ${SHMEM_DIR})
            set(ENV{PKG_CONFIG_PATH} ${SHMEM_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH})
        endif()

        # Currently only support Sandia OpenSHMEM
        pkg_check_modules(SANDIA_OPENSHMEM REQUIRED IMPORTED_TARGET sandia-openshmem>=1.5.3)
        pkg_get_variable(_SANDIA_OPENSHMEM_PCFILEDIR sandia-openshmem pcfiledir)

        # Incorrect pkgconfig prefix - need to fix defined variables
        get_filename_component(_SANDIA_OPENSHMEM_PCFILEDIR_PARENT ${_SANDIA_OPENSHMEM_PCFILEDIR} DIRECTORY)
        get_filename_component(SANDIA_OPENSHMEM_DIR ${_SANDIA_OPENSHMEM_PCFILEDIR_PARENT} DIRECTORY)

        string(REPLACE "${SANDIA_OPENSHMEM_PREFIX}" "${SANDIA_OPENSHMEM_DIR}" SANDIA_OPENSHMEM_INCLUDE_DIRS "${SANDIA_OPENSHMEM_INCLUDE_DIRS}")

        set(OPENSHMEM_INCLUDE_DIRS "${SANDIA_OPENSHMEM_INCLUDE_DIRS}" PARENT_SCOPE)

        if (NOT DEFINED ISHMEM_DEFAULT_RUNTIME_UPPER)
            set(ISHMEM_DEFAULT_RUNTIME_UPPER "OPENSHMEM")
        endif()
        if ("${ISHMEM_DEFAULT_RUNTIME_UPPER}" STREQUAL "OPENSHMEM")
            set(DEFAULT_CONFIRMED TRUE)
            set(ISHMEM_DEFAULT_RUNTIME_STR "OPENSHMEM" PARENT_SCOPE)
            set(ISHMEM_DEFAULT_RUNTIME_VAL ISHMEMX_RUNTIME_OPENSHMEM PARENT_SCOPE)
        endif()
    endif()

    if (ENABLE_MPI)
        if (EXISTS ${MPI_DIR})
            set(ENV{MPI_HOME} ${MPI_DIR})
            if (DEFINED ENV{I_MPI_ROOT} AND NOT "$ENV{I_MPI_ROOT}" STREQUAL "")
                if (NOT ENV{I_MPI_ROOT} STREQUAL MPI_DIR)
                    message(WARNING " \${MPI_DIR} is set but does not match \${I_MPI_ROOT}.\n"
                                    " FindMPI will likely select \${I_MPI_ROOT} over \${MPI_DIR}.\n"
                                    " To ensure \${MPI_DIR} is used, please remove \${I_MPI_ROOT} from your environment.\n"
                                    " MPI_DIR=${MPI_DIR}\n"
                                    " I_MPI_ROOT=$ENV{I_MPI_ROOT}")
                endif()
            endif()
        endif()

        set(MPI_CXX_SKIP_MPICXX TRUE)
        find_package(MPI COMPONENTS REQUIRED CXX)

        if (NOT DEFINED ISHMEM_DEFAULT_RUNTIME_UPPER)
            set(ISHMEM_DEFAULT_RUNTIME_UPPER "MPI")
        endif()
        if ("${ISHMEM_DEFAULT_RUNTIME_UPPER}" STREQUAL "MPI")
            set(DEFAULT_CONFIRMED TRUE)
            set(ISHMEM_DEFAULT_RUNTIME_STR "MPI" PARENT_SCOPE)
            set(ISHMEM_DEFAULT_RUNTIME_VAL "ISHMEMX_RUNTIME_MPI" PARENT_SCOPE)
        endif()
    endif()

    if (NOT DEFINED DEFAULT_CONFIRMED)
        if (NOT ENABLE_OPENSHMEM AND "${ISHMEM_DEFAULT_RUNTIME}" STREQUAL "OPENSHMEM")
            message(FATAL_ERROR "Attempted to set '${ISHMEM_DEFAULT_RUNTIME}' as default when ENABLE_OPENSHMEM is disabled.")
        elseif (NOT ENABLE_MPI AND "${ISHMEM_DEFAULT_RUNTIME}" STREQUAL "MPI")
            message(FATAL_ERROR "Attempted to set '${ISHMEM_DEFAULT_RUNTIME}' as default when ENABLE_MPI is disabled.")
        else()
            message(FATAL_ERROR " Attempted to set unknown runtime '${ISHMEM_DEFAULT_RUNTIME}' as default.\n"
                                " Supported options: \"OPENSHMEM\", \"MPI\".\n")
        endif()
    endif()
endfunction(setup_runtime_backends)
