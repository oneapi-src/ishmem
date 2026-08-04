# Copyright (C) 2024-2026 Intel Corporation
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

    separate_arguments(ISHMEM_SYCL_COMPILE_OPTIONS UNIX_COMMAND "${SYCL_FLAGS}")
    separate_arguments(ISHMEM_SYCL_LINK_OPTIONS UNIX_COMMAND "${SYCL_LINK_FLAGS}")

    set(ISHMEM_CMAKE_REQUIRED_FLAGS_SAVE "${CMAKE_REQUIRED_FLAGS}")
    string(REPLACE ";" " " ISHMEM_SYCL_REQUIRED_FLAGS "${ISHMEM_SYCL_COMPILE_OPTIONS}")
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${ISHMEM_SYCL_REQUIRED_FLAGS}")
    check_cxx_compiler_flag(-fsycl-host-only COMPILER_FLAG_FSYCL_HOST_ONLY)
    if (COMPILER_FLAG_FSYCL_HOST_ONLY)
        set(ISHMEM_SYCL_HOST_ONLY_FLAG -fsycl-host-only PARENT_SCOPE)
    else()
        check_cxx_compiler_flag(--offload-host-only COMPILER_FLAG_OFFLOAD_HOST_ONLY)
        if (NOT COMPILER_FLAG_OFFLOAD_HOST_ONLY)
            message(FATAL_ERROR
                "Neither '-fsycl-host-only' nor '--offload-host-only' is supported by your compiler")
        endif()
        set(ISHMEM_SYCL_HOST_ONLY_FLAG --offload-host-only PARENT_SCOPE)
    endif()
    set(CMAKE_REQUIRED_FLAGS "${ISHMEM_CMAKE_REQUIRED_FLAGS_SAVE}")

    # Set compiler settings
    set(ISHMEM_CPU_COMPILE_OPTIONS
        -mmovdir64b
        -mavx512f
        -mwaitpkg)
    set(ISHMEM_WARN_COMPILE_OPTIONS
        -Wall
        -Wextra
        -Wconversion
        -Wno-unused-parameter
        -Wformat
        -Wformat-security)
    set(ISHMEM_DEFAULT_COMPILE_OPTIONS
        -D_GNU_SOURCE
        -fvisibility=internal)
    set(ISHMEM_DEBUG_COMPILE_OPTIONS
        -g
        -DENABLE_DEBUG
        -Rno-debug-disables-optimization)
    set(ISHMEM_RELEASE_COMPILE_OPTIONS
        -O3)
    set(ISHMEM_RELWITHDEBINFO_COMPILE_OPTIONS
        -O2
        -g)

    set(ISHMEM_RELEASE_LINK_OPTIONS
        LINKER:-z,noexecstack
        LINKER:-z,nodlopen)

    set(ISHMEM_AOT_COMPILE_OPTIONS)
    set(ISHMEM_AOT_INTERFACE_OPTIONS)
    if (ENABLE_AOT_COMPILATION)
        list(APPEND ISHMEM_AOT_COMPILE_OPTIONS
            -fsycl-targets=spir64_gen
            --start-no-unused-arguments
            -Xs
            "-device ${ISHMEM_AOT_DEVICE_TYPES}"
            --end-no-unused-arguments)
        list(APPEND ISHMEM_AOT_INTERFACE_OPTIONS
            -fsycl-targets=spir64_gen
            "SHELL:--start-no-unused-arguments -Xs \"-device ${ISHMEM_AOT_DEVICE_TYPES}\" --end-no-unused-arguments")
        # Suppress the `Compilation from IR - skipping loading of FCL` outputs:
        list(APPEND ISHMEM_AOT_COMPILE_OPTIONS
            --start-no-unused-arguments
            -Xsycl-target-backend
            -q
            --end-no-unused-arguments)
        list(APPEND ISHMEM_AOT_INTERFACE_OPTIONS
            "SHELL:--start-no-unused-arguments -Xsycl-target-backend -q --end-no-unused-arguments")
    endif()

    set(ISHMEM_CPU_COMPILE_OPTIONS ${ISHMEM_CPU_COMPILE_OPTIONS} PARENT_SCOPE)
    set(ISHMEM_WARN_COMPILE_OPTIONS ${ISHMEM_WARN_COMPILE_OPTIONS} PARENT_SCOPE)
    set(ISHMEM_DEFAULT_COMPILE_OPTIONS ${ISHMEM_DEFAULT_COMPILE_OPTIONS} PARENT_SCOPE)
    set(ISHMEM_DEBUG_COMPILE_OPTIONS ${ISHMEM_DEBUG_COMPILE_OPTIONS} PARENT_SCOPE)
    set(ISHMEM_RELEASE_COMPILE_OPTIONS ${ISHMEM_RELEASE_COMPILE_OPTIONS} PARENT_SCOPE)
    set(ISHMEM_RELWITHDEBINFO_COMPILE_OPTIONS ${ISHMEM_RELWITHDEBINFO_COMPILE_OPTIONS} PARENT_SCOPE)
    set(ISHMEM_RELEASE_LINK_OPTIONS ${ISHMEM_RELEASE_LINK_OPTIONS} PARENT_SCOPE)
    set(ISHMEM_SYCL_COMPILE_OPTIONS ${ISHMEM_SYCL_COMPILE_OPTIONS} PARENT_SCOPE)
    set(ISHMEM_SYCL_LINK_OPTIONS ${ISHMEM_SYCL_LINK_OPTIONS} PARENT_SCOPE)
    set(ISHMEM_AOT_COMPILE_OPTIONS ${ISHMEM_AOT_COMPILE_OPTIONS} PARENT_SCOPE)
    set(ISHMEM_AOT_INTERFACE_OPTIONS ${ISHMEM_AOT_INTERFACE_OPTIONS} PARENT_SCOPE)
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
                # Normalize paths for accurate comparison
                get_filename_component(normalized_i_mpi_root "$ENV{I_MPI_ROOT}" REALPATH)
                get_filename_component(normalized_mpi_dir "${MPI_DIR}" REALPATH)
                if (NOT normalized_i_mpi_root STREQUAL normalized_mpi_dir)
                    message(WARNING " \${MPI_DIR} is set but does not match \${I_MPI_ROOT}.\n"
                                    " FindMPI will likely select \${I_MPI_ROOT} over \${MPI_DIR}.\n"
                                    " To ensure \${MPI_DIR} is used, please remove \${I_MPI_ROOT} from your environment.\n"
                                    " MPI_DIR=${MPI_DIR}\n"
                                    " I_MPI_ROOT=$ENV{I_MPI_ROOT}")
                endif()
            endif()
        endif()

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
