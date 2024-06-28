# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

include_directories("${L0_INSTALL_PREFIX}/include")

#needed for _movdir64b
add_compile_options(-mmovdir64b)
#needed for avx512 for ring use
add_compile_options(-mavx512f)
add_compile_options(-mwaitpkg)

if (ENABLE_OPENSHMEM)
    include_directories("${SHMEM_INSTALL_PREFIX}/include")
endif()
if (ENABLE_MPI)
    SET(CMAKE_EXE_LINKER_FLAGS "${MPI_CXX_LINK_FLAGS}")
    include_directories("${MPI_CXX_INCLUDE_PATH}")
endif()

set(ISHMEM_SOURCES ishmem.cpp
                   rma.cpp
                   ipc.cpp
                   runtime_ipc.cpp
                   env_utils.cpp
                   accelerator.cpp
                   amo.cpp
		   amo_nbi.cpp
                   malloc.cpp
                   memory.cpp
                   proxy.cpp
                   proxy_func.cpp
                   runtime.cpp
                   collectives.cpp
                   nbi.cpp
                   signaling.cpp
                   synchronization.cpp
                   teams.cpp
                   collectives/alltoall.cpp
                   collectives/barrier.cpp
                   collectives/broadcast.cpp
                   collectives/collect.cpp
                   collectives/reduce.cpp
                   collectives/sync.cpp
                   err.cpp
                   timestamp.cpp
                   gpuprint.cpp)

# Disable warnings
set_source_files_properties(malloc.cpp PROPERTIES COMPILE_FLAGS -w)

if (ENABLE_MPI)
  set(ISHMEM_SOURCES ${ISHMEM_SOURCES}
                     runtime/runtime_mpi.cpp
                     runtime/wrapper_mpi.cpp)
endif()
if (ENABLE_OPENSHMEM)
  set(ISHMEM_SOURCES ${ISHMEM_SOURCES}
                     runtime/runtime_openshmem.cpp
                     runtime/wrapper_openshmem.cpp)
endif()
if (ENABLE_PMI)
  set(ISHMEM_SOURCES ${ISHMEM_SOURCES}
                     runtime/runtime_pmi.cpp
                     runtime/wrapper_pmi.cpp)
endif()

# We build ishmem-objects separately to support static and dynamic library creation
# Note: SYCL does not currently support dynamic library creation with device kernels
# https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-2/example-gpu-commands.html
add_library(ishmem-objects OBJECT ${ISHMEM_SOURCES})
set_target_properties(ishmem-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_include_directories(ishmem-objects PRIVATE "${PROJECT_BINARY_DIR}")

add_library(ishmem-static STATIC $<TARGET_OBJECTS:ishmem-objects>)
set_target_properties(ishmem-static PROPERTIES OUTPUT_NAME ishmem)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(ishmem-static PUBLIC ${EXTRA_LIBS}
                      INTERFACE ze_loader
                      INTERFACE ${CMAKE_DL_LIBS}
                      PUBLIC Threads::Threads)
if (ENABLE_MPI)
  target_link_libraries(ishmem-static PUBLIC ${EXTRA_LIBS} INTERFACE mpi)
endif()


if (ENABLE_PMI)
    target_link_libraries(ishmem-static PUBLIC pmi-simple)
endif()

target_include_directories(ishmem-static PUBLIC "${PROJECT_BINARY_DIR}")

include_directories(${CMAKE_SOURCE_DIR}/src)

install(TARGETS ishmem-static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(FILES ishmem.h ishmemx.h DESTINATION include)
