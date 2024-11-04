# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

if (ENABLE_OPENSHMEM)
    add_subdirectory(jacobi)
endif()

if (ENABLE_MPI)
endif()

if (ENABLE_PMI)
endif()
