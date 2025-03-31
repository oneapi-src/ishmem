# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

if (ENABLE_OPENSHMEM OR ENABLE_MPI)
    add_subdirectory(jacobi)
endif()

if (ENABLE_PMI)
endif()
