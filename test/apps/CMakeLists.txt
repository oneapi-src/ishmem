# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

if (ENABLE_OPENSHMEM OR ENABLE_MPI)
    add_subdirectory(jacobi)
endif()

if (ENABLE_PMI)
endif()
