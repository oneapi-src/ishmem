# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

if (ENABLE_OPENSHMEM)
    add_subdirectory(SHMEM)
endif()

if (ENABLE_MPI)
endif()

if (ENABLE_PMI)
endif()
