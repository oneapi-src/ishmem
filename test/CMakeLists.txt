# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

include_directories("include")

add_subdirectory(unit)
add_subdirectory(apps)
if (BUILD_PERF_TEST)
  add_subdirectory(performance)
endif ()
