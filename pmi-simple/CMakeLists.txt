# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

if (ENABLE_PMI)
    include(CheckIncludeFiles)
    set(CMAKE_C_COMPILER clang)
    set(USE_PMI_PORT 1)

    check_include_files(assert.h HAVE_ASSERT_H)
    check_include_files(unistd.h HAVE_UNISTD_H)
    check_include_files(stdlib.h HAVE_STDLIB_H)
    check_include_files(string.h HAVE_STRING_H)
    check_include_files(strings.h HAVE_STRINGS_H)
    check_include_files(sys/socket.h HAVE_SYS_SOCKET_H)
    check_include_files(sys/types.h HAVE_SYS_TYPES_H)
    check_include_files(arpa/inet.h HAVE_ARPA_INET_H)
    configure_file(pmi-simple_config.h.in pmi-simple_config.h)

    set(PMI_SIMPLE_SOURCES simple_pmi.cpp
                           simple_pmiutil.cpp)
    add_library(pmi-simple ${PMI_SIMPLE_SOURCES})

    target_include_directories(pmi-simple
                               PUBLIC "${CMAKE_SOURCE_DIR}/src"
                               PUBLIC "${PROJECT_BINARY_DIR}"
                               PUBLIC "${EXTRA_INCS}")

    set_target_properties(pmi-simple PROPERTIES PRIVATE_HEADER simple_pmiutil.h
                                          PRIVATE_HEADER pmi.h
                                          PRIVATE_HEADER mpl.h)

    install(TARGETS pmi-simple DESTINATION lib)
endif()
