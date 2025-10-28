/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portions derived from Sandia OpenSHMEM (https://github.com/Sandia-OpenSHMEM/SOS)
 * For license and copyright information, see the LICENSE and third-party-programs.txt
 * files in the top level directory of this distribution.
 */

/* General definitions */
ISHMEMI_ENV_DEF(DEBUG, bool, false, "Enable debugging messages")
ISHMEMI_ENV_DEF(ENABLE_VERBOSE_PRINT, bool, false, "Include file and line info in debug messages")
ISHMEMI_ENV_DEF(STACK_PRINT_LIMIT, size_t, 10,
                "How many stack frames to print on RAISE_ERROR can be 10 to 50")

/* IPC definitions */
ISHMEMI_ENV_DEF(ENABLE_GPU_IPC, bool, true, "Enable intra-node inter-GPU IPC implementation")
ISHMEMI_ENV_DEF(ENABLE_GPU_IPC_PIDFD, bool, true,
                "Enable pidfd implementation for IPC handle exchange")

/* Symmetric Heap definitions */
ISHMEMI_ENV_DEF(SYMMETRIC_SIZE, size_t, 512 * 1024 * 1024, "Symmetric heap size")
ISHMEMI_ENV_DEF(ENABLE_ACCESSIBLE_HOST_HEAP, bool, false,
                "Enable shared symmetric heap in host and device")

/* Tuning parameters */
ISHMEMI_ENV_DEF(NBI_COUNT, size_t, 1024, "NBI operations between GC")
ISHMEMI_ENV_DEF(MWAIT_BURST, size_t, 0, "Use UMONITOR UMWAIT in proxy thread, burst count")

/* Library name definitions */
ISHMEMI_ENV_DEF(SHMEM_LIB_NAME, std::string, "libsma.so", "SHMEM Library name")
ISHMEMI_ENV_DEF(MPI_LIB_NAME, std::string, "libmpi.so", "MPI Library name")
ISHMEMI_ENV_DEF(PMI_LIB_NAME, std::string, "libpmi.so", "PMI Library name")

ISHMEMI_ENV_DEF(TEAMS_MAX, size_t, 64, "Maximum number of teams per PE")
ISHMEMI_ENV_DEF(TEAM_SHARED_ONLY_SELF, bool, false,
                "Include only the self PE in ISHMEM_TEAM_SHARED")

/* Runtime definitions */
ISHMEMI_ENV_DEF(RUNTIME, std::string, ISHMEM_DEFAULT_RUNTIME_STR,
                "The default runtime to use for scale-out communication")
ISHMEMI_ENV_DEF(RUNTIME_USE_OSHMPI, bool, false, "Specify the OpenSHMEM backend as OSHMPI")
