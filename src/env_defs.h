/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portions derived from Sandia OpenSHMEM (https://github.com/Sandia-OpenSHMEM/SOS)
 * For license and copyright information, see the LICENSE and third-party-programs.txt
 * files in the top level directory of this distribution.
 */

/* General definitions */
ISHMEMI_ENV_DEF(DEBUG, bool, false, ISHMEMI_ENV_CAT_OTHER, "Enable debugging messages")
ISHMEMI_ENV_DEF(ENABLE_VERBOSE_PRINT, bool, false, ISHMEMI_ENV_CAT_OTHER,
                "Include file and line info in debug messages")

/* IPC definitions */
ISHMEMI_ENV_DEF(ENABLE_GPU_IPC, bool, true, ISHMEMI_ENV_CAT_IPC,
                "Enable intra-node inter-GPU IPC implementation")
ISHMEMI_ENV_DEF(ENABLE_GPU_IPC_PIDFD, bool, true, ISHMEMI_ENV_CAT_IPC,
                "Enable pidfd implementation for IPC handle exchange")

/* Symmetric Heap definitions */
ISHMEMI_ENV_DEF(SYMMETRIC_SIZE, size, 512 * 1024 * 1024, ISHMEMI_ENV_CAT_OPENSHMEM,
                "Symmetric heap size")
ISHMEMI_ENV_DEF(ENABLE_ACCESSIBLE_HOST_HEAP, bool, false, ISHMEMI_ENV_CAT_OTHER,
                "Enable shared symmetric heap in host and device")
/* Tuning parameters */
ISHMEMI_ENV_DEF(NBI_COUNT, size, 1024, ISHMEMI_ENV_CAT_OPENSHMEM, "NBI operations between GC")
ISHMEMI_ENV_DEF(MWAIT_BURST, size, 0, ISHMEMI_ENV_CAT_OPENSHMEM,
                "use UMONITOR UMWAIT in proxy thread, burst count")

/* Library name definitions */
ISHMEMI_ENV_DEF(SHMEM_LIB_NAME, string, "libsma.so", ISHMEMI_ENV_CAT_ATTRIBUTE,
                "SHMEM Library name")
ISHMEMI_ENV_DEF(MPI_LIB_NAME, string, "libmpi.so", ISHMEMI_ENV_CAT_ATTRIBUTE, "MPI Library name")
ISHMEMI_ENV_DEF(PMI_LIB_NAME, string, "libpmi.so", ISHMEMI_ENV_CAT_ATTRIBUTE, "PMI Library name")

ISHMEMI_ENV_DEF(TEAMS_MAX, size, 64, ISHMEMI_ENV_CAT_OTHER, "Maximum number of teams per PE")
ISHMEMI_ENV_DEF(TEAM_SHARED_ONLY_SELF, bool, false, ISHMEMI_ENV_CAT_OTHER,
                "Include only the self PE in ISHMEM_TEAM_SHARED")
