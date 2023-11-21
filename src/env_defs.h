/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 *
 * Copyright (c) 2022 Intel Corporation. All rights reserved.
 *
 * Copyright (c) 2022 Cornelis Networks, Inc. All rights reserved.
 *
 * This software is available to you under the BSD license.
 *
 * COPYRIGHT
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer listed
 *   in this license in the documentation and/or other materials
 *   provided with the distribution.
 *
 * - Neither the name of the copyright holders nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * The copyright holders provide no reassurances that the source code
 * provided does not infringe any patent, copyright, or any other
 * intellectual property rights of third parties.  The copyright holders
 * disclaim any liability to any recipient for claims brought against
 * recipient by any third party for infringement of that parties
 * intellectual property rights.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* General definitions */
ISHMEMI_ENV_DEF(DEBUG, bool, false, ISHMEMI_ENV_CAT_OTHER, "Enable debugging messages")

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

/* Library name definitions */
ISHMEMI_ENV_DEF(SHMEM_LIB_NAME, string, "libsma.so", ISHMEMI_ENV_CAT_ATTRIBUTE,
                "SHMEM Library name")
ISHMEMI_ENV_DEF(MPI_LIB_NAME, string, "libmpi.so", ISHMEMI_ENV_CAT_ATTRIBUTE, "MPI Library name")
ISHMEMI_ENV_DEF(PMI_LIB_NAME, string, "libpmi.so", ISHMEMI_ENV_CAT_ATTRIBUTE, "PMI Library name")
