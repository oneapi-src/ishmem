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

#ifndef ISHMEM_ENV_UTILS_H
#define ISHMEM_ENV_UTILS_H

#include <stddef.h>
#include <string>
#include "ishmem_config.h"

typedef size_t ishmemi_env_size;
typedef bool ishmemi_env_bool;
typedef std::string ishmemi_env_string;

enum ishmemi_env_categories {
    ISHMEMI_ENV_CAT_OPENSHMEM,
    ISHMEMI_ENV_CAT_ATTRIBUTE,
    ISHMEMI_ENV_CAT_IPC,
    ISHMEMI_ENV_CAT_OTHER
};

struct ishmemi_params_s {
#define ISHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) ishmemi_env_##KIND NAME;
#include "env_defs.h"
#undef ISHMEMI_ENV_DEF

#define ISHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) bool NAME##_provided;
#include "env_defs.h"
#undef ISHMEMI_ENV_DEF
};

extern struct ishmemi_params_s ishmemi_params;

int ishmemi_parse_env(void);

#endif /* ISHMEM_ENV_UTILS_H */
