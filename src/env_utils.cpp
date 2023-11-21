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

#include <cstdlib>
#include <cmath>
#include <iostream>
#include "env_utils.h"
#include "internal.h"
#include <string.h>

struct ishmemi_params_s ishmemi_params;

/* atol() + optional scaled suffix recognition: 1K, 2M, 3G, 1T */
static int atol_scaled(char *str, ishmemi_env_size *out)
{
    int scale, n, ncheck;
    double p = -1.0;
    char f;
    char tmp;

    n = sscanf(str, "%lf%c", &p, &f);
    ncheck = sscanf(str, "%lf%c%c", &p, &f, &tmp);

    if (n == 2) {
        if (ncheck == 3) {
            return 1;
        }

        switch (f) {
            case 'k':
            case 'K':
                scale = 10;
                break;
            case 'm':
            case 'M':
                scale = 20;
                break;
            case 'g':
            case 'G':
                scale = 30;
                break;
            case 't':
            case 'T':
                scale = 40;
                break;
            default:
                return 1;
        }
    } else if (p < 0) {
        return 1;
    } else {
        scale = 0;
    }

    *out = (ishmemi_env_size) ceil(p * static_cast<double>(1lu << scale));
    return 0;
}

static char *ishmemi_getenv(const char *name)
{
    char *env_name, *env_value;
    int ret;

    ret = asprintf(&env_name, "ISHMEM_%s", name);
    if (ret < 0) {
        RAISE_ERROR_MSG("Error in asprintf: ISHMEM_%s\n", name);
    }
    env_value = getenv(env_name);
    free(env_name);
    if (env_value != nullptr) {
        return env_value;
    }

    return nullptr;
}

static int ishmemi_getenv_size(const char *name, ishmemi_env_size default_val,
                               ishmemi_env_size *out, bool *provided)
{
    char *env = ishmemi_getenv(name);
    *provided = (env != nullptr);
    if (*provided) {
        int ret = atol_scaled(env, out);
        if (ret) {
            RAISE_ERROR_MSG("Could not parse '%s' as a valid a value for ISHMEM_%s\n", env, name);
        }
    } else {
        *out = default_val;
    }
    return 0;
}

static int ishmemi_getenv_bool(const char *name, ishmemi_env_bool default_val,
                               ishmemi_env_bool *out, bool *provided)
{
    char *env = ishmemi_getenv(name);
    bool val;
    *provided = (env != nullptr);
    if (*provided) {
        if (strcmp(env, "0") == 0) {
            val = false;
        } else if (strcasecmp(env, "false") == 0) {
            val = false;
        } else {
            val = true;
        }
    } else {
        val = default_val;
    }
    *out = val;
    return 0;
}

static int ishmemi_getenv_string(const char *name, ishmemi_env_string default_val,
                                 ishmemi_env_string *out, bool *provided)
{
    char *env = ishmemi_getenv(name);
    *provided = (env != nullptr);
    *out = (*provided) ? env : std::move(default_val);
    return 0;
}

int ishmemi_parse_env(void)
{
    int ret;
#define ISHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)                                 \
    ret = ishmemi_getenv_##KIND(#NAME, DEFAULT, &(ishmemi_params.NAME),                            \
                                &(ishmemi_params.NAME##_provided));                                \
    if (ret) return ret;
#include "env_defs.h"
#undef ISHMEMI_ENV_DEF
    return 0;
}
