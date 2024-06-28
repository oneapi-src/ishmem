/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portions derived from Sandia OpenSHMEM (https://github.com/Sandia-OpenSHMEM/SOS)
 * For license and copyright information, see the LICENSE and third-party-programs.txt
 * files in the top level directory of this distribution.
 */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include "env_utils.h"
#include "ishmem/err.h"
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

static long errchk_atol(char *s)
{
    long val;
    char *e;
    errno = 0;

    val = strtol(s, &e, 0);
    if (errno != 0 || e == s) {
        RAISE_ERROR_MSG("Environment variable conversion failed (%s)\n", s);
    }

    return val;
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

__attribute__((unused)) static int ishmemi_getenv_long(const char *name,
                                                       ishmemi_env_long default_val,
                                                       ishmemi_env_long *out, bool *provided)
{
    char *env = ishmemi_getenv(name);
    *provided = (env != NULL);
    *out = (*provided) ? errchk_atol(env) : default_val;
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
