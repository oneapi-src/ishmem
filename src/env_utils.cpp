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
#include <utility>
#include "ishmem/env_utils.h"
#include "ishmem/err.h"
#include <cstring>

struct ishmemi_params_s ishmemi_params;
std::map<std::string, std::pair<ishmemi_env_val, bool>> ishmemi_env;

std::unordered_set<std::string> env_ignore = {"ROOT"};

extern char **environ;

/* atol() + optional scaled suffix recognition: 1K, 2M, 3G, 1T */
static int atol_scaled(char *str, size_t *out)
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

    *out = static_cast<size_t>(ceil(p * static_cast<double>(1lu << scale)));
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

/* Note - this function should only be called after ishmemi_parse_env */
ishmemx_runtime_type_t ishmemi_env_get_runtime()
{
    const char *value = std::get<std::string>(ishmemi_env["RUNTIME"].first).c_str();
    ishmemx_runtime_type_t ret = ISHMEMX_RUNTIME_INVALID;

    /* Note - proper checking of compiled runtime support is handled in runtime.cpp */
    if (strcasecmp(value, "OPENSHMEM") == 0) {
        ret = ISHMEMX_RUNTIME_OPENSHMEM;
    } else if (strcasecmp(value, "MPI") == 0) {
        ret = ISHMEMX_RUNTIME_MPI;
    } else if (strcasecmp(value, "PMI") == 0) {
        ret = ISHMEMX_RUNTIME_PMI;
    } else {
        /* Default value */
#if defined(ENABLE_OPENSHMEM)
        return ISHMEMX_RUNTIME_OPENSHMEM;
#endif
#if defined(ENABLE_MPI)
        return ISHMEMX_RUNTIME_MPI;
#endif
#if defined(ENABLE_PMI)
        return ISHMEMX_RUNTIME_PMI;
#endif
    }

    return ret;
}

static int set_env(std::string name,
                   std::map<std::string, std::pair<ishmemi_env_val, bool>>::iterator &iter)
{
    int ret = 0;
    char *env_value = nullptr;

    env_value = ishmemi_getenv(name.c_str());

    if (env_value != nullptr) {
        if (std::holds_alternative<size_t>(ishmemi_env[name].first)) {
            size_t value;

            ret = atol_scaled(env_value, &value);
            ISHMEM_CHECK_GOTO_MSG(ret != 0, fn_exit,
                                  "Could not parse '%s' as a valid value for ISHMEM_%s\n",
                                  env_value, name.c_str());

            iter->second = std::make_pair(value, true);
        } else if (std::holds_alternative<long>(ishmemi_env[name].first)) {
            long value = std::atol(env_value);
            iter->second = std::make_pair(value, true);
        } else if (std::holds_alternative<bool>(ishmemi_env[name].first)) {
            bool value;

            if (strcmp(env_value, "0") == 0) {
                value = false;
            } else if (strcasecmp(env_value, "false") == 0) {
                value = false;
            } else {
                value = true;
            }

            iter->second = std::make_pair(value, true);
        } else if (std::holds_alternative<std::string>(ishmemi_env[name].first)) {
            iter->second = std::pair<std::string, bool>(std::move(env_value), true);
        } else {
            ISHMEM_CHECK_GOTO_MSG(true, fn_fail, "Could not determine type of ishmemi_env[%s]\n",
                                  name.c_str());
        }
    }

fn_exit:
    return ret;
fn_fail:
    ret = 1;
    goto fn_exit;
}

int ishmemi_parse_env(void)
{
    int ret = 0;
    const char *prefix = "ISHMEM_";
    size_t prefix_len = std::strlen(prefix);

    /* Fill ishemmi_env with the default values for environment variables */
#define ISHMEMI_ENV_DEF(NAME, KIND, DEFAULT, SHORT_DESC)                                           \
    ishmemi_env[#NAME] = std::make_pair(ishmemi_env_val(std::in_place_type<KIND>, DEFAULT), false);
#include "ishmem/env_defs.h"
#undef ISHMEMI_ENV_DEF

    /* Parse environment variables provided by the user */
    for (char **env_var = environ; *env_var != nullptr; env_var++) {
        /* Skip any non-ISHMEM environment variables */
        if (std::strncmp(*env_var, prefix, prefix_len) != 0) continue;

        std::string::size_type eq_pos;
        std::string env(*env_var);
        env = env.substr(prefix_len);

        eq_pos = env.find("=");
        if (eq_pos != std::string::npos) {
            env = env.substr(0, eq_pos);
        }

        /* Check if the environment variable should be ignored */
        if (env_ignore.find(env) != env_ignore.end()) continue;

        /* Check that ISHMEM environment variable is defined */
        auto iter = ishmemi_env.find(env);

        if (iter == ishmemi_env.end()) {
            ISHMEM_WARN_MSG("Environment variable 'ISHMEM_%s' is not a supported variable\n",
                            env.c_str());
        } else {
            ret = set_env(env, iter);
            ISHMEM_CHECK_GOTO_MSG(ret != 0, fn_exit,
                                  "Failed to initialize variable for 'ISHMEM_%s'\n", env.c_str());
        }
    }

    /* Fill ishmemi_params with the default/user values */
#define ISHMEMI_ENV_DEF(NAME, KIND, DEFAULT, SHORT_DESC)                                           \
    ishmemi_params.NAME = std::get<KIND>(ishmemi_env[#NAME].first);
#include "ishmem/env_defs.h"
#undef ISHMEMI_ENV_DEF

fn_exit:
    return ret;
}
