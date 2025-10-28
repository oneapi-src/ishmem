/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portions derived from Sandia OpenSHMEM (https://github.com/Sandia-OpenSHMEM/SOS)
 * For license and copyright information, see the LICENSE and third-party-programs.txt
 * files in the top level directory of this distribution.
 */

#ifndef ISHMEM_ENV_UTILS_H
#define ISHMEM_ENV_UTILS_H

#include <stddef.h>
#include <string>
#include <map>
#include <unordered_set>
#include <variant>
#include "ishmem/config.h"
#include "ishmemx.h"

typedef std::variant<size_t, long, bool, std::string> ishmemi_env_val;

extern std::map<std::string, std::pair<ishmemi_env_val, bool>> ishmemi_env;

struct ishmemi_params_s {
#define ISHMEMI_ENV_DEF(NAME, KIND, DEFAULT, SHORT_DESC) KIND NAME;
#include "ishmem/env_defs.h"
#undef ISHMEMI_ENV_DEF
};

extern struct ishmemi_params_s ishmemi_params;

int ishmemi_parse_env(void);
ishmemx_runtime_type_t ishmemi_env_get_runtime(void);

#endif /* ISHMEM_ENV_UTILS_H */
