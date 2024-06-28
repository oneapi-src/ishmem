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
#include "ishmem_config.h"

typedef long ishmemi_env_long;
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
