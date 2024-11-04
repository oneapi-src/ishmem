/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "runtime.h"
#if defined(ENABLE_OPENSHMEM)
#include "runtime/runtime_openshmem.h"
#endif
#if defined(ENABLE_MPI)
#include "runtime/runtime_mpi.h"
#endif
#if defined(ENABLE_PMI)
#include "runtime/runtime_pmi.h"
#endif

/* The instance of the runtime backend */
ishmemi_runtime_type *ishmemi_runtime = nullptr;

int ishmemi_runtime_type::unsupported(ishmemi_request_t *msg, ishmemi_ringcompletion_t *comp)
{
    RAISE_ERROR_MSG("Encountered type '%s (%d)' unsupported for operation '%s (%d)'\n",
                    ishmemi_type_str[msg->type], msg->type, ishmemi_op_str[msg->op], msg->op);
    ishmemi_cpu_info->proxy_state = EXIT;
    return -1;
}

const char *ishmemi_runtime_type::team_predefined_string(ishmemi_runtime_team_predefined_t val)
{
    switch (val) {
        case WORLD:
            return "WORLD";
        case SHARED:
            return "SHARED";
        case NODE:
            return "NODE";
        default:
            ISHMEM_ERROR_MSG("Unknown team passed to ishmemi_runtime_team_predefined_string\n");
            return "";
    }
}

int ishmemi_runtime_init(ishmemx_attr_t *attr)
{
    int ret = 0;

    switch (attr->runtime) {
        case ISHMEMX_RUNTIME_MPI:
#if defined(ENABLE_MPI)
            ishmemi_runtime = new ishmemi_runtime_mpi(attr->initialize_runtime, attr->mpi_comm);
            ISHMEM_CHECK_GOTO_MSG(ishmemi_runtime == nullptr, fn_fail,
                                  "MPI runtime instance allocation failed\n");
#else
            ISHMEM_ERROR_MSG(
                "MPI runtime was selected, but Intel® SHMEM is not built with MPI support\n");
            ret = -1;
#endif
            break;
        case ISHMEMX_RUNTIME_OPENSHMEM:
#if defined(ENABLE_OPENSHMEM)
            ishmemi_runtime = new ishmemi_runtime_openshmem(
                attr->initialize_runtime, std::get<bool>(ishmemi_env["RUNTIME_USE_OSHMPI"].first));
            ISHMEM_CHECK_GOTO_MSG(ishmemi_runtime == nullptr, fn_fail,
                                  "SHMEM runtime instance allocation failed\n");
#else
            ISHMEM_ERROR_MSG(
                "OpenSHMEM runtime was selected, but Intel® SHMEM is not built with OpenSHMEM "
                "support\n");
            ret = -1;
#endif
            break;
        case ISHMEMX_RUNTIME_PMI:
#if defined(ENABLE_PMI)
            ishmemi_runtime = new ishmemi_runtime_pmi(attr->initialize_runtime);
            ISHMEM_CHECK_GOTO_MSG(ishmemi_runtime == nullptr, fn_fail,
                                  "PMI runtime instance allocation failed\n");
#else
            ISHMEM_ERROR_MSG(
                "PMI runtime was selected, but Intel® SHMEM is not built with PMI support\n");
            ret = -1;
#endif
            break;
        default:
            ISHMEM_ERROR_MSG("Invalid runtime selection\n");
            ret = -1;
    }

fn_exit:
    return ret;
fn_fail:
    ret = -1;
    goto fn_exit;
}

int ishmemi_runtime_fini()
{
    delete ishmemi_runtime;
    return 0;
}
