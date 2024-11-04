/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portions derived from Sandia OpenSHMEM (https://github.com/Sandia-OpenSHMEM/SOS)
 * For license and copyright information, see the LICENSE and third-party-programs.txt
 * files in the top level directory of this distribution.
 */

#include "teams.h"
#include "ishmem/err.h"
#include "runtime.h"
#include "runtime/runtime_types.h"
#include "collectives.h"
#include "collectives/reduce_impl.h"
#include "accelerator.h"

#define ISHMEMI_TEAMS_MIN   3 /* The number of pre-defined teams */
#define ISHMEMI_DIAG_STRLEN 1024
#define N_PSYNC_BYTES       8
/* TODO determine ISHMEMI_SYNC_SIZE at configuration */
#define ISHMEMI_SYNC_SIZE 32
#define PSYNC_CHUNK_SIZE  (N_PSYNCS_PER_TEAM * ISHMEMI_SYNC_SIZE)

static int team_ret_val; /* These are globals so they can be used in a collective */
static int team_ret_val_reduced;
static unsigned char psync_pool_avail[N_PSYNC_BYTES];
static unsigned char psync_pool_avail_reduced[N_PSYNC_BYTES];

/* Checks whether a PE has a consistent stride given (start, stride, size).
 * This function is useful within a loop across PE IDs, and sets 'start',
 * 'stride' and 'size' accordingly upon exiting the loop. It also assumes
 * 'start' and 'stride' are initialized to a negative number and 'size' to 0.
 * If an inconsistent stride is found, returns -1. */
static inline int check_for_linear_stride(int pe, int *start, int *stride, int *size)
{
    if (*start < 0) {
        *start = pe;
        (*size)++;
    } else if (*stride < 0) {
        *stride = pe - *start;
        (*size)++;
    } else if ((pe - *start) % *stride != 0) {
        ISHMEM_WARN_MSG("Detected non-uniform stride inserting PE %d into <%d, %d, %d>\n", pe,
                        *start, *stride, *size);
        return -1;
    } else {
        (*size)++;
    }
    return 0;
}

/* Return -1 if `global_pe` is not in the given active set.
 * If `global_pe` is in the active set, return the PE index within this set.
 * A stride of 0 causes divide-by-zero, which is asserted only with ENABLE_ERROR_CHECKING */

/* count local PEs */
int ishmemi_team_local_pes(int start, int stride, int size)
{
    int local = 0;
    if (stride > 0) {
        for (int pe = start; pe < start + (size * stride); pe += stride) {
            if (ishmemi_ptr(ishmemi_heap_base, pe) != nullptr) local += 1;
        }
    } else {
        for (int pe = start; pe > start + (size * stride); pe += stride) {
            if (ishmemi_ptr(ishmemi_heap_base, pe) != nullptr) local += 1;
        }
    }
    return (local);
}

static int team_init(ishmem_team_t team, int my_pe, int start, int stride, int size)
{
    /* team_host is in the device symmetric heap, so we need to copy to it, rather than store */
    if (team < 0 || team >= ISHMEMI_N_TEAMS) return -1;
    ishmemi_team_host_t *team_host = &ishmemi_cpu_info->team_host_pool[team];
    /* team_host is in the device symmetric heap, so we need to copy to it, rather than store */

    ishmemi_team_device_t *team_device = &ishmemi_mmap_gpu_info->team_device_pool[team];
    ::memset(team_host, 0, sizeof(ishmemi_team_host_t));
    ishmem_zero((void *) team_device, sizeof(ishmemi_team_device_t));
    team_host->start = start;
    team_host->stride = stride;
    team_host->size = size;
    team_host->n_local_pes =
        (ishmemi_params.ENABLE_GPU_IPC) ? ishmemi_team_local_pes(start, stride, size) : 1;
    team_host->only_intra =
        ishmemi_params.ENABLE_GPU_IPC && (team_host->n_local_pes == team_host->size);
    team_host->last_pe = start + (stride * (size - 1));
    team_host->my_pe = my_pe;
    ISHMEM_DEBUG_MSG("init team %d, my_pe %d, start %d, stride %d, size %d\n", (int) team, my_pe,
                     start, stride, size);
    ISHMEM_DEBUG_MSG("init team %d, only_intra %d, n_local_pes %d\n", (int) team,
                     (team_host->only_intra) ? 1 : 0, team_host->n_local_pes);

    SET_HEAP_FIELD(team_device->start, team_host->start);
    SET_HEAP_FIELD(team_device->stride, team_host->stride);
    SET_HEAP_FIELD(team_device->size, team_host->size);
    SET_HEAP_FIELD(team_device->n_local_pes, team_host->n_local_pes);
    SET_HEAP_FIELD(team_device->only_intra, team_host->only_intra);
    SET_HEAP_FIELD(team_device->last_pe, team_host->last_pe);
    SET_HEAP_FIELD(team_device->my_pe, team_host->my_pe);

    return (0);
}

int ishmemi_team_init(void)
{
    int ret = 0;
    int start = -1;
    int stride = -1;
    int size = 0;
    ishmemi_runtime_team_t runtime_team;
    int my_team_pe;
    ishmemi_team_host_t *host_team;

    size_t n_teams = static_cast<size_t>(ishmemi_params.TEAMS_MAX);
    if (ishmemi_params.TEAMS_MAX < ISHMEMI_TEAMS_MIN) ishmemi_params.TEAMS_MAX = ISHMEMI_TEAMS_MIN;
    ishmemi_cpu_info->n_teams = n_teams;
    ishmemi_mmap_gpu_info->n_teams = n_teams;

    ::memset(psync_pool_avail, 0, N_PSYNC_BYTES);
    for (size_t i = 0; i < ishmemi_mmap_gpu_info->n_teams; i++) {
        ishmemi_bit_set(psync_pool_avail, N_PSYNC_BYTES, i);
    }

    /* Set the bits for SHMEM_TEAM_WORLD, SHMEM_TEAM_SHARED, and SHMEMX_TEAM_NODE to 0: */
    ishmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, ISHMEM_TEAM_WORLD);
    ishmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, ISHMEM_TEAM_SHARED);
    ishmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, ISHMEMX_TEAM_NODE);

    ishmemi_cpu_info->team_host_pool =
        (ishmemi_team_host_t *) ishmemi_runtime->malloc(sizeof(ishmemi_team_host_t) * n_teams);
    ISHMEM_CHECK_GOTO_MSG(ishmemi_cpu_info->team_host_pool == nullptr, cleanup,
                          "Allocation of team host pool failed\n");
    ::memset(ishmemi_cpu_info->team_host_pool, 0, sizeof(ishmemi_team_host_t) * n_teams);

    ishmemi_mmap_gpu_info->team_device_pool =
        (ishmemi_team_device_t *) ishmemi_calloc(n_teams, sizeof(ishmemi_team_device_t));
    ISHMEM_CHECK_GOTO_MSG(ishmemi_mmap_gpu_info->team_device_pool == nullptr, cleanup,
                          "Allocation of device team pool failed\n");

    /* Initialize ISHMEM_TEAM_WORLD */
    start = 0;
    stride = 1;
    size = ishmemi_n_pes;
    my_team_pe = ishmemi_my_pe;

    ret = ishmemi_runtime->team_predefined_set(&runtime_team,
                                               ishmemi_runtime_team_predefined_t::WORLD,
                                               /* expected size */ size,
                                               /* expected world pe */ ishmemi_my_pe,
                                               /* expected team pe */ my_team_pe);

    ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "ishmemi_runtime_team_predefined_set WORLD failed\n");
    team_init(ISHMEM_TEAM_WORLD, ishmemi_my_pe, 0, 1, ishmemi_n_pes);
    host_team = &ishmemi_cpu_info->team_host_pool[ISHMEM_TEAM_WORLD];
    host_team->runtime_team = runtime_team;
    ISHMEM_DEBUG_MSG("ISHMEM_TEAM_WORLD: start=%d, stride=%d, size=%d, n_local=%d\n",
                     host_team->start, host_team->stride, host_team->size, host_team->n_local_pes);

    /* Initialize ISHMEM_TEAM_SHARED */

    if (ishmemi_params.TEAM_SHARED_ONLY_SELF) {
        start = ishmemi_my_pe;
        stride = 1;
        size = 1;
        my_team_pe = -1;
    } else { /* Search for shared-memory peer PEs while checking for a consistent stride */
        start = -1;
        stride = -1;
        size = 0;

        for (int pe = 0; pe < ishmemi_n_pes; pe++) {
            void *ret_ptr = ishmemi_ptr(ishmemi_heap_base, pe);
            if (ret_ptr == NULL) continue;

            ret = check_for_linear_stride(pe, &start, &stride, &size);
            if (ret < 0) {
                start = ishmemi_my_pe;
                stride = 1;
                size = 1;
                break;
            }
        }

        if (!(size > 0 && size <= ishmemi_runtime->get_node_size())) {
            RAISE_ERROR_MSG("size of shared memory team out of range");
        }

        stride = (stride == -1) ? 1 : stride;

        my_team_pe = ishmemi_pe_in_active_set(ishmemi_my_pe, start, stride, size);

        assert(my_team_pe >= 0);
    }
    ret = ishmemi_runtime->team_predefined_set(&runtime_team,
                                               ishmemi_runtime_team_predefined_t::SHARED,
                                               /* expected size */ size,
                                               /* expected world pe */ ishmemi_my_pe,
                                               /* expected team pe */ my_team_pe);
    ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "ishmemi_runtime_team_predefined_set SHARED failed\n");

    team_init(ISHMEM_TEAM_SHARED, my_team_pe, start, stride, size);
    host_team = &ishmemi_cpu_info->team_host_pool[ISHMEM_TEAM_SHARED];
    host_team->runtime_team = runtime_team;
    ISHMEM_DEBUG_MSG("ISHMEM_TEAM_SHARED: start=%d, stride=%d, size=%d, n_local=%d\n",
                     host_team->start, host_team->stride, host_team->size, host_team->n_local_pes);

    /* Initialize ISHMEM_TEAM_NODE */
    /* Search for on-node peer PEs while checking for a consistent stride */
    start = -1;
    stride = -1;
    size = 0;
    for (int pe = 0; pe < ishmemi_n_pes; pe++) {
        ret = ishmemi_runtime->get_node_rank(pe);
        if (ret < 0) continue;

        ret = check_for_linear_stride(pe, &start, &stride, &size);
        if (ret < 0) {
            start = ishmemi_my_pe;
            stride = 1;
            size = 1;
            break;
        }
    }
    assert(size > 0 && size == ishmemi_runtime->get_node_size());

    stride = (stride == -1) ? 1 : stride;

    my_team_pe = ishmemi_pe_in_active_set(ishmemi_my_pe, start, stride, size);
    assert(my_team_pe >= 0);

    ret = ishmemi_runtime->team_predefined_set(
        &runtime_team, ishmemi_runtime_team_predefined_t::NODE, size, ishmemi_my_pe, my_team_pe);
    ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "ishmemi_runtime_team_predefined_set NODE failed\n");

    team_init(ISHMEMX_TEAM_NODE, my_team_pe, start, stride, size);
    host_team = &ishmemi_cpu_info->team_host_pool[ISHMEMX_TEAM_NODE];
    host_team->runtime_team = runtime_team;
    ISHMEM_DEBUG_MSG("ISHMEMX_TEAM_NODE: start=%d, stride=%d, size=%d, n_local=%d\n",
                     host_team->start, host_team->stride, host_team->size, host_team->n_local_pes);

    if (ishmemi_params.TEAMS_MAX > N_PSYNC_BYTES * CHAR_BIT) {
        ISHMEM_ERROR_MSG("Requested %ld teams, but only %d are supported\n",
                         ishmemi_params.TEAMS_MAX, N_PSYNC_BYTES * CHAR_BIT);
        goto cleanup;
    }
    return 0;

cleanup:
    ISHMEMI_FREE(ishmemi_runtime->free, ishmemi_cpu_info->team_host_pool);
    ISHMEMI_FREE(ishmem_free, ishmemi_mmap_gpu_info->team_device_pool);

    return -1;
}

void ishmemi_team_destroy(ishmem_team_t team)
{
    if (team <= ISHMEM_TEAM_INVALID || team >= ISHMEMI_N_TEAMS) return;

    if (team == ISHMEM_TEAM_WORLD || team == ISHMEM_TEAM_SHARED || team == ISHMEMX_TEAM_NODE) {
        ISHMEM_WARN_MSG("User attempted to destroy a pre-defined team.\n");
        return;
    }
    ishmemi_team_host_t *host_team = &ishmemi_cpu_info->team_host_pool[team];
    if (host_team->size > 0) {
        host_team->size = 0;
        ishmemi_bit_set(psync_pool_avail, N_PSYNC_BYTES, static_cast<size_t>(team));

        ISHMEM_DEBUG_MSG("destroy team %d runtime_team\n", team);
        ishmemi_runtime->team_destroy(ishmemi_cpu_info->team_host_pool[team].runtime_team);
    }
}
int ishmemi_team_fini(void)
{
    /* Destroy all undestroyed user-created teams */
    for (size_t i = ISHMEMI_TEAMS_MIN; i < ISHMEMI_N_TEAMS; i++) {
        ishmemi_team_host_t *team = &ishmemi_cpu_info->team_host_pool[i];
        if (team->size > 0) {
            ishmem_team_destroy((ishmem_team_t) i);
        }
    }

    /* Free the device team resources */
    ISHMEMI_FREE(ishmem_free, ishmemi_mmap_gpu_info->team_device_pool);
    /* Free the host team resources */
    ISHMEMI_FREE(ishmemi_runtime->free, ishmemi_cpu_info->team_host_pool);

    return 0;
}

int ishmemi_team_split_strided(ishmem_team_t parent_team_idx, int PE_start, int PE_stride,
                               int PE_size, const ishmem_team_config_t *config, long config_mask,
                               ishmem_team_t *new_team)
{
    if (parent_team_idx == ISHMEM_TEAM_INVALID) return 1;

    ishmemi_team_host_t *parent_team = &ishmemi_cpu_info->team_host_pool[parent_team_idx];
    *new_team = ISHMEM_TEAM_INVALID;
    ishmemi_runtime_team_t new_runtime_team;
    int ret = -1;
    PE_stride = (PE_stride == 0 || PE_size == 1) ? 1 : PE_stride;

    int global_PE_start = ishmemi_team_pe(parent_team, PE_start);
    int global_PE_stride = parent_team->stride * PE_stride;
    int global_PE_end = global_PE_start + global_PE_stride * (PE_size - 1);
    if (PE_start < 0 || PE_start >= parent_team->size || PE_size <= 0 ||
        PE_size > parent_team->size) {
        ISHMEM_WARN_MSG("Invalid <start, stride, size>: child <%d, %d, %d>, parent <%d, %d, %d>\n",
                        PE_start, PE_stride, PE_size, parent_team->start, parent_team->stride,
                        parent_team->size);
        return -1;
    }

    if (global_PE_start < 0 || global_PE_start >= ishmemi_n_pes) {
        ISHMEM_WARN_MSG("Starting global PE (%d) is invalid\n", global_PE_start);
        return -1;
    }

    if (global_PE_end < 0 || global_PE_end >= ishmemi_n_pes) {
        ISHMEM_WARN_MSG("Ending global PE (%d) is invalid\n", global_PE_end);
        return -1;
    }

    /* do the bit reduction to find a team slot, then fill it in */

    team_ret_val = 0;
    team_ret_val_reduced = 0;
    bool psync_pool_bit_cleared = false;

    int my_team_pe =
        ishmemi_pe_in_active_set(ishmemi_my_pe, global_PE_start, global_PE_stride, PE_size);
    /* my_team_pe == -1 means we are not in the new team */
    ishmemi_team_host_t *host_team = nullptr;
    ishmemi_team_device_t *device_team = nullptr;

    /* call runtime team_split to obtain the corresponding runtime team */
    /* all PEs in the parent team call this, so some may not be in the new team */
    ret = ishmemi_runtime->team_split_strided(
        parent_team->runtime_team, PE_start, PE_stride, PE_size,
        reinterpret_cast<const ishmemi_runtime_team_config_t *>(config), config_mask,
        &new_runtime_team);
    if (ret != 0) return 1;
    bool runtime_team_set = true;

    if (my_team_pe != -1) {
        char bit_str[ISHMEMI_DIAG_STRLEN];

        /* the members of the new team do this reduction to find a team number */
        int ret = ishmemi_runtime->uchar_and_reduce(new_runtime_team, psync_pool_avail_reduced,
                                                    psync_pool_avail, N_PSYNC_BYTES);
        ISHMEM_CHECK_RETURN_MSG(ret, "Call to ishmemi_runtime_uchar_and_reduce failed\n");

        ishmemi_bit_to_string(bit_str, ISHMEMI_DIAG_STRLEN, psync_pool_avail_reduced,
                              N_PSYNC_BYTES);
        ISHMEM_DEBUG_MSG("My pSyncs  [ %s ]\n", bit_str);

        /* Select the least signficant nonzero bit, which corresponds to an available pSync. */
        *new_team = static_cast<ishmem_team_t>(
            ishmemi_bit_1st_nonzero(psync_pool_avail_reduced, N_PSYNC_BYTES));
        /* ishmemi_bit_1st_nonzero returns -1 if it cannot find any non-zero bits */

        ishmemi_bit_to_string(bit_str, ISHMEMI_DIAG_STRLEN, psync_pool_avail_reduced,
                              N_PSYNC_BYTES);
        ISHMEM_DEBUG_MSG("All pSyncs [ %s ], allocated %d\n", bit_str, *new_team);

        if (*new_team == ISHMEM_TEAM_INVALID || *new_team >= ishmemi_params.TEAMS_MAX) {
            ISHMEM_WARN_MSG("No more teams available (max = %ld), try increasing SHMEM_TEAMS_MAX\n",
                            ishmemi_params.TEAMS_MAX);
            /* No psync was available, but must call barrier across parent team before returning. */
            *new_team = ISHMEM_TEAM_INVALID;
            team_ret_val = 1;
        } else {
            /* Set the selected psync bit to 0, reserving that slot */
            ishmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, static_cast<size_t>(*new_team));
            psync_pool_bit_cleared = true;
            /* only now can we fill in the new team */
            team_init(*new_team, my_team_pe, global_PE_start, global_PE_stride, PE_size);
            host_team = &ishmemi_cpu_info->team_host_pool[*new_team];
            device_team = &ishmemi_mmap_gpu_info->team_device_pool[*new_team];
        }
    }

    ishmem_team_config_t defaults;
    ISHMEM_CHECK_RETURN_MSG(ret, "Proxy call to team_split_strided failed\n");
    if (config_mask == 0) {
        if (config != NULL) {
            ISHMEM_WARN_MSG("%s %s\n", "team_split_strided operation encountered an unexpected",
                            "non-NULL config structure passed with a config_mask of 0.");
        }
        defaults.num_contexts = 0;
    } else {
        if (config_mask != ISHMEM_TEAM_NUM_CONTEXTS) {
            ISHMEM_WARN_MSG("Invalid team_split_strided config_mask (%ld)\n", config_mask);
            return 1;  // everyone should hit this if anyone does
        } else {
            assert(config->num_contexts >= 0);
        }
    }

    if (*new_team != ISHMEM_TEAM_INVALID) {
        /* sets the shmem_team_t field on the new team for use with the proxy: */
        ISHMEM_DEBUG_MSG("team %d new_runtime_team\n", *new_team);
        if (host_team != nullptr && device_team != nullptr) {
            host_team->runtime_team = new_runtime_team;
            host_team->config_mask = (size_t) config_mask;
            if (config_mask == 0) {
                ::memcpy(&host_team->config, &defaults, sizeof(ishmem_team_config_t));
            } else {
                host_team->config = *config;
            }
            SET_HEAP_FIELD(device_team->config_mask, host_team->config_mask);
            SET_HEAP_FIELD(device_team->config, host_team->config);
        } else {
            ISHMEM_WARN_MSG("Host team or device team reference is NULL\n");
            goto cleanup;
        }
    }
    /* All PEs in the parent team are doing these things */
    /* This barrier on the parent team eliminates problematic race conditions
     * during psync allocation between back-to-back team creations. */
    ishmemi_runtime->quiet();
    ret = ishmemi_runtime->team_sync(parent_team->runtime_team);
    ISHMEM_CHECK_RETURN_MSG(ret, "Call to ishmemi_runtime->team_sync failed\n");

    /* This MAX reduction assures all PEs return the same value.  */
    ret = ishmemi_runtime->int_max_reduce(parent_team->runtime_team, &team_ret_val_reduced,
                                          &team_ret_val, 1);
    ISHMEM_CHECK_RETURN_MSG(ret, "Call to ishmemi_int_max_reduce failed\n");

    /* If no team was available, print some team triplet info and return nonzero. */
    if (my_team_pe >= 0 && *new_team == ISHMEM_TEAM_INVALID) {
        if (host_team != nullptr) {
            ISHMEM_WARN_MSG("Team split strided failed: child <%d, %d, %d>, parent <%d, %d, %d>\n",
                            host_team->start, host_team->stride, host_team->size,
                            parent_team->start, parent_team->stride, parent_team->size);
        } else {
            ISHMEM_WARN_MSG("Host team reference is NULL\n");
        }
        goto cleanup;
    }
    return team_ret_val_reduced;

cleanup:
    if (runtime_team_set) ishmemi_runtime->team_destroy(new_runtime_team);
    if (psync_pool_bit_cleared)
        ishmemi_bit_set(psync_pool_avail, N_PSYNC_BYTES, static_cast<size_t>(*new_team));
    return -1;
}

int ishmemi_team_split_2d(ishmem_team_t parent_team_idx, int xrange,
                          const ishmem_team_config_t *xaxis_config, long xaxis_mask,
                          ishmem_team_t *xaxis_team, const ishmem_team_config_t *yaxis_config,
                          long yaxis_mask, ishmem_team_t *yaxis_team)
{
    if (parent_team_idx == ISHMEM_TEAM_INVALID) {
        return 1;
    }
    ishmemi_team_host_t *parent_team = &ishmemi_cpu_info->team_host_pool[parent_team_idx];
    *xaxis_team = ISHMEM_TEAM_INVALID;
    *yaxis_team = ISHMEM_TEAM_INVALID;

    if (xrange > parent_team->size) {
        xrange = parent_team->size;
    }

    const int parent_size = parent_team->size;
    const int num_xteams =
        static_cast<int>(ceil(static_cast<float>(parent_size) / static_cast<float>(xrange)));
    const int num_yteams = xrange;

    int start = 0;
    int ret = 0;

    for (int i = 0; i < num_xteams; i++) {
        ishmem_team_t my_xteam;
        int xsize = (i == num_xteams - 1 && parent_size % xrange) ? parent_size % xrange : xrange;

        ret = ishmem_team_split_strided(parent_team_idx, start, 1, xsize, xaxis_config, xaxis_mask,
                                        &my_xteam);
        if (ret) {
            ISHMEM_ERROR_MSG("Creation of x-axis team %d of %d failed\n", i + 1, num_xteams);
        }
        start += xrange;

        if (my_xteam != ISHMEM_TEAM_INVALID) {
            assert(*xaxis_team == ISHMEM_TEAM_INVALID);
            *xaxis_team = my_xteam;
        }
    }

    start = 0;

    for (int i = 0; i < num_yteams; i++) {
        ishmem_team_t my_yteam;
        int remainder = parent_size % xrange;
        int yrange = parent_size / xrange;
        int ysize = (remainder && i < remainder) ? yrange + 1 : yrange;

        ret = ishmem_team_split_strided(parent_team_idx, start, xrange, ysize, yaxis_config,
                                        yaxis_mask, &my_yteam);
        if (ret) {
            ISHMEM_ERROR_MSG("Creation of y-axis team %d of %d failed\n", i + 1, num_yteams);
        }
        start += 1;

        if (my_yteam != ISHMEM_TEAM_INVALID) {
            assert(*yaxis_team == ISHMEM_TEAM_INVALID);
            *yaxis_team = my_yteam;
        }
    }

    ret = ishmemi_runtime->team_sync(parent_team->runtime_team);
    ISHMEM_CHECK_RETURN_MSG(ret, "Call to ishmemi_runtime->team_sync failed\n");

    return 0;
}

int ishmem_team_my_pe(ishmem_team_t team)
{
    if constexpr (enable_error_checking) validate_init();
    if (team <= ISHMEM_TEAM_INVALID || team >= ISHMEMI_N_TEAMS) return -1;
    else
#ifdef __SYCL_DEVICE_ONLY__
        return global_info->team_device_pool[team].my_pe;
#else
        return ishmemi_cpu_info->team_host_pool[team].my_pe;
#endif
}

int ishmem_team_n_pes(ishmem_team_t team)
{
    if constexpr (enable_error_checking) validate_init();
    if (team <= ISHMEM_TEAM_INVALID || team >= ISHMEMI_N_TEAMS) return -1;
    else
#ifdef __SYCL_DEVICE_ONLY__
        return global_info->team_device_pool[team].size;
#else
        return ishmemi_cpu_info->team_host_pool[team].size;
#endif
}

int ishmem_team_get_config(ishmem_team_t team, long config_mask, ishmem_team_config_t *config)
{
    if constexpr (enable_error_checking) validate_init();
    if (team <= ISHMEM_TEAM_INVALID || team >= ISHMEMI_N_TEAMS) return -1;

#ifdef __SYCL_DEVICE_ONLY__
    ishmemi_team_device_t *team_ptr = &global_info->team_device_pool[team];
#else
    ishmemi_team_host_t *team_ptr = &ishmemi_cpu_info->team_host_pool[team];
#endif
    if (config_mask != 0) {
        if (config_mask != ISHMEM_TEAM_NUM_CONTEXTS) {
            ISHMEM_WARN_MSG("Invalid team config mask (%ld)\n", config_mask);
            return -1;
        }
        if (config == NULL) {
            ISHMEM_WARN_MSG("NULL config pointer passed to shmem_team_get_config\n");
            return -1;
        }
        ::memcpy(config, &team_ptr->config, sizeof(ishmem_team_config_t));
    } else if (config != NULL) {
        ISHMEM_WARN_MSG("%s %s\n", "ishmem_team_get_config encountered an unexpected",
                        "non-NULL config structure passed with a config_mask of 0.");
    }
    return 0;
}

int ishmem_team_translate_pe(ishmem_team_t src_team, int src_pe, ishmem_team_t dest_team)
{
    if constexpr (enable_error_checking) validate_init();
    if (src_team <= ISHMEM_TEAM_INVALID || dest_team <= ISHMEM_TEAM_INVALID ||
        src_team >= ISHMEMI_N_TEAMS || dest_team >= ISHMEMI_N_TEAMS)
        return -1;

#if __SYCL_DEVICE_ONLY__
    ishmemi_team_device_t *from_team = &global_info->team_device_pool[src_team];
    ishmemi_team_device_t *to_team = &global_info->team_device_pool[dest_team];
#else
    ishmemi_team_host_t *from_team = &ishmemi_cpu_info->team_host_pool[src_team];
    ishmemi_team_host_t *to_team = &ishmemi_cpu_info->team_host_pool[dest_team];
#endif

    int src_pe_world, dest_pe = -1;

    if (src_pe > from_team->size) return -1;

    src_pe_world = from_team->start + src_pe * from_team->stride;

    if constexpr (enable_error_checking) {
        /* world_size is unused in a Release build type, because NDEBUG is
         * defined and the assertion below is disabled */
        int world_size __attribute__((unused)) = -1;
#if __SYCL_DEVICE_ONLY__
        world_size = global_info->n_pes;
#else
        /* During ishmemi_teams_init, the team_pool may not be allocated yet */
        world_size = ishmemi_n_pes;
#endif
        if (!(src_pe_world >= from_team->start && (src_pe_world < world_size))) {
            RAISE_ERROR_MSG("translation error, pe out of range");
        }
    }

    dest_pe =
        ishmemi_pe_in_active_set(src_pe_world, to_team->start, to_team->stride, to_team->size);

    return dest_pe;
}

/* Teams Management Routines */
int ishmem_team_split_strided(ishmem_team_t parent_team, int PE_start, int PE_stride, int PE_size,
                              const ishmem_team_config_t *config, long config_mask,
                              ishmem_team_t *new_team)
{
    if constexpr (enable_error_checking) validate_init();
    if (parent_team <= ISHMEM_TEAM_INVALID || parent_team >= ISHMEMI_N_TEAMS ||
        (PE_stride == 0 && PE_size != 1))
        return -1;

    return ishmemi_team_split_strided(parent_team, PE_start, PE_stride, PE_size, config,
                                      config_mask, new_team);
}

int ishmem_team_split_2d(ishmem_team_t parent_team, int xrange,
                         const ishmem_team_config_t *xaxis_config, long xaxis_mask,
                         ishmem_team_t *xaxis_team, const ishmem_team_config_t *yaxis_config,
                         long yaxis_mask, ishmem_team_t *yaxis_team)
{
    if constexpr (enable_error_checking) validate_init();
    if (parent_team <= ISHMEM_TEAM_INVALID || parent_team >= ISHMEMI_N_TEAMS) return -1;

    return ishmemi_team_split_2d(parent_team, xrange, xaxis_config, xaxis_mask, xaxis_team,
                                 yaxis_config, yaxis_mask, yaxis_team);
}

void ishmem_team_destroy(ishmem_team_t team)
{
    if constexpr (enable_error_checking) validate_init();
    ishmemi_team_destroy(team);
}
