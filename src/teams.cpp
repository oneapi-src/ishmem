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
#include "collectives.h"
#include "collectives/reduce_impl.h"
#include "accelerator.h"

#define ISHMEMI_TEAMS_MIN   3 /* The number of pre-defined teams */
#define ISHMEMI_DIAG_STRLEN 1024
#define N_PSYNC_BYTES       8
/* TODO determine ISHMEMI_SYNC_SIZE at configuration */
#define ISHMEMI_SYNC_SIZE 32
#define PSYNC_CHUNK_SIZE  (N_PSYNCS_PER_TEAM * ISHMEMI_SYNC_SIZE)

ishmemi_team_t *ishmemi_team_world = nullptr;
ishmemi_team_t *ishmemi_team_shared = nullptr;
ishmemi_team_t *ishmemi_team_node = nullptr;

void *reduce_buffs;
void *reduce_source;
void *reduce_dest;
size_t *collect_team_mynelems;
size_t *collect_team_nelems;
size_t *collect_team_my_size;
size_t *collect_team_sizes;

static unsigned char *psync_pool_avail;
static unsigned char *psync_pool_avail_reduced;

static int *team_ret_val;
static int *team_ret_val_reduced;

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

static inline int pe_in_active_set(int global_pe, int PE_start, int PE_stride, int PE_size)
{
    if (PE_size == 1) return PE_start == global_pe ? 0 : -1;
    if (PE_stride == 0) return -1;
    int n = (global_pe - PE_start) / PE_stride;
    if ((global_pe < PE_start && PE_stride > 0) || (global_pe > PE_start && PE_stride < 0) ||
        (global_pe - PE_start) % PE_stride || n >= PE_size)
        return -1;
    else {
        return n;
    }
}

int ishmemi_team_init(void)
{
    int ret = 0;
    size_t psync_len = 0;
    int start = -1, stride = -1, size = 0;
    size_t pool_idx = 0;

    size_t n_teams = static_cast<size_t>(ishmemi_params.TEAMS_MAX);

    /* Initialize ISHMEM_TEAM_WORLD */
    ret = ishmemi_usm_alloc_host((void **) &ishmemi_team_world, sizeof(ishmemi_team_t));
    ISHMEM_CHECK_RETURN_MSG(ret, "ISHMEM_TEAM_WORLD allocation (usm_alloc_host) failed\n");
    memset((void *) ishmemi_team_world, 0, sizeof(ishmem_team_t));

    ishmemi_team_world->psync_idx = ISHMEM_TEAM_WORLD;
    ishmemi_team_world->start = 0;
    ishmemi_team_world->stride = 1;
    ishmemi_team_world->size = ishmemi_n_pes;
    ishmemi_team_world->my_pe = ishmemi_my_pe;
    ishmemi_team_world->config_mask = 0;
    memset(&ishmemi_team_world->config, 0, sizeof(ishmem_team_config_t));
    for (size_t i = 0; i < N_PSYNCS_PER_TEAM; i++)
        ishmemi_team_world->psync_avail[i] = 1;
    ret = ishmemi_runtime_team_predefined_set(&ishmemi_team_world->shmem_team,
                                              ishmemi_runtime_team_predefined_t::WORLD,
                                              ishmemi_n_pes, ishmemi_my_pe, ishmemi_my_pe);
    ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "ishmemi_runtime_team_predefined_set WORLD failed\n");

    /* Pre-allocate contiguous reduction bounce buffers to be used by all teams */
    reduce_buffs = (void *) ishmem_malloc(ISHMEM_REDUCE_BUFFER_SIZE * n_teams);
    ISHMEM_CHECK_GOTO_MSG(reduce_buffs == nullptr, cleanup,
                          "Allocation of team device work buffer pool failed\n");

    reduce_dest = (void *) ishmemi_runtime_malloc(ISHMEM_REDUCE_BUFFER_SIZE * n_teams);
    ISHMEM_CHECK_GOTO_MSG(reduce_dest == nullptr, cleanup,
                          "Allocation of team dest host bounce buffer pool failed\n");

    reduce_source = (void *) ishmemi_runtime_malloc(ISHMEM_REDUCE_BUFFER_SIZE * n_teams);
    ISHMEM_CHECK_GOTO_MSG(reduce_source == nullptr, cleanup,
                          "Allocation of team source host bounce buffer pool failed\n");

    /* Pre-allocate collect/fcollect scratch buffers for all teams */
    collect_team_mynelems = (size_t *) ishmem_calloc(n_teams, sizeof(size_t));
    ISHMEM_CHECK_GOTO_MSG(collect_team_mynelems == nullptr, cleanup,
                          "Allocation of collect_team_mynelems failed\n");

    collect_team_nelems = (size_t *) ishmem_calloc(MAX_LOCAL_PES * n_teams, sizeof(size_t));
    ISHMEM_CHECK_GOTO_MSG(collect_team_nelems == nullptr, cleanup,
                          "Allocation of collect_team_nelems failed\n");

    collect_team_my_size = (size_t *) ishmemi_runtime_calloc(n_teams, sizeof(size_t));
    ISHMEM_CHECK_GOTO_MSG(collect_team_my_size == nullptr, cleanup,
                          "Allocation of collect_team_my_size failed\n");

    collect_team_sizes = (size_t *) ishmemi_runtime_calloc(MAX_LOCAL_PES * n_teams, sizeof(size_t));
    ISHMEM_CHECK_GOTO_MSG(collect_team_sizes == nullptr, cleanup,
                          "Allocation of collect_team_sizes failed\n");

    ishmemi_team_world->buffer = reduce_buffs;
    ishmemi_team_world->dest = reduce_dest;
    ishmemi_team_world->source = reduce_source;
    ishmemi_team_world->collect_mynelems = collect_team_mynelems;
    ishmemi_team_world->collect_nelems = collect_team_nelems;
    ishmemi_team_world->collect_my_size = collect_team_my_size;
    ishmemi_team_world->collect_sizes = collect_team_sizes;

    /* Initialize ISHMEM_TEAM_SHARED */
    ret = ishmemi_usm_alloc_host((void **) &ishmemi_team_shared, sizeof(ishmemi_team_t));
    ISHMEM_CHECK_RETURN_MSG(ret, "ISHMEM_TEAM_SHARED allocation (usm_alloc_host) failed\n");
    memset((void *) ishmemi_team_shared, 0, sizeof(ishmem_team_t));

    ishmemi_team_shared->psync_idx = ISHMEM_TEAM_SHARED;
    ishmemi_team_shared->my_pe = 0;
    ishmemi_team_shared->config_mask = 0;
    memset(&ishmemi_team_shared->config, 0, sizeof(ishmem_team_config_t));
    for (size_t i = 0; i < N_PSYNCS_PER_TEAM; i++)
        ishmemi_team_shared->psync_avail[i] = 1;

    pool_idx = static_cast<size_t>(ISHMEM_TEAM_SHARED) * ISHMEM_REDUCE_BUFFER_SIZE;
    ishmemi_team_shared->buffer = static_cast<uint8_t *>(reduce_buffs) + pool_idx;
    ishmemi_team_shared->dest = static_cast<uint8_t *>(reduce_dest) + pool_idx;
    ishmemi_team_shared->source = static_cast<uint8_t *>(reduce_source) + pool_idx;

    pool_idx = static_cast<size_t>(ISHMEM_TEAM_SHARED);
    ishmemi_team_shared->collect_mynelems = collect_team_mynelems + pool_idx;
    ishmemi_team_shared->collect_my_size = collect_team_my_size + pool_idx;

    pool_idx = static_cast<size_t>(ISHMEM_TEAM_SHARED) * MAX_LOCAL_PES;
    ishmemi_team_shared->collect_nelems = collect_team_nelems + pool_idx;
    ishmemi_team_shared->collect_sizes = collect_team_sizes + pool_idx;

    /* Initialize ISHMEM_TEAM_NODE */
    ret = ishmemi_usm_alloc_host((void **) &ishmemi_team_node, sizeof(ishmemi_team_t));
    ISHMEM_CHECK_RETURN_MSG(ret, "ISHMEMX_TEAM_NODE allocation (usm_alloc_host) failed\n");
    memset((void *) ishmemi_team_node, 0, sizeof(ishmem_team_t));

    ishmemi_team_node->psync_idx = ISHMEMX_TEAM_NODE;
    ishmemi_team_node->my_pe = 0;
    ishmemi_team_node->config_mask = 0;
    memset(&ishmemi_team_node->config, 0, sizeof(ishmem_team_config_t));
    for (size_t i = 0; i < N_PSYNCS_PER_TEAM; i++)
        ishmemi_team_node->psync_avail[i] = 1;

    if (ishmemi_params.TEAM_SHARED_ONLY_SELF) {
        ishmemi_team_shared->start = ishmemi_my_pe;
        ishmemi_team_shared->stride = 1;
        ishmemi_team_shared->size = 1;
    } else { /* Search for shared-memory peer PEs while checking for a consistent stride */
        int start = -1, stride = -1, size = 0;

        for (int pe = 0; pe < ishmemi_n_pes; pe++) {
            void *ret_ptr = ishmem_ptr(ishmemi_heap_base, pe);
            if (ret_ptr == NULL) continue;

            ret = check_for_linear_stride(pe, &start, &stride, &size);
            if (ret < 0) {
                start = ishmemi_my_pe;
                stride = 1;
                size = 1;
                break;
            }
        }
        assert(size > 0 && size <= ishmemi_runtime_get_node_size());

        ishmemi_team_shared->start = start;
        ishmemi_team_shared->stride = (stride == -1) ? 1 : stride;
        ishmemi_team_shared->size = size;
        ishmemi_team_shared->my_pe =
            ishmemi_team_translate_pe(ishmemi_team_world, ishmemi_my_pe, ishmemi_team_shared);
        assert(ishmemi_team_shared->my_pe >= 0);

        ISHMEM_DEBUG_MSG("ISHMEM_TEAM_SHARED: start=%d, stride=%d, size=%d\n",
                         ishmemi_team_shared->start, ishmemi_team_shared->stride,
                         ishmemi_team_shared->size);
    }
    ret = ishmemi_runtime_team_predefined_set(
        &ishmemi_team_shared->shmem_team, ishmemi_runtime_team_predefined_t::SHARED,
        ishmemi_team_shared->size, ishmemi_my_pe, ishmemi_team_shared->my_pe);
    ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "ishmemi_runtime_team_predefined_set SHARED failed\n");

    /* Search for on-node peer PEs while checking for a consistent stride */
    start = -1, stride = -1, size = 0;
    for (int pe = 0; pe < ishmemi_n_pes; pe++) {
        ret = ishmemi_runtime_get_node_rank(pe);
        if (ret < 0) continue;

        ret = check_for_linear_stride(pe, &start, &stride, &size);
        if (ret < 0) {
            start = ishmemi_my_pe;
            stride = 1;
            size = 1;
            break;
        }
    }
    assert(size > 0 && size == ishmemi_runtime_get_node_size());

    ishmemi_team_node->start = start;
    ishmemi_team_node->stride = (stride == -1) ? 1 : stride;
    ishmemi_team_node->size = size;
    ishmemi_team_node->my_pe =
        ishmemi_team_translate_pe(ishmemi_team_world, ishmemi_my_pe, ishmemi_team_node);
    ret = ishmemi_runtime_team_predefined_set(&ishmemi_team_node->shmem_team,
                                              ishmemi_runtime_team_predefined_t::NODE, size,
                                              ishmemi_my_pe, ishmemi_team_node->my_pe);
    ISHMEM_CHECK_GOTO_MSG(ret, cleanup, "ishmemi_runtime_team_predefined_set NODE failed\n");

    pool_idx = static_cast<size_t>(ISHMEMX_TEAM_NODE) * ISHMEM_REDUCE_BUFFER_SIZE;
    ishmemi_team_node->buffer = static_cast<uint8_t *>(reduce_buffs) + pool_idx;
    ishmemi_team_node->dest = static_cast<uint8_t *>(reduce_dest) + pool_idx;
    ishmemi_team_node->source = static_cast<uint8_t *>(reduce_source) + pool_idx;

    pool_idx = static_cast<size_t>(ISHMEMX_TEAM_NODE);
    ishmemi_team_node->collect_mynelems = collect_team_mynelems + pool_idx;
    ishmemi_team_node->collect_my_size = collect_team_my_size + pool_idx;

    pool_idx = static_cast<size_t>(ISHMEMX_TEAM_NODE) * MAX_LOCAL_PES;
    ishmemi_team_node->collect_nelems = collect_team_nelems + pool_idx;
    ishmemi_team_node->collect_sizes = collect_team_sizes + pool_idx;

    ISHMEM_DEBUG_MSG("ISHMEMX_TEAM_NODE: start=%d, stride=%d, size=%d\n", ishmemi_team_node->start,
                     ishmemi_team_node->stride, ishmemi_team_node->size);

    if (ishmemi_params.TEAMS_MAX > N_PSYNC_BYTES * CHAR_BIT) {
        ISHMEM_ERROR_MSG("Requested %ld teams, but only %d are supported\n",
                         ishmemi_params.TEAMS_MAX, N_PSYNC_BYTES * CHAR_BIT);
        goto cleanup;
    }

    if (ishmemi_params.TEAMS_MAX < ISHMEMI_TEAMS_MIN) ishmemi_params.TEAMS_MAX = ISHMEMI_TEAMS_MIN;

    ishmemi_mmap_gpu_info->n_teams = ishmemi_params.TEAMS_MAX;

    ret = ishmemi_usm_alloc_host(
        (void **) &(ishmemi_mmap_gpu_info->team_pool),
        static_cast<size_t>(ishmemi_params.TEAMS_MAX) * sizeof(ishmemi_team_t *));
    ISHMEM_CHECK_RETURN_MSG(ret, "team pool allocation (usm_alloc_host) failed\n");

    for (long i = 0; i < ishmemi_mmap_gpu_info->n_teams; i++) {
        ishmemi_mmap_gpu_info->team_pool[i] = NULL;
    }
    ishmemi_mmap_gpu_info->team_pool[ISHMEM_TEAM_WORLD] = ishmemi_team_world;
    ishmemi_mmap_gpu_info->team_pool[ISHMEM_TEAM_SHARED] = ishmemi_team_shared;
    ishmemi_mmap_gpu_info->team_pool[ISHMEMX_TEAM_NODE] = ishmemi_team_node;

    /* clang-format off */
    /* Allocate pSync pool, each with the maximum possible size requirement */
    /* Create two pSyncs per team for back-to-back collectives and one for barriers.
     * Array organization:
     *
     * [ (world) (shared) (node) (team 1) (team 2) ...  (world) (shared) (node) (team 1) (team 2) ... ]
     *  <----------- groups 1 & 2------------------->|<------------- group 3 ------------------------->
     *  <--- (bcast, collect, reduce, etc.) -------->|<------ (barriers and syncs) ------------------->
     * */
    /* clang-format on */
    psync_len = ishmemi_mmap_gpu_info->n_teams * (PSYNC_CHUNK_SIZE + ISHMEMI_SYNC_SIZE);
    ishmemi_mmap_gpu_info->psync_pool = (long *) ishmem_calloc(sizeof(long), psync_len);
    ISHMEM_CHECK_GOTO_MSG(ishmemi_mmap_gpu_info->psync_pool == nullptr, cleanup,
                          "Allocation of internal psync pool failed\n");

    /* Convenience pointer to the group-3 pSync array (for barriers and syncs): */
    ishmemi_mmap_gpu_info->psync_barrier_pool =
        &ishmemi_mmap_gpu_info->psync_pool[PSYNC_CHUNK_SIZE * ishmemi_mmap_gpu_info->n_teams];

    psync_pool_avail = (unsigned char *) ishmemi_runtime_calloc(N_PSYNC_BYTES, 2);
    ISHMEM_CHECK_GOTO_MSG(psync_pool_avail == nullptr, cleanup,
                          "Allocation of internal psync pool failed\n");
    psync_pool_avail_reduced = &psync_pool_avail[N_PSYNC_BYTES];

    for (size_t i = 0; i < ishmemi_mmap_gpu_info->n_teams; i++) {
        ishmemi_bit_set(psync_pool_avail, N_PSYNC_BYTES, i);
    }

    /* Set the bits for SHMEM_TEAM_WORLD, SHMEM_TEAM_SHARED, and SHMEMX_TEAM_NODE to 0: */
    ishmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, ISHMEM_TEAM_WORLD);
    ishmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, ISHMEM_TEAM_SHARED);
    ishmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, ISHMEMX_TEAM_NODE);

    /* Initialize an integer used to agree on an equal return value across PEs in team creation: */
    team_ret_val = (int *) ishmemi_runtime_malloc(sizeof(int) * 2);
    ISHMEM_CHECK_GOTO_MSG(team_ret_val == nullptr, cleanup,
                          "Allocation of team return value failed\n");

    team_ret_val_reduced = &team_ret_val[1];

    return 0;

cleanup:
    /* collect buffers */
    ISHMEMI_FREE(ishmem_free, collect_team_mynelems);
    ISHMEMI_FREE(ishmem_free, collect_team_nelems);
    ISHMEMI_FREE(ishmemi_runtime_free, collect_team_my_size);
    ISHMEMI_FREE(ishmemi_runtime_free, collect_team_sizes);

    /* reduction buffers */
    ISHMEMI_FREE(ishmem_free, reduce_buffs);
    ISHMEMI_FREE(ishmemi_runtime_free, reduce_dest);
    ISHMEMI_FREE(ishmemi_runtime_free, reduce_source);

    /* psync pool */
    ISHMEMI_FREE(ishmemi_usm_free, ishmemi_mmap_gpu_info->team_pool);
    ISHMEMI_FREE(ishmem_free, ishmemi_mmap_gpu_info->psync_pool);
    ISHMEMI_FREE(ishmemi_runtime_free, psync_pool_avail);
    ISHMEMI_FREE(ishmemi_runtime_free, team_ret_val);

    /* pre-defined teams */
    ISHMEMI_FREE(ishmemi_usm_free, ishmemi_team_world);
    ISHMEMI_FREE(ishmemi_usm_free, ishmemi_team_shared);
    ISHMEMI_FREE(ishmemi_usm_free, ishmemi_team_node);

    return -1;
}

int ishmemi_team_fini(void)
{
    /* Destroy all undestroyed user-created teams */
    for (size_t i = ISHMEMI_TEAMS_MIN; i < ishmemi_mmap_gpu_info->n_teams; i++) {
        if (ishmemi_mmap_gpu_info->team_pool[i] != NULL)
            ishmemi_team_destroy(ishmemi_mmap_gpu_info->team_pool[i]);
    }

    /* Free the collect scratch buffers */
    ISHMEMI_FREE(ishmem_free, collect_team_mynelems);
    ISHMEMI_FREE(ishmem_free, collect_team_nelems);
    ISHMEMI_FREE(ishmemi_runtime_free, collect_team_my_size);
    ISHMEMI_FREE(ishmemi_runtime_free, collect_team_sizes);

    /* Free the reduction bounce buffers */
    ISHMEMI_FREE(ishmem_free, reduce_buffs);
    ISHMEMI_FREE(ishmemi_runtime_free, reduce_dest);
    ISHMEMI_FREE(ishmemi_runtime_free, reduce_source);

    /* Destroy all pre-defined teams except world. */
    for (int i = ISHMEMI_TEAMS_MIN - 1; i > 0; i--) {
        if (ishmemi_mmap_gpu_info->team_pool[i] != NULL)
            ishmemi_team_destroy(ishmemi_mmap_gpu_info->team_pool[i]);
    }

    /* Free the psync and team pool resources */
    ISHMEMI_FREE(ishmemi_runtime_free, psync_pool_avail);
    ISHMEMI_FREE(ishmemi_runtime_free, team_ret_val);
    ISHMEMI_FREE(ishmem_free, ishmemi_mmap_gpu_info->psync_pool);
    ISHMEMI_FREE(ishmemi_usm_free, ishmemi_mmap_gpu_info->team_pool);

    /* Free the world team object */
    ISHMEMI_FREE(ishmemi_usm_free, ishmemi_team_world);

    return 0;
}

int ishmemi_team_destroy(ishmemi_team_t *team)
{
    if (team == nullptr) return -1;
    if (team->psync_idx == ISHMEM_TEAM_INVALID) return -1;

    ishmemi_bit_set(psync_pool_avail, N_PSYNC_BYTES, static_cast<size_t>(team->psync_idx));
    ishmemi_mmap_gpu_info->team_pool[team->psync_idx] = nullptr;

    ISHMEMI_FREE(ishmemi_usm_free, team);

    return 0;
}

int ishmemi_team_translate_pe(ishmemi_team_t *src_team, int src_pe, ishmemi_team_t *dest_team)
{
    int src_pe_world, dest_pe = -1;

    if (src_team == nullptr || dest_team == nullptr) return -1;

    if (src_pe > src_team->size) return -1;

    src_pe_world = src_team->start + src_pe * src_team->stride;

    if constexpr (enable_error_checking) {
        /* world_size is unused in a Release build type, because NDEBUG is
         * defined and the assertion below is disabled */
        int world_size __attribute__((unused)) = -1;
#if __SYCL_DEVICE_ONLY__
        ishmemi_team_t *team_world = global_info->team_pool[ISHMEM_TEAM_WORLD];
        world_size = team_world->size;
#else
        /* During ishmemi_teams_init, the team_pool may not be allocated yet */
        world_size = ishmemi_n_pes;
#endif
        assert(src_pe_world >= src_team->start && src_pe_world < world_size);
    }

    dest_pe = pe_in_active_set(src_pe_world, dest_team->start, dest_team->stride, dest_team->size);

    return dest_pe;
}

int ishmemi_team_split_strided(ishmemi_team_t *parent_team, int PE_start, int PE_stride,
                               int PE_size, const ishmem_team_config_t *config, long config_mask,
                               ishmem_team_t *new_team)
{
    *new_team = ISHMEM_TEAM_INVALID;

    if (parent_team == nullptr) {
        return 1;
    }

    int global_PE_start = ishmemi_team_pe(parent_team, PE_start);
    int global_PE_end = global_PE_start + PE_stride * (PE_size - 1);
    int my_pe = pe_in_active_set(ishmemi_my_pe, global_PE_start, PE_stride, PE_size);

    if (PE_start < 0 || PE_start >= parent_team->size || PE_size <= 0 ||
        PE_size > parent_team->size || (PE_stride == 0 && PE_size != 1)) {
        ISHMEM_WARN_MSG("Invalid <start, stride, size>: child <%d, %d, %d>, parent <%d, %d, %d>\n",
                        PE_start, PE_stride, PE_size, parent_team->start, parent_team->stride,
                        parent_team->size);
        return -1;
    }

    if (global_PE_start >= ishmemi_n_pes || global_PE_end >= ishmemi_n_pes) {
        ISHMEM_WARN_MSG("Starting PE (%d) or ending PE (%d) is invalid\n", global_PE_start,
                        global_PE_end);
        return -1;
    }

    ishmemi_team_t *myteam = nullptr;
    int ret = ishmemi_usm_alloc_host((void **) &myteam, sizeof(ishmemi_team_t));
    ISHMEM_CHECK_GOTO_MSG(ret || myteam == nullptr, cleanup,
                          "Allocation of new team split strided object failed\n");
    memset((void *) myteam, 0, sizeof(ishmem_team_t));

    /* sets the ishmemi_team_t fields: */
    myteam->my_pe = my_pe;
    myteam->start = global_PE_start;
    myteam->stride = (PE_stride == 0 || PE_size == 1) ? 1 : PE_stride;
    myteam->size = PE_size;
    if (config_mask == 0) {
        if (config != NULL) {
            ISHMEM_WARN_MSG("%s %s\n", "team_split_strided operation encountered an unexpected",
                            "non-NULL config structure passed with a config_mask of 0.");
        }
        ishmem_team_config_t defaults;
        myteam->config_mask = 0;
        defaults.num_contexts = 0;
        memcpy(&myteam->config, &defaults, sizeof(ishmem_team_config_t));
    } else {
        if (config_mask != ISHMEM_TEAM_NUM_CONTEXTS) {
            ISHMEM_WARN_MSG("Invalid team_split_strided config_mask (%ld)\n", config_mask);
            goto cleanup;
        } else {
            assert(config->num_contexts >= 0);
            myteam->config = *config;
            myteam->config_mask = config_mask;
        }
    }

    myteam->psync_idx = ISHMEM_TEAM_INVALID;

    /* sets the shmem_team_t field on the new team for use with the proxy: */
    ret = ishmemi_runtime_team_split_strided(
        parent_team->shmem_team, PE_start, myteam->stride, PE_size,
        reinterpret_cast<const ishmemi_runtime_team_config_t *>(config), config_mask,
        &myteam->shmem_team);

    ISHMEM_CHECK_RETURN_MSG(ret, "Proxy call to team_split_strided failed\n");

    *team_ret_val = 0;
    *team_ret_val_reduced = 0;

    if (my_pe != -1) {
        char bit_str[ISHMEMI_DIAG_STRLEN];

        int ret = ishmemi_runtime_uchar_and_reduce(myteam->shmem_team, psync_pool_avail_reduced,
                                                   psync_pool_avail, N_PSYNC_BYTES);
        ISHMEM_CHECK_RETURN_MSG(ret, "Call to ishmemi_runtime_uchar_and_reduce failed\n");

        ishmemi_bit_to_string(bit_str, ISHMEMI_DIAG_STRLEN, psync_pool_avail_reduced,
                              N_PSYNC_BYTES);
        ISHMEM_DEBUG_MSG("My pSyncs  [ %s ]\n", bit_str);

        /* Select the least signficant nonzero bit, which corresponds to an available pSync. */
        myteam->psync_idx = static_cast<ishmem_team_t>(
            ishmemi_bit_1st_nonzero(psync_pool_avail_reduced, N_PSYNC_BYTES));

        ishmemi_bit_to_string(bit_str, ISHMEMI_DIAG_STRLEN, psync_pool_avail_reduced,
                              N_PSYNC_BYTES);
        ISHMEM_DEBUG_MSG("All pSyncs [ %s ], allocated %d\n", bit_str, myteam->psync_idx);

        if (myteam->psync_idx == -1 || myteam->psync_idx >= ishmemi_params.TEAMS_MAX) {
            ISHMEM_WARN_MSG("No more teams available (max = %ld), try increasing SHMEM_TEAMS_MAX\n",
                            ishmemi_params.TEAMS_MAX);
            /* No psync was available, but must call barrier across parent team before returning. */
            myteam->psync_idx = -1;
            *team_ret_val = 1;
        } else {
            /* Set the selected psync bit to 0, reserving that slot */
            ishmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES,
                              static_cast<size_t>(myteam->psync_idx));

            for (size_t i = 0; i < N_PSYNCS_PER_TEAM; i++)
                myteam->psync_avail[i] = 1;

            /* Set the reduction bounce buffers from pre-allocated objects */
            size_t pool_idx = static_cast<size_t>(myteam->psync_idx) * ISHMEM_REDUCE_BUFFER_SIZE;
            myteam->buffer = static_cast<uint8_t *>(reduce_buffs) + pool_idx;
            myteam->dest = static_cast<uint8_t *>(reduce_dest) + pool_idx;
            myteam->source = static_cast<uint8_t *>(reduce_source) + pool_idx;
            ishmemi_mmap_gpu_info->team_pool[myteam->psync_idx] = myteam;

            /* Set the collect scratch buffers from pre-allocated objects */
            pool_idx = static_cast<size_t>(myteam->psync_idx);
            myteam->collect_mynelems = &collect_team_mynelems[pool_idx];
            myteam->collect_nelems = &collect_team_nelems[MAX_LOCAL_PES * pool_idx];
            myteam->collect_my_size = &collect_team_my_size[pool_idx];
            myteam->collect_sizes = &collect_team_sizes[MAX_LOCAL_PES * pool_idx];

            *new_team = myteam->psync_idx;
        }
    }

    /* This barrier on the parent team eliminates problematic race conditions
     * during psync allocation between back-to-back team creations. */
    ishmemi_runtime_quiet();
    ret = ishmemi_runtime_team_sync(parent_team->shmem_team);
    ISHMEM_CHECK_RETURN_MSG(ret, "Call to ishmemi_runtime_team_sync failed\n");

    /* This MAX reduction assures all PEs return the same value.  */
    ret = ishmemi_runtime_int_max_reduce(parent_team->shmem_team, team_ret_val_reduced,
                                         team_ret_val, 1);
    ISHMEM_CHECK_RETURN_MSG(ret, "Call to ishmemi_int_max_reduce failed\n");

    /* If no team was available, print some team triplet info and return nonzero. */
    if (my_pe >= 0 && myteam != NULL && myteam->psync_idx == -1) {
        ISHMEM_WARN_MSG("Team split strided failed: child <%d, %d, %d>, parent <%d, %d, %d>\n",
                        myteam->start, myteam->stride, myteam->size, parent_team->start,
                        parent_team->stride, parent_team->size);
        goto cleanup;
    }
    return *team_ret_val_reduced;

cleanup:
    ISHMEMI_FREE(ishmemi_usm_free, myteam);
    return -1;
}

int ishmemi_team_split_2d(ishmemi_team_t *parent_team, int xrange,
                          const ishmem_team_config_t *xaxis_config, long xaxis_mask,
                          ishmem_team_t *xaxis_team, const ishmem_team_config_t *yaxis_config,
                          long yaxis_mask, ishmem_team_t *yaxis_team)
{
    *xaxis_team = ISHMEM_TEAM_INVALID;
    *yaxis_team = ISHMEM_TEAM_INVALID;

    if (parent_team == nullptr) {
        return 1;
    }

    if (xrange > parent_team->size) {
        xrange = parent_team->size;
    }

    const int parent_stride = parent_team->stride;
    const int parent_size = parent_team->size;
    const int num_xteams =
        static_cast<int>(ceil(static_cast<float>(parent_size) / static_cast<float>(xrange)));
    const int num_yteams = xrange;

    int start = 0;
    int ret = 0;

    for (int i = 0; i < num_xteams; i++) {
        ishmem_team_t my_xteam;
        int xsize = (i == num_xteams - 1 && parent_size % xrange) ? parent_size % xrange : xrange;

        ret = ishmemi_team_split_strided(parent_team, start, parent_stride, xsize, xaxis_config,
                                         xaxis_mask, &my_xteam);
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

        ret = ishmemi_team_split_strided(parent_team, start, xrange * parent_stride, ysize,
                                         yaxis_config, yaxis_mask, &my_yteam);
        if (ret) {
            ISHMEM_ERROR_MSG("Creation of y-axis team %d of %d failed\n", i + 1, num_yteams);
        }
        start += 1;

        if (my_yteam != ISHMEM_TEAM_INVALID) {
            assert(*yaxis_team == ISHMEM_TEAM_INVALID);
            *yaxis_team = my_yteam;
        }
    }

    ret = ishmemi_runtime_team_sync(parent_team->shmem_team);
    ISHMEM_CHECK_RETURN_MSG(ret, "Call to ishmemi_runtime_team_sync failed\n");

    return 0;
}

/* Returns a psync from the given team that can be safely used for the
 * specified collective operation. */
long *ishmemi_team_choose_psync(ishmemi_team_t *team, ishmemi_team_op_t op)
{
    switch (op) {
        case TEAM_OP_SYNC:
#if __SYCL_DEVICE_ONLY__
            return &global_info->psync_barrier_pool[team->psync_idx * ISHMEMI_SYNC_SIZE];
#else
            return &ishmemi_mmap_gpu_info->psync_barrier_pool[team->psync_idx * ISHMEMI_SYNC_SIZE];
#endif

        default:
            for (int i = 0; i < N_PSYNCS_PER_TEAM; i++) {
                if (team->psync_avail[i]) {
                    team->psync_avail[i] = 0;
#if __SYCL_DEVICE_ONLY__
                    return &global_info->psync_pool[(team->psync_idx + i) * PSYNC_CHUNK_SIZE];
#else
                    return &ishmemi_mmap_gpu_info
                                ->psync_pool[(team->psync_idx + i) * PSYNC_CHUNK_SIZE];
#endif
                }
            }

            /* No psync is available, so we must quiesce communication across all psyncs on this
             * team. */
            ishmem_quiet();

            size_t psync = static_cast<size_t>(team->psync_idx * ISHMEMI_SYNC_SIZE);
            // FIXME: assure a correct psync is used here by ishmemi_team_sync(team)...
            // ishmemi_sync(team->start, team->stride, team->size,
            //                     &ishmemi_psync_barrier_pool[psync]);
            ishmemi_team_sync(team);

            for (int i = 0; i < N_PSYNCS_PER_TEAM; i++) {
                team->psync_avail[i] = 1;
            }
            team->psync_avail[0] = 0;

#if __SYCL_DEVICE_ONLY__
            return &global_info->psync_pool[psync];
#else
            return &ishmemi_mmap_gpu_info->psync_pool[psync];
#endif
    }
}

void ishmemi_team_release_psyncs(ishmemi_team_t *team, ishmemi_team_op_t op)
{
    switch (op) {
        case TEAM_OP_SYNC:
            for (size_t i = 0; i < N_PSYNCS_PER_TEAM; i++) {
                team->psync_avail[i] = 1;
            }
            break;
        default:
            break;
    }

    return;
}
