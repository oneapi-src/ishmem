/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portions derived from Sandia OpenSHMEM (https://github.com/Sandia-OpenSHMEM/SOS)
 * For license and copyright information, see the LICENSE and third-party-programs.txt
 * files in the top level directory of this distribution.
 */

#ifndef ISHMEM_TEAMS_H
#define ISHMEM_TEAMS_H

#include <cstddef>
#include "ishmem.h"
#include "ishmem/util.h"
#include "runtime/runtime_types.h"

#define N_PSYNCS_PER_TEAM 2

struct ishmemi_team_t {
    int start, stride, size;
    int my_pe;                 // PE ID in the *team* space
    int n_local_pes;           // number of local PEs in the team
    void *buffer;              // reduction buffer for in-place intra-node
    void *source;              // host bounce buffer for internode
    void *dest;                // host bounce buffer for internode
    size_t *collect_mynelems;  // device symmetric scratch buffer for my PE's nelems
    size_t *collect_nelems;    // device symmetric scratch buffer for all PE's nelems
    size_t *collect_my_size;   // host symmetric scratch buffer for my PE's data size
    size_t *collect_sizes;     // host symmetric scratch buffer for all PE's data sizes
    ishmem_team_t psync_idx;   // ishmem_team_t is just an integer index into the team pool
    int psync_avail[N_PSYNCS_PER_TEAM];  // indicates whetiher the psync buffer is available
    ishmem_team_config_t config;
    long config_mask;
    union {
        ishmemi_runtime_team_t shmem_team;
    };
};
typedef struct ishmemi_team_t ishmemi_team_t;

extern ishmemi_team_t *ishmemi_team_world;
extern ishmemi_team_t *ishmemi_team_shared;
extern ishmemi_team_t *ishmemi_team_node;

typedef enum ishmemi_team_op_t {
    TEAM_OP_SYNC = 0,
    TEAM_OP_BCAST,
    TEAM_OP_REDUCE,
    TEAM_OP_COLLECT,
    TEAM_OP_ALLTOALL
} ishmemi_team_op_t;

/* Team Management Routines */
int ishmemi_team_init(void);
int ishmemi_team_fini(void);
int ishmemi_team_my_pe(ishmemi_team_t *team);
int ishmemi_team_n_pes(ishmemi_team_t *team);
int ishmemi_team_get_config(ishmemi_team_t *team, long config_mask, ishmem_team_config_t *config);
int ishmemi_team_split_strided(ishmemi_team_t *parent_team, int PE_start, int PE_stride,
                               int PE_size, const ishmem_team_config_t *config, long config_mask,
                               ishmem_team_t *new_team);
int ishmemi_team_split_2d(ishmemi_team_t *parent_team, int xrange,
                          const ishmem_team_config_t *xaxis_config, long xaxis_mask,
                          ishmem_team_t *xaxis_team, const ishmem_team_config_t *yaxis_config,
                          long yaxis_mask, ishmem_team_t *yaxis_team);
int ishmemi_team_destroy(ishmemi_team_t *team);

ISHMEM_DEVICE_ATTRIBUTES int ishmemi_team_translate_pe(ishmemi_team_t *src_team, int src_pe,
                                                       ishmemi_team_t *dest_team);
ISHMEM_DEVICE_ATTRIBUTES long *ishmemi_team_choose_psync(ishmemi_team_t *team,
                                                         ishmemi_team_op_t op);
ISHMEM_DEVICE_ATTRIBUTES void ishmemi_team_release_psyncs(ishmemi_team_t *team,
                                                          ishmemi_team_op_t op);

static inline int ishmemi_team_pe(ishmemi_team_t *team, int pe)
{
    return team->start + team->stride * pe;
}

#endif /* ISHMEM_TEAMS_H */
