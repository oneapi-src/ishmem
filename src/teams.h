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
#include "collectives.h"

#define N_PSYNCS_PER_TEAM 2

/* every team has an ishmemi_team_device_t which is device resident, and an ishmemi_team_host_t,
 * which is host resident.  They contain fields which are duplicated between host and device.  In
 * addition, the host structure has bounce buffers "source and dest" which are used for host
 * reductions of device data. The host structure is in the host symmetric heap, and the device
 * structure is in the device symmetric heap
 *
 * The fields start, stride, size, my_pe, n_local_pes, only_intra, last_pe, config_mask, and config
 * are read only. They are set at team init time and thereafter remain constant
 *
 * Psync: Each team has two longs for psync, which are used for ishmem_team_sync.  Use alternates
 * between them because back to back calls to sync can be partly overlapping across the team.  Sync
 * really only means that no pe can return from sync until every pe in the team has entered.
 * Therefore it is possile for a pe to enter the following sync before some other pe has left the
 * first one.  You need two copies of psync.
 *
 * With psync either all pes are running on-device or all pes are running on-host.  Calls to sync
 * that are split between device code and host code do not work, because the different pes would be
 * using different psync words. The two copies of psync_idx are synchronized across the team, but
 * not synchronized between host and device.
 *
 * The fields collect_mynelems and collect_nelems are used to fcollect the various nelems values
 * with which different pes call collect.  The array is allocated with MAX_LOCAL_PES because this
 * path is only used for intranode teams. Other cases of collect call the runtime version of
 * collect.
 *
 * The "buffer" is used to implement in-place reductions.  First, the input buffer is copied to the
 * buffer, then an out of place reduction is used.
 */

#define SET_HEAP_FIELD(to, value) ishmem_copy(&to, &value, sizeof(to))

/* Create two different team data structures, one for host memory and one for device memory
 * They contain duplicate fields as needed to avoid remote references.
 * These will be allocated in host symmetric memory and in device symmetric memory
 * so that the included psync and collect fields can be accessed from other PEs
 */
struct ishmemi_team_device_t {
    int start, stride, size;
    int my_pe;        // PE ID in the *team* space
    int n_local_pes;  // number of local PEs in the team
    bool only_intra;  // all team members are local
    int last_pe;      // start + (stride * (size-1))
    int psync_idx;    // ishmem_team_t is just an integer index into the team pool
    long psync[N_PSYNCS_PER_TEAM];
    size_t config_mask;
    ishmem_team_config_t config;
    size_t collect_mynelems;               // device symmetric scratch buffer for my PE's nelems
    size_t collect_nelems[MAX_LOCAL_PES];  // device symmetric scratch buffer for all PE's nelems
    uint8_t buffer[ISHMEM_REDUCE_BUFFER_SIZE];  // reduction buffer for in-place intra-node
};

/* The host structure has all the fields of the device structure, plus a few more */
struct ishmemi_team_host_t : ishmemi_team_device_t {
    ishmemi_runtime_team_t runtime_team;
    uint8_t source[ISHMEM_REDUCE_BUFFER_SIZE];  // host bounce buffer for internode
    uint8_t dest[ISHMEM_REDUCE_BUFFER_SIZE];    // host bounce buffer for internode
};

typedef struct ishmemi_team_device_t ishmemi_team_device_t;
typedef struct ishmemi_team_host_t ishmemi_team_host_t;

/* Team Management Routines */
int ishmemi_team_init(void);
int ishmemi_team_fini(void);
void ishmemi_team_destroy(ishmem_team_t team);

int ishmemi_team_split_strided(ishmem_team_t parent_team, int PE_start, int PE_stride, int PE_size,
                               const ishmem_team_config_t *config, long config_mask,
                               ishmem_team_t *new_team);
int ishmemi_team_split_2d(ishmem_team_t parent_team, int xrange,
                          const ishmem_team_config_t *xaxis_config, long xaxis_mask,
                          ishmem_team_t *xaxis_team, const ishmem_team_config_t *yaxis_config,
                          long yaxis_mask, ishmem_team_t *yaxis_team);

#define ishmemi_team_pe(team_ptr, pe) (team_ptr->start + (team_ptr->stride * pe))

static inline int ishmemi_pe_in_active_set(int global_pe, int PE_start, int PE_stride, int PE_size)
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

#endif /* ISHMEM_TEAMS_H */
