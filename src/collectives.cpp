/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cstddef>
#include "ishmem/util.h"
#include "ishmem/err.h"
#include "memory.h"
#include "runtime.h"
#include "collectives.h"
#include "ishmem/env_utils.h"

size_t *ishmemi_my_collect_size;
size_t *ishmemi_collect_sizes;

struct ishmem_local_info_t {
    int local_rank;
    int next_pe;
};

/* table gleaned from KVS */
struct ishmem_local_info_t *ishmemi_local_info = nullptr;

struct ishmem_topology_t {
    int global_pe[MAX_LOCAL_PES];
};

int ishmemi_n_hosts; /* number of supernodes */
/* ishmemi_topology is a two D array of pe's indexed by  [host][local_rank] */
struct ishmem_topology_t *ishmemi_topology = nullptr;  // indexed by hostid

#define USE_KVS 0

int ishmemi_topology_init()
{
    /* This implements an fcollect on everyone's local_info */
    /* construct a local_info_t and save it in KVS */
    int local_rank = ishmemi_runtime->get_node_rank(ishmemi_my_pe);
    int local_size = ishmemi_runtime->get_node_size();
    struct ishmem_local_info_t *local_info = nullptr;
    size_t topology_size;
    int *pes = nullptr;
    int pes_found;
    int host;

    local_info = (struct ishmem_local_info_t *) ishmemi_runtime->calloc(
        1, sizeof(struct ishmem_local_info_t));
    ISHMEM_CHECK_GOTO_MSG(local_info == nullptr, fn_fail,
                          "Allocation of local info for collectives failed\n");

    local_info->local_rank = local_rank;
    local_info->next_pe = -1; /* next PE on this node */
    /* linear scan to find the pe with a local rank one higher than ours */
    for (int pe = 0; pe < ishmemi_cpu_info->n_pes; pe += 1) {
        int local_idx = ishmemi_runtime->get_node_rank(pe);
        if (local_idx == -1) continue;
        if (local_idx == ((local_rank + 1) % local_size)) {
            local_info->next_pe = pe;
            break;
        }
    }

    ISHMEM_CHECK_GOTO_MSG(local_info->next_pe == -1, fn_fail,
                          "Next PE in local info is not found.\n");
    ISHMEM_DEBUG_MSG("local_info->local_rank %d next_pe %d\n", local_info->local_rank,
                     local_info->next_pe);
    /* allocate space for all the local_info records */
    ishmemi_local_info = (struct ishmem_local_info_t *) ishmemi_runtime->calloc(
        static_cast<size_t>(ishmemi_n_pes), sizeof(struct ishmem_local_info_t));
    ISHMEM_CHECK_GOTO_MSG(ishmemi_local_info == nullptr, fn_fail,
                          "Allocation of ishmemi_local_info for collectives failed\n");
#if USE_KVS
    ret = shmem_runtime_put("local_info", &local_info, sizeof(ishmem_local_info_t));
    ISHMEM_CHECK_GOTO_MSG(ret != 0, fn_fail, "Failed during local_info store to KVS: (%d)", ret);
    /* runtime exchange */
    /* download all the local_info records from KVS */
    for (int pe = 0; pe < ishmemi_cpu_info->n_pes; pe += 1) {
        /* fetch local_idx, fetch next_pe */
        ret = shmem_runtime_get(pe, "hostname", &ishmemi_local_info[pe],
                                sizeof(struct ishmem_local_info_t));
        ISHMEM_CHECK_GOTO_MSG(ret != 0, fn_fail,
                              "Failed during pe %d local_info read from KVS (%d)\n", pe, ret);
    }
#else /* instead of using KVS, use fcollect */
    ishmemi_runtime->fcollect(ishmemi_local_info, local_info, sizeof(ishmem_local_info_t));
#endif
    ISHMEMI_FREE(ishmemi_runtime->free, local_info);
    /* now we have, for every pe, its local_rank and which pe is next in its local_node
     * the PEs with local_rank 0 are the proxies for their hosts, first count them
     * there should be n_pes / local_size
     */
    ishmemi_n_hosts = 0;
    for (int pe = 0; pe < ishmemi_cpu_info->n_pes; pe += 1) {
        int local_rank = ishmemi_local_info[pe].local_rank;
        int next_pe = ishmemi_local_info[pe].next_pe;
        ISHMEM_CHECK_GOTO_MSG((local_rank < 0) || (local_rank >= MAX_LOCAL_PES) || (next_pe < 0) ||
                                  (next_pe >= MAX_LOCAL_PES),
                              fn_fail, "PE %d bad local_info [%d, %d]\n", pe, local_rank, next_pe);
        if (local_rank == 0) ishmemi_n_hosts += 1;
    }
    ISHMEM_CHECK_GOTO_MSG((ishmemi_n_hosts * local_size) != ishmemi_n_pes, fn_fail,
                          "Total calculated PEs and NPES are not equal\n");
    ISHMEM_DEBUG_MSG("Total number of hosts: %d\n", ishmemi_n_hosts);
    /* now build the real topology data structure */
    topology_size = static_cast<size_t>(MAX_LOCAL_PES * ishmemi_n_hosts) * sizeof(int);
    ishmemi_topology = (struct ishmem_topology_t *) ::malloc(topology_size);
    ISHMEM_CHECK_GOTO_MSG(ishmemi_topology == nullptr, fn_fail,
                          "Allocation of ishmemi_topology failed\n");
    ::memset(ishmemi_topology, 0, topology_size);
    host = 0;
    for (int pe = 0; pe < ishmemi_cpu_info->n_pes; pe += 1) {
        if (ishmemi_local_info[pe].local_rank == 0) {
            int tpe = pe;
            int count = 0;
            /* follow the chain of next_pe entering nodes into ishmemi_topology */
            do {
                ISHMEM_CHECK_GOTO_MSG(count > MAX_LOCAL_PES, fn_fail,
                                      "Bad local_info for host %d\n", host);
                ishmemi_topology[host].global_pe[ishmemi_local_info[tpe].local_rank] = tpe;
                tpe = ishmemi_local_info[tpe].next_pe;
                count += 1;
            } while (tpe != pe);
            host += 1;
            ISHMEM_CHECK_GOTO_MSG(host > ishmemi_n_hosts, fn_fail, "Found too many hosts %d\n",
                                  host);
        }
    }
    ISHMEM_CHECK_GOTO_MSG(host != ishmemi_n_hosts, fn_fail, "Found too few hosts %d\n", host);

    /* Now cross check data for each host to show all pes are accounted for */
    pes = (int *) ::malloc(static_cast<size_t>(ishmemi_n_pes) * sizeof(int));
    ISHMEM_CHECK_GOTO_MSG(pes == nullptr, fn_fail, "Allocation of PE array failed\n");
    for (int pe = 0; pe < ishmemi_n_pes; pe += 1)
        pes[pe] = -1;
    pes_found = 0;
    for (int host = 0; host < ishmemi_n_hosts; host += 1) {
        for (int local_rank = 0; local_rank < local_size; local_rank += 1) {
            int pe = ishmemi_topology[host].global_pe[local_rank];
            ISHMEM_CHECK_GOTO_MSG((pe < 0) || (pe >= ishmemi_n_pes), fn_fail,
                                  "Bad pe in topology %d\n", pe);
            ISHMEM_CHECK_GOTO_MSG((pes[pe] != -1), fn_fail, "Duplicate pe in topology %d\n", pe);
            pes[pe] = pe;
            pes_found += 1;
        }
    }
    ISHMEM_CHECK_GOTO_MSG((pes_found != ishmemi_n_pes), fn_fail,
                          "Wrong number of pes in topology %d\n", pes_found);

    ISHMEMI_FREE(::free, pes);
    ISHMEMI_FREE(ishmemi_runtime->free, local_info);
    return (0);
fn_fail:
    ishmemi_collectives_fini();
    ISHMEMI_FREE(::free, pes);
    ISHMEMI_FREE(ishmemi_runtime->free, local_info);
    return (-1);
}

int ishmemi_collectives_init()
{
    if (ishmemi_topology_init()) goto fn_fail;
    return (0);
fn_fail:
    ishmemi_collectives_fini();
    return (-1);
}

int ishmemi_collectives_fini()
{
    ISHMEMI_FREE(::free, ishmemi_topology);
    ISHMEMI_FREE(ishmemi_runtime->free, ishmemi_local_info);
    return (0);
}
