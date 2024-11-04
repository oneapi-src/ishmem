/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_OPENSHMEM_H
#define ISHMEM_RUNTIME_OPENSHMEM_H

/* Wrappers to interface with OpenSHMEM runtime */
#include "ishmem/config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "runtime.h"
#include "wrapper.h"
#include "collectives.h"
#include "ishmem/err.h"
#include "proxy_impl.h"

/* Operations that need multiple function pointers:
 * - p          (uint8, uint16, uint32, uint64, ulonglong)
 * - g          (uint8, uint16, uint32, uint64, ulonglong)
 * - reductions (uint8, uint16, uint32, uint64, ulonglong,
 *               int8, int16, int32, int64, longlong,
 *               float, double, longdouble)
 * - amos       (uint32, uint64, ulonglong,
 *               int32, int64, longlong,
 *               float, double, longdouble)
 * - test       (uint32, uint64, ulonglong)
 * - wait_until (uint32, uint64, ulonglong)
 */

/* Enabling SHMEMX_TEAM_NODE by default which provides a team that shares a compute node */
#define ISHMEMI_TEAM_NODE ishmemi_openshmem_wrappers::SHMEMX_TEAM_NODE

class ishmemi_runtime_openshmem : public ishmemi_runtime_type {
  public:
    ishmemi_runtime_openshmem(bool, bool = false);

    ~ishmemi_runtime_openshmem(void);

    void heap_create(void *, size_t) override;

    /* Query APIs */
    int get_rank(void) override;
    int get_size(void) override;
    int get_node_rank(int) override;
    int get_node_size(void) override;
    bool is_local(int) override;
    bool is_symmetric_address(const void *) override;

    /* Memory APIs */
    void *malloc(size_t) override;
    void *calloc(size_t, size_t) override;
    void free(void *) override;

    /* Team APIs */
    int team_sync(ishmemi_runtime_team_t) override;
    int team_predefined_set(ishmemi_runtime_team_t *, ishmemi_runtime_team_predefined_t, int, int,
                            int) override;
    int team_split_strided(ishmemi_runtime_team_t, int, int, int,
                           const ishmemi_runtime_team_config_t *, long,
                           ishmemi_runtime_team_t *) override;
    void team_destroy(ishmemi_runtime_team_t) override;

    /* Operation APIs */
    void abort(int, const char[]) override;
    int get_kvs(int, char *, void *, size_t) override;
    int uchar_and_reduce(ishmemi_runtime_team_t, unsigned char *, const unsigned char *,
                         size_t) override;
    int int_max_reduce(ishmemi_runtime_team_t, int *, const int *, size_t) override;
    void bcast(void *, size_t, int) override;
    void node_bcast(void *, size_t, int) override;
    void fcollect(void *, void *, size_t) override;
    void node_fcollect(void *, void *, size_t) override;
    void barrier_all(void) override;
    void node_barrier(void) override;
    void fence(void) override;
    void quiet(void) override;
    void sync(void) override;

    void progress(void) override;

  private:
    int team_sanity_checks(ishmemi_runtime_team_predefined_t, int, int, int, int, int, int);

    void funcptr_init(void);
    void funcptr_fini(void);

    int rank = -1;
    int size = 0;
    bool initialized = false;
    bool oshmpi = false;
};

#endif /* ISHMEM_RUNTIME_OPENSHMEM_H */
