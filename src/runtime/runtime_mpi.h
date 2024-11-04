/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_MPI_H
#define ISHMEM_RUNTIME_MPI_H

/* Wrappers to interface with MPI runtime */
#include "ishmem/config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <map>

#include "runtime.h"
#include "wrapper.h"
#include "uthash.h"
#include "runtime_mpi_types.h"

class ishmemi_runtime_mpi : public ishmemi_runtime_type {
  public:
    ishmemi_runtime_mpi(bool, void *);

    ~ishmemi_runtime_mpi(void);

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

    typedef struct {
        int nelems;
        int stride;
        int block_size;
        int extent;
        MPI_Datatype base;
    } datatype_key_t;

    typedef struct {
        datatype_key_t key;
        MPI_Datatype datatype;
        UT_hash_handle hh;
    } datatype_entry_t;

    typedef struct team_t {
        MPI_Comm comm = MPI_COMM_NULL;
        MPI_Group group = MPI_GROUP_NULL;
        int rank = -1;
        int size = 0;
    } team_t;

  private:
    /* Private functions */
    void funcptr_init(void);
    void funcptr_fini(void);

    /* Variables that are only needed within class methods */
    bool initialized = false;
    bool gpu_non_contig_support = false; /* TODO: add check once supported by Intel® MPI Library */
    static datatype_entry_t *datatype_map;

  public:
    /* Variables that are needed outside of class methods */
    static std::map<ishmemi_runtime_mpi_types::team_t, team_t> teams;
    static constexpr ishmemi_runtime_mpi_types::team_t team_undefined = -1;
    static ishmemi_runtime_mpi_types::team_t world_team;
    static ishmemi_runtime_mpi_types::team_t node_team;
    static ishmemi_runtime_mpi_types::team_t shared_team;
    static ishmemi_runtime_mpi_types::team_t team_idx;

    static MPI_Win global_win;
    static void *global_win_base_addr;
    static size_t global_win_size;

  public:
    /* Functions that are needed outside of class methods that aren't overrides of the base class */
    static void get_strided_dt(size_t, ptrdiff_t, size_t, int, MPI_Datatype, MPI_Datatype *);
};
#endif /* ISHMEM_RUNTIME_MPI_H */
