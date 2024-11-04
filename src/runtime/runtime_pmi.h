/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_PMI_H
#define ISHMEM_RUNTIME_PMI_H

/* Wrappers to interface with MPI runtime */
#include "ishmem/config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "runtime.h"
#include "wrapper.h"
#include "uthash.h"

class ishmemi_runtime_pmi : public ishmemi_runtime_type {
  public:
    ishmemi_runtime_pmi(bool);

    ~ishmemi_runtime_pmi(void);

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
    void funcptr_init(void);
    void funcptr_fini(void);

    int rank = -1;
    int size = 0;
    int node_size = 0;
    char *kvs_name = nullptr;
    char *kvs_key = nullptr;
    char *kvs_value = nullptr;
    int max_name_len = 0;
    int max_key_len = 0;
    int max_val_len = 0;
    int *location_array = nullptr;
    bool initialized = false;
    static constexpr int SINGLETON_KEY_LEN = 128;
    static constexpr int SINGLETON_VAL_LEN = 1024;

    typedef struct {
        char key[SINGLETON_KEY_LEN];
        char val[SINGLETON_VAL_LEN];
        UT_hash_handle hh;
    } singleton_kvs_t;

    singleton_kvs_t *singleton_kvs = nullptr;
};

#endif /* ISHMEM_RUNTIME_MPI_H */
