/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_RUNTIME_H
#define ISHMEM_RUNTIME_H

#include "ishmemx.h"
#include "proxy_impl.h"

typedef int (*ishmemi_runtime_proxy_func_t)(ishmemi_request_t *, ishmemi_ringcompletion_t *);

typedef enum {
    WORLD,
    SHARED,
    NODE
} ishmemi_runtime_team_predefined_t;

int ishmemi_runtime_init(ishmemx_attr_t *);
int ishmemi_runtime_fini(void);

class ishmemi_runtime_type {
  public:
    /* Base class constructor */
    ishmemi_runtime_type(void) {}

    /* Base class destructor */
    virtual ~ishmemi_runtime_type(void) {}

    /* Pre-initialize Heap */
    virtual void heap_create(void *, size_t) = 0;

    /* Query APIs */
    virtual int get_rank(void) = 0;
    virtual int get_size(void) = 0;
    virtual int get_node_rank(int) = 0;
    virtual int get_node_size(void) = 0;
    virtual bool is_local(int) = 0;
    virtual bool is_symmetric_address(const void *) = 0;

    /* Memory APIs */
    virtual void *malloc(size_t) = 0;
    virtual void *calloc(size_t, size_t) = 0;
    virtual void free(void *) = 0;

    /* Team APIs */
    const char *team_predefined_string(ishmemi_runtime_team_predefined_t);
    virtual int team_sync(ishmemi_runtime_team_t) = 0;
    virtual int team_predefined_set(ishmemi_runtime_team_t *, ishmemi_runtime_team_predefined_t,
                                    int, int, int) = 0;
    virtual int team_split_strided(ishmemi_runtime_team_t, int, int, int,
                                   const ishmemi_runtime_team_config_t *, long,
                                   ishmemi_runtime_team_t *) = 0;
    virtual void team_destroy(ishmemi_runtime_team_t) = 0;

    /* Operation APIs */
    virtual void abort(int, const char[]) = 0;
    virtual int get_kvs(int, char *, void *, size_t) = 0; /* Get KVS value */
    virtual int uchar_and_reduce(ishmemi_runtime_team_t, unsigned char *, const unsigned char *,
                                 size_t) = 0;
    virtual int int_max_reduce(ishmemi_runtime_team_t, int *, const int *, size_t) = 0;
    virtual void bcast(void *, size_t, int) = 0;
    virtual void node_bcast(void *, size_t, int) = 0;
    virtual void fcollect(void *, void *, size_t) = 0;
    virtual void node_fcollect(void *, void *, size_t) = 0;
    virtual void barrier_all(void) = 0;
    virtual void node_barrier(void) = 0;
    virtual void fence(void) = 0;
    virtual void quiet(void) = 0;
    virtual void sync(void) = 0;

    virtual void progress(void) = 0;

    static int unsupported(ishmemi_request_t *, ishmemi_ringcompletion_t *);

    /* Function table */
    ishmemi_runtime_proxy_func_t **proxy_funcs = nullptr;

    /* This sums the total number of base types used in defining the proxy function pointers:
     * - void, uint8_t, uint16_t, uint32_t, uint64_t, unsigned long long, int8_t, int16_t, int32_t,
     *   int64_t, long long, float, double, long double
     */
    static constexpr size_t proxy_func_num_types = 14;
};

/* The instance of the runtime backend */
extern ishmemi_runtime_type *ishmemi_runtime;

#endif /* ISHMEM_RUNTIME_H */
