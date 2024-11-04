/* Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_TEST_RUNTIME_OPENSHMEM_H
#define ISHMEM_TEST_RUNTIME_OPENSHMEM_H

#include <shmem.h>
#include "runtime.h"

namespace ishmemi_test_openshmem_wrappers {
    /* Setup APIs */
    extern void (*init)(void);
    extern void (*finalize)(void);

    /* Memory APIs */
    extern void *(*calloc)(size_t, size_t);
    extern void *(*malloc)(size_t);
    extern void (*free)(void *);

    /* Operation APIs */
    extern void (*sync_all)(void);
    extern void (*broadcastmem)(shmem_team_t, void *, void *, size_t, int);
    extern void (*uint64_sum_reduce)(shmem_team_t, uint64_t *, uint64_t *, size_t);
    extern void (*float_sum_reduce)(shmem_team_t, float *, float *, size_t);
}  // namespace ishmemi_test_openshmem_wrappers

class ishmemi_test_runtime_openshmem : public ishmemi_test_runtime_type {
  public:
    ishmemi_test_runtime_openshmem(void);
    ~ishmemi_test_runtime_openshmem(void);

    /* Helper APIs */
    ishmemx_runtime_type_t get_type(void) override;

    /* Setup APIs */
    void init(void) override;
    void finalize(void) override;

    /* Memory APIs */
    void *calloc(size_t, size_t) override;
    void *malloc(size_t) override;
    void free(void *) override;

    /* Operation APIs */
    void sync(void) override;
    void broadcast(void *, void *, size_t, int) override;
    void uint64_sum_reduce(uint64_t *, uint64_t *, size_t) override;
    void float_sum_reduce(float *, float *, size_t) override;
};

#endif
