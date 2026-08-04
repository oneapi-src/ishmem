/* Copyright (C) 2023-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ishmem.h>
#include "proxy_impl.h"
#include "memory.h"

#define NSEC_IN_SEC 1000000000.0
double getduration(const sycl::event &e)
{
    uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    double duration = static_cast<double>(end - start) / NSEC_IN_SEC;
    return (duration);
}

SYCL_EXTERNAL void write_debug(uint64_t val)
{
    ishmemi_info_t *info = global_info;
    info->debug[0] = (long) val;
}

SYCL_EXTERNAL void my_atomic_add(uint64_t *dest, uint64_t val, int pe)
{
    uint8_t local_index = ISHMEMI_LOCAL_PES[pe];

    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        if (local_index != 0 && info->only_intra_node) {
            uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, dest);
            sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            atomic_p += val;
            return;
        }
    }
}

SYCL_EXTERNAL void my_atomic_add_2(uint64_t *dest, uint64_t val, int pe)
{
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        uint8_t local_index = info->local_pes[pe];
        if (local_index != 0 && info->only_intra_node) {
            uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, dest);
            sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            atomic_p += val;
            return;
        }
    }
}

SYCL_EXTERNAL void my_atomic_add_3(uint64_t *dest, uint64_t val, int pe)
{
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        uint8_t local_index = info->local_pes[pe];
        if (local_index != 0 && info->only_intra_node) {
            uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, dest);
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            atomic_p += val;
            return;
        }
    }
}

SYCL_EXTERNAL void my_atomic_add_4(uint64_t *dest, uint64_t val, int pe)
{
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        uint8_t local_index = info->local_pes[pe];
        if (local_index != 0 && info->only_intra_node) {
            uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, dest);
            sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            atomic_p += val;
            return;
        }
    }
}

SYCL_EXTERNAL void my_atomic_add_5(uint64_t *dest, uint64_t val, int pe)
{
    /* Node-local, on-device implementation */
    if constexpr (ishmemi_is_device) {
        ishmemi_info_t *info = global_info;
        uint8_t local_index = info->local_pes[pe];
        if (local_index != 0 && info->only_intra_node) {
            uint64_t *p = ISHMEMI_ADJUST_PTR(uint64_t, local_index, dest);
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                atomic_p(*p);
            atomic_p += val;
            return;
        }
    }
}

int main(int argc, char *argv[])
{
    ishmem_init();
    sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
    sycl::queue q(prop_list);

    uint64_t *remote = (uint64_t *) ishmem_calloc(1, sizeof(uint64_t));
    ishmem_barrier_all();
    size_t iterations = 10000;
    sycl::event e;

    e = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (size_t i = 0; i < iterations; i += 1) {
                ishmem_uint64_atomic_add(remote, 1L, 0);
            }
        });
    });
    e.wait_and_throw();
    double duration = 1000000000.0 * getduration(e) / (double) iterations;
    printf("ishmem %f ns\n", duration);

    e = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (size_t i = 0; i < iterations; i += 1) {
                write_debug(15);
            }
        });
    });
    e.wait_and_throw();
    duration = 1000000000.0 * getduration(e) / (double) iterations;
    printf("write_debug %f ns\n", duration);

    e = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (size_t i = 0; i < iterations; i += 1) {
                my_atomic_add(remote, (uint64_t) 1L, 0);
            }
        });
    });
    e.wait_and_throw();
    duration = 1000000000.0 * getduration(e) / (double) iterations;
    printf("my_atomic_add %f ns\n", duration);

    e = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (size_t i = 0; i < iterations; i += 1) {
                my_atomic_add_2(remote, (uint64_t) 1L, 0);
            }
        });
    });
    e.wait_and_throw();
    duration = 1000000000.0 * getduration(e) / (double) iterations;
    printf("my_atomic_add_2 %f ns\n", duration);

    e = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (size_t i = 0; i < iterations; i += 1) {
                my_atomic_add_3(remote, (uint64_t) 1L, 0);
            }
        });
    });
    e.wait_and_throw();
    duration = 1000000000.0 * getduration(e) / (double) iterations;
    printf("my_atomic_add_3 %f ns\n", duration);

    e = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (size_t i = 0; i < iterations; i += 1) {
                my_atomic_add_4(remote, (uint64_t) 1L, 0);
            }
        });
    });
    e.wait_and_throw();
    duration = 1000000000.0 * getduration(e) / (double) iterations;
    printf("my_atomic_add_4 %f ns\n", duration);

    e = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (size_t i = 0; i < iterations; i += 1) {
                my_atomic_add_5(remote, (uint64_t) 1L, 0);
            }
        });
    });
    e.wait_and_throw();
    duration = 1000000000.0 * getduration(e) / (double) iterations;
    printf("my_atomic_add_5 %f ns\n", duration);

    ishmem_free(remote);

    ishmem_finalize();
    return 0;
}
