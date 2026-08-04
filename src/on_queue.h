/* Copyright (C) 2024-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_ON_QUEUE_H
#define ISHMEM_ON_QUEUE_H

#include "accelerator.h"
#include <map>

struct ishmemi_on_queue_map_entry_t {
    const size_t max_work_group_size;
    sycl::event event;
    ishmemi_on_queue_map_entry_t(sycl::queue q)
        : max_work_group_size(q.get_device().get_info<sycl::info::device::max_work_group_size>())
    {
    }
};

template <bool atomic, typename Fn>
inline void ishmemi_submit_on_queue(Fn func, sycl::handler &cgh)
{
    ishmemi_info_t *info = ishmemi_mmap_gpu_info;

    bool device_task = info->only_intra_node;
    if constexpr (atomic) {
        device_task &= info->device_atomics;
    }

    if (device_task) {
        cgh.single_task(func);
    } else {
        cgh.host_task(func);
    }
}

inline void set_cmd_grp_dependencies(sycl::handler &cgh, bool entry_already_exists, sycl::event &e,
                                     const std::vector<sycl::event> &deps)
{
    if (entry_already_exists) {
        cgh.depends_on(e);
    }
    cgh.depends_on(deps);
}

class ishmemi_on_queue_map : public std::map<sycl::queue *, ishmemi_on_queue_map_entry_t *> {
  public:
    ishmemi_on_queue_map() {};

    ~ishmemi_on_queue_map()
    {
        for (auto iter = begin(); iter != end(); ++iter) {
            delete iter->second;
        }
    }

    inline ishmemi_on_queue_map::iterator get_entry_info(sycl::queue &q, bool &entry_already_exists)
    {
        ishmemi_validate_queue_device(q);

        auto iter = find(&q);
        if (iter != end()) {
            entry_already_exists = true;
        } else {
            auto entry = new ishmemi_on_queue_map_entry_t(q);
            auto [entry_iter, insert_succeeded] =
                insert(std::pair<sycl::queue *, ishmemi_on_queue_map_entry_t *>(&q, entry));
            if (!insert_succeeded) {
                RAISE_ERROR_MSG("Failed to insert entry into on_queue API event map\n");
            }
            iter = entry_iter;
            entry_already_exists = false;
        }
        return iter;
    }

    std::mutex map_mtx;
};

extern ishmemi_on_queue_map ishmemi_on_queue_events_map;

#endif  // ISHMEM_ON_QUEUE_H
