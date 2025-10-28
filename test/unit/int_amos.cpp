/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <common.h>

constexpr int data_size = 14;

int check_error_bit(int val, int bit)
{
    return ((val & (1 << bit)) != 0);
}

int main(int argc, char **argv)
{
    int exit_code = 0;

    ishmem_init();

    uint32_t my_pe = static_cast<uint32_t>(ishmem_my_pe());
    uint32_t npes = static_cast<uint32_t>(ishmem_n_pes());

    sycl::queue q;

    std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Selected vendor: " << q.get_device().get_info<sycl::info::device::vendor>()
              << std::endl;

    uint32_t *source = (uint32_t *) ishmem_malloc(data_size * sizeof(uint32_t));
    CHECK_ALLOC(source);
    uint32_t *target = (uint32_t *) ishmem_malloc(data_size * sizeof(uint32_t));
    CHECK_ALLOC(target);
    int *errors = sycl::malloc_host<int>(1, q);
    CHECK_ALLOC(errors);

    /* Initialize data */
    auto e_init = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>{data_size, data_size}, [=](sycl::nd_item<1> idx) {
            uint32_t i = static_cast<uint32_t>(idx.get_global_id()[0]);
            source[i] = (my_pe << 16) + i;
            target[i] = (my_pe << 16) + 0xface;
        });
    });
    e_init.wait_and_throw();

    ishmem_barrier_all();

    /* Perform atomic operations
     * Each PE is performing the atomic operation on the next PE's memory.
     * 14 atomic operations are performed on 14 memory locations on target, respectively.
     * For atomic operations that require an input value, the source buffer's corresponding
     * location is passed. Fetch atomics capture the old value of the remote target in its
     * source location. */
    auto e_shmem = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            int my_dev_pe = ishmem_my_pe();
            int my_dev_npes = ishmem_n_pes();
            int target_pe = (my_dev_pe + 1) % my_dev_npes;

            /* Fetch */
            target[0] = ishmem_uint_atomic_fetch(&source[0], target_pe);
            /* Set */
            ishmem_uint_atomic_set(&target[1], source[1], target_pe);
            /* Compare swap */
            source[2] = ishmem_uint_atomic_compare_swap(
                &target[2],
                static_cast<unsigned int>(((my_dev_pe + 1) % my_dev_npes) << 16) + 0xface,
                source[2], target_pe);
            /* Swap */
            source[3] = ishmem_uint_atomic_swap(&target[3], source[3], target_pe);
            /* Fetch increment */
            source[4] = ishmem_uint_atomic_fetch_inc(&target[4], target_pe);
            /* Increment */
            ishmem_uint_atomic_inc(&target[5], target_pe);
            /* Fetch add */
            source[6] = ishmem_uint_atomic_fetch_add(&target[6], source[6], target_pe);
            /* Add */
            ishmem_uint_atomic_add(&target[7], source[7], target_pe);
            /* Fetch and */
            source[8] = ishmem_uint_atomic_fetch_and(&target[8], source[8], target_pe);
            /* And */
            ishmem_uint_atomic_and(&target[9], source[9], target_pe);
            /* Fetch or */
            source[10] = ishmem_uint_atomic_fetch_or(&target[10], source[10], target_pe);
            /* Or */
            ishmem_uint_atomic_or(&target[11], source[11], target_pe);
            /* Fetch xor */
            source[12] = ishmem_uint_atomic_fetch_xor(&target[12], source[12], target_pe);
            /* Xor */
            ishmem_uint_atomic_xor(&target[13], source[13], target_pe);
        });
    });
    e_shmem.wait_and_throw();

    ishmem_barrier_all();

    *errors = 0;
    /* Verify data
     * For each operation, the target is checked to have the previous PE's ID << 16 + index
     * For fetching atomics, the source is checked to have the next PE's ID << 16 + 0xface
     * errors is an integer that uses its 14 least significant bits to store the operation
     * correctness (0 for correct, 1 for error) for 14 operations */
    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            uint32_t target_pe = static_cast<uint32_t>(((my_pe + 1) % npes) << 16);
            uint32_t source_pe = static_cast<uint32_t>(((my_pe - 1 + npes) % npes) << 16);
            /* Fetch verify */
            if (target[0] != ((target_pe) + 0)) {
                *errors = *errors + (1 << 0);
            }
            /* Set verify */
            if (target[1] != ((source_pe) + 1)) {
                *errors = *errors + (1 << 1);
            }
            /* Compare-swap verify */
            if ((target[2] != ((source_pe) + 2)) || (source[2] != ((target_pe) + 0xface))) {
                *errors = *errors + (1 << 2);
            }
            /* Swap verify */
            if ((target[3] != ((source_pe) + 3)) || (source[3] != ((target_pe) + 0xface))) {
                *errors = *errors + (1 << 3);
            }
            /* Fetch increment verify */
            if ((target[4] != ((my_pe << 16) + 0xface + 1)) ||
                (source[4] != ((target_pe) + 0xface))) {
                *errors = *errors + (1 << 4);
            }
            /* Increment verify */
            if (target[5] != ((my_pe << 16) + 0xface + 1)) {
                *errors = *errors + (1 << 5);
            }
            /* Fetch add verify */
            if ((target[6] != ((my_pe << 16) + 0xface) + ((source_pe) + 6)) ||
                (source[6] != ((target_pe) + 0xface))) {
                *errors = *errors + (1 << 6);
            }
            /* Add verify */
            if (target[7] != ((my_pe << 16) + 0xface) + ((source_pe) + 7)) {
                *errors = *errors + (1 << 7);
            }
            /* Fetch and verify */
            if ((target[8] != (((my_pe << 16) + 0xface) & ((source_pe) + 8))) ||
                (source[8] != ((target_pe) + 0xface))) {
                *errors = *errors + (1 << 8);
            }
            /* And verify */
            if (target[9] != (((my_pe << 16) + 0xface) & ((source_pe) + 9))) {
                *errors = *errors + (1 << 9);
            }
            /* Fetch or verify */
            if ((target[10] != (((my_pe << 16) + 0xface) | ((source_pe) + 10))) ||
                (source[10] != ((target_pe) + 0xface))) {
                *errors = *errors + (1 << 10);
            }
            /* Or verify */
            if (target[11] != (((my_pe << 16) + 0xface) | ((source_pe) + 11))) {
                *errors = *errors + (1 << 11);
            }
            /* Fetch xor verify */
            if ((target[12] != (((my_pe << 16) + 0xface) ^ ((source_pe) + 12))) ||
                (source[12] != ((target_pe) + 0xface))) {
                *errors = *errors + (1 << 12);
            }
            /* Xor verify */
            if (target[13] != (((my_pe << 16) + 0xface) ^ ((source_pe) + 13))) {
                *errors = *errors + (1 << 13);
            }
        });
    });
    e_verify.wait_and_throw();

    if (*errors != 0) {
        std::cerr << "[PE " << my_pe << "] Validation check(s) failed" << std::endl;

        uint32_t *hosttarget = sycl::malloc_host<uint32_t>(data_size, q);
        CHECK_ALLOC(hosttarget);
        uint32_t *hostsource = sycl::malloc_host<uint32_t>(data_size, q);
        CHECK_ALLOC(hostsource);
        q.memcpy(hosttarget, target, sizeof(uint32_t) * data_size).wait_and_throw();
        q.memcpy(hostsource, source, sizeof(uint32_t) * data_size).wait_and_throw();
        int err_val = *errors;
        uint32_t exp1 = ((my_pe + 1) % npes) << 16;
        uint32_t exp2 = ((my_pe - 1 + npes) % npes) << 16;

        if (check_error_bit(err_val, 0))
            fprintf(stdout, "[PE %d] Fetch failed: target[0] = 0x%08x, Expected = 0x%08x\n", my_pe,
                    hosttarget[0], ((exp1) + 0));
        if (check_error_bit(err_val, 1))
            fprintf(stdout, "[PE %d] Set failed: target[1] = 0x%08x, Expected = 0x%08x\n", my_pe,
                    hosttarget[1], ((exp2) + 1));
        if (check_error_bit(err_val, 2))
            fprintf(stdout,
                    "[PE %d] Compare-swap failed: target[2] = 0x%08x, Expected = 0x%08x; source[2] "
                    "= 0x%08x, Expected = 0x%08x\n",
                    my_pe, hosttarget[2], ((exp2) + 2), hostsource[2], ((exp1) + 0xface));
        if (check_error_bit(err_val, 3))
            fprintf(stdout,
                    "[PE %d] Swap failed: target[3] = 0x%08x, Expected = 0x%08x; source[3] = "
                    "0x%08x, Expected = 0x%08x\n",
                    my_pe, hosttarget[3], ((exp2) + 3), hostsource[3], ((exp1) + 0xface));
        if (check_error_bit(err_val, 4))
            fprintf(stdout,
                    "[PE %d] Fetch-inc failed: target[4] = 0x%08x, Expected = 0x%08x; source[4] = "
                    "0x%08x, Expected = 0x%08x\n",
                    my_pe, hosttarget[4], ((my_pe << 16) + 0xface + 1), hostsource[4],
                    ((exp1) + 0xface));
        if (check_error_bit(err_val, 5))
            fprintf(stdout, "[PE %d] Inc failed: target[5] = 0x%08x, Expected = 0x%08x\n", my_pe,
                    hosttarget[5], ((my_pe << 16) + 0xface + 1));
        if (check_error_bit(err_val, 6))
            fprintf(stdout,
                    "[PE %d] Fetch-add failed: target[6] = 0x%08x, Expected = 0x%08x; source[6] = "
                    "0x%08x, Expected = 0x%08x\n",
                    my_pe, hosttarget[6], ((my_pe << 16) + 0xface) + ((exp2) + 6), hostsource[6],
                    ((exp1) + 0xface));
        if (check_error_bit(err_val, 7))
            fprintf(stdout, "[PE %d] Add failed: target[7] = 0x%08x, Expected = 0x%08x\n", my_pe,
                    hosttarget[7], ((my_pe << 16) + 0xface) + ((exp2) + 7));
        if (check_error_bit(err_val, 8))
            fprintf(stdout,
                    "[PE %d] Fetch-and failed: target[8] = 0x%08x, Expected = 0x%08x; source[8] = "
                    "0x%08x, Expected = 0x%08x\n",
                    my_pe, hosttarget[8], (((my_pe << 16) + 0xface) & ((exp2) + 8)), hostsource[8],
                    ((exp1) + 0xface));
        if (check_error_bit(err_val, 9))
            fprintf(stdout, "[PE %d] And failed: target[9] = 0x%08x, Expected = 0x%08x\n", my_pe,
                    hosttarget[9], (((my_pe << 16) + 0xface) & ((exp2) + 9)));
        if (check_error_bit(err_val, 10))
            fprintf(stdout,
                    "[PE %d] Fetch-or failed: target[10] = 0x%08x, Expected = 0x%08x; source[10] = "
                    "0x%08x, Expected = 0x%08x\n",
                    my_pe, hosttarget[10], (((my_pe << 16) + 0xface) | ((exp2) + 10)),
                    hostsource[10], ((exp1) + 0xface));
        if (check_error_bit(err_val, 11))
            fprintf(stdout, "[PE %d] Or failed: target[11] = 0x%08x, Expected = 0x%08x\n", my_pe,
                    hosttarget[11], (((my_pe << 16) + 0xface) | ((exp2) + 11)));
        if (check_error_bit(err_val, 12))
            fprintf(stdout,
                    "[PE %d] Fetch-xor failed: target[12] = 0x%08x, Expected = 0x%08x; source[12] "
                    "= 0x%08x, Expected = 0x%08x\n",
                    my_pe, hosttarget[12], (((my_pe << 16) + 0xface) ^ ((exp2) + 12)),
                    hostsource[12], ((exp1) + 0xface));
        if (check_error_bit(err_val, 13))
            fprintf(stdout, "[PE %d] Xor failed: target[13] = 0x%08x, Expected = 0x%08x\n", my_pe,
                    hosttarget[13], (((my_pe << 16) + 0xface) ^ ((exp2) + 13)));

        sycl::free(hostsource, q);
        sycl::free(hosttarget, q);
        exit_code = 1;
    } else {
        std::cout << "[PE " << my_pe << "] No errors" << std::endl;
    }

    fflush(stdout);
    sycl::free(errors, q);
    ishmem_free(source);
    ishmem_free(target);

    ishmem_finalize();

    return exit_code;
}
