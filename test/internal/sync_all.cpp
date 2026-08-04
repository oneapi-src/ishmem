/* Copyright (C) 2024-2026 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stdio.h>
#include <unistd.h>
#include <shmem.h>

int main(int argc, char **argv)
{
    int exit_code = 0;
    shmem_init();
    setbuf(stdout, NULL);
    printf("pe %d of %d pid %d\n", shmem_my_pe(), shmem_n_pes(), getpid());

    for (size_t od = 0; od < 65536; od += 1) {
        printf("host sync_all %ld\n", od);
        shmem_sync_all();
    }

    shmem_finalize();

    return exit_code;
}
