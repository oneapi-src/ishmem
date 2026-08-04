#include <stdio.h>
#include <shmem.h>
#include <shmemx.h>
#include <stdlib.h>

int main()
{
    int ret = 0;

    sycl::queue Q;
    std::cout << "Device : " << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

    size_t external_heap_length = 64 * 1024;
    size_t *external_heap = sycl::malloc_device<size_t>(external_heap_length, Q);

    int *src = (int *) malloc(1 * sizeof(int));
    int *errors = (int *) sycl::malloc_host<int>(1, Q);

    *errors = 0;

    shmemx_heap_create(external_heap, external_heap_length, SHMEMX_EXTERNAL_HEAP_ZE, 0);
    shmem_init();

    int me = shmem_my_pe();
    int npes = shmem_n_pes();

    auto e1 = Q.submit([&](sycl::handler &h) { h.single_task([=]() { external_heap[0] = 55; }); });
    e1.wait_and_throw();

    shmem_barrier_all();
    src[0] = me + 5;

    shmem_int_put((int *) external_heap, src, 1, (me + 1) % npes);

    shmem_barrier_all();

    auto e2 =
        Q.submit([&](sycl::handler &h) { h.single_task([=]() { *errors = external_heap[0]; }); });
    e2.wait_and_throw();

    if (*errors != (me - 1 + npes) % npes + 5) {
        std::cout << "ERROR: [PE " << me << "] Data in device heap " << *errors
                  << ", expected = " << (me - 1 + npes) % npes + 5 << std::endl;
        ret = 1;
    } else {
        std::cout << "[PE " << me << "] GDR put test successful" << std::endl;
    }

    shmem_barrier_all();
    int local = -1;
    shmem_int_get(&local, (int *) external_heap, 1, (me - 1 + npes) % npes);
    shmem_barrier_all();

    if (local != (me - 2 + npes) % npes + 5) {
        std::cout << "ERROR: [PE " << me << "] Data received " << local
                  << ", expected = " << (me - 2 + npes) % npes + 5 << std::endl;
        ret = 1;
    } else {
        std::cout << "[PE " << me << "] GDR get test successful" << std::endl;
    }

    shmem_barrier_all();
    local = me + 101;
    shmem_int_atomic_set((int *) external_heap, local, (me + 1) % npes);
    shmem_barrier_all();

    auto e3 =
        Q.submit([&](sycl::handler &h) { h.single_task([=]() { *errors = external_heap[0]; }); });
    e3.wait_and_throw();

    if (*errors != (me - 1 + npes) % npes + 101) {
        std::cout << "ERROR: [PE " << me << "] Data in device heap " << *errors
                  << ", expected = " << (me - 1 + npes) % npes + 101 << std::endl;
    } else {
        std::cout << "[PE " << me << "] GDR atomic set test successful" << std::endl;
    }

    sycl::free(errors, Q);
    free(src);
    sycl::free(external_heap, Q);
    shmem_finalize();

    return ret;
}
