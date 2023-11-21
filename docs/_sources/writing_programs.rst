.. _writing_programs:

========================================
Writing Intel® SHMEM Programs
========================================

Intel® SHMEM Programs require including the following header files::

#include <CL/sycl.hpp>
#include <ishmem.h>

Here is how to :ref:`initialize<library_setup_exit_query_routines>` the
``ishmem`` library with an OpenSHMEM runtime::

    ishmem_init();

Now we can query for the PE identifier and total number of PEs::

    int my_pe = ishmem_my_pe();
    int npes = ishmem_n_pes();

    std::cout << "Hello from PE " << my_pe << std::endl;

To perform ``ishmem`` operations, we must first allocate some symmetric
objects::

    int *src = (int *) ishmem_malloc(array_size * sizeof(int));
    int *dst = (int *) ishmem_calloc(array_size, sizeof(int));

Now let's initialize these source and destination buffers from within a
parallel SYCL kernel::

    auto e_init = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>{array_size, array_size}, [=](sycl::nd_item<1> idx) {
            int i = idx.get_global_id()[0];
            src[i] = (my_pe << 16) + i;
            dst[i] = (my_pe << 16) + 0xface;
        });
    });
    e_init.wait_and_throw();

Now we must perform a barrier operation to assure that all the source data is initialized before doing any communication::

    ishmem_barrier_all();

Let's perform a simple ring-style communication pattern; that is, have each PE send its source data to the subsequent PE (the PE with the largest identifier value will send to PE 0)::

    /* Perform put operation */
    auto e1 = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            int my_dev_pe = ishmem_my_pe();
            int my_dev_npes = ishmem_n_pes();

            ishmem_int_put(dst, src, array_size, (my_dev_pe + 1) % my_dev_npes);
        });
    });
    e1.wait_and_throw();

Before verifying the correct results, we need to perform another barrier operation, to assure all the communication is complete::

    ishmem_barrier_all();

    int *errors = (int *) sycl::malloc_host<int>(1, q);
    *errors = 0;

    /* Verify data */
    auto e_verify = q.submit([&](sycl::handler &h) {
        h.single_task([=]() {
            for (int i = 0; i < array_size; ++i) {
                if (dst[i] != (((my_pe + 1) % npes) << 16) + i) {
                    *errors = *errors + 1;
                }
            }
        });
    });
    e_verify.wait_and_throw();

    if (*errors > 0) {
        std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
    }

Finally, we can free all allocated memory and finalize the library.
For symmetric ``ishmem`` objects, we must call ``ishmem_free``::

    ishmem_free(source);
    ishmem_free(target);
    sycl::free(errors, q);

    ishmem_finalize();

For an overview of more APIs and how they are used in applications, the |ishmem_examples| provide an excellent resource.

.. |ishmem_examples| raw:: html

   <a href="https://github.com/oneapi-src/ishmem/tree/main/examples/SHMEM" target="_blank">Intel® SHMEM examples</a>

