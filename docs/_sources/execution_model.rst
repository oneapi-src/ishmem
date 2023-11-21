.. _execution_model:

===============
Execution Model
===============

An Intel® SHMEM program consists of a set of processes called PEs.
While not required by Intel® SHMEM, in typical usage, PEs are
executed using a single program, multiple data (SPMD) model.
SPMD requires each PE to use the same executable; however, PEs are able to
follow divergent control paths.
PEs are often implemented using OS processes and PEs are permitted to create
additional threads on the host, when supported by the host :ref:`runtime
library<ishmemx_runtime_type_t>`.

.. important:: Intel® SHMEM requires a one-to-one mapping of PEs
   to SYCL devices. This implies that Intel® SHMEM executions must
   launch with a number of processes on each compute node that is no more
   than the number of available SYCL devices on each one of those nodes.
   Intel® SHMEM provides a launcher script called ``ishmrun``
   that assigns the environment variable **ZE_AFFINITY_MASK** so that each
   PE is assigned a single SYCL device. Usage of this script is described in
   Section :ref:`Compiling and Running
   Programs<compiling_and_running_programs>`.

.. note:: Intel® Data Center GPU Max Series devices utilize a multi-tile
   architecture (as of Intel®  SHMEM |version| with 1 or 2 tiles).  By default,
   the Intel® Intel® SHMEM runtime considers each individual `tile` to make up
   a single SYCL device.  However, setting the environment variable
   **enableImplicitScaling=1** enables implicit scaling mode, where each
   multi-tile `GPU` is considered to be a single SYCL device. When implicit
   scaling is enabled, the GPU driver automatically distributes SYCL device
   kernel thread-groups across all tiles.

   In the default case, it is usually reasonable to execute up to 1 PE per
   tile. When implicit scaling is enabled, it is usually reasonable to
   execute up to 1 PE per multi-tile GPU device.


.. FIXME: additional threads, when supported by the Intel® SHMEM library.

PE execution is loosely coupled, relying on ``ishmem`` routines to
communicate and synchronize among executing PEs.
The Intel® SHMEM phase in a program begins with a call to the
initialization routine ``ishmem_init``, which must be performed before using
any of the other ``ishmem`` routines.
An Intel® SHMEM program concludes its use of the library when all
PEs call ``ishmem_finalize``.
During a call to ``ishmem_finalize``, the Intel® SHMEM library
must complete all pending communication and release all the resources
associated to the library using an implicit collective synchronization across
PEs.
Calling any ``ishmem`` routine before initialization or after
``ishmem_finalize`` leads to undefined behavior.
After finalization, a subsequent initialization call also leads to undefined
behavior.

.. important:: Because SYCL kernel execution is non-blocking on the host, all
   kernels performing ``ishmem`` calls must first `complete` (by calling
   ``wait`` or ``wait_and_throw`` on the SYCL queue) before calling
   ``ishmem_finalize``.

.. FIXME: ishmem_init OR ishmem_init_threads / ishmem_finalize OR ishmem_global_exit.

The PEs of the Intel® SHMEM program are identified by unique
integers.
The identifiers are integers assigned in a monotonically increasing manner from
zero to one less than the total number of PEs.
PE identifiers are used for Intel® SHMEM calls (e.g., to specify
`Put` or `Get` routines on symmetric data objects, collective synchronization
calls) or to dictate a control flow for PEs using constructs of C/C++.
The identifiers are fixed for the duration of the Intel® SHMEM phase
of a program.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Invoking Intel® SHMEM Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pointer arguments to ``ishmem`` routines that point to non-`const` data
must not overlap in memory with other arguments to the same routine, with the
exception of in-place reductions as described in the :ref:`Reductions
Section<reductions>`.
Otherwise, the behavior is undefined.
Two arguments overlap in memory if any of their data elements are contained
in the same physical memory locations.
In particular, pointers to identical symmetric objects on different PEs do not
overlap, but different pointers on the same PE overlap if they point to the
same memory.
For example, consider an address a returned by the ``ishmem_ptr`` operation
for symmetric object `A` on PE `i`.
Providing the local address `a` and the symmetric address of object `A` to an
``ishmem`` routine targeting PE `i` results in undefined behavior.

Buffers provided to ``ishmem`` routines are `in-use` until the corresponding
operation has completed at the calling PE.
Updates to a buffer that is in-use, including updates performed through
locally and remotely issued ``ishmem`` operations, result in undefined
behavior.
Similarly, reads from a buffer that is in-use are allowed only when the
buffer was provided as a `const`-qualified argument to the ``ishmem``
routine for which it is in-use.
Otherwise, the behavior is undefined.
Exceptions are made for buffers that are in-use by AMOs, as described in
:ref:`Atomicity Guarantees<amo_guarantees>`.
For information regarding the completion of Intel® SHMEM
operations, see :ref:`Memory Ordering<memory_ordering>`.

.. ``ishmem`` routines with multiple symmetric object arguments do not require
.. these symmetric objects to be located within the same symmetric memory
.. segment.
.. For example, objects located in the symmetric data segment and objects
.. located in the symmetric heap can be provided as arguments to the same OpenSHMEM operation.

