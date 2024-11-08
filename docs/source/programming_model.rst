.. _programming_model:

==========================
Programming Model Overview
==========================

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Intel® SHMEM APIs Overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Intel® SHMEM implements PGAS by defining remotely accessible data objects
as mechanisms to share information among Intel® SHMEM processes, or
*processing elements* (PEs), and private data objects that are accessible by
only the PE itself. The API allows communication and synchronization operations
on both private (local to the PE initiating the operation) and remotely
accessible data objects. The key feature of Intel® SHMEM is that data
transfer operations are *one-sided* in nature. This means that a local PE
executing a data transfer routine does not require the participation of the
remote PE to complete the routine. This allows for overlap between
communication and computation to hide data transfer latencies, which makes
Intel® SHMEM ideal for unstructured, small-to-medium-sized data
communication patterns. The Intel® SHMEM library has the potential to
provide a low-latency, high-bandwidth communication API for use in highly
parallelized scalable programs.

Intel® SHMEM's interfaces can be used to implement SPMD style programs.  It
provides interfaces to start the Intel® SHMEM PEs in parallel and
communication and synchronization interfaces to access remotely accessible data
objects across PEs. These interfaces can be leveraged to divide a problem into
multiple sub-problems that can be solved independently or with coordination
using the communication and synchronization interfaces.  The Intel® SHMEM
specification defines library calls, constants, variables, and language
bindings for SYCL and DPC++/C++.
Unlike Unified Parallel C, Fortran 2008, Titanium, X10, and Chapel, which are
all PGAS languages, Intel® SHMEM relies on the user to use the library
calls to implement the correct semantics of its programming model.

.. important:: Intel® SHMEM does not yet support all of the
   following routines. Please refer to the :ref:`Supported
   Features<supported_features>` section to see which interfaces are supported
   in the |release| version of Intel® SHMEM.

An overview of the Intel® SHMEM routines is described below:

#. **Library Setup and Query**

        a. *Initialization and Finalization*: The Intel® SHMEM library
        environment is initialized and finalized.  Users may optionally set
        runtime attributes during initialization.

        b. *Query*: The local PE may get the number of PEs running the same
        program and its unique integer identifier.

        c. *Accessibility*: The local PE can find out if a remote PE is
        executing the same binary, or if a particular symmetric data object can
        be accessed by a remote PE, or may obtain a pointer to a symmetric data
        object on the specified remote PE on shared memory systems.

#. **Symmetric Data Object Management**

        a. *Allocation*: OpenSHMEM routines that require all PEs to call them
        at the same time are called *collective operations*.  Allocation and
        deallocation of symmetric data objects are collective operations.

        b. *Deallocation*: All executing PEs must participate in the
        deallocation of the same symmetric data object with identical
        arguments.

        c.  *Reallocation*: All executing PEs must participate in the
        reallocation of the same symmetric data object with identical
        arguments.

#. **Communication Management**

        a.  *Contexts*: Contexts are containers for communication operations.
        Each context provides an environment where the operations performed on
        that context are ordered and completed independently of other
        operations performed by the application.

#. **Team Management**

        a.  *Teams*: Teams are PE subsets created by grouping a set of PEs.
        Teams are involved in both collective and point-to-point communication
        operations. Collective communication operations are performed on all
        PEs in a valid team and point-to-point communication operations are
        performed between a local and remote PE with team-based PE numbering
        through team-based contexts.

#. **Remote Memory Access (RMA)**

        a.  *PUT*: The local PE specifies the **source** data object, private
        or symmetric, that is copied to the symmetric data object on the remote
        PE.
        
        b.  *GET*: The local PE specifies the symmetric data object on the
        remote PE that is copied to a data object, private or symmetric, on the
        local PE.

#. **Atomic Memory Operations (AMO)**

        a.  *Swap*: The PE initiating the swap gets the old value of a
        symmetric data object from a remote PE and copies a new value to
        that symmetric data object on the remote PE.

        b.  *Increment*: The PE initiating the increment adds 1 to the
        symmetric data object on the remote PE.

        c.  *Add*: The PE initiating the add specifies the value to be
        added to the symmetric data object on the remote PE.

        d.  *Bitwise Operations*: The PE initiating the bitwise operation
        specifies the operand value to the bitwise operation to be performed on
        the symmetric data object on the remote PE.

        e.  *Compare and Swap*: The PE initiating the swap gets the old
        value of the symmetric data object based on a value to be compared and
        copies a new value to the symmetric data object on the remote PE.

        f.  *Fetch and Increment*: The PE initiating the increment adds 1
        to the symmetric data object on the remote PE and returns with the
        old value.

        g.  *Fetch and Add*: The PE initiating the add specifies the value to
        be added to the symmetric data object on the remote PE and returns with
        the old value.

        h.  *Fetch and Bitwise Operations*: The PE initiating the bitwise
        operation specifies the operand value to the bitwise operation to be
        performed on the symmetric data object on the remote PE and
        returns the old value.

#. **Signaling Operations**

        a.  *Signaling Put*: The **source** data is copied to the symmetric
        object on the remote PE and a flag on the remote PE is subsequently
        updated to signal completion.

#. **Synchronization and Ordering**

        a.  *Fence*: The PE calling fence ensures ordering of PUT, AMO,
        and memory store operations to symmetric data objects.

        b.  *Quiet*: The PE calling quiet ensures remote completion of remote
        access operations and stores to symmetric data objects.

        c.  *Barrier*: All PEs collectively synchronize and ensure completion
        of all remote and local updates prior to any PE returning from the
        call.

        d.  *Wait and Test*: A PE calling a point-to-point synchronization
        routine ensures the value of a local symmetric object meets a specified
        condition.  Wait operations block until the specified condition is met,
        whereas test operations return immediately and indicate whether or not
        the specified condition is met.

#. **Collective Communication**

        a.  *Broadcast*: The *root* PE specifies a symmetric data object to
        be copied to a symmetric data object on one or more remote PEs
        (not including itself).

        b.  *Collection*: All PEs participating in the routine get the
        result of concatenated symmetric objects contributed by each of the
        PEs in another symmetric data object.

        c.  *Reduction*: All PEs participating in the routine get the
        result of an associative binary routine over elements of the specified
        symmetric data object on another symmetric data object.

        d.  *All-to-All*: All PEs participating in the routine exchange a
        fixed amount of contiguous or strided data with all other PEs.

.. #. **Mutual Exclusion**
        a.  *Set Lock*: The PE acquires exclusive access to the region bounded
        by the symmetric *lock* variable.

        b.  *Test Lock*: The PE tests the symmetric *lock* variable for
        availability.

        c.  *Clear Lock*: The PE which has previously acquired the *lock*
        releases it.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``on_queue`` APIs Overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the standard RMA and collective routines, Intel®
SHMEM also provides corresponding ``on_queue`` API extensions
such as ``ishmemx_put_on_queue`` and
``ishmemx_barrier_all_on_queue``.
All ``on_queue`` APIs take a SYCL queue as an argument and, optionally, a list
of SYCL events.
For any ``on_queue`` API, it is the implementation's choice to either execute
the operation on the provided queue as a device kernel or to execute the
operation as a host kernel.
Additionally, the implementation will ensure that the operation is executed
only after any SYCL events in the provided event list are complete and after
kernels submitted to the provided queue via previously invoked ``on_queue``
operations are complete.
The ``on_queue`` APIs are only callable from the host.
However, ``on_queue`` APIs support global or static variables as valid
arguments only to read-only, scalar parameters that are copied by-value to the
device.
Additionally, such global or static variables must have been declared with the
``const`` modifier.

.. _on_queue_api_completion_semantics:

**on_queue API Completion Semantics**
To guarantee completion of an operation launched via an ``on_queue`` API,
the user must do one of the following:

  - Call ``ishmem_quiet`` on the host after calling ``wait`` or
    ``wait_and_throw`` on the SYCL event returned by the ``on_queue`` API.
  - Call ``ishmem_quiet`` or ``ishmemx_quiet_work_group`` on the device after
    calling ``wait`` or ``wait_and_throw`` on the SYCL event returned by the
    ``on_queue`` API.
    Then, ``wait`` or ``wait_and_throw`` must be called on the SYCL event
    returned by the kernel that initiated the ``ishmem_quiet`` or
    ``ishmemx_quiet_work_group``.
  - Call ``ishmemx_quiet_on_queue`` and then call ``wait`` or
    ``wait_and_throw`` on the SYCL event returned by the
    ``ishmemx_quiet_on_queue`` API.
    If the SYCL queue passed to the ``ishmemx_quiet_on_queue`` call is not the
    same as the one passed to the ``on_queue`` API that the user wishes to
    ensure completion of, then the event returned by the original ``on_queue``
    API must be included in the **deps** parameter passed to the
    ``ishmemx_quiet_on_queue`` call.

**NOTE:** For every occasion in the table above where the user is instructed to
call ``wait`` or ``wait_and_throw`` on a SYCL event, the user may also call
``wait`` or ``wait_and_throw`` on the SYCL queue that generated the event.

The following is a complete list of the ``on_queue`` APIs provided by
Intel® SHMEM.
TYPENAME corresponds to the types specified by Table
:ref:`Standard RMA Types<stdrmatypes>`, OP corresponds to the relevant
operations specified by Table :ref:`Reduction Types, Names, and Supporting
Operations<reducetypes>`, and SIZE is one of 8, 16, 32, 64, or 128:

  - ``ishmemx_put_on_queue``
  - ``ishmemx_TYPENAME_put_on_queue``
  - ``ishmemx_putmem_on_queue``
  - ``ishmemx_putSIZE_on_queue``
  - ``ishmemx_iput_on_queue``
  - ``ishmemx_TYPENAME_iput_on_queue``
  - ``ishmemx_iputSIZE_on_queue``
  - ``ishmemx_ibput_on_queue``
  - ``ishmemx_TYPENAME_ibput_on_queue``
  - ``ishmemx_ibputSIZE_on_queue``
  - ``ishmemx_get_on_queue``
  - ``ishmemx_TYPENAME_get_on_queue``
  - ``ishmemx_getmem_on_queue``
  - ``ishmemx_getSIZE_on_queue``
  - ``ishmemx_iget_on_queue``
  - ``ishmemx_TYPENAME_iget_on_queue``
  - ``ishmemx_igetSIZE_on_queue``
  - ``ishmemx_ibget_on_queue``
  - ``ishmemx_TYPENAME_ibget_on_queue``
  - ``ishmemx_ibgetSIZE_on_queue``
  - ``ishmemx_put_nbi_on_queue``
  - ``ishmemx_TYPENAME_put_nbi_on_queue``
  - ``ishmemx_putmem_nbi_on_queue``
  - ``ishmemx_putSIZE_nbi_on_queue``
  - ``ishmemx_get_nbi_on_queue``
  - ``ishmemx_TYPENAME_get_nbi_on_queue``
  - ``ishmemx_getmem_nbi_on_queue``
  - ``ishmemx_getSIZE_nbi_on_queue``
  - ``ishmemx_put_signal_on_queue``
  - ``ishmemx_TYPENAME_put_signal_on_queue``
  - ``ishmemx_putmem_signal_on_queue``
  - ``ishmemx_putSIZE_signal_on_queue``
  - ``ishmemx_put_signal_nbi_on_queue``
  - ``ishmemx_TYPENAME_put_signal_nbi_on_queue``
  - ``ishmemx_putmem_signal_nbi_on_queue``
  - ``ishmemx_putSIZE_signal_nbi_on_queue``
  - ``ishmemx_barrier_all_on_queue``
  - ``ishmemx_sync_all_on_queue``
  - ``ishmemx_team_sync_on_queue``
  - ``ishmemx_quiet_on_queue``
  - ``ishmemx_alltoall_on_queue``
  - ``ishmemx_TYPENAME_alltoall_on_queue``
  - ``ishmemx_alltoallmem_on_queue``
  - ``ishmemx_broadcast_on_queue``
  - ``ishmemx_TYPENAME_broadcast_on_queue``
  - ``ishmemx_broadcastmem_on_queue``
  - ``ishmemx_fcollect_on_queue``
  - ``ishmemx_TYPENAME_fcollect_on_queue``
  - ``ishmemx_fcollectmem_on_queue``
  - ``ishmemx_collect_on_queue``
  - ``ishmemx_TYPENAME_collect_on_queue``
  - ``ishmemx_collectmem_on_queue``
  - ``ishmemx_OP_reduce_work_group``
  - ``ishmemx_TYPENAME_OP_reduce_on_queue``
  - ``ishmemx_wait_until_on_queue``
  - ``ishmemx_TYPENAME_wait_until_on_queue``
  - ``ishmemx_wait_until_all_on_queue``
  - ``ishmemx_TYPENAME_wait_until_all_on_queue``
  - ``ishmemx_wait_until_any_on_queue``
  - ``ishmemx_TYPENAME_wait_until_any_on_queue``
  - ``ishmemx_wait_until_some_on_queue``
  - ``ishmemx_TYPENAME_wait_until_some_on_queue``
  - ``ishmemx_signal_wait_until_on_queue``


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``work_group`` APIs Overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the standard RMA and collective routines, Intel®
SHMEM also provides corresponding ``work_group`` API extensions
such as ``ishmemx_put_work_group`` and
``ishmemx_barrier_all_work_group``.
Unlike the standard RMA and collective routines, the ``work_group`` APIs are
only callable from device kernels and require passing a **group** argument.
This argument corresponds to either a |sycl_spec_groups| in the
context of explicit ND-range parallel kernels.
All work-items in an ND-range kernel are organized into
`work-groups`, which execute independently and in any order on
the underlying hardware.
For more comprehensive information, please refer to the
|sycl_book| or |online_resources|, which introduce and explain
key programming concepts and language details about SYCL, such as
groups and subgroups.

.. |sycl_spec_groups| raw:: html

   <a href="https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#group-class" target="_blank">SYCL group or sub_group</a>

.. |sycl_book| raw:: html

   <a href="https://link.springer.com/book/10.1007%2F978-1-4842-5574-2" target="_blank">Data Parallel C++ book</a>

.. |online_resources| raw:: html

   <a href="https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-0/data-parallelism-in-c-using-sycl.html" target="_blank">other online resources</a>

Depending on the use-case, the ``work_group`` APIs may achieve
better performance than the corresponding standard APIs.
For example, when calling the RMA ``work_group`` routines across
GPUs connected by Intel® :math:`\text{X}^e` Link fabric, the
implementation may perform the data transfer by having each
thread in the `work-group` copy a respective chunk of the source
buffer directly to the destination buffer.
On the other hand, if the destination buffer resides across the
network, the ``work_group`` APIs allow the implementation to
restrict the expensive RMA operations to `only` the leader
threads of each **work_group** or **sub_group**.
This latter optimization can have a dramatic effect on
performance when compared to the case where all threads within a
SYCL kernel to simultaneously post small RMA requests to the
network.
Similar `work-group` optimizations also apply to the collective
operations.

The ``work_group`` APIs must be called by every thread in the
**work_group** or **sub_group** with identical arguments.
``work_group`` functions that return values will return the same value on all
threads.
The **source** buffer must be ready for transmission across all
threads within the **group** before the ``work_group`` API is
invoked, which may require synchronization in the application.
In the implementation, each ``work_group`` API includes a
**group** barrier at both the start and the end of the routine.
Furthermore, users must assure each `work-group` can execute to
completion.
For example, within a kernel there can be no dependencies between
threads across different `work-groups`.

The following is a complete list of the ``work_group`` APIs provided by
Intel® SHMEM.
TYPENAME corresponds to the types specified by Table
:ref:`Standard RMA Types<stdrmatypes>`, OP corresponds to the relevant
operations specified by Table :ref:`Reduction Types, Names, and Supporting
Operations<reducetypes>`, and SIZE is one of 8, 16, 32, 64, or 128:

  - ``ishmemx_put_work_group``
  - ``ishmemx_TYPENAME_put_work_group``
  - ``ishmemx_putmem_work_group``
  - ``ishmemx_putSIZE_work_group``
  - ``ishmemx_iput_work_group``
  - ``ishmemx_TYPENAME_iput_work_group``
  - ``ishmemx_iputSIZE_work_group``
  - ``ishmemx_ibput_work_group``
  - ``ishmemx_TYPENAME_ibput_work_group``
  - ``ishmemx_ibputSIZE_work_group``
  - ``ishmemx_get_work_group``
  - ``ishmemx_TYPENAME_get_work_group``
  - ``ishmemx_getmem_work_group``
  - ``ishmemx_getSIZE_work_group``
  - ``ishmemx_put_nbi_work_group``
  - ``ishmemx_TYPENAME_put_nbi_work_group``
  - ``ishmemx_putmem_nbi_work_group``
  - ``ishmemx_putSIZE_nbi_work_group``
  - ``ishmemx_get_nbi_work_group``
  - ``ishmemx_TYPENAME_get_nbi_work_group``
  - ``ishmemx_getmem_nbi_work_group``
  - ``ishmemx_getSIZE_nbi_work_group``
  - ``ishmemx_iget_work_group``
  - ``ishmemx_TYPENAME_iget_work_group``
  - ``ishmemx_igetSIZE_work_group``
  - ``ishmemx_ibget_work_group``
  - ``ishmemx_TYPENAME_ibget_work_group``
  - ``ishmemx_ibgetSIZE_work_group``
  - ``ishmemx_put_signal_work_group``
  - ``ishmemx_TYPENAME_put_signal_work_group``
  - ``ishmemx_putmem_signal_work_group``
  - ``ishmemx_putSIZE_signal_work_group``
  - ``ishmemx_put_signal_nbi_work_group``
  - ``ishmemx_TYPENAME_put_signal_nbi_work_group``
  - ``ishmemx_putmem_signal_nbi_work_group``
  - ``ishmemx_putSIZE_signal_nbi_work_group``
  - ``ishmemx_barrier_all_work_group``
  - ``ishmemx_sync_all_work_group``
  - ``ishmemx_team_sync_work_group``
  - ``ishmemx_fence_work_group``
  - ``ishmemx_quiet_work_group``
  - ``ishmemx_alltoall_work_group``
  - ``ishmemx_TYPENAME_alltoall_work_group``
  - ``ishmemx_alltoallmem_work_group``
  - ``ishmemx_broadcast_work_group``
  - ``ishmemx_TYPENAME_broadcast_work_group``
  - ``ishmemx_broadcastmem_work_group``
  - ``ishmemx_collect_work_group``
  - ``ishmemx_TYPENAME_collect_work_group``
  - ``ishmemx_collectmem_work_group``
  - ``ishmemx_fcollect_work_group``
  - ``ishmemx_TYPENAME_fcollect_work_group``
  - ``ishmemx_fcollectmem_work_group``
  - ``ishmemx_OP_reduce_work_group``
  - ``ishmemx_TYPENAME_OP_reduce_work_group``
  - ``ishmemx_wait_until_work_group``
  - ``ishmemx_TYPENAME_wait_until_work_group``
  - ``ishmemx_wait_until_all_work_group``
  - ``ishmemx_TYPENAME_wait_until_all_work_group``
  - ``ishmemx_wait_until_any_work_group``
  - ``ishmemx_TYPENAME_wait_until_any_work_group``
  - ``ishmemx_wait_until_some_work_group``
  - ``ishmemx_TYPENAME_wait_until_some_work_group``
  - ``ishmemx_test_work_group``
  - ``ishmemx_TYPENAME_test_work_group``
  - ``ishmemx_test_all_work_group``
  - ``ishmemx_TYPENAME_test_all_work_group``
  - ``ishmemx_test_any_work_group``
  - ``ishmemx_TYPENAME_test_any_work_group``
  - ``ishmemx_test_some_work_group``
  - ``ishmemx_TYPENAME_test_some_work_group``
  - ``ishmemx_signal_wait_until_work_group``
