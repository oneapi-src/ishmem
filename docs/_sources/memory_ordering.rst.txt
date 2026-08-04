.. _memory_ordering:

---------------
Memory Ordering
---------------

.. _mem_ordering_impact:

**List of Operations affected by Memory Ordering Routines:**

==============================   ======   ======
Operations                       Fence    Quiet
==============================   ======   ======
Memory Store                     X        X
Blocking `Put`                   X        X
Blocking `Get`
Blocking Fetching `AMO`
Blocking Non-fetching `AMO`      X        X
Blocking `put-with-signal`       X        X
Nonblocking `Put`                X        X
Nonblocking `Get`                         X
Nonblocking `AMO`                X [#]_   X
Nonblocking `put-with-signal`    X        X 
==============================   ======   ======

This section introduces Intel® SHMEM interfaces that provide mechanisms to
ensure ordering and/or delivery of completion on memory store, blocking,
and nonblocking routines. Table :ref:`List of Operations affected by 
Memory Ordering Routines<mem_ordering_impact>` lists the operations
affected by Intel® SHMEM memory ordering routines.
 

^^^^^^^^^^^^^^
ISHMEM_FENCE
^^^^^^^^^^^^^^

Ensures ordering of delivery of operations on symmetric data objects.

.. cpp:function:: void ishmem_fence(void)

  :parameters: None.
  :returns: None.

Callable from the **host** and **device**.

**Description:**
This routine ensures ordering of delivery of operations on symmetric data
objects.

.. TODO:

.. Table "mem-order" lists the operations that are ordered by the
.. ``ishmem_fence`` routine.

All operations on symmetric data objects issued to a particular PE prior to the
call to ``ishmem_fence`` are guaranteed to be delivered before any subsequent
operations on symmetric data objects to the same PE.
``ishmem_fence`` guarantees order of delivery, not completion.
It does not guarantee order of delivery of nonblocking `Get` or values fetched
by nonblocking AMO routines.

^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_FENCE_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^

Ensures ordering of delivery of operations on symmetric data objects.

.. cpp:function:: void ishmemx_fence_work_group(const Group& group)

  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the fence operation.
  :returns: None.

Callable from the **device**.

**Description:**
This routine ensures ordering of delivery of operations on symmetric data
objects.

.. TODO:

.. Table "mem-order" lists the operations that are ordered by the
.. ``ishmem_fence`` routine.

All operations on symmetric data objects issued to a particular PE prior to the
call to ``ishmemx_fence_work_group`` are guaranteed to be delivered before any subsequent
operations on symmetric data objects to the same PE.
``ishmemx_fence_work_group`` guarantees order of delivery, not completion.
It does not guarantee order of delivery of nonblocking `Get` or values fetched
by nonblocking AMO routines.

.. _ishmem_quiet:

^^^^^^^^^^^^
ISHMEM_QUIET
^^^^^^^^^^^^

Waits for completion of outstanding operations on symmetric data objects
issued by a PE.

.. cpp:function:: void ishmem_quiet(void)

  :parameters: None.
  :returns: None.

Callable from the **host** and **device**.

**Description:**
The ``ishmem_quiet`` routine ensures completion of all operations on
symmetric data objects issued by the calling PE.

.. TODO:

.. Table "mem-order" lists the operations that are ordered by the
.. ``ishmem_fence`` routine.

All operations on symmetric data objects are guaranteed to be complete and
visible to all PEs when ``ishmem_quiet`` returns.

A host-initiated ``ishmem_quiet`` will only guarantee completion of
device-initiated operations for which the corresponding SYCL kernel has
completed execution.

.. _ishmemx_quiet_on_queue:

^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_QUIET_ON_QUEUE
^^^^^^^^^^^^^^^^^^^^^^

Waits for completion of outstanding operations on symmetric data objects
issued by a PE.

.. cpp:function:: sycl::event ishmemx_quiet_on_queue(sycl::queue& q, const std::vector<sycl::event>& deps)

  :param q: The SYCL queue on which to execute the operation. **q** must be mapped to the GPU tile assigned to the calling PE.
  :param deps: An optional vector of SYCL events that the operation depends on.
  :returns: The SYCL event created upon submitting the operation to the SYCL runtime.

Callable from the **host**.

**Description:**
The ``ishmemx_quiet_on_queue`` routine ensures completion of all operations on
symmetric data objects issued by the calling PE.

.. TODO:

.. Table "mem-order" lists the operations that are ordered by the
.. ``ishmem_fence`` routine.

All operations on symmetric data objects are guaranteed to be complete and
visible to all PEs when ``ishmemx_quiet_on_queue`` returns.

To ensure the quiet operation has completed, refer to the
:ref:`on_queue API Completion Semantics<on_queue_api_completion_semantics>`
section.

.. _ishmemx_quiet_work_group:

^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_QUIET_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^

Waits for completion of outstanding operations on symmetric data objects
issued by a PE.

.. cpp:function:: void ishmemx_quiet_work_group(const Group& group)

  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the quiet operation.
  :returns: None.

Callable from the **device**.

**Description:**
The ``ishmemx_quiet_work_group`` routine ensures completion of all operations on
symmetric data objects issued by the calling PE.

.. TODO:

.. Table "mem-order" lists the operations that are ordered by the
.. ``ishmem_fence`` routine.

All operations on symmetric data objects are guaranteed to be complete and
visible to all PEs when ``ishmemx_quiet_work_group`` returns.

.. [#] Intel® SHMEM fence routines does not guarantee order of delivery of
   values fetched by nonblocking AMO routines.
