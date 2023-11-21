.. _memory_ordering:

---------------
Memory Ordering
---------------

^^^^^^^^^^^^^^
ISHMEM_FENCE
^^^^^^^^^^^^^^

Ensures ordering of delivery of operations on symmetric data objects.

.. cpp:function:: void ishmem_fence()

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

^^^^^^^^^^^^
ISHMEM_QUIET
^^^^^^^^^^^^

Waits for completion of outstanding operations on symmetric data objects
issued by a PE.

.. cpp:function:: void ishmem_quiet()

Callable from the **host** and **device**.

**Description:**
The ``ishmem_quiet`` routine ensures completion of all operations on
symmetric data objects issued by the calling PE.

.. TODO:

.. Table "mem-order" lists the operations that are ordered by the
.. ``ishmem_fence`` routine.

All operations on symmetric data objects are guaranteed to be complete and
visible to all PEs when ``ishmem_quiet`` returns.

^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_QUIET_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^

Waits for completion of outstanding operations on symmetric data objects
issued by a PE.

.. cpp:function:: void ishmemx_quiet_work_group(const Group& group)

  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the quiet operation.

Callable from the **device**.

**Description:**
The ``ishmemx_quiet_work_group`` routine ensures completion of all operations on
symmetric data objects issued by the calling PE.

.. TODO:

.. Table "mem-order" lists the operations that are ordered by the
.. ``ishmem_fence`` routine.

All operations on symmetric data objects are guaranteed to be complete and
visible to all PEs when ``ishmemx_quiet_work_group`` returns.
