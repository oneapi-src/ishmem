.. _memory_management:

-----------------
Memory Management
-----------------
Intel® SHMEM provides a set of APIs for managing the symmetric heap.
The APIs allow one to dynamically allocate, deallocate, and align symmetric
data objects in the symmetric heap.

The ``ishmem_malloc``, ``ishmem_align``, and ``ishmem_free`` routines are
provided  so that multiple PEs in a program can allocate symmetric, remotely
accessible memory blocks.
These memory blocks can then be used with ``ishmem`` communication routines.
When no action is performed, these routines return without performing a
barrier.
Otherwise, each of these routines includes at least one call to a procedure
that is semantically equivalent to ``ishmem_barrier_all``:
``ishmem_malloc`` and ``ishmem_align`` call a barrier on exit;
and ``ishmem_free`` calls a barrier on entry.
This ensures that all PEs participate in the memory allocation, and that the
memory on other PEs can be used as soon as the local PE returns.
The user is responsible for calling these routines with identical argument(s)
on all PEs; if differing **ptr**, **size**, or **alignment** arguments are
used, the behavior of the call and any subsequent ``ishmem`` calls is
undefined.

.. The implicit barriers performed by these routines quiet the default context.
.. It is the user's responsibility to ensure that no communication operations
.. involving the given memory block are pending on other contexts prior to calling
.. the ``ishmem_free`` and ``ishmem_realloc`` routines.

^^^^^^^^^^^^^^^
ISHMEM_MALLOC
^^^^^^^^^^^^^^^

.. cpp:function:: void* ishmem_malloc(size_t size)

  :param size: The size, in bytes, of a block to be allocated from the symmetric heap.
  :returns: The symmetric address of the allocated space; otherwise, it returns a null pointer.

Callable from the **host**.

**Description:**
The ``ishmem_malloc`` routine returns the symmetric address of a block of at
least **size** bytes, which shall be suitably aligned so that it may be
assigned to a pointer to any type of object.  This space is allocated from the
symmetric heap (in contrast to malloc, which allocates from the private heap).
The memory space is uninitialized.
When **size** is zero, the ``ishmem_malloc`` routine performs no action and
returns a null pointer; otherwise, ``ishmem_malloc`` calls a barrier on exit.

The value of the **size** argument must be identical on all PEs; otherwise, the
behavior is undefined.

^^^^^^^^^^^^^
ISHMEM_FREE
^^^^^^^^^^^^^

.. cpp:function:: void ishmem_free(void* ptr)

  :param ptr: Symmetric address of an object in the symmetric heap.

Callable from the **host**.

**Description:**
The ``ishmem_free`` routine causes the block to which **ptr** points to be
deallocated, that is, made available for further allocation.  If **ptr** is a
null pointer, no action is performed; otherwise, ``ishmem_free`` calls a
barrier on entry.  It is the user's responsibility to ensure that no
communication operations involving the given memory block are pending on other
communication contexts prior to calling ``ishmem_free``.

The value of the **ptr** argument must be identical on all PEs; otherwise, the
behavior is undefined.

.. note:: The |release| version of the ``ishmem_free`` routine does not
   release memory for use in subsequent allocations.  This issue will be
   fixed in a future version of Intel® SHMEM.

.. ^^^^^^^^^^^^^^^^
.. ISHMEM_REALLOC
.. ^^^^^^^^^^^^^^^^
..
.. .. cpp:function:: void ishmem_realloc(void* ptr, size_t size)
..
..   :param ptr:
..   :param size:

^^^^^^^^^^^^^^
ISHMEM_ALIGN
^^^^^^^^^^^^^^

.. cpp:function:: void ishmem_align(size_t alignment, size_t size)

  :param alignment: Byte alignment of the block allocated on the symmetric heap.
  :param size: The size, in bytes, of a block to be allocated from the symmetric heap.
  :returns: An aligned symmetric address whose value is a multiple of alignment; otherwise returns a null pointer.

Callable from the **host**.

**Description**
The ``ishmem_align`` routine allocates a block in the symmetric heap that has
a byte alignment specified by the **alignment** argument. The value of
**alignment** shall be a multiple of ``sizeof(void *)`` that is also a power of
two.  Otherwise, the behavior is undefined. When size is zero, the
``ishmem_align`` routine performs no action and returns a null pointer;
otherwise, ``ishmem_align`` call a barrier on exit.
The memory space is uninitialized.

.. ^^^^^^^^^^^^^^^^^^^^^^^^^^
.. ISHMEM_MALLOC_WITH_HINTS
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^
..
.. .. cpp:function:: void ishmem_malloc_with_hints(size_t size, long hints)
..
..  :param size:
..  :param hints:

^^^^^^^^^^^^^^^
ISHMEM_CALLOC
^^^^^^^^^^^^^^^

.. cpp:function:: void* ishmem_calloc(size_t count, size_t size)

  :param count: The number of elements to allocate.
  :param size: The size in bytes of each element to allocate.
  :returns: A pointer to the lowest byte address of the allocated space; otherwise, it returns a null pointer.

Callable from the **host**.

**Description:**
  The ``ishmem_calloc`` routine is a collective operation that allocates a
  region of remotely-accessible memory for an array of **count** objects of
  **size** bytes each and returns a pointer to the lowest byte address of the
  allocated symmetric memory. The space is initialized to all bits zero.

  If the allocation succeeds, the pointer returned shall be suitably aligned so
  that it may be assigned to a pointer to any type of object.  If the
  allocation does not succeed, or either **count** or **size** is 0, the return
  value is a null pointer.

  The values for **count** and **size** shall each be equal across all PEs
  calling ``ishmem_calloc``; otherwise, the behavior is undefined.

  When **count** or **size** is 0, the ``ishmem_calloc`` routine returns
  without performing a barrier.  Otherwise, this routine calls a procedure that
  is semantically equivalent to a barrier on exit.

