.. _signaling:

--------------------
Signaling Operations
--------------------


This section specifies the Intel® SHMEM support for
`put-with-signal`, nonblocking `put-with-signal`, and `signal-fetch` routines.
The put-with-signal routines provide a method for copying data from a
contiguous local data object to a data object on a specified PE and
subsequently updating a remote flag to signal completion.
The signal-fetch routine provides support for fetching a signal update
operation.

.. Intel® SHMEM `put-with-signal` routines specified in this section
.. have two variants.
.. In one of the variants, the context handle, **ctx**, is explicitly passed as an
.. argument.
.. In this variant, the operation is performed on the specified context.
.. If the context handle **ctx** does not correspond to a valid context, the
.. behavior is undefined.
.. In the other variant, the context handle is not explicitly passed and thus, the
.. operations are performed on the default context.

.. _signal_atomicity:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Atomicity Guarantees for Signaling Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All signaling operations put-with-signal, nonblocking put-with-signal, and
signal-fetch are performed on a signal data object, a remotely accessible
symmetric object of type **uint64_t**.
A signal operator in the put-with-signal routine is an Intel® SHMEM
library constant that determines the type of update to be performed as a signal
on the signal data object.

All signaling operations on the signal data object complete as if performed
atomically with respect to the following:

* other blocking or nonblocking variant of the put-with-signal routine that
  updates the signal data object using the same signal update operator;
* signal-fetch routine that fetches the signal data object; and
* any point-to-point synchronization routine that accesses the signal data
  object.

.. _signal_operators:

^^^^^^^^^^^^^^^^^^^^^^^^^^
Available Signal Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^

With the atomicity guarantees as described in Section :ref:`Atomicity
Guarantees for Signaling Operations<signal_atomicity>`, the following options
can be used as a signal operator.

* ``ISHMEM_SIGNAL_SET`` An update to signal data object is an atomic set
  operation. It writes an unsigned 64-bit value as a signal into the signal
  data object on a remote PE as an atomic operation.

* ``ISHMEM_SIGNAL_ADD`` An update to signal data object is an atomic add
  operation. It adds an unsigned 64-bit value as a signal into the signal data
  object on a remote PE as an atomic operation.

^^^^^^^^^^^^^^^^^^^
Blocking Put-Signal
^^^^^^^^^^^^^^^^^^^

"""""""""""""""""
ISHMEM_PUT_SIGNAL
"""""""""""""""""
The `put-with-signal` routines provide a method for copying data from a
contiguous local data object to a data object on a specified PE and
subsequently updating a remote flag to signal completion.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`, and SIZE is one of 8, 16, 32, 64, 128.

.. cpp:function:: template<typename TYPE> void ishmem_put_signal(TYPE* dest, const TYPE* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe)

.. cpp:function:: void ishmem_TYPENAME_put_signal(TYPE* dest, const TYPE* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe)

.. cpp:function:: void ishmem_putSIZE_signal(void* dest, const void* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe)

.. cpp:function:: void ishmem_putmem_signal(void* dest, const void* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param source: Local address of the data object containing the data to be copied. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param nelems: Number of elements in the **dest** and **source** arrays. For ``ishmem_putmem_signal``, elements are bytes.
  :param sig_addr: Symmetric address of the signal data object to be updated on the remote PE as a signal.
  :param signal: Unsigned 64-bit value that is used for updating the remote **sig_addr** signal data object.
  :param sig_op: Signal operator that represents the type of update to be performed on the remote **sig_addr** signal data object.
  :param pe: PE number of the remote PE.
  :returns: None.

Callable from the **host** and **device**.

**Description:**
The `put-with-signal` routines provide a method for copying data from a
contiguous local data object to a data object on a specified PE and
subsequently updating a remote flag to signal completion.
The routines return after the data has been copied out of the **source** array
on the local PE.

The **sig_op** signal operator determines the type of update to be performed on
the remote **sig_addr** signal data object.
The completion of signal update based on the **sig_op** signal operator using
the **signal** flag on the remote PE indicates the delivery of its
corresponding **dest** data words into the data object on the remote PE.

An update to the **sig_addr** signal data object through a `put-with-signal`
routine completes as if performed atomically as described in Section :ref:`Atomicity
Guarantees for Signaling Operations<signal_atomicity>`.
The various options as described in Section :ref:`Available Signal
Operators<signal_operators>` can be used as the **sig_op** signal operator.

The **dest** and **sig_addr** data objects must both be remotely accessible and
may not be overlapping in memory.

The completion of signal update using the **signal** flag on the remote PE
indicates only the delivery of its corresponding **dest** data words into the
data object on the remote PE.
Without a memory-ordering operation, there is no implied ordering between the
signal update of a `put-with-signal` routine and another data transfer.
For example, the completion of the signal update in a sequence consisting of a
put routine followed by a `put-with-signal` routine does not imply delivery of
the `put` routine's data.

"""""""""""""""""""""""""""""
ISHMEMX_PUT_SIGNAL_WORK_GROUP
"""""""""""""""""""""""""""""
The `put-with-signal` routines provide a method for copying data from a
contiguous local data object to a data object on a specified PE and
subsequently updating a remote flag to signal completion.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`, and SIZE is one of 8, 16, 32, 64, 128.

.. cpp:function:: template<typename TYPE, typename Group> void ishmemx_put_signal_work_group(TYPE* dest, const TYPE* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_TYPENAME_put_signal_work_group(TYPE* dest, const TYPE* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_putSIZE_signal_work_group(void* dest, const void* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_putmem_signal_work_group(void* dest, const void* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, const Group& group)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param source: Local address of the data object containing the data to be copied. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param nelems: Number of elements in the **dest** and **source** arrays. For ``ishmemx_putmem_signal_work_group``, elements are bytes.
  :param sig_addr: Symmetric address of the signal data object to be updated on the remote PE as a signal.
  :param signal: Unsigned 64-bit value that is used for updating the remote **sig_addr** signal data object.
  :param sig_op: Signal operator that represents the type of update to be performed on the remote **sig_addr** signal data object.
  :param pe: PE number of the remote PE.
  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the `Put` operation.
  :returns: None.

Callable from the **device**.

**Description:**

The `put-with-signal` routines provide a method for copying data from a
contiguous local data object to a data object on a specified PE and
subsequently updating a remote flag to signal completion.
The routines return after the data has been copied out of the **source** array
on the local PE.

The **sig_op** signal operator determines the type of update to be performed on
the remote **sig_addr** signal data object.
The completion of signal update based on the **sig_op** signal operator using
the **signal** flag on the remote PE indicates the delivery of its
corresponding **dest** data words into the data object on the remote PE.

An update to the **sig_addr** signal data object through a `put-with-signal`
routine completes as if performed atomically as described in Section :ref:`Atomicity
Guarantees for Signaling Operations<signal_atomicity>`.
The various options as described in Section :ref:`Available Signal
Operators<signal_operators>` can be used as the **sig_op** signal operator.

The **dest** and **sig_addr** data objects must both be remotely accessible and
may not be overlapping in memory.

The completion of signal update using the **signal** flag on the remote PE
indicates only the delivery of its corresponding **dest** data words into the
data object on the remote PE.
Without a memory-ordering operation, there is no implied ordering between the
signal update of a `put-with-signal` routine and another data transfer.
For example, the completion of the signal update in a sequence consisting of a
put routine followed by a `put-with-signal` routine does not imply delivery of
the `put` routine's data.

^^^^^^^^^^^^^^^^^^^^^^
Nonblocking Put-Signal
^^^^^^^^^^^^^^^^^^^^^^

"""""""""""""""""""""
ISHMEM_PUT_SIGNAL_NBI
"""""""""""""""""""""
The `nonblocking put-with-signal` routines provide a method for copying data from a
contiguous local data object to a data object on a specified PE and
subsequently updating a remote flag to signal completion.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`, and SIZE is one of 8, 16, 32, 64, 128.

.. cpp:function:: template<typename TYPE> void ishmem_put_signal_nbi(TYPE* dest, const TYPE* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe)

.. cpp:function:: void ishmem_TYPENAME_put_signal_nbi(TYPE* dest, const TYPE* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe)

.. cpp:function:: void ishmem_putSIZE_signal_nbi(void* dest, const void* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe)

.. cpp:function:: void ishmem_putmem_signal_nbi(void* dest, const void* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE of TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param source: Local address of the data object containing the data to be copied. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param nelems: Number of elements in the **dest** and **source** arrays. For **ishmem_putmem**, elements are bytes.
  :param sig_addr: Symmetric address of the signal data object to be updated on the remote PE as a signal.
  :param signal: Unsigned 64-bit value that is used for updating the remote **sig_addr** signal data object.
  :param sig_op: Signal operator that represents the type of update to be performed on the remote **sig_addr** signal data object.
  :param pe: PE number of the remote PE.
  :returns: None.

Callable from the **host** and **device**.

**Description:**
The `nonblocking put-with-signal` routines provide a method for copying data
from a contiguous local data object to a data object on a specified PE and
subsequently updating a remote flag to signal completion.

The routines return after initiating the operation.
The operation is considered complete after a subsequent call to
``ishmem_quiet`` or ``ishmemx_quiet_work_group``.
At the completion of the quiet operation, the data has been copied out of the
**source** array on the local PE and delivered into the **dest** array on the
destination PE.

The delivery of the **signal** flag on the remote PE indicates only the
delivery of its corresponding **dest** data words into the data object on the
remote PE.
Furthermore, two successive nonblocking `put-with-signal` routines, or a
nonblocking `put-with-signal` routine with another data transfer may deliver
data out of order unless a call to ``ishmem_fence`` or
``ishmemx_fence_work_group`` is introduced between the two calls.

The **sig_op** signal operator determines the type of update to be performed on
the remote **sig_addr** signal data object.

An update to the **sig_addr** signal data object through a nonblocking
`put-with-signal` routine completes as if performed atomically as described in
Section :ref:`Atomicity Guarantees for Signaling Operations<signal_atomicity>`.
The various options as described in Section :ref:`Available Signal
Operators<signal_operators>` can be used as the **sig_op** signal operator.

The **dest** and **sig_addr** data objects must both be remotely accessible and
may not be overlapping in memory.

"""""""""""""""""""""""""""""""""
ISHMEMX_PUT_SIGNAL_NBI_WORK_GROUP
"""""""""""""""""""""""""""""""""
The `nonblocking put-with-signal` routines provide a method for copying data
from a contiguous local data object to a data object on a specified PE and
subsequently updating a remote flag to signal completion.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`, and SIZE is one of 8, 16, 32, 64, 128.

.. cpp:function:: template<typename TYPE, typename Group> void ishmemx_put_signal_nbi_work_group(TYPE* dest, const TYPE* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_TYPENAME_put_signal_nbi_work_group(TYPE* dest, const TYPE* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_putSIZE_signal_nbi_work_group(void* dest, const void* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_putmem_signal_nbi_work_group(void* dest, const void* source, size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, const Group& group)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE of TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`. 
  :param source: Local address of the data object containing the data to be copied. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param nelems: Number of elements in the **dest** and **source** arrays. For **ishmem_putmem**, elements are bytes.
  :param sig_addr: Symmetric address of the signal data object to be updated on the remote PE as a signal.
  :param signal: Unsigned 64-bit value that is used for updating the remote **sig_addr** signal data object.
  :param sig_op: Signal operator that represents the type of update to be performed on the remote **sig_addr** signal data object.
  :param pe: PE number of the remote PE.
  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the `Put` operation.
  :returns: None.

Callable from the **device**.

**Description:**

The `nonblocking put-with-signal` routines provide a method for copying data
from a contiguous local data object to a data object on a specified PE and
subsequently updating a remote flag to signal completion.

The routines return after initiating the operation.
The operation is considered complete after a subsequent call to
``ishmem_quiet`` or ``ishmemx_quiet_work_group``.
At the completion of the quiet operation, the data has been copied out of the
**source** array on the local PE and delivered into the **dest** array on the
destination PE.

The delivery of the **signal** flag on the remote PE indicates only the
delivery of its corresponding **dest** data words into the data object on the
remote PE.
Furthermore, two successive nonblocking `put-with-signal` routines, or a
nonblocking `put-with-signal` routine with another data transfer may deliver
data out of order unless a call to ``ishmem_fence`` or
``ishmemx_fence_work_group`` is introduced between the two calls.

The **sig_op** signal operator determines the type of update to be performed on
the remote **sig_addr** signal data object.

An update to the **sig_addr** signal data object through a nonblocking
`put-with-signal` routine completes as if performed atomically as described in
Section :ref:`Atomicity Guarantees for Signaling Operations<signal_atomicity>`.
The various options as described in Section :ref:`Available Signal
Operators<signal_operators>` can be used as the **sig_op** signal operator.

The **dest** and **sig_addr** data objects must both be remotely accessible and
may not be overlapping in memory.


^^^^^^^^^^^^^^^^^^^
ISHMEMX_SIGNAL_ADD
^^^^^^^^^^^^^^^^^^^

Adds to a signal value of a remote date object.

.. cpp:function:: void ishmemx_signal_add(uint64_t * sig_addr, uint64_t signal, int pe)

  :param sig_addr: Symmetric address of the signal data object to be updated on the remote PE. 
  :param signal: Unsigned 64-bit value that is used for updating the remote **sig_addr**
		 signal data object.
  :param pe: PE number of the remote PE.
  :returns: None.

Callable from the **host** and **device**.

**Description:**
``ishmemx_signal_add`` adds **value** to the signal data object pointed to by **sig_addr**
on PE **pe**.
The update to **sig_addr** signal object at the calling PE is expected to satisfy
the atomicity guarantees as described in Section :ref:`Atomicity Guarantees for
Signaling Operations<signal_atomicity>`.


^^^^^^^^^^^^^^^^^^^
ISHMEM_SIGNAL_FETCH
^^^^^^^^^^^^^^^^^^^

Fetches the signal update on a local data object.

.. cpp:function:: uint64_t ishmem_signal_fetch(const uint64_t * sig_addr)

  :param sig_addr: Local address of the remotely accessible signal variable.
  :returns: The contents of the signal data object, **sig_addr**, at the calling PE.

Callable from the **host** and **device**.

**Description:**
``ishmem_signal_fetch`` performs a fetch operation and returns the contents of
the **sig_addr** signal data object.
Access to **sig_addr** signal object at the calling PE is expected to satisfy
the atomicity guarantees as described in Section :ref:`Atomicity Guarantees for
Signaling Operations<signal_atomicity>`.


^^^^^^^^^^^^^^^^^^^
ISHMEMX_SIGNAL_SET
^^^^^^^^^^^^^^^^^^^

Sets the signal value of a remote date object.

.. cpp:function:: void ishmemx_signal_set(uint64_t * sig_addr, uint64_t signal, int pe)

  :param sig_addr: Symmetric address of the signal data object to be updated on the remote PE.
  :param signal: Unsigned 64-bit value that is used for updating the remote **sig_addr**
		 signal data object.
  :param pe: PE number of the remote PE. 
  :returns: None.

Callable from the **host** and **device**.

**Description:**
``ishmemx_signal_set`` writes **value** into the signal data object pointed to by **sig_addr**
on PE **pe**.
The update to **sig_addr** signal object at the calling PE is expected to satisfy
the atomicity guarantees as described in Section :ref:`Atomicity Guarantees for
Signaling Operations<signal_atomicity>`.
