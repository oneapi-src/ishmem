.. _library_setup_exit_query_routines:

---------------------------------------
Library Setup, Exit, and Query Routines
---------------------------------------

The library setup, exit, and query interfaces initialize and monitor the
parallel environment of the PEs.

.. _ishmem_init:

^^^^^^^^^^^
ISHMEM_INIT
^^^^^^^^^^^

.. cpp:function:: void ishmem_init(void)

  :parameters: None.
  :returns: None.

Callable from the **host**.

**Description:**
Initializes the ``ishmem`` library.
:ref:`ishmem_init<ishmem_init>` and
:ref:`ishmemx_init_attr<ishmemx_init_attr>` are collective operations that
all PEs must call before any other ``ishmem`` routine may be called.

At the end of the ``ishmem`` program which it initialized, the call to
``ishmem_init`` must be matched with a call to ``ishmem_finalize``.
After the first call to ``ishmem_init``, a subsequent call to
``ishmem_init`` or ``ishmemx_init_attr`` in the same program results in
undefined behavior.

.. _ishmemx_runtime_type_t:

^^^^^^^^^^^^^^^^^
ISHMEMX_INIT_ATTR
^^^^^^^^^^^^^^^^^

.. cpp:enum:: ishmemx_runtime_type_t

  .. c:macro:: ISHMEM_RUNTIME_OPENSHMEM
  .. c:macro:: ISHMEM_RUNTIME_MPI
  .. c:macro:: ISHMEM_RUNTIME_PMI

Callable from the **host**.

**Description:**
Indicates which runtime is used to initialize Intel® SHMEM: either
OpenSHMEM, MPI, or PMI.

.. important:: As of Intel® SHMEM |version| only ISHMEM_RUNTIME_OPENSHMEM and ISHMEM_RUNTIME_MPI are supported.

.. _ishmemx_attr_t:
.. cpp:struct:: ishmemx_attr_t

  .. c:var:: ishmemx_runtime_type_t runtime
  .. c:var:: bool initialize_runtime = true
  .. c:var:: bool gpu = true
  .. c:var:: void *mpi_comm

**Description:**
A struct declaration describing attributes for initialization.
A valid **runtime** enumeration value must be set by the user and must
correspond to a runtime that is enabled within the build of the ``ishmem``
library.
By default, the parallel runtime is initialized by Intel® SHMEM
(**initialize_runtime** default is ``true``).
The **gpu** boolean indicates whether to use GPU memory for the symmetric
heap (default is ``true``). **mpi_comm** is a pointer to the corresponding
MPI communicator for representing ``ISHMEM_TEAM_WORLD`` when used with
``ISHMEM_RUNTIME_MPI`` (default is ``MPI_COMM_WORLD``).

.. _ishmemx_init_attr:
.. cpp:function:: void ishmemx_init_attr(ishmemx_attr_t * attr)

  :param attr: a struct of type :ref:`ishmemx_attr_t<ishmemx_attr_t>` specifying initialization attributes
  :returns: None.

Callable from the **host**.

**Description:**
Initializes the ``ishmem`` library.
:ref:`ishmem_init<ishmem_init>` and
:ref:`ishmemx_init_attr<ishmemx_init_attr>` are collective operations that
all PEs must call before any other ``ishmem`` routine may be called.
At the end of the ``ishmem`` program which it initialized, the call to
``ishmem_init`` must be matched with a call to ``ishmem_finalize``.
After the first call to ``ishmemx_init_attr``, a subsequent call to
``ishmemx_init_attr`` or ``ishmem_init`` in the same program results in
undefined behavior.

.. _ishmem_my_pe:

^^^^^^^^^^^^
ISHMEM_MY_PE
^^^^^^^^^^^^

.. cpp:function:: int ishmem_my_pe(void)

  :parameters: None.
  :returns: The PE number.

Callable from the **host** and **device**.

**Description:**
This routine returns the PE number of the calling PE.  The result is an
integer between 0 and *npes* - 1, where *npes* is the total number of PEs
executing the current program.

.. _ishmem_n_pes:

^^^^^^^^^^^^
ISHMEM_N_PES
^^^^^^^^^^^^

.. cpp:function:: int ishmem_n_pes(void)

  :parameters: None.
  :returns: The number of total PEs running in the program.

Callable from the **host** and **device**.

**Description:**
The routine returns the number of PEs running in the program.


.. _ishmem_finalize:

^^^^^^^^^^^^^^^
ISHMEM_FINALIZE
^^^^^^^^^^^^^^^

.. cpp:function:: void ishmem_finalize(void)

  :parameters: None.
  :returns: None.


Callable from the **host**.

**Description:**
``ishmem_finalize`` is a collective operation that ends the ``ishmem``
portion of a program previously initialized by
:ref:`ishmem_init<ishmem_init>` or
:ref:`ishmemx_init_attr<ishmemx_init_attr>` and releases all resources used
by the ``ishmem`` library.
This collective operation requires all PEs to participate in the call.
There is an implicit global barrier in ``ishmem_finalize`` to ensure that
pending communications are completed and that no resources are released until
all PEs have entered ``ishmem_finalize``.
This routine destroys all teams created by the ``ishmem`` program.
``ishmem_finalize`` must be the last ``ishmem`` library call encountered in
the ``ishmem`` portion of a program.
A call to ``ishmem_finalize`` will release all resources initialized by a
corresponding call to ``ishmem_init`` or ``ishmemx_init_attr``. All
processes that represent the PEs will still exist after the call to
``ishmem_finalize`` returns, but they will no longer have access to resources
that have been released.

.. FIXME after contexts added:
.. As a result, all shareable contexts are destroyed.
.. The user is responsible for destroying all contexts with the
.. SHMEM_CTX_PRIVATE option enabled prior to calling this routine; otherwise,
.. the behavior is undefined.

.. note:: Because SYCL kernel execution is non-blocking on the host, all
   kernels performing ``ishmem`` calls must first `complete` (for example, by
   calling ``wait`` or ``wait_and_throw`` on the SYCL queue) before calling
   ``ishmem_finalize``.

.. ^^^^^^^^^^^^^^^^^^^^
.. ISHMEM_GLOBAL_EXIT
.. ^^^^^^^^^^^^^^^^^^^^

.. ^^^^^^^^^^^^^^^^^^^^^^
.. ISHMEM_PE_ACCESSIBLE
.. ^^^^^^^^^^^^^^^^^^^^^^

.. ^^^^^^^^^^^^^^^^^^^^^^^^
.. ISHMEM_ADDR_ACCESSIBLE
.. ^^^^^^^^^^^^^^^^^^^^^^^^

.. _ishmem_query_initialized:

^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEM_QUERY_INITIALIZED
^^^^^^^^^^^^^^^^^^^^^^^^

.. cpp:function:: void ishmem_query_initialized(int * initialized)

  :param initialized: Nonzero if the library is in the initialized state. Zero otherwise.
  :returns: None.

Callable from the **host**.

**Description:**
The ``ishmem_query_initialized`` call returns the initialization status of the 
``ishmem`` library. If the application has called an initialization routine and 
has not yet made the corresponding call to ``ishmem_finalize``, this routine 
returns nonzero. Otherwise, it returns zero.
This function may be called at any time, regardless of the thread safety level 
or the current initialized state of the library.

.. _ishmem_ptr:

^^^^^^^^^^
ISHMEM_PTR
^^^^^^^^^^

.. cpp:function:: void* ishmem_ptr(const void* dest, int pe)

  :param dest: The symmetric address of the remotely accessible data object to be referenced
  :param pe: An integer that indicates the PE number on which **dest** is to be accessed.
  :returns:  A local pointer to the remotely accessible **dest** data object is returned when it can be accessed using memory loads and stores.  Otherwise, a null pointer is returned.

Callable from the **host** and **device**.

**Description:**
``ishmem_ptr`` returns a **device** address that may be used to directly
reference **dest** on the specified PE in the world team.
This address can be assigned to a pointer.
After that, ordinary loads and stores to **dest** may be performed from
within the device kernel.
The address returned by ``ishmem_ptr`` is a local address to a remotely
accessible data object.
Providing this address to an argument of a ``ishmem`` routine that requires
a symmetric address results in undefined behavior.

The ``ishmem_ptr`` routine can provide an efficient means to accomplish
communication, for example when a sequence of reads and writes to a data
object on a remote PE does not match the access pattern provided in a
``ishmem`` data transfer routine like ``ishmem_put`` or
``ishmem_iget``.

.. ^^^^^^^^^^^^^^^^^
.. ISHMEM_TEAM_PTR
.. ^^^^^^^^^^^^^^^^^


.. _ishmem_info_get_version:

^^^^^^^^^^^^^^^^^^^^^^^
ISHMEM_INFO_GET_VERSION
^^^^^^^^^^^^^^^^^^^^^^^

.. cpp:function:: void ishmem_info_get_version(int* major, int* minor)

  :param major: The major version of the ``ishmem`` specification in use.
  :param minor: The minor version of the ``ishmem`` specification in use.
  :returns: None.

Callable from the **host** and **device**.

**Description:**
Returns the major and minor version of the ``ishmem`` specification in use.
For a given library implementation, the major and minor version returned by
these calls are consistent with the library constants ISHMEM_MAJOR_VERSION
and ISHMEM_MINOR_VERSION.


.. _ishmem_info_get_name:

^^^^^^^^^^^^^^^^^^^^
ISHMEM_INFO_GET_NAME
^^^^^^^^^^^^^^^^^^^^

.. cpp:function:: void ishmem_info_get_name(char* name)

  :param name: The vendor defined string.
  :returns: None.

Callable from the **host** and **device**.

**Description:**
This routine returns the vendor defined name string of size defined by the
library constant ISHMEM_MAX_NAME_LEN. The program calling this function
provides the **name** memory buffer of at least size ISHMEM_MAX_NAME_LEN. The
implementation copies the vendor defined string of size at most
ISHMEM_MAX_NAME_LEN to **name**. The string is terminated by a null
character.  If the **name** memory buffer is provided with size less than
ISHMEM_MAX_NAME_LEN, behavior is undefined. For a given library
implementation, the vendor string returned is consistent with the library
constant ISHMEM_VENDOR_STRING.

