.. _library_constants:

=================
Library Constants
=================

.. .. c:macro:: ISHMEM_THREAD_SINGLE
..
.. The thread support level which specifies that the program must not be multithreaded.
..
.. .. c:macro:: ISHMEM_THREAD_FUNNELED
..
.. The thread support level which specifies that the program may be multithreaded
.. but must ensure that only the main thread invokes the ``ishmem`` interfaces.
..
.. .. c:macro:: ISHMEM_THREAD_SERIALIZED
..
.. The thread support level which specifies that the program may be multithreaded
.. but must ensure that the ``ishmem`` interfaces are not invoked concurrently
.. by multiple threads.
..
.. .. c:macro:: ISHMEM_THREAD_MULTIPLE
..
.. The thread support level which specifies that the program may be multithreaded
.. and any thread may invoke the ``ishmem`` interfaces.

.. c:macro:: ISHMEM_MAJOR_VERSION

Integer representing the major version of Intel® SHMEM Specification
in use.

.. c:macro:: ISHMEM_MINOR_VERSION

Integer representing the minor version of Intel® SHMEM Specification
in use.

.. c:macro:: ISHMEM_MAX_NAME_LEN

Integer representing the maximum length of ``ISHMEM_VENDOR_STRING``.

.. c:macro:: ISHMEM_VENDOR_STRING

String representing vendor defined information of size at  most
``ISHMEM_MAX_NAME_LEN``. The string ``ISHMEM_VENDOR_STRING`` is terminated by a
null character.

.. .. c:macro:: ISHMEM_CMP_EQ
..
.. An integer constant expression corresponding to the “equal to” comparison
.. operation.
.. See Section :ref:`Point-to-Point Synchronization<point_to_point>` for more
.. detail about its use.
..
.. .. c:macro:: ISHMEM_CMP_NE
..
.. An integer constant expression corresponding to the “not equal to” comparison
.. operation.
.. See Section :ref:`Point-to-Point Synchronization<point_to_point>` for more
.. detail about its use.
..
.. .. c:macro:: ISHMEM_CMP_LT
..
.. An integer constant expression corresponding to the “less than” comparison
.. operation.
.. See Section :ref:`Point-to-Point Synchronization<point_to_point>` for more
.. detail about its use.
..
.. .. c:macro:: ISHMEM_CMP_LE
..
.. An integer constant expression corresponding to the “less than or equal to”
.. comparison operation.
.. See Section :ref:`Point-to-Point Synchronization<point_to_point>` for more
.. detail about its use.
..
.. .. c:macro:: ISHMEM_CMP_GT
..
.. An integer constant expression corresponding to the “greater than” comparison
.. operation.
.. See Section :ref:`Point-to-Point Synchronization<point_to_point>` for more
.. detail about its use.
..
.. .. c:macro:: ISHMEM_CMP_GE
..
.. An integer constant expression corresponding to the “greater than or equal to”
.. comparison operation.
.. See Section :ref:`Point-to-Point Synchronization<point_to_point>` for more
.. detail about its use.

.. ISHMEM_TEAM_NUM_CONTEXTS
.. ISHMEM_TEAM_INVALID
.. ISHMEM_CTX_INVALID
.. ISHMEM_CTX_SERIALIZED
.. ISHMEM_CTX_PRIVATE
.. ISHMEM_CTX_NOSTORE
.. ISHMEM_SIGNAL_SET
.. ISHMEM_SIGNAL_ADD
.. ISHMEM_MALLOC_ATOMICS_REMOTE
.. ISHMEM_MALLOC_SIGNAL_REMOTE


.. ===============
.. Library Handles
.. ===============
..
.. .. c:macro:: ISHMEM_TEAM_WORLD
..
.. .. c:macro:: ISHMEM_TEAM_SHARED
..
.. .. c:macro:: ISHMEM_CTX_DEFAULT

.. _env_vars:

=====================
Environment Variables
=====================

The Intel® SHMEM specification provides a set of environment
variables that allows users to configure the implementation and receive
information about the implementation.

.. c:macro:: ISHMEM_VERSION

If set to any value, print the library version at startup.

.. c:macro:: ISHMEM_INFO

If set to any value, print helpful text about all these environment variables.

.. c:macro:: ISHMEM_SYMMETRIC_SIZE

Specifies the size (in bytes) of the symmetric heap memory per PE.
The resulting size is implementation-defined and must be at least as large as
the integer ceiling of the product of the numeric prefix and the scaling
factor.
The allowed character suffixes for the scaling factor are as follows:

* k or K multiplies by :math:`2^{10}` (kibibytes)

* m or M multiplies by :math:`2^{20}` (mebibytes)

* g or G multiplies by :math:`2^{30}` (gibibytes)

* t or T multiplies by :math:`2^{40}` (tebibytes)

For example, string "20m" is equivalent to the integer value 20971520, or 20
mebibytes.
Similarly the string "3.1M" is equivalent to the integer value 3250586.
Only one multiplier is recognized and any characters following the multiplier
are ignored, so "20kk" will not produce the same result as "20m".
Usage of string ".5m" will yield the same result as the string "0.5m".
An invalid value for ``ISHMEM_SYMMETRIC_SIZE`` is an error, which causes the
Intel® SHMEM library to terminate the program.

.. c:macro:: ISHMEM_DEBUG

If set to any value, enable debugging messages.

.. c:macro:: ISHMEM_SHMEM_LIB_NAME

Informs the Intel® SHMEM library of the shared object name (e.g.
``libshmem.so``) of the host-side OpenSHMEM library to be dynamically loaded.
The default value is ``libsma.so``.

.. c:macro:: ISHMEM_ENABLE_GPU_IPC

Enables the intra-node inter-process communication (IPC) implementation.
The default value is 1 which enables use of the Intel® :math:`\text{X}^e` Link
fabric for inter-GPU communications on the same super-node.
The value can be set to 0 for situations in which Intel® :math:`\text{X}^e`
Link fabric is not available or does not connect all the GPUs.

.. c:macro:: ISHMEM_ENABLE_GPU_IPC_PIDFD

Enables the pidfd implementation of IPC.
This is enabled by default, but will fail on older Linux kernels that do not
support the necessary systiem calls.
In such cases, use ISHMEM_ENABLE_GPU_IPC_PIDFD=0

.. c:macro:: ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP

Place symmetric heap in `host` unified shared memory (allocated on the host and
accessible by the host and device).
