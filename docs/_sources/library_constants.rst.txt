.. _library_constants:

=================
Library Constants
=================

.. c:macro:: ISHMEM_THREAD_SINGLE

The thread support level which specifies that the program must not be multithreaded.
See Section :ref:`Thread Support<thread_support_routines>` for more
detail about its use.

.. c:macro:: ISHMEM_THREAD_FUNNELED

The thread support level which specifies that the program may be multithreaded
but must ensure that only the main thread invokes the ``ishmem`` interfaces.
See Section :ref:`Thread Support<thread_support_routines>` for more
detail about its use.

.. c:macro:: ISHMEM_THREAD_SERIALIZED

The thread support level which specifies that the program may be multithreaded
but must ensure that the ``ishmem`` interfaces are not invoked concurrently
by multiple threads. See Section :ref:`Thread Support<thread_support_routines>` for more
detail about its use.

.. c:macro:: ISHMEM_THREAD_MULTIPLE

The thread support level which specifies that the program may be multithreaded
and any thread may invoke the ``ishmem`` interfaces. See 
Section :ref:`Thread Support<thread_support_routines>` for more
detail about its use.

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

.. c:macro:: ISHMEM_CMP_EQ

An integer constant expression corresponding to the “equal to” comparison
operation.
See Section :ref:`Point-to-Point Synchronization<point_to_point>` for more
detail about its use.

.. c:macro:: ISHMEM_CMP_NE

An integer constant expression corresponding to the “not equal to” comparison
operation.
See Section :ref:`Point-to-Point Synchronization<point_to_point>` for more
detail about its use.

.. c:macro:: ISHMEM_CMP_LT

An integer constant expression corresponding to the “less than” comparison
operation.
See Section :ref:`Point-to-Point Synchronization<point_to_point>` for more
detail about its use.

.. c:macro:: ISHMEM_CMP_LE

An integer constant expression corresponding to the “less than or equal to”
comparison operation.
See Section :ref:`Point-to-Point Synchronization<point_to_point>` for more
detail about its use.

.. c:macro:: ISHMEM_CMP_GT

An integer constant expression corresponding to the “greater than” comparison
operation.
See Section :ref:`Point-to-Point Synchronization<point_to_point>` for more
detail about its use.

.. c:macro:: ISHMEM_CMP_GE

An integer constant expression corresponding to the “greater than or equal to”
comparison operation.
See Section :ref:`Point-to-Point Synchronization<point_to_point>` for more
detail about its use.

.. ISHMEM_TEAM_NUM_CONTEXTS

.. c:macro:: ISHMEM_TEAM_INVALID

A value corresponding to an invalid team.
This value can be used to initialize or update team handles to indicate that
they do not reference a valid team.
When managed in this way, applications can use an equality comparison to test
whether a given team handle references a valid team.
See Section :ref:`Team Management Routines<team>` for more detail about its
use.

.. ISHMEM_CTX_INVALID
.. ISHMEM_CTX_SERIALIZED
.. ISHMEM_CTX_PRIVATE
.. ISHMEM_CTX_NOSTORE
.. ISHMEM_SIGNAL_SET
.. ISHMEM_SIGNAL_ADD
.. ISHMEM_MALLOC_ATOMICS_REMOTE
.. ISHMEM_MALLOC_SIGNAL_REMOTE

.. _library_handles:

===============
Library Handles
===============

.. c:macro:: ISHMEM_TEAM_WORLD

Handle of type **ishmem_team_t** that corresponds to the world team that
contains all PEs in the ``ishmem`` program.
All point-to-point communication operations occur on PE numbers relative to the
world team, and all collective synchronizations that do not specify a team are
performed on the world team.
See Section :ref:`Team Management Routines<team>` for more detail about its
use.

.. c:macro:: ISHMEM_TEAM_SHARED

Handle of type **ishmem_team_t** that corresponds to a team of PEs that share
a memory domain.
``ISHMEM_TEAM_SHARED`` refers to the team of all PEs that would mutually
return a non-null address from a call to :ref:`ishmem_ptr<ishmem_ptr>` for
all symmetric heap objects.
That is, :ref:`ishmem_ptr<ishmem_ptr>` must return a non-null pointer to the
local PE for all symmetric heap objects on all target PEs in the team.
This means that symmetric heap objects on each PE are directly load/store
accessible by all PEs in the team.
See Section :ref:`Team Management Routines<team>` for more detail about its
use.

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
support the necessary system calls.
In such cases, use ISHMEM_ENABLE_GPU_IPC_PIDFD=0

.. c:macro:: ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP

Places symmetric heap in `host` unified shared memory (allocated on the host and
accessible by the host and device).

.. c:macro:: ISHMEM_ENABLE_VERBOSE_PRINT

Includes the file, line, and function along with messages printed by the utility
routines and other output for debug, warning, or error reporting. 

.. c:macro:: ISHMEM_RUNTIME

Selects the host back-end library. Valid options are ``OPENSHMEM`` or ``MPI``.
These options are case-insensitive strings, so ``OPENSHMEM``, ``OpenSHMEM``,
``opEnshmem``, etc., select the OpenSHMEM back-end, and ``MPI``, ``mpi``,
``mPi``, etc., select the MPI back-end.
See :ref:`Building Intel® SHMEM<building_ishmem>` for more information about
using this variable.

.. c:macro:: ISHMEM_RUNTIME_USE_OSHMPI

Indicates whether the host back-end is the OSHMPI library.
