.. _collectives:

---------------------
Collective Operations
---------------------

`Collective routines` are defined as coordinated communication or
synchronization operations performed by a group of PEs.

Intel® SHMEM provides two types of collective routines:

#. Collective routines that operate on teams using a team handle parameter to
   determine which PEs will participate in the routine, and use resources
   encapsulated by the team object to perform operations.
   See Section :ref:`Team Management Routines<team>` for details on team
   management.

#. Collective routines that do not accept a team handle argument nor active set
   parameters, which implicitly operate on the world team,
   ``ISHMEM_TEAM_WORLD``.

.. FIXME : above, add "and, as required, the default context." if/when contexts

Concurrent accesses to symmetric memory by an Intel® SHMEM collective routine
and any other means of access---where at least one PE or a thread within a PE
updates the symmetric memory---results in undefined behavior.
Since PEs can enter and exit collectives at different times, accessing such
memory remotely may require additional synchronization.

.. important:: All collective operations must complete before another SYCL
   kernel calls collective operations.

.. important:: A collective call must be either all host-initiated or
   device-initiated. For example, a program that initiates a collective
   operation from the host on some PEs but from the device on other PEs has
   undefined behavior.

^^^^^^^^^^^^^^^^^^
ISHMEM_BARRIER_ALL
^^^^^^^^^^^^^^^^^^

Registers the arrival of a PE at a barrier and blocks the PE until all other
PEs arrive at the barrier and all local updates and remote memory updates are
completed.

.. cpp:function:: void ishmem_barrier_all(void)

  :parameters: None.
  :returns: None.

Callable from the **host** and **device**.

**Description:**
The ``ishmem_barrier_all`` routine is a mechanism for synchronizing all PEs
in the world team at once.
This routine blocks the calling PE until all PEs in the world team have called
``ishmem_barrier_all``.

.. In a multithreaded Intel® SHMEM program, only the calling thread is blocked,
.. however, it may not be called concurrently by multiple threads in the same
.. PE.

Prior to synchronizing with other PEs, ``ishmem_barrier_all`` ensures
completion of all previously issued memory stores, and of all local and remote
memory updates issued via ``ishmem`` AMO and RMA routine calls such as
``ishmem_int_add``, ``ishmem_put_nbi``, and ``ishmem_get_nbi``.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_BARRIER_ALL_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Registers the arrival of a PE at a barrier and blocks the PE until all other
PEs arrive at the barrier and all local updates and remote memory updates are
completed.

.. cpp:function:: template<typename Group> void ishmemx_barrier_all_work_group(const Group& group)

  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the barrier operation.
  :returns: None.

Callable from the **device**.

**Description:**
The ``ishmemx_barrier_all_work_group`` routine is a mechanism for
synchronizing all PEs.
Unlike ``ishmem_barrier_all``, ``ishmemx_barrier_all_work_group`` allows
for the device threads within **group** to cooperate towards the barrier
operation.
This may be more performant; for example, when ``ishmem_barrier_all``
requires `all` device threads in the kernel to invoke RMA operations.
This routine blocks the calling PE until all PEs in the world team have called
``ishmemx_barrier_all_work_group``.
All threads in **group** must call the routine with identical arguments.

^^^^^^^^^^^^^^^
ISHMEM_SYNC_ALL
^^^^^^^^^^^^^^^

Registers the arrival of a PE at a synchronization point and suspends
execution until all other PEs arrive at the synchronization point.

.. cpp:function:: void ishmem_sync_all(void)

  :parameters: None.
  :returns: None.

Callable from the **host** and the **device**.

**Description:**
This routine blocks the calling PE until all PEs in the world team have called
``ishmem_sync_all``.

.. In a multithreaded Intel® SHMEM program, only the calling thread is blocked.

In contrast with the ``ishmem_barrier_all`` routines, ``ishmem_sync_all``
only ensures completion and visibility of previously issued memory
stores and does not ensure completion of remote memory updates issued via
``ishmem`` routines.


^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_SYNC_ALL_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Registers the arrival of a PE at a synchronization point and suspends
execution until all other PEs arrive at the synchronization point.

.. cpp:function:: template<typename Group> void ishmemx_sync_all_work_group(const Group& group)

  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the barrier operation.
  :returns: None.

Callable from the **device**.

**Description:**
This routine blocks the calling PE until all PEs in the world team have called
``ishmemx_sync_all_work_group``.

.. In a multithreaded Intel® SHMEM program, only the calling thread is blocked.

In contrast with the ``ishmem_sync_all`` routine, ``ishmemx_sync_all_work_group`` allows for the device threads within **group** to cooperate towards the sync operation.
This may be more performant; for example, when ``ishmem_sync_all``
requires `all` device threads in the kernel to invoke RMA operations.
``ishmemx_sync_all_work_group`` only ensures completion and visibility of
previously issued memory stores and does not ensure completion of remote
memory updates issued via ``ishmem`` routines.
All threads in **group** must call the routine with identical arguments.

^^^^^^^^^^^^^^^^
ISHMEM_TEAM_SYNC
^^^^^^^^^^^^^^^^

Registers the arrival of a PE at a synchronization point and suspends execution
until all other PEs in a given ``ishmem`` team arrive at this synchronization
point.

.. cpp:function:: int ishmem_team_sync(ishmem_team_t team)

  :param team: The team over which to perform the operation.
  :returns: Zero on successful local completion. Nonzero otherwise.

Callable from the **host** and the **device**.

**Description:**
``ishmem_team_sync`` is a collective synchronization routine over an existing
``ishmem`` team.
The routine registers the arrival of a PE at a synchronization point in the
program.
This is a fast mechanism for synchronizing all PEs that participate in this
collective call.
The routine blocks the calling PE until all PEs in the specified **team** have
called ``ishmem_team_sync``.

.. In a multithreaded Intel® SHMEM program, only the calling thread is blocked.

All PEs in the provided **team** must participate in the sync operation.
If **team** compares equal to ``ISHMEM_TEAM_INVALID`` or is otherwise invalid,
the behavior is undefined.
If a PE not in the team calls ``ishmem_team_sync``, the behavior is undefined.

In contrast with the ``ishmem_barrier_all`` routine, ``ishmem_team_sync`` only
ensures completion and visibility of previously issued memory stores and does
not ensure completion of remote memory updates issued via ``ishmem`` routines.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_TEAM_SYNC_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Registers the arrival of a PE at a synchronization point and suspends
execution until all other PEs arrive at the synchronization point.

.. cpp:function:: template<typename Group> void ishmemx_team_sync_work_group(ishmem_team_t team, const Group& group)

  :param team: The team over which to perform the operation.
  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the barrier operation.

Callable from the **device**.

**Description:**
This routine blocks the calling PE until all PEs in **team** have called
``ishmemx_team_sync_work_group``.

.. In a multithreaded Intel® SHMEM program, only the calling thread is blocked.

In contrast with the ``ishmem_team_sync`` routine,
``ishmemx_team_sync_work_group`` allows for the device threads within **group**
to cooperate towards the sync operation.
This may be more performant; for example, when ``ishmem_team_sync`` requires
`all` device threads in the kernel to invoke RMA operations.
All PEs in the provided **team** must participate in the sync operation.
If **team** compares equal to ``ISHMEM_TEAM_INVALID`` or is otherwise invalid,
the behavior is undefined.
``ishmemx_team_sync_work_group`` only ensures completion and visibility of
previously issued memory stores and does not ensure completion of remote memory
updates issued via ``ishmem`` routines.
All threads in **group** must call the routine with identical arguments.


^^^^^^^^^^^^^^^
ISHMEM_ALLTOALL
^^^^^^^^^^^^^^^

Exchanges a fixed amount of contiguous data blocks between all pairs of PEs
participating in the collective routine.

.. cpp:function:: int ishmem_TYPENAME_alltoall(TYPE* dest, const TYPE* source, size_t nelems)

.. cpp:function:: int ishmem_TYPENAME_alltoall(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nelems)

.. cpp:function:: int ishmem_alltoallmem(void* dest, const void* source, size_t nelems)

.. cpp:function:: int ishmem_alltoallmem(ishmem_team_t team, void* dest, const void* source, size_t nelems)

   :param dest: Symmetric address of a data object large enough to receive the combined total of **nelems** elements from each PE. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
   :param source: Symmetric address of a data object that contains **nelems** elements of data for each PE, ordered according to destination PE. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
   :param nelems: The number of elements to exchange for each PE. For ``ishmem_alltoallmem``, elements are bytes.
   :param team: A valid ``ishmem`` team handle to a team.
   :returns:  Zero on successful local completion; otherwise, nonzero.

Callable from the **host** and **device**.

**Description:**
The ``ishmem_alltoall`` routines are collective routines. Each PE
participating in the operation exchanges **nelems** data elements
with all other PEs participating in the operation.
The size of a data element is 8 bits for ``ishmem_alltoallmem``.

The data being sent and received are stored in a contiguous symmetric data
object.
The total size of each PE's **source** object and **dest** object is **nelems**
times the size of an element times `N`, where `N` equals the number of PEs
participating in the operation.
The **source** object contains `N` blocks of data (where the size of each block
is defined by **nelems**) and each block of data is sent to a different PE.

The same **dest** and **source** arrays, and same value for **nelems** must be
passed by all PEs that participate in the collective.

.. FIXME: TEAMS

.. Given a PE `i` that is the `k`:sup:`th` PE participating in the operation and a
.. PE `j` that is the `l`:sup:`th` PE participating in the operation, PE `i` sends
.. the `l`:sup:`th` block of its **source** object to the `k`:sup:`th` block of
.. the **dest** object of PE `j`.

Given a PE `i` that is the `i`:sup:`th` PE participating in the operation and a
PE `j` that is the `j`:sup:`th` PE participating in the operation, PE `i` sends
the `j`:sup:`th` block of its **source** object to the `i`:sup:`th` block of
the **dest** object of PE `j`.

If no **team** argument is passed to ``ishmem_alltoall`` or
``ishmem_alltoallmem``, all PEs in the world team must participate in the
collective.
Collective routines that accept a **team** argument operate over all PEs in the
provided team.
All PEs in the provided team must participate in the collective.
If **team** compares equal to ``ISHMEM_TEAM_INVALID`` or is otherwise invalid,
the behavior is undefined.

Before any PE calls a ``ishmem_alltoall`` routine, the following conditions must
be ensured:

#. The **dest** data object on all PEs in the **team** is ready to accept the
   ``ishmem_alltoall`` data.
#. The **source** data object on all PEs in the **team** is ready to send.

Otherwise, the behavior is undefined.

Upon return from a ``ishmem_alltoall`` routine, the following is true for
the local PE:

#. Its **dest** symmetric data object is completely updated.
#. The data has been copied out of the **source** data object.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_ALLTOALL_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exchanges a fixed amount of contiguous data blocks between all pairs of PEs
participating in the collective routine.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_alltoall_work_group(TYPE* dest, const TYPE* source, size_t nelems, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_alltoall_work_group(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nelems, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_alltoallmem_work_group(void* dest, const void* source, size_t nelems, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_alltoallmem_work_group(ishmem_team_t team, void* dest, const void* source, size_t nelems, const Group& group)

   :param dest: Symmetric address of a data object large enough to receive the combined total of **nelems** elements from each PE. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
   :param source: Symmetric address of a data object that contains **nelems** elements of data for each PE, ordered according to destination PE. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
   :param nelems: The number of elements to exchange for each PE. For ``ishmem_alltoallmem``, elements are bytes.
   :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the barrier operation.
   :param team: A valid ``ishmem`` team handle to a team.
   :returns:  Zero on successful local completion; otherwise, nonzero.

Callable from the **device**.

**Description:**
The ``ishmemx_alltoall_work_group`` routines have similar semantics and
requirements as the ``ishmem_alltoall`` routines.
In contrast with the ``ishmem_alltoall`` routines,
``ishmemx_alltoall_work_group`` allows for the device threads within **group**
to cooperate towards the all-to-all operation.
This may be more performant; for example, when ``ishmem_alltoall``
requires `all` device threads in the kernel to invoke RMA operations.
This routine blocks the calling PE until all PEs in the team have called
``ishmemx_alltoall_work_group``.
If no **team** argument is passed to ``ishmemx_alltoall_work_group`` or
``ishmemx_alltoallmem_work_group``, all PEs in the world team must participate
in the collective.
Collective routines that accept a **team** argument operate over all PEs in the
provided team.
All PEs in the provided team must participate in the collective.
If **team** compares equal to ``ISHMEM_TEAM_INVALID`` or is otherwise invalid,
the behavior is undefined.
``ishmemx_alltoall_work_group`` only ensures completion and visibility of
previously issued memory stores and does not ensure completion of remote memory
updates issued via ``ishmem`` routines.
All threads in **group** must call the routine with identical arguments.

^^^^^^^^^^^^^^^^
ISHMEM_BROADCAST
^^^^^^^^^^^^^^^^

Broadcasts a block of data from one PE to one or more destination PEs.

Below, TYPE is one of the standard RMA types and has a corresponding TYPENAME
specified by Table :ref:`Standard RMA Types<stdrmatypes>`.

.. cpp:function:: int ishmem_TYPENAME_broadcast(TYPE* dest, const TYPE* source, size_t nelems, int PE_root)

.. cpp:function:: int ishmem_TYPENAME_broadcast(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nelems, int PE_root)

.. cpp:function:: int ishmem_broadcastmem(void* dest, const void* source, size_t nelems, int PE_root)

.. cpp:function:: int ishmem_broadcastmem(ishmem_team_t team, void* dest, const void* source, size_t nelems, int PE_root)

   :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`. 
   :param source: Symmetric address of the source data object. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
   :param nelems: The number of elements in the **source** and **dest** arrays. For ``ishmem_broadcastmem``, elements are bytes.
   :param PE_root: The PE from which the data is copied.
   :param team: A valid ``ishmem`` team handle to a team.
   :returns:  Zero on successful local completion; otherwise, nonzero.

Callable from the **host** and **device**.

**Description:**
The broadcast routines are collective routines across all PEs in a valid
``ishmem`` team.
They copy the **source** data object on the PE specified by **PE_root** to
the **dest** data object on the PEs participating in the collective
operation.
The same **dest** and **source** data objects and the same value of
**PE_root** must be passed by all PEs participating in the collective
operation.

For broadcasts:

* The **dest** object is updated on all PEs in the ``ishmem`` team.

* All PEs in the **team** must participate in the operation.

* If **team** compares equal to ``ISHMEM_TEAM_INVALID`` or is otherwise
  invalid, the behavior is undefined.

* PE numbering is relative to the team. The specified root PE must be a valid
  PE number for the team, between :math:`0` and :math:`N-1`, where :math:`N` is the
  size of the team.

* The values of argument **PE_root** must be the same value on all PEs in the
  **team**.

Before any PE calls a broadcast routine, the following conditions must be
ensured:

* The **dest** array on all PEs in the **team** is ready to accept the
  broadcast data.

Otherwise, the behavior is undefined.

Upon return from a broadcast routine, the following are true for the local PE:

* The **dest** data object is updated on all PEs in the **team**.

* The **source** data object may be safely reused.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_BROADCAST_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Broadcasts a block of data from one PE to one or more destination PEs.

Below, TYPE is one of the standard RMA types and has a corresponding TYPENAME
specified by Table :ref:`Standard RMA Types<stdrmatypes>`.

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_broadcast_work_group(TYPE* dest, const TYPE* source, size_t nelems, int PE_root, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_broadcast_work_group(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nelems, int PE_root, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_broadcastmem_work_group(void* dest, const void* source, size_t nelems, int PE_root, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_broadcastmem_work_group(ishmem_team_t team, void* dest, const void* source, size_t nelems, int PE_root, const Group& group)

   :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`. 
   :param source: Symmetric address of the source data object. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
   :param nelems: The number of elements in the **source** and **dest** arrays. For ``ishmemx_broadcastmem_work_group``, elements are bytes.
   :param PE_root: The PE from which the data is copied.
   :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the barrier operation.
   :param team: A valid ``ishmem`` team handle to a team.
   :returns:  Zero on successful local completion; otherwise, nonzero.

Callable from the **device**.

**Description:**
The ``ishmemx_broadcast_work_group`` and
``ishmemx_broadcastmem_work_group`` routines have similar semantics and
requirements as the ``ishmem_broadcast`` routines.
In contrast with the ``ishmem_broadcast`` routines,
``ishmemx_broadcast_work_group`` and ``ishmemx_broadcastmem_work_group``
allow for the device threads within **group** to cooperate towards the broadcast operation.
This routine blocks the calling PE until all PEs in the team have called
``ishmemx_broadcast_work_group``.
If no **team** argument is passed to ``ishmemx_broadcast_work_group`` or
``ishmemx_broadcastmem_work_group``, all PEs in the world team must participate
in the broadcast operation.
Broadcast routines that accept a **team** argument operate over all PEs in the
provided team.
All PEs in the provided team must participate in the broadcast.
If **team** compares equal to ``ISHMEM_TEAM_INVALID`` or is otherwise invalid,
the behavior is undefined.
``ishmemx_broadcast_work_group`` only ensures completion and visibility of
previously issued memory stores and does not ensure completion of remote memory
updates issued via ``ishmem`` routines.
All threads in **group** must call the routine with identical arguments.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEM_COLLECT, ISHMEM_FCOLLECT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Concatenates blocks of data from multiple PEs to an array in every PE
participating in the collective routine.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: int ishmem_TYPENAME_collect(TYPE* dest, const TYPE* source, size_t nelems)

.. cpp:function:: int ishmem_TYPENAME_collect(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nelems)

.. cpp:function:: int ishmem_TYPENAME_fcollect(TYPE* dest, const TYPE* source, size_t nelems)

.. cpp:function:: int ishmem_TYPENAME_fcollect(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nelems)

.. cpp:function:: int ishmem_collectmem(void* dest, const void* source, size_t nelems)

.. cpp:function:: int ishmem_collectmem(ishmem_team_t team, void* dest, const void* source, size_t nelems)

.. cpp:function:: int ishmem_fcollectmem(void* dest, const void* source, size_t nelems)

.. cpp:function:: int ishmem_fcollectmem(ishmem_team_t team, void* dest, const void* source, size_t nelems)

  :param dest: Symmetric address of an array large enough to accept the concatenation of the **source** arrays on all participating PEs. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param source: Symmetric address of the **source** data object. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param nelems: The number of elements in **source** array. For ``ishmem_collectmem`` and ``ishmem_fcollectmem``, elements are bytes.
  :param team: A valid ``ishmem`` team handle to a team.
  :returns: Zero on successful local completion. Nonzero otherwise.

Callable from the **host** and **device**.

**Description:**
The ``ishmem_collect`` and ``ishmem_fcollect`` routines perform a collective
operation to concatenate **nelems** data items from the **source** array into
the **dest** array, over all PEs in a valid ``ishmem`` team in processor number
order.

.. For a team, the data from PE number 0 in the team is first, then the contribution from PE 1 in the team, and so on.

The collected result is written to the **dest** array for all PEs in the team.
The same **dest** and **source** arrays must be passed by all PEs that
participate in the operation.

The ``ishmem_fcollect`` routines require that **nelems** be the same value
in all participating PEs, while the ``ishmem_collect`` routines allow
**nelems** to vary from PE to PE.

If no **team** argument is passed to either ``ishmem_collect`` or
``ishmem_fcollect``, then all PEs in the world team must participate in the
collective.
Collect and fcollect routines that accept a **team** argument operate over all
PEs in the provided team.
All PEs in the provided team must participate in the collective.
If **team** compares equal to ``ISHMEM_TEAM_INVALID`` or is otherwise invalid,
the behavior is undefined.

Upon return from a collective routine, the following are true for the local
PE:

* The **dest** array is updated and the **source** array may be safely
  reused. 


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_COLLECT_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_FCOLLECT_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Concatenates blocks of data from multiple PEs to an array in every PE
participating in the collective routine.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_collect_work_group(TYPE* dest, const TYPE* source, size_t nelems, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_collect_work_group(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nelems, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_fcollect_work_group(TYPE* dest, const TYPE* source, size_t nelems, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_fcollect_work_group(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nelems, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_collectmem_work_group(void* dest, const void* source, size_t nelems, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_collectmem_work_group(ishmem_team_t team, void* dest, const void* source, size_t nelems, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_fcollectmem_work_group(void* dest, const void* source, size_t nelems, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_fcollectmem_work_group(ishmem_team_t team, void* dest, const void* source, size_t nelems, const Group& group)

  :param dest: Symmetric address of an array large enough to accept the concatenation of the **source** arrays on all participating PEs. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param source: Symmetric address of the **source** data object. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param nelems: The number of elements in **source** array. For ``ishmemx_collectmem_work_group`` and ``ishmemx_fcollectmem_work_group``, elements are bytes.
  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the barrier operation.
  :param team: A valid ``ishmem`` team handle to a team.
  :returns: Zero on successful local completion. Nonzero otherwise.

Callable from the **device**.

**Description:**
The ``ishmemx_collect_work_group`` and ``ishmemx_fcollect_work_group`` routines
have similar semantics and requirements as the ``ishmem_collect`` and
``ishmem_fcollect`` routines, respectively.
In contrast with the ``ishmem_collect`` and ``ishmem_fcollect`` routines,
``ishmemx_collect_work_group`` and ``ishmemx_fcollect_work_group`` allow for
the device threads within **group** to cooperate towards the operation.
This may be more performant; for example, when ``ishmem_collect``
requires `all` device threads in the kernel to invoke RMA operations.
The ``ishmemx_collect_work_group`` and ``ishmemx_fcollect_work_group`` routines
block the calling PE until all PEs in the team have called
``ishmemx_collect_work_group`` or ``ishmemx_fcollect_work_group``,
respectively.
If no **team** argument is passed to ``ishmemx_collect_work_group``,
``ishmemx_fcollect_work_group``, or ``ishmemx_fcollectmem_work_group``, or
``ishmemx_fcollectmem_work_group``, then all PEs in the world team must
participate in the collective.
Collect routines that accept a **team** argument operate over all PEs in the
provided team.
All PEs in the provided team must participate in the collective.
If **team** compares equal to ``ISHMEM_TEAM_INVALID`` or is otherwise invalid,
the behavior is undefined.
``ishmemx_collect_work_group`` and ``ishmemx_fcollect_work_group`` only ensure
completion and visibility of previously issued memory stores and does not
ensure completion of remote memory updates issued via ``ishmem`` routines.
All threads in **group** must call the routine with identical arguments.

.. _reductions:

^^^^^^^^^^^^^
ISHMEM_REDUCE
^^^^^^^^^^^^^

.. _reducetypes:

**Reduction Types, Names, and Supporting Operations:**

==================   ==========   ===================================
TYPE                 TYPENAME     Operations Supporting TYPE
==================   ==========   ===================================
char                 char                        MAX, MIN,  SUM, PROD
signed char          schar                       MAX, MIN,  SUM, PROD
short                short                       MAX, MIN,  SUM, PROD
int                  int                         MAX, MIN,  SUM, PROD
long                 long                        MAX, MIN,  SUM, PROD
long long            longlong                    MAX, MIN,  SUM, PROD
ptrdiff_t            ptrdiff                     MAX, MIN,  SUM, PROD
unsigned char        uchar        AND, OR, XOR,  MAX, MIN,  SUM, PROD
unsigned short       ushort       AND, OR, XOR,  MAX, MIN,  SUM, PROD
unsigned int         uint         AND, OR, XOR,  MAX, MIN,  SUM, PROD
unsigned long        ulong        AND, OR, XOR,  MAX, MIN,  SUM, PROD
unsigned long long   ulonglong    AND, OR, XOR,  MAX, MIN,  SUM, PROD
int8_t               int8         AND, OR, XOR,  MAX, MIN,  SUM, PROD
int16_t              int16        AND, OR, XOR,  MAX, MIN,  SUM, PROD
int32_t              int32        AND, OR, XOR,  MAX, MIN,  SUM, PROD
int64_t              int64        AND, OR, XOR,  MAX, MIN,  SUM, PROD
uint8_t              uint8        AND, OR, XOR,  MAX, MIN,  SUM, PROD
uint16_t             uint16       AND, OR, XOR,  MAX, MIN,  SUM, PROD
uint32_t             uint32       AND, OR, XOR,  MAX, MIN,  SUM, PROD
uint64_t             uint64       AND, OR, XOR,  MAX, MIN,  SUM, PROD
size_t               size         AND, OR, XOR,  MAX, MIN,  SUM, PROD
float                float                       MAX, MIN,  SUM, PROD
double               double                      MAX, MIN,  SUM, PROD
==================   ==========   ===================================

.. long double       longdouble                  MAX, MIN,  SUM, PROD
.. double _Complex   complexd                               SUM, PROD
.. float  _Complex   complexf                               SUM, PROD

The following functions perform reduction operations across all PEs in a given
``ishmem`` team.

In the functions below, TYPE is one of the reduction types and has a
corresponding TYPENAME specified by Table :ref:`Reduction Types, Names, and Supporting Operations<reducetypes>`.

.. cpp:function:: int ishmem_TYPENAME_and_reduce(TYPE* dest, const TYPE* source, size_t nreduce)

.. cpp:function:: int ishmem_TYPENAME_and_reduce(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce)

.. cpp:function:: int ishmem_TYPENAME_or_reduce(TYPE* dest, const TYPE* source, size_t nreduce)

.. cpp:function:: int ishmem_TYPENAME_or_reduce(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce)

.. cpp:function:: int ishmem_TYPENAME_xor_reduce(TYPE* dest, const TYPE* source, size_t nreduce)

.. cpp:function:: int ishmem_TYPENAME_xor_reduce(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce)

.. cpp:function:: int ishmem_TYPENAME_max_reduce(TYPE* dest, const TYPE* source, size_t nreduce)

.. cpp:function:: int ishmem_TYPENAME_max_reduce(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce)

.. cpp:function:: int ishmem_TYPENAME_min_reduce(TYPE* dest, const TYPE* source, size_t nreduce)

.. cpp:function:: int ishmem_TYPENAME_min_reduce(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce)

.. cpp:function:: int ishmem_TYPENAME_sum_reduce(TYPE* dest, const TYPE* source, size_t nreduce)

.. cpp:function:: int ishmem_TYPENAME_sum_reduce(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce)

.. cpp:function:: int ishmem_TYPENAME_prod_reduce(TYPE* dest, const TYPE* source, size_t nreduce)

.. cpp:function:: int ishmem_TYPENAME_prod_reduce(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce)

  :param dest: Symmetric address of an array, of length **nreduce** elements, to receive the result of the reduction routines. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Reduction Types<reducetypes>`.
  :param source: Symmetric address of an array, of length **nreduce** elements, that contains one element for each separate reduction routine. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Reduction Types<reducetypes>`.
  :param nreduce: The number of elements in the **dest** and **source** arrays. **nreduce** must be of type **size_t** and have the same value across all PEs.
  :param team: A valid ``ishmem`` team handle to a team.
  :returns: Zero on successful local completion. Nonzero otherwise.

Callable from the **host** and **device**.

**Description:**
``ishmem`` reduction routines are collective routines over all PEs in an
existing ``ishmem`` team that compute one or more reductions across symmetric
arrays.
A reduction performs an associative binary routine across a set of values.

The **nreduce** argument determines the number of separate reductions to
perform.
The **source** array on all PEs participating in the reduction provides one
element for each reduction.
The results of the reductions are placed in the **dest** array on all PEs
participating in the reduction.

The **source** and **dest** arguments must either be the same symmetric
address, or two different symmetric addresses corresponding to buffers that
do not overlap in memory. That is, they must be completely overlapping or
completely disjoint.

If no **team** argument is passed to a reduction routine, all PEs in the world
team must participate in the reduction.
Reduction routines that accept a **team** argument operate over all PEs in the
provided team.
All PEs in the provided team must participate in the collective.
If **team** compares equal to ``ISHMEM_TEAM_INVALID`` or is otherwise invalid,
the behavior is undefined.

Before any PE calls a reduction routine, the following conditions must be
ensured:

* The **dest** array on all PEs participating in the reduction is ready to
  accept the results of the reduction.

Otherwise, the behavior is undefined.

Upon return from a reduction routine, the following are true for the local
PE:

* The **dest** array is updated and the **source** array may be safely
  reused.

^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_REDUCE_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^

The following functions perform reduction operations across all PEs in a given
``ishmem`` team.

In the functions below, TYPE is one of the reduction types and has a
corresponding TYPENAME specified by Table :ref:`Reduction Types, Names, and Supporting Operations<reducetypes>`.

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_and_reduce_work_group(TYPE* dest, const TYPE* source, size_t nreduce, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_and_reduce_work_group(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_or_reduce_work_group(TYPE* dest, const TYPE* source, size_t nreduce, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_or_reduce_work_group(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_xor_reduce_work_group(TYPE* dest, const TYPE* source, size_t nreduce, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_xor_reduce_work_group(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_max_reduce_work_group(TYPE* dest, const TYPE* source, size_t nreduce, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_max_reduce_work_group(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_min_reduce_work_group(TYPE* dest, const TYPE* source, size_t nreduce, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_min_reduce_work_group(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_sum_reduce_work_group(TYPE* dest, const TYPE* source, size_t nreduce, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_sum_reduce_work_group(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_prod_reduce_work_group(TYPE* dest, const TYPE* source, size_t nreduce, const Group& group)

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_prod_reduce_work_group(ishmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce, const Group& group)

  :param dest: Symmetric address of an array, of length **nreduce** elements, to receive the result of the reduction routines. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Reduction Types<reducetypes>`.
  :param source: Symmetric address of an array, of length **nreduce** elements, that contains one element for each separate reduction routine. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Reduction Types<reducetypes>`.
  :param nreduce: The number of elements in the **dest** and **source** arrays. **nreduce** must be of type **size_t** and have the same value across all PEs.
  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the barrier operation.
  :param team: A valid ``ishmem`` team handle to a team.
  :returns: Zero on successful local completion. Nonzero otherwise.

Callable from the **device**.

**Description:**
The ``ishmemx_reduce_work_group`` routines have similar semantics and
requirements as the ``ishmem_reduce`` routines.
In contrast with the ``ishmem_reduce`` routines,
``ishmemx_reduce_work_group`` allows for the device threads within
**group** to cooperate towards the reduction operation.
This may be more performant; for example, when ``ishmem_reduce``
requires `all` device threads in the kernel to invoke RMA operations.
This routine blocks the calling PE until all PEs in the team have called
``ishmemx_reduce_work_group``.
If no **team** argument is passed to a reduction routine, all PEs in the world
team must participate in the collective.
Reduction routines that accept a **team** argument operate over all PEs in the
provided team.
All PEs in the provided team must participate in the reduction.
If **team** compares equal to ``ISHMEM_TEAM_INVALID`` or is otherwise invalid,
the behavior is undefined.
``ishmemx_reduce_work_group`` only ensures completion and visibility of
previously issued memory stores and does not ensure completion of remote
memory updates issued via ``ishmem`` routines.
All threads in **group** must call the routine with identical arguments.

.. important:: For the reduction operations ``sum`` and ``prod``, the order of
   reduction may not be the same across all participating PEs, so the results
   for floating point datatypes may differ slightly. This is because floating
   addition and multiplication are not associative operations.
