.. _team:

------------------------
Team Management Routines
------------------------

The PEs in an ``ishmem`` program communicate using either point-to-point
routines, such as RMA and AMO routines, that specify the PE number of the
target PE, or collective routines that operate over a set of PEs.
Intel® SHMEM teams allow programs to group a set of PEs for communication.
Team-based collective operations include all PEs in a valid team.
Point-to-point communication can make use of team-relative PE numbering via PE
number translation.

.. FIXME : add "through team-based contexts" after contexts are supported

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Predefined and Application-Defined Teams
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An Intel® SHMEM team may be predefined (i.e., provided by the Intel® SHMEM
library) or defined by the Intel® SHMEM application.
An application-defined team is created by `splitting` a parent team into one or
more new teams---each with some subset of PEs of the parent team---via one of
the ``ishmem_team_split_*`` routines.

All predefined teams are valid for the duration of the ``ishmem`` portion of an
application.
Any team successfully created by a ``ishmem_team_split_*`` routine is valid
until it is destroyed.
All valid teams have at least one member.

^^^^^^^^^^^^
Team Handles
^^^^^^^^^^^^

A `team handle` is an opaque object with type **ishmem_team_t** that is used to
reference a team.
Team handles are not remotely accessible objects.
The predefined teams may be accessed via the team handles listed in
Section :ref:`Library Handles<library_handles>`.

Intel® SHMEM communication routines that do not accept a team handle argument
operate on the world team, which may be accessed through the
``ISHMEM_TEAM_WORLD`` handle.
The world team encompasses the set of all PEs in the ``ishmem`` program, and a
given PE's number in the world team is equal to the value returned by
``ishmem_my_pe``.

A team handle may be initialized to or assigned the value
``ISHMEM_TEAM_INVALID`` to indicate that handle does not reference a valid
team.
When managed in this way, applications can use an equality comparison to test
whether a given team handle references a valid team.


^^^^^^^^^^^^^
Thread Safety
^^^^^^^^^^^^^

When it is allowed by the threading model provided by the Intel® SHMEM
library, a team may be used concurrently in non-collective operations
(e.g., :ref:`ishmem_team_my_pe<ishmem_team_my_pe>`) by multiple threads within the
PE where it was created.
A team may not be used concurrently by multiple threads in the same PE for
collective operations. However, multiple collective operations on different
teams may be performed in parallel.

^^^^^^^^^^^^^^^^^^^
Collective Ordering
^^^^^^^^^^^^^^^^^^^

In Intel® SHMEM, a team object encapsulates resources used to communicate
between PEs in collective operations.
When calling multiple subsequent collective operations on a team, the
collective operations---along with any relevant team based resources---are
matched across the PEs in the team based on ordering of collective routine
calls.
It is the responsibility of the user to ensure that team-based collectives
occur in the same program order across all PEs in a team.

For a full discussion of collective semantics, see Section
:ref:`Collectives<collectives>`.

^^^^^^^^^^^^^
Team Creation
^^^^^^^^^^^^^

Team creation is a collective operation on the parent team object that occurs
on the host.
New teams result from an ``ishmem_team_split_*`` routine, which takes a parent
team and other arguments and produces new teams that contain a subset of the
PEs that are members of the parent team.
All PEs in a parent team must participate in a split operation to create new
teams.
If a PE from the parent team is not a member of any resulting new teams, it
will receive a value of ``ISHMEM_TEAM_INVALID`` as the value for the new team
handle.

Teams that are created by a ``ishmem_team_split_*`` routine may be provided a
configuration argument that specifies attributes of each new team.
This configuration argument is of type **ishmem_team_config_t**, which is
detailed further in Section :ref:`ishmem_team_config_t<ishmem_team_config_t>`.

PEs in a newly created team are consecutively numbered starting with PE number
0.
PEs are ordered by their PE number in the parent team.
Team-relative PE numbers can be used for point-to-point operations by using the
translation routine ``ishmem_team_translate_pe``.

.. FIXME : add "through team-based contexts" above after/if contexts supported

Split operations are collective and are subject to the constraints on
team-based collectives specified in Section :ref:`Collectives<collectives>`.
In particular, in multithreaded executions, threads at a given PE must not
perform simultaneous split operations on the same parent team.
Team creation operations are matched across participating PEs based
on the order in which they are performed.
Thus, team creation events must also occur in the same order on all PEs in the
parent team.

Upon completion of a team creation operation, the parent and any resulting
child teams will be immediately usable for any team-based operations, including
creating new child teams, without any intervening synchronization.


.. _ishmem_team_my_pe:

^^^^^^^^^^^^^^^^^
ISHMEM_TEAM_MY_PE
^^^^^^^^^^^^^^^^^

Returns the number of the calling PE within a specified team.

.. cpp:function:: int ishmem_team_my_pe(ishmem_team_t team)

  :param team: An ``ishmem`` team handle.
  :returns: The number of the calling PE within the specified team, or the value ``-1`` if the team handle compares equal to ``ISHMEM_TEAM_INVALID``.

Callable from the **host** and **device**.

**Description:**
When **team** specifies a valid team, the ``ishmem_team_my_pe`` routine returns
the number of the calling PE within the specified team.
The number is an integer between **0** and **N-1** for a team containing **N**
PEs.
Each member of the team has a unique number.

If **team** compares equal to ``ISHMEM_TEAM_INVALID``, then the value **-1** is
returned.
If **team** is otherwise invalid, the behavior is undefined.

**Notes:**
For the world team, this routine will return the same value as
``ishmem_my_pe``.

^^^^^^^^^^^^^^^^^
ISHMEM_TEAM_N_PES
^^^^^^^^^^^^^^^^^

Returns the number of PEs in a specified team.

.. cpp:function:: int ishmem_team_n_pes(ishmem_team_t team)

  :param team: An ``ishmem`` team handle.
  :returns: The number of PEs in the specified team, or the value **-1** if the team handle compares equal to ``ISHMEM_TEAM_INVALID``.

Callable from the **host** and **device**.

**Description:**
When **team** specifies a valid team, the ``ishmem_team_n_pes`` routine returns
the number of PEs in the team.
This will always be a value between **1** and **N**, where **N** is the total
number of PEs running in the ``ishmem`` program.

If **team** compares equal to ``ISHMEM_TEAM_INVALID``, then the value **-1** is
returned.
If **team** is otherwise invalid, the behavior is undefined.

**Notes:**
For the world team, this routine will return the same value as
``ishmem_n_pes``.

.. _ishmem_team_config_t:

^^^^^^^^^^^^^^^^^^^^
ISHMEM_TEAM_CONFIG_T
^^^^^^^^^^^^^^^^^^^^

.. important:: Intel® SHMEM |release| ignores ``ishmem_team_config_t``, because
   communication contexts are not yet supported.

A structure type representing team configuration arguments.

.. cpp:struct:: ishmem_team_config_t

  .. c:var:: int num_contexts

**Description:**
A team configuration object is provided as an argument to
``ishmem_team_split_*`` routines.
It specifies the requested capabilities of the team to be created.

The **num_contexts** member specifies the total number of simultaneously
existing contexts that the program requests to create from this team.  These
contexts may be created in any number of threads.
Successful creation of a team configured with **num_contexts** of **N** means
that the implementation will make a best effort to reserve enough resources to
support **N** contexts created from the team in existence at any given time.

.. It is not a guarantee that **N** calls to ``ishmem_team_create_ctx`` will succeed.
.. See Section~\ref{sec:ctx} for more on communication contexts and
.. Section~\ref{subsec:shmem_team_create_ctx} for team-based context creation.
.. FIXME above

When using the configuration structure to create teams, a mask parameter
controls which fields may be accessed by the Intel® SHMEM library.
Any configuration parameter value that is not indicated in the mask will be
ignored, and the default value will be used instead.
Therefore, a program must set only the fields for which it does not want the
default value.

A configuration mask is created through a bitwise OR operation of the following library constants.
A configuration mask value of **0** indicates that the team should be created with the default values for all configuration parameters.

============================     =======================================================
Constant Name:                   Constant Description:
============================     =======================================================
``ISHMEM_TEAM_NUM_CONTEXTS``     The team should be created using the value of the
                                 **num_contexts** member of the configuration parameter
                                 **config** as a requirement.
============================     =======================================================

The default values for configuration parameters are:

========================    =======================================================
Parameter Default Value:    Parameter Value Description:
========================    =======================================================
**num_contexts = 0**        By default, no contexts can be created on a new team
========================    =======================================================

^^^^^^^^^^^^^^^^^^^^^^
ISHMEM_TEAM_GET_CONFIG
^^^^^^^^^^^^^^^^^^^^^^

Return the configuration parameters of a given team.

.. cpp:function:: int ishmem_team_get_config(ishmem_team_t team, long config_mask, ishmem_team_config_t* config)

  :param team: An ``ishmem`` team handle.
  :param config_mask: The bitwise mask representing the set of configuration parameters to fetch from the given team.
  :param config: A pointer to the configuration parameters for the given team.
  :returns: If **team** does not compare equal to ``ISHMEM_TEAM_INVALID``, then ``ishmem_team_get_config`` returns **0**; otherwise, it returns nonzero.

**Description:**
``ishmem_team_get_config`` returns through the **config** argument the
configuration parameters as described by the mask, which were assigned
according to input configuration parameters when the team was created.

If **team** compares equal to ``ISHMEM_TEAM_INVALID``, then no operation is
performed.
If **team** is otherwise invalid, the behavior is undefined.

^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEM_TEAM_TRANSLATE_PE
^^^^^^^^^^^^^^^^^^^^^^^^

Translate a given PE number from one team to the corresponding PE number in
another team.

.. cpp:function:: int ishmem_team_translate_pe(ishmem_team_t src_team, int src_pe, ishmem_team_t dest_team)

  :param src_team: An ``ishmem`` team handle.
  :param src_pe: A PE number in **src_team**.
  :param dest_team: An ``ishmem`` team handle.
  :returns: The specified PE's number in the **dest_team**, or a value of **-1** if any team handle arguments are invalid or the **src_pe** is not in both the source and destination teams.

Callable from the **host** and **device**.

**Description:**
The ``ishmem_team_translate_pe`` routine will translate a given PE number in
one team into the corresponding PE number in another team.
Specifically, given the **src_pe** in **src_team**, this routine returns that
PE's number in **dest_team**.
If **src_pe** is not a member of both **src_team** and **dest_team**, a value
of **-1** is returned.

If at least one of **src_team** and **dest_team** compares equal to
``ISHMEM_TEAM_INVALID``, then **-1** is returned.
If either of the **src_team** or **dest_team** handles are otherwise invalid,
the behavior is undefined.

**Notes:**
If ``ISHMEM_TEAM_WORLD`` is provided as the **dest_team** parameter, this
routine acts as a global PE number translator and will return the corresponding
``ISHMEM_TEAM_WORLD`` number.

^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEM_TEAM_SPLIT_STRIDED
^^^^^^^^^^^^^^^^^^^^^^^^^

Create a new Intel® SHMEM team from a subset of the existing parent team PEs,
where the subset is defined by the PE triplet (**start**, **stride**, and
**size**) supplied to the routine.

.. cpp:function:: int ishmem_team_split_strided(ishmem_team_t parent_team, int start, int stride, int size, const ishmem_team_config_t* config, long config_mask, ishmem_team_t* new_team)

  :param parent_team: An ``ishmem`` team handle.
  :param start: The lowest PE number of the subset of PEs from the parent team that will form the new team.
  :param stride: The stride between team PE numbers in the parent team that comprise the subset of PEs that will form the new team.
  :param size: The number of PEs from the parent team in the subset PEs that will form the new team. **size** must be a positive integer.
  :param config: A pointer to the configuration parameters for the new team.
  :param config_mask: The bitwise mask representing the set of configuration parameters to use **config**.
  :param new_team: An ``ishmem`` team handle. Upon successful creation, it references an ``ishmem`` team that contains the subset of all PEs in the parent team specified by the PE triplet provided.
  :returns: Zero on successful creation of **new_team**; otherwise, nonzero.

Callable from the **host**. 

**Description:**
The ``ishmem_team_split_strided`` routine is a collective routine.
It creates a new ``ishmem`` team from an existing parent team,
where the PE subset of the resulting team is defined by the triplet of arguments
:math:`(start, stride, size)`.
A valid triplet is one such that:

.. math::

   start + stride \cdot i \in \mathbb{Z}_{N-1}
   \;
   \forall
   \;
   i \in \mathbb{Z}_{size-1}

where :math:`\mathbb{Z}` is the set of natural numbers (:math:`0, 1, \dots`),
:math:`N` is the number of PEs in the parent team, :math:`size` is a positive
number indicating the number of PEs in the new team, and :math:`stride` is an integer.
The index :math:`i` specifies the number of the given PE in the new team.
When :math:`stride` is greater than zero, PEs in the new team remain in the
same relative order as in the parent team.
When :math:`stride` is less than zero, PEs in the new team are in *reverse*
relative order with respect to the parent team.
If a :math:`stride` value equal to 0 is passed to
``ishmem_team_split_strided``, then the `size` argument passed must be 1, or
the behavior is undefined.

.. A valid :math:`(start, stride, size)` triplet passed to ``ishmem_team_split_strided``
.. must produce a subset of PEs from the parent team with no duplicate members;
.. otherwise, the triplet is invalid.

This routine must be called by all PEs in the parent team.
All PEs must provide the same values for the PE triplet.
On successful creation of the new team:

#. The **new_team** handle will reference a valid team for the subset of PEs in
   the parent team that are members of the new team.
#. Those PEs in the parent team that are not members of the new team will have
   **new_team** assigned to ``ISHMEM_TEAM_INVALID``.
#. ``ishmem_team_split_strided`` will return zero to all PEs in the parent
   team.

If the new team cannot be created or an invalid PE triplet is provided, then
**new_team** will be assigned the value ``ISHMEM_TEAM_INVALID`` and
``ishmem_team_split_strided`` will return a nonzero value on all PEs in the
parent team.

The **config** argument specifies team configuration parameters, which are
described in Section :ref:`ishmem_team_config_t<ishmem_team_config_t>`.

The **config_mask** argument is a bitwise mask representing the set of
configuration parameters to use from **config**.
A **config_mask** value of **0** indicates that the team should be created with the default values for all configuration parameters.
See Section :ref:`ishmem_team_config_t<ishmem_team_config_t>` for field mask
names and default configuration parameters.

If **parent_team** compares equal to ``ISHMEM_TEAM_INVALID``, then no new team
will be created, **new_team** will be assigned the value
``ISHMEM_TEAM_INVALID``, and ``ishmem_team_split_strided`` will return a
nonzero value.
If **parent_team** is otherwise invalid, the behavior is undefined.

**Notes:**
The ``ishmem_team_split_strided`` operation uses an arbitrary **stride**
argument, whereas the deprecated OpenSHMEM `logPE_stride` argument to the
active set collective operations only permits strides that are a power of two.
Arbitrary strides allow a greater number of PE subsets to be expressed
and can support a broader range of usage models.

See the description of team handles and predefined teams in Section :ref:`Teams
Management Routines<team>` for more information about team handle semantics and
usage.

^^^^^^^^^^^^^^^^^^^^
ISHMEM_TEAM_SPLIT_2D
^^^^^^^^^^^^^^^^^^^^

Create two new teams by splitting an existing parent team into two subsets
based on a 2D Cartesian space defined by the **xrange** argument and a `y`
dimension that is derived from **xrange** and the parent team size.

.. cpp:function:: int ishmem_team_split_2d(ishmem_team_t parent_team, int xrange, const ishmem_team_config_t* xaxis_config, long xaxis_mask, ishmem_team_t* xaxis_team, const ishmem_team_config_t* yaxis_config, long yaxis_mask, ishmem_team_t* yaxis_team)

  :param parent_team: An ``ishmem`` team handle.
  :param xrange: A positive integer representing the number of elements in the first dimension.
  :param xaxis_config: A pointer to the configuration parameters for the new **xaxis_team** object.
  :param xaxis_mask: The bitwise mask representing the set of configuration parameters to use from **xaxis_config**.
  :param xaxis_team: A new team handle representing the subset of all PEs that share the same coordinate along the `y`-axis as the calling PE.
  :param yaxis_config: A pointer to the configuration parameters for the new **yaxis_team**.
  :param yaxis_mask: The bitwise mask representing the set of configuration parameters to use from **yaxis_config**.
  :param yaxis_team: A new team handle representing the subset of all PEs that share the same coordinate along the `x`-axis as the calling PE.
  :returns: Zero on successful creation of all **xaxis_team** and **yaxis_team** objects; otherwise, nonzero.

Callable from the **host**. 

**Description:**
The ``ishmem_team_split_2d`` routine is a collective operation.
It returns two new teams to the calling PE by splitting an existing parent team
into subsets based on a 2D Cartesian space.
The user provides the size of the `x` dimension, which is then used to derive
the size of the `y` dimension based on the size of the parent team.
The size of the `y` dimension will be equal to :math:`\lceil N \div xrange
\rceil`, where :math:`N` is the size of the parent team.
In other words, :math:`xrange \times yrange \geq N`, so that every PE in the
parent team has a unique :math:`(x,y)` location in the 2D Cartesian space.
The resulting **xaxis_team** and **yaxis_team** correspond to the calling PE's
row and column, respectively, in the 2D Cartesian space.

The mapping of PE number to coordinates is :math:`(x, y) = ( pe \mod xrange,
\lfloor pe \div xrange \rfloor )`, where :math:`pe` is the PE number in the
parent team.
For example, if :math:`xrange = 3`, then the first 3 PEs in the
parent team will form the first **xteam**, the second three PEs in the
parent team form the second **xteam**, and so on.

Thus, after the split operation, each of the new **xteam**'s will contain all
PEs that have the same coordinate along the `y`-axis as the calling PE.
Each of the new **yteam**'s will contain all PEs with the same coordinate along
the `x`-axis as the calling PE.

The PEs are numbered in the new teams based on the coordinate of the PE along
the given axis.
As a result, the value returned by ``ishmem_team_my_pe(xteam)`` is the
`x`-coordinate and the value returned by ``ishmem_team_my_pe(yteam)`` is the
`y`-coordinate of the calling PE.

Any valid Intel® SHMEM team can be used as the parent team.
This routine must be called by all PEs in the parent team.
The value of **xrange** must be positive and all PEs in the parent team must
pass the same value for **xrange**.  When **xrange** is greater than the size
of the parent team, ``ishmem_team_split_2d`` behaves as though **xrange** were
equal to the size of the parent team.

The **xaxis_config** and **yaxis_config** arguments specify team configuration
parameters for the `x`- and `y`-axis teams, respectively.
These parameters are described in Section
:ref:`ishmem_team_config_t<ishmem_team_config_t>`.
All PEs that will be in the same resultant team must specify the same
configuration parameters.
The PEs in the parent team `do not` have to all provide the same parameters for
new teams.

The **xaxis_mask** and **yaxis_mask** arguments are a bitwise masks
representing the set of configuration parameters to use from **xaxis_config**
and **yaxis_config**, respectively.
A mask value of **0** indicates that the team should be created with the
default values for all configuration parameters.
See Section :ref:`ishmem_team_config_t<ishmem_team_config_t>` for field mask
names and default configuration parameters.

If **parent_team** compares equal to ``ISHMEM_TEAM_INVALID``, then no new teams
will be created, both **xaxis_team** and **yaxis_team** will be assigned the
value ``ISHMEM_TEAM_INVALID``, and ``ishmem_team_split_2d`` will return a
nonzero value.
If **parent_team** is otherwise invalid, the behavior is undefined.

If any **xaxis_team** or **yaxis_team** on any PE in **parent_team** cannot be
created, then both team handles on all PEs in **parent_team** will be assigned
the value ``ISHMEM_TEAM_INVALID`` and ``ishmem_team_split_2d`` will return a
nonzero value.

**Notes:**
Since the split may result in a 2D space with more points than there are
members of the parent team, there may be a final, incomplete row of the 2D
mapping of the parent team.
This means that the resultant **yteam**'s may vary in size by up to 1 PE, and
that there may be one resultant **xteam** of smaller size than all of the other
**xteam**'s.

The following grid shows the 12 teams that would result from splitting a parent
team of size 10 with **xrange** of 3.
The numbers in the grid cells are the PE numbers in the parent team.
The rows are the **xteam**'s. The columns are the **yteam**'s.

+------------+-------------+-------------+-------------+
|            | yteam, x=0  | yteam, x=1  | yteam, x=2  |
+============+=============+=============+=============+
| xteam, y=0 | 0           | 1           | 2           |
+------------+-------------+-------------+-------------+
| xteam, y=1 | 3           | 4           | 5           |
+------------+-------------+-------------+-------------+
| xteam, y=2 | 6           | 7           | 8           |
+------------+-------------+-------------+-------------+
| xteam, y=3 | 9           |             |             |
+------------+-------------+-------------+-------------+

It would be legal, for example, if PEs 0, 3, 6, 9 specified a different value
for **yaxis_config** than all of the other PEs, as long as the configuration
parameters match for all PEs in each of the new teams.

See the description of team handles and predefined teams in Section :ref:`Team
Management Routines<team>` for more information about team handle semantics and
usage.

^^^^^^^^^^^^^^^^^^^
ISHMEM_TEAM_DESTROY
^^^^^^^^^^^^^^^^^^^

Destroy an existing team.

.. cpp:function:: void ishmem_team_destroy(ishmem_team_t team)

  :param team: An ``ishmem`` team handle.
  :returns: None.

Callable from the **host**.

**Description:**
The ``ishmem_team_destroy`` routine is a collective operation that destroys the
team referenced by the team handle argument **team**.
Upon return, the referenced team is invalid.

.. FIXME : add the following if/when contexts are supported

.. This routine destroys all shareable contexts created from the referenced team.
.. The user is responsible for destroying all contexts created from this team with
.. the ``ISHMEM_CTX_PRIVATE`` option enabled prior to calling this routine;
.. otherwise, the behavior is undefined.

If **team** compares equal to ``ISHMEM_TEAM_WORLD`` or any other predefined
team, the behavior is undefined.

If **team** compares equal to ``ISHMEM_TEAM_INVALID``, then no operation is
performed.
If **team** is otherwise invalid, the behavior is undefined.
