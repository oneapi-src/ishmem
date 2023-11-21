.. _point_to_point:

------------------------------
Point-To-Point Synchronization
------------------------------

The following section discusses Intel® SHMEM interfaces that provide
a mechanism for synchronization between two PEs based on the value of a
symmetric data object.
The point-to-point synchronization routines can be used to portably ensure
that memory access operations observe remote updates in the order enforced by
the initiator PE.

.. Where appropriate compiler support is available, Intel® SHMEM provides
.. type-generic point-to-point synchronization interfaces via `C11` generic
.. selection. Such type-generic routines are supported for the
.. standard AMO types identified in Table :ref:`Standard AMO Types<stdamotypes>`.

.. The standard AMO types include some of the exact-width integer types defined in
.. the C++ standard library header ``<atomic>``.

.. The ishmem_test_any and ishmem_wait_until_any routines
.. require the SIZE_MAX macro defined in stdint.h by
.. C99 S7.18.3 and C11 S7.20.3.

The point-to-point synchronization routines support values (the **ivar** and
**cmp_value** arguments) having a type in the :ref:`Standard AMO Types
Table<stdamotypes>`.

The point-to-point synchronization interface provides named constants whose
values are integer constant expressions that specify the comparison operators
used by the ``ishmem`` synchronization routines.
The constant names and associated operations are
presented in Table :ref:`Point-to-point Comparison Constants<p2p_consts>`.

.. _p2p_consts:

**Point-to-point Comparison Constants**

===============   =======================
Constant Name     Comparison
===============   =======================
ISHMEM_CMP_EQ     Equal
ISHMEM_CMP_NE     Not equal
ISHMEM_CMP_GT     Greater than
ISHMEM_CMP_GE     Greater than or equal to
ISHMEM_CMP_LT     Less than
ISHMEM_CMP_LE     Less than or equal to
===============   =======================

^^^^^^^^^^^^^^^^^^^
ISHMEM_WAIT_UNTIL
^^^^^^^^^^^^^^^^^^^

Wait for a variable on the local PE to change.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: void ishmem_TYPENAME_wait_until(TYPE* ivar, int cmp, TYPE cmp_value)

  :param ivar: Symmetric address of a remotely accessible data object. The type of **ivar** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivar** with **cmp_value**.
  :param cmp_value: The value to be compared with **ivar**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.

Callable from the **host** and the **device**.

**Description:**
The ``ishmem_wait_until`` operation blocks until the value contained in the
symmetric data object, **ivar**, at the calling PE satisfies the wait
condition.
The **ivar** object at the calling PE may be updated by an AMO performed by a
thread located within the calling PE or within another PE.

These routines can be used to implement point-to-point synchronization between
PEs or between threads within the same PE.
A call to ``ishmem_wait_until`` blocks until the value of **ivar** at the
calling PE satisfies the wait condition specified by the comparison operator,
**cmp**, and comparison value, **cmp_value**.

Implementations must ensure that ``ishmem_wait_until`` does not return
before the update of the memory indicated by **ivar** is fully complete.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_WAIT_UNTIL_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wait for a variable on the local PE to change.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: template<typename Group> void ishmemx_TYPENAME_wait_until_work_group(TYPE* ivar, int cmp, TYPE cmp_value)

  :param ivar: Symmetric address of a remotely accessible data object. The type of **ivar** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivar** with **cmp_value**.
  :param cmp_value: The value to be compared with **ivar**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the wait operation.

Callable from the **device**.

**Description:**
The ``ishmemx_wait_until_work_group`` operation blocks until the value
contained in the symmetric data object, **ivar**, at the calling PE satisfies
the wait condition.
The **ivar** object at the calling PE may be updated by an AMO performed by a
thread located within the calling PE or within another PE.

These routines can be used to implement point-to-point synchronization between
PEs or between threads within the same PE.
A call to ``ishmemx_wait_until_work_group`` blocks until the value of
**ivar** at the calling PE satisfies the wait condition specified by the
comparison operator, **cmp**, and comparison value, **cmp_value**.

Implementations must ensure that ``ishmemx_wait_until_work_group`` does not
return before the update of the memory indicated by **ivar** is fully
complete.

^^^^^^^^^^^^^
ISHMEM_TEST
^^^^^^^^^^^^^

Indicate whether a variable on the local PE meets the specified condition.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: int ishmem_TYPENAME_test(TYPE* ivar, int cmp, TYPE cmp_value)

  :param ivar: Symmetric address of a remotely accessible data object. The type of **ivar** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivar** with **cmp_value**.
  :param cmp_value: The value to be compared with **ivar**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :returns: ``ishmem_test`` returns 1 if the comparison of the symmetric object pointed to by **ivar** with the value **cmp_value** according to the comparison operator **cmp** evaluates to true; otherwise, it returns 0.

Callable from the **host** and **device**.

**Description:**
``ishmem_test`` tests the numeric comparison of the symmetric object
pointed to by **ivar** with the value **cmp_value** according to the
comparison operator **cmp**.
The **ivar** object at the calling PE may be updated by an AMO performed by a thread located within the calling PE or within another PE.

Implementations must ensure that ``ishmem_test`` does not return 1 before
the update of the memory indicated by **ivar** is fully complete.

^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_TEST_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^

Indicate whether a variable on the local PE meets the specified condition.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: int ishmemx_TYPENAME_test_work_group(TYPE* ivar, int cmp, TYPE cmp_value)

  :param ivar: Symmetric address of a remotely accessible data object. The type of **ivar** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivar** with **cmp_value**.
  :param cmp_value: The value to be compared with **ivar**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the wait operation.
  :returns: ``ishmem_test`` returns 1 if the comparison of the symmetric object pointed to by **ivar** with the value **cmp_value** according to the comparison operator **cmp** evaluates to true; otherwise, it returns 0.

Callable from the **device**.

**Description:**
``ishmemx_test_work_group`` tests the numeric comparison of the symmetric
object pointed to by **ivar** with the value **cmp_value** according to the
comparison operator **cmp**.
The **ivar** object at the calling PE may be updated by an AMO performed by a thread located within the calling PE or within another PE.

Implementations must ensure that ``ishmemx_test_work_group`` does not return
1 before the update of the memory indicated by **ivar** is fully complete.

