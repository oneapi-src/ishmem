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
  :returns: None.

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

.. cpp:function:: template<typename Group> void ishmemx_TYPENAME_wait_until_work_group(TYPE* ivar, int cmp, TYPE cmp_value, const Group& group)

  :param ivar: Symmetric address of a remotely accessible data object. The type of **ivar** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivar** with **cmp_value**.
  :param cmp_value: The value to be compared with **ivar**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param group: The SYCL ``group`` or ``sub_group`` with which to collectively perform the wait operation.
  :returns: None.

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

^^^^^^^^^^^^^^^^^^^^^
ISHMEM_WAIT_UNTIL_ALL
^^^^^^^^^^^^^^^^^^^^^

Wait for an array of variables on the local PE until all variables meet the specified wait condition.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: void ishmem_TYPENAME_wait_until_all(TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE cmp_value)

  :param ivars: Symmetric address of an array of remotely accessible data objects. The type of **ivars** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param nelems: The number of elements in the **ivars** array.
  :param status: Local address of an optional mask array of length **nelems** that indicates which elements in **ivars** are excluded from the wait set.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivars** with **cmp_value**.
  :param cmp_value: The value to be compared with the objects pointed to by **ivars**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :returns: None.

Callable from the **host** and the **device**.

**Description:**
The ``ishmem_wait_until_all`` routine waits until all entries in the wait set
specified by **ivars** and **status** have satisfied the wait condition at 
the calling PE.
The **ivars** objects at the calling PE may be updated by an AMO performed by
a thread located within the calling PE or within another PE.
If **nelems** is 0, the wait set is empty and this routine returns immediately.
This routine compares each element of the **ivars** array in the wait set with
the value **cmp_value** according to the comparison operator **cmp** at the
calling PE.
This routine is semantically similar to ``ishmem_wait_until``, but adds support
for point-to-point synchronization involving an array of symmetric data objects.

The optional **status** is a mask array of length **nelems** where each element
corresponds to the respective element in **ivars** and indicates whether the
element is excluded from the wait set.
Elements of **status** set to 0 will be included in the wait set, and elements
set to a nonzero value will be ignored.
If all elements in **status** are nonzero or **nelems** is 0, the wait set is
empty and this routine returns immediately.
If **status** is a null pointer, it is ignored and all elements in **ivars**
are included in the wait set.
The **ivars** and **status** arrays must not overlap in memory.

Implementations must ensure that ``ishmem_wait_until_all`` does not return
before the update of the memory indicated by **ivars** is fully complete.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_WAIT_UNTIL_ALL_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wait for an array of variables on the local PE until all variables meet the specified wait condition.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: template<typename Group> void ishmemx_TYPENAME_wait_until_all_work_group(TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE cmp_value, const Group& group)

  :param ivars: Symmetric address of an array of remotely accessible data objects. The type of **ivars** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param nelems: The number of elements in the **ivars** array.
  :param status: Local address of an optional mask array of length **nelems** that indicates which elements in **ivars** are excluded from the wait set.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivars** with **cmp_value**.
  :param cmp_value: The value to be compared with the objects pointed to by **ivars**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param group: The SYCL ``group`` or ``sub_group`` with which to collectively perform the wait operation.
  :returns: None.

Callable from the **device**.

**Description:**
The ``ishmemx_wait_until_all_work_group`` routine waits until all entries
in the wait set specified by **ivars** and **status** have satisfied the
wait condition at the calling PE.
The **ivars** objects at the calling PE may be updated by an AMO performed by
a thread located within the calling PE or within another PE.
If **nelems** is 0, the wait set is empty and this routine returns immediately.
This routine compares each element of the **ivars** array in the wait set with
the value **cmp_value** according to the comparison operator **cmp** at the
calling PE.
This routine is semantically similar to ``ishmemx_wait_until_work_group``, but
adds support for point-to-point synchronization involving an array of symmetric
data objects.

The optional **status** is a mask array of length **nelems** where each element
corresponds to the respective element in **ivars** and indicates whether the
element is excluded from the wait set.
Elements of **status** set to 0 will be included in the wait set, and elements
set to a nonzero value will be ignored.
If all elements in **status** are nonzero or **nelems** is 0, the wait set is
empty and this routine returns immediately.
If **status** is a null pointer, it is ignored and all elements in **ivars**
are included in the wait set.
The **ivars** and **status** arrays must not overlap in memory.

Implementations must ensure that ``ishmemx_wait_until_all_work_group`` does
not return before the update of the memory indicated by **ivars** is fully
complete.

^^^^^^^^^^^^^^^^^^^^^
ISHMEM_WAIT_UNTIL_ANY
^^^^^^^^^^^^^^^^^^^^^

Wait for an array of variables on the local PE until any one variable meets the specified wait condition.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: size_t ishmem_TYPENAME_wait_until_any(TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE cmp_value)

  :param ivars: Symmetric address of an array of remotely accessible data objects. The type of **ivars** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param nelems: The number of elements in the **ivars** array.
  :param status: Local address of an optional mask array of length **nelems** that indicates which elements in **ivars** are excluded from the wait set.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivars** with **cmp_value**.
  :param cmp_value: The value to be compared with the objects pointed to by **ivars**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :returns: ``ishmem_wait_until_any`` returns the index of an element in the **ivars** array that satisfies the wait condition. If the wait set is empty, this routine returns **SIZE_MAX**.

Callable from the **host** and the **device**.

**Description:**
The ``ishmem_wait_until_any`` routine waits until any one entry in the wait set
specified by **ivars** and **status** satisfies the wait condition at the
calling PE.
The **ivars** objects at the calling PE may be updated by an AMO performed by a
thread located within the calling PE or within another PE.
This routine compares each element of the **ivars** array in the wait set with
the value **cmp_value** according to the comparison operator **cmp** at the
calling PE.
The order in which these elements are tested is unspecified.
If an entry **i** in **ivars** within the wait set satisfies the wait condition,
a series of calls to ``ishmem_wait_until_any`` must eventually return **i**.

The optional **status** is a mask array of length **nelems** where each element
corresponds to the respective element in **ivars** and indicates whether the
element is excluded from the wait set.
Elements of **status** set to 0 will be included in the wait set, and elements
set to a nonzero value will be ignored.
If all elements in **status** are nonzero or **nelems** is 0, the wait set is
empty and this routine returns **SIZE_MAX**.
If **status** is a null pointer, it is ignored and all elements in **ivars**
are included in the wait set.
The **ivars** and **status** arrays must not overlap in memory.

Implementations must ensure that ``ishmem_wait_until_any`` does not return
before the update of the memory indicated by **ivars** is fully complete.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_WAIT_UNTIL_ANY_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wait for an array of variables on the local PE until any one variable meets the specified wait condition.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: template<typename Group> size_t ishmemx_TYPENAME_wait_until_any_work_group(TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE cmp_value, const Group& group)

  :param ivars: Symmetric address of an array of remotely accessible data objects. The type of **ivars** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param nelems: The number of elements in the **ivars** array.
  :param status: Local address of an optional mask array of length **nelems** that indicates which elements in **ivars** are excluded from the wait set.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivars** with **cmp_value**.
  :param cmp_value: The value to be compared with the objects pointed to by **ivars**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param group: The SYCL ``group`` or ``sub_group`` with which to collectively perform the wait operation.
  :returns: ``ishmemx_wait_until_any_work_group`` returns the index of an element in the **ivars** array that satisfies the wait condition. If the wait set is empty, this routine returns **SIZE_MAX**.

Callable from the **device**.

**Description:**
The ``ishmemx_wait_until_any_work_group`` routine waits until any one entry in
the wait set specified by **ivars** and **status** satisfies the wait
condition at the calling PE.
The **ivars** objects at the calling PE may be updated by an AMO performed by a
thread located within the calling PE or within another PE.
This routine compares each element of the **ivars** array in the wait set with
the value **cmp_value** according to the comparison operator **cmp** at the
calling PE.
The order in which these elements are tested is unspecified.
If an entry **i** in **ivars** within the wait set satisfies the wait condition,
a series of calls to ``ishmemx_wait_until_any_work_group`` must eventually
return **i**.

The optional **status** is a mask array of length **nelems** where each element
corresponds to the respective element in **ivars** and indicates whether the
element is excluded from the wait set.
Elements of **status** set to 0 will be included in the wait set, and elements
set to a nonzero value will be ignored.
If all elements in **status** are nonzero or **nelems** is 0, the wait set is
empty and this routine returns **SIZE_MAX**.
If **status** is a null pointer, it is ignored and all elements in **ivars**
are included in the wait set.
The **ivars** and **status** arrays must not overlap in memory.

Implementations must ensure that ``ishmemx_wait_until_any_work_group`` does not
return before the update of the memory indicated by **ivars** is fully
complete.

^^^^^^^^^^^^^^^^^^^^^^
ISHMEM_WAIT_UNTIL_SOME
^^^^^^^^^^^^^^^^^^^^^^

Wait on an array of variables on the local PE until at least one variable meets the specified wait condition.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: size_t ishmem_TYPENAME_wait_until_some(TYPE* ivars, size_t nelems, size_t* indices, const int* status, int cmp, TYPE cmp_value)

  :param ivars: Symmetric address of an array of remotely accessible data objects. The type of **ivars** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param nelems: The number of elements in the **ivars** array.
  :param indices: Local address of an array of indices of length at least **nelems** into **ivars** that satisfied the wait condition.
  :param status: Local address of an optional mask array of length **nelems** that indicates which elements in **ivars** are excluded from the wait set.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivars** with **cmp_value**.
  :param cmp_value: The value to be compared with the objects pointed to by **ivars**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :returns: ``ishmem_wait_until_some`` returns the number of indices returned in the **indices** array. If the wait set is empty, this routine returns 0.

Callable from the **host** and the **device**.

**Description:**
The ``ishmem_wait_until_some`` routine waits until at least one entry in the
wait set specified by **ivars** and **status** satisfies the wait condition at
the calling PE.
The **ivars** objects at the calling PE may be updated by an AMO performed by a
thread located within the calling PE or within another PE.
This routine compares each element of the **ivars** array in the wait set with
the value **cmp_value** according to the comparison operator **cmp** at the
calling PE.
This routine tests all elements of **ivars** in the wait set at least once, and
the order in which the elements are tested is unspecified.

Upon return, the **indices** array contains the indices of at least one element
in the wait set that satisfied the wait condition during the call to 
``ishmem_wait_until_some``.
The return value of ``ishmem_wait_until_some`` is equal to the total number of
these satisfied elements.
For a given return value **N**, the first **N** elements of the **indices**
array contain those unique indices that satisfied the wait condition.
These first **N** elements of **indices** may be unordered with respect to the
corresponding indices of **ivars**.
The array pointed to by **indices** must be at least **nelems** long.
If an entry **i** in **ivars** within the wait set satisfies the wait
condition, a series of calls to ``ishmem_wait_until_some`` must eventually
include **i** in the **indices** array.

The optional **status** is a mask array of length **nelems** where each element
corresponds to the respective element in **ivars** and indicates whether the
element is excluded from the wait set.
Elements of **status** set to 0 will be included in the wait set, and elements
set to a nonzero value will be ignored.
If all elements in **status** are nonzero or **nelems** is 0, the wait set is
empty and this routine returns 0.
If **status** is a null pointer, it is ignored and all elements in **ivars**
are included in the wait set.
The **ivars**, **indices**, and **status** arrays must not overlap in memory.

Implementations must ensure that ``ishmem_wait_until_some`` does not return
before the update of the memory indicated by **ivars** is fully complete.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_WAIT_UNTIL_SOME_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wait on an array of variables on the local PE until at least one variable meets the specified wait condition.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: template<typename Group> size_t ishmemx_TYPENAME_wait_until_some_work_group(TYPE* ivars, size_t nelems, size_t* indices, const int* status, int cmp, TYPE cmp_value, const Group& group)

  :param ivars: Symmetric address of an array of remotely accessible data objects. The type of **ivars** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param nelems: The number of elements in the **ivars** array.
  :param indices: Local address of an array of indices of length at least **nelems** into **ivars** that satisfied the wait condition.
  :param status: Local address of an optional mask array of length **nelems** that indicates which elements in **ivars** are excluded from the wait set.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivars** with **cmp_value**.
  :param cmp_value: The value to be compared with the objects pointed to by **ivars**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param group: The SYCL ``group`` or ``sub_group`` with which to collectively perform the wait operation.
  :returns: ``ishmemx_wait_until_some_work_group`` returns the number of indices returned in the **indices** array. If the wait set is empty, this routine returns 0.

Callable from the **device**.

**Description:**
The ``ishmemx_wait_until_some_work_group`` routine waits until at least one
entry in the wait set specified by **ivars** and **status** satisfies the wait
condition at the calling PE.
The **ivars** objects at the calling PE may be updated by an AMO performed by a
thread located within the calling PE or within another PE.
This routine compares each element of the **ivars** array in the wait set with
the value **cmp_value** according to the comparison operator **cmp** at the
calling PE.
This routine tests all elements of **ivars** in the wait set at least once, and
the order in which the elements are tested is unspecified.

Upon return, the **indices** array contains the indices of at least one element
in the wait set that satisfied the wait condition during the call to 
``ishmemx_wait_until_some_work_group``.
The return value of ``ishmemx_wait_until_some_work_group`` is equal to the total
number of these satisfied elements.
For a given return value **N**, the first **N** elements of the **indices**
array contain those unique indices that satisfied the wait condition.
These first **N** elements of **indices** may be unordered with respect to the
corresponding indices of **ivars**.
The array pointed to by **indices** must be at least **nelems** long.
If an entry **i** in **ivars** within the wait set satisfies the wait
condition, a series of calls to ``ishmemx_wait_until_some_work_group`` must
eventually include **i** in the **indices** array.

The optional **status** is a mask array of length **nelems** where each element
corresponds to the respective element in **ivars** and indicates whether the
element is excluded from the wait set.
Elements of **status** set to 0 will be included in the wait set, and elements
set to a nonzero value will be ignored.
If all elements in **status** are nonzero or **nelems** is 0, the wait set is
empty and this routine returns 0.
If **status** is a null pointer, it is ignored and all elements in **ivars**
are included in the wait set.
The **ivars**, **indices**, and **status** arrays must not overlap in memory.

Implementations must ensure that ``ishmemx_wait_until_some_work_group`` does not
return before the update of the memory indicated by **ivars** is fully
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

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_test_work_group(TYPE* ivar, int cmp, TYPE cmp_value, const Group& group)

  :param ivar: Symmetric address of a remotely accessible data object. The type of **ivar** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivar** with **cmp_value**.
  :param cmp_value: The value to be compared with **ivar**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param group: The SYCL ``group`` or ``sub_group`` with which to collectively perform the test operation.
  :returns: ``ishmemx_test_work_group`` returns 1 if the comparison of the symmetric object pointed to by **ivar** with the value **cmp_value** according to the comparison operator **cmp** evaluates to true; otherwise, it returns 0.

Callable from the **device**.

**Description:**
``ishmemx_test_work_group`` tests the numeric comparison of the symmetric
object pointed to by **ivar** with the value **cmp_value** according to the
comparison operator **cmp**.
The **ivar** object at the calling PE may be updated by an AMO performed by a thread located within the calling PE or within another PE.

Implementations must ensure that ``ishmemx_test_work_group`` does not return
1 before the update of the memory indicated by **ivar** is fully complete.

^^^^^^^^^^^^^^^^
ISHMEM_TEST_ALL
^^^^^^^^^^^^^^^^

Indicate whether all variables within an array of variables on the local PE meet a specified test condition.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: int ishmem_TYPENAME_test_all(TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE cmp_value)

  :param ivars: Symmetric address of an array of remotely accessible data objects. The type of **ivars** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param nelems: The number of elements in the **ivars** array.
  :param status: Local address of an optional mask array of length **nelems** that indicates which elements in **ivars** are excluded from the wait set.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivar** with **cmp_value**.
  :param cmp_value: The value to be compared with **ivar**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :returns: ``ishmem_test_all`` returns 1 if all variables in **ivars** satisfy the test condition or if **nelems** is 0, otherwise this routine returns 0.

Callable from the **host** and **device**.

**Description:**
The ``ishmem_test_all`` routine indicates whether all entries in the test set
specified by **ivars** and **status** have satisfied the test condition at the
calling PE.
The **ivars** objects at the calling PE may be updated by an AMO performed by a
thread located within the calling PE or within another PE.
This routine does not block and returns zero if not all entries in **ivars**
satisfied the test condition.
This routine compares each element of the **ivars** array in the test set with
the value **cmp_value** according to the comparison operator **cmp** at the
calling PE.

If **nelems** is 0, the test set is empty and this routine returns 1.

The optional **status** is a mask array of length **nelems** where each element
corresponds to the respective element in **ivars** and indicates whether the
element is excluded from the test set.
Elements of **status** set to 0 will be included in the test set, and elements
set to a nonzero value will be ignored.
If all elements in **status** are nonzero or **nelems** is 0, the test set is
empty and this routine returns 0.
If **status** is a null pointer, it is ignored and all elements in **ivars**
are included in the test set.
The **ivars** and **status** arrays must not overlap in memory.

Implementations must ensure that ``ishmem_test_all`` does not return 1 before
the update of the memory indicated by **ivars** is fully complete.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_TEST_ALL_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Indicate whether all variables within an array of variables on the local PE meet a specified test condition.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: template<typename Group> int ishmemx_TYPENAME_test_all_work_group(TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE cmp_value, const Group& group)

  :param ivars: Symmetric address of an array of remotely accessible data objects. The type of **ivars** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param nelems: The number of elements in the **ivars** array.
  :param status: Local address of an optional mask array of length **nelems** that indicates which elements in **ivars** are excluded from the wait set.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivar** with **cmp_value**.
  :param cmp_value: The value to be compared with **ivar**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param group: The SYCL ``group`` or ``sub_group`` with which to collectively perform the test operation.
  :returns: ``ishmemx_test_all_work_group`` returns 1 if all variables in **ivars** satisfy the test condition or if **nelems** is 0, otherwise this routine returns 0.

Callable from the **device**.

**Description:**
The ``ishmemx_test_all_work_group`` routine indicates whether all entries in the
test set specified by **ivars** and **status** have satisfied the test
condition at the calling PE.
The **ivars** objects at the calling PE may be updated by an AMO performed by a
thread located within the calling PE or within another PE.
This routine does not block and returns zero if not all entries in **ivars**
satisfied the test condition.
This routine compares each element of the **ivars** array in the test set with
the value **cmp_value** according to the comparison operator **cmp** at the
calling PE.

If **nelems** is 0, the test set is empty and this routine returns 1.

The optional **status** is a mask array of length **nelems** where each element
corresponds to the respective element in **ivars** and indicates whether the
element is excluded from the test set.
Elements of **status** set to 0 will be included in the test set, and elements
set to a nonzero value will be ignored.
If all elements in **status** are nonzero or **nelems** is 0, the test set is
empty and this routine returns 0.
If **status** is a null pointer, it is ignored and all elements in **ivars**
are included in the test set.
The **ivars** and **status** arrays must not overlap in memory.

Implementations must ensure that ``ishmemx_test_all_work_group`` does not
return 1 before the update of the memory indicated by **ivars** is fully
complete.

^^^^^^^^^^^^^^^^
ISHMEM_TEST_ANY
^^^^^^^^^^^^^^^^

Indicate whether any one variable within an array of variables on the local PE meets a specified test condition.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: size_t ishmem_TYPENAME_test_any(TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE cmp_value)

  :param ivars: Symmetric address of an array of remotely accessible data objects. The type of **ivars** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param nelems: The number of elements in the **ivars** array.
  :param status: Local address of an optional mask array of length **nelems** that indicates which elements in **ivars** are excluded from the wait set.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivar** with **cmp_value**.
  :param cmp_value: The value to be compared with **ivar**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :returns: ``ishmem_test_any`` returns the index of an element in the **ivars** array that satisfies the test condition. If the test set is empty or no conditions in the test set are satisfied, this routine returns **SIZE_MAX**.

Callable from the **host** and **device**.

**Description:**
The ``ishmem_test_any`` routine indicates whether any entry in the test set
specified by **ivars** and **status** has satisfied the test condition at the
calling PE.
The **ivars** objects at the calling PE may be updated by an AMO performed by a
thread located within the calling PE or within another PE.
This routine does not block and returns **SIZE_MAX** if no entries in **ivars**
satisfied the test condition.
This routine compares each element of the **ivars** array in the test set with
the value **cmp_value** according to the comparison operator **cmp** at the
calling PE.
The order in which these elements are tested is unspecified.
If an entry **i** in **ivars** within the test set satisfies the test
condition, a series of calls to ``ishmem_test_any`` must eventually return **i**.

The optional **status** is a mask array of length **nelems** where each element
corresponds to the respective element in **ivars** and indicates whether the
element is excluded from the test set.
Elements of **status** set to 0 will be included in the test set, and elements
set to a nonzero value will be ignored.
If all elements in **status** are nonzero or **nelems** is 0, the test set is
empty and this routine returns **SIZE_MAX**.
If **status** is a null pointer, it is ignored and all elements in **ivars**
are included in the test set.
The **ivars** and **status** arrays must not overlap in memory.

Implementations must ensure that ``ishmem_test_any`` does not return an index
before the update of the memory indicated by the corresponding **ivars**
element is fully complete.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_TEST_ANY_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Indicate whether any one variable within an array of variables on the local PE meets a specified test condition.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: template<typename Group> size_t ishmemx_TYPENAME_test_any_work_group(TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE cmp_value, const Group& group)

  :param ivars: Symmetric address of an array of remotely accessible data objects. The type of **ivars** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param nelems: The number of elements in the **ivars** array.
  :param status: Local address of an optional mask array of length **nelems** that indicates which elements in **ivars** are excluded from the wait set.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivar** with **cmp_value**.
  :param cmp_value: The value to be compared with **ivar**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param group: The SYCL ``group`` or ``sub_group`` with which to collectively perform the test operation.
  :returns: ``ishmemx_test_any_work_group`` returns the index of an element in the **ivars** array that satisfies the test condition. If the test set is empty or no conditions in the test set are satisfied, this routine returns **SIZE_MAX**.

Callable from the **device**.

**Description:**
The ``ishmemx_test_any_work_group`` routine indicates whether any entry in the
test set specified by **ivars** and **status** has satisfied the test
condition at the calling PE.
The **ivars** objects at the calling PE may be updated by an AMO performed by a
thread located within the calling PE or within another PE.
This routine does not block and returns **SIZE_MAX** if no entries in **ivars**
satisfied the test condition.
This routine compares each element of the **ivars** array in the test set with
the value **cmp_value** according to the comparison operator **cmp** at the
calling PE.
The order in which these elements are tested is unspecified.
If an entry **i** in **ivars** within the test set satisfies the test
condition, a series of calls to ``ishmemx_test_any_work_group`` must eventually
return **i**.

The optional **status** is a mask array of length **nelems** where each element
corresponds to the respective element in **ivars** and indicates whether the
element is excluded from the test set.
Elements of **status** set to 0 will be included in the test set, and elements
set to a nonzero value will be ignored.
If all elements in **status** are nonzero or **nelems** is 0, the test set is
empty and this routine returns **SIZE_MAX**.
If **status** is a null pointer, it is ignored and all elements in **ivars**
are included in the test set.
The **ivars** and **status** arrays must not overlap in memory.

Implementations must ensure that ``ishmemx_test_any_work_group`` does not return
an index before the update of the memory indicated by the corresponding
**ivars** element is fully complete.

^^^^^^^^^^^^^^^^
ISHMEM_TEST_SOME
^^^^^^^^^^^^^^^^

Indicate whether at least one variable within an array of variables on the local PE meets a specified test condition.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: size_t ishmem_TYPENAME_test_some(TYPE* ivars, size_t nelems, size_t* indices, const int* status, int cmp, TYPE cmp_value)

  :param ivars: Symmetric address of an array of remotely accessible data objects. The type of **ivars** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param nelems: The number of elements in the **ivars** array.
  :param indices: Local address of an array of indices of length at least **nelems** into **ivars** that satisfied the wait condition.
  :param status: Local address of an optional mask array of length **nelems** that indicates which elements in **ivars** are excluded from the wait set.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivars** with **cmp_value**.
  :param cmp_value: The value to be compared with the objects pointed to by **ivars**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :returns: ``ishmem_test_some`` returns the number of indices returned in the **indices** array. If the test set is empty, this routine returns 0.

Callable from the **host** and **device**.

**Description:**
The ``ishmem_test_some`` routine indicates whether at least one entry in the
test set specified by **ivars** and **status** satisfies the test condition at
the calling PE.
The **ivars** objects at the calling PE may be updated by an AMO performed by a
thread located within the calling PE or within another PE.
This routine does not block and returns zero if no entries in **ivars**
satisfied the test condition.
This routine compares each element of the **ivars** array in the test set with
the value **cmp_value** according to the comparison operator **cmp** at the
calling PE.
This routine tests all elements of **ivars** in the test set at least once, and
the order in which the elements are tested is unspecified.

Upon return, the **indices** array contains the indices of the elements in the
test set that satisfied the test condition during the call to
``ishmem_test_some``.
The return value of ``ishmem_test_some`` is equal to the total number of these
satisfied elements.
If the return value is **N**, then the first **N** elements of the **indices**
array contain those unique indices that satisfied the test condition.
These first **N** elements of **indices** may be unordered with respect to the
corresponding indices of **ivars**.
The array pointed to by **indices** must be at least **nelems** long.
If an entry **i** in **ivars** within the test set satisfies the test
condition, a series of calls to ``ishmem_test_some`` must eventually include
**i** in the **indices** array.

The optional **status** is a mask array of length **nelems** where each element
corresponds to the respective element in **ivars** and indicates whether the
element is excluded from the test set.
Elements of **status** set to 0 will be included in the test set, and elements
set to a nonzero value will be ignored.
If all elements in **status** are nonzero or **nelems** is 0, the test set is
empty and this routine returns 0.
If **status** is a null pointer, it is ignored and all elements in **ivars**
are included in the test set.
The **ivars**, **indices**, and **status** arrays must not overlap in memory.

Implementations must ensure that ``ishmem_test_some`` does not return indices
before the updates of the memory indicated by the corresponding **ivars**
elements are fully complete.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_TEST_SOME_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Indicate whether at least one variable within an array of variables on the local PE meets a specified test condition.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by Table :ref:`Standard AMO Types<stdamotypes>`.

.. cpp:function:: template<typename Group> size_t ishmemx_TYPENAME_test_some_work_group(TYPE* ivars, size_t nelems, size_t* indices, const int* status, int cmp, TYPE cmp_value, const Group& group)

  :param ivars: Symmetric address of an array of remotely accessible data objects. The type of **ivars** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param nelems: The number of elements in the **ivars** array.
  :param indices: Local address of an array of indices of length at least **nelems** into **ivars** that satisfied the wait condition.
  :param status: Local address of an optional mask array of length **nelems** that indicates which elements in **ivars** are excluded from the wait set.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **ivars** with **cmp_value**.
  :param cmp_value: The value to be compared with the objects pointed to by **ivars**. The type of **cmp_value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param group: The SYCL ``group`` or ``sub_group`` with which to collectively perform the test operation.
  :returns: ``ishmemx_test_some_work_group`` returns the number of indices returned in the **indices** array. If the test set is empty, this routine returns 0.

Callable from the **device**.

**Description:**
The ``ishmemx_test_some_work_group`` routine indicates whether at least one
entry in the test set specified by **ivars** and **status** satisfies the
test condition at the calling PE.
The **ivars** objects at the calling PE may be updated by an AMO performed by a
thread located within the calling PE or within another PE.
This routine does not block and returns zero if no entries in **ivars**
satisfied the test condition.
This routine compares each element of the **ivars** array in the test set with
the value **cmp_value** according to the comparison operator **cmp** at the
calling PE.
This routine tests all elements of **ivars** in the test set at least once, and
the order in which the elements are tested is unspecified.

Upon return, the **indices** array contains the indices of the elements in the
test set that satisfied the test condition during the call to
``ishmemx_test_some_work_group``.
The return value of ``ishmemx_test_some_work_group`` is equal to the total
number of these satisfied elements.
If the return value is **N**, then the first **N** elements of the **indices**
array contain those unique indices that satisfied the test condition.
These first **N** elements of **indices** may be unordered with respect to the
corresponding indices of **ivars**.
The array pointed to by **indices** must be at least **nelems** long.
If an entry **i** in **ivars** within the test set satisfies the test
condition, a series of calls to ``ishmemx_test_some_work_group`` must eventually
include **i** in the **indices** array.

The optional **status** is a mask array of length **nelems** where each element
corresponds to the respective element in **ivars** and indicates whether the
element is excluded from the test set.
Elements of **status** set to 0 will be included in the test set, and elements
set to a nonzero value will be ignored.
If all elements in **status** are nonzero or **nelems** is 0, the test set is
empty and this routine returns 0.
If **status** is a null pointer, it is ignored and all elements in **ivars**
are included in the test set.
The **ivars**, **indices**, and **status** arrays must not overlap in memory.

Implementations must ensure that ``ishmemx_test_some_work_group`` does not
return indices before the updates of the memory indicated by the
corresponding **ivars** elements are fully complete.

^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEM_SIGNAL_WAIT_UNTIL
^^^^^^^^^^^^^^^^^^^^^^^^

Wait for a variable on the local PE to change from a signaling operation.

.. cpp:function:: uint64_t ishmem_signal_wait_until(uint64_t* sig_addr, int cmp, uint64_t cmp_value)

  :param sig_addr: Local, symmetric address of the source signal variable.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **sig_addr** with **cmp_value**.
  :param cmp_value: The value against which the object pointed to by **sig_addr** will be compared.
  :returns: ``ishmem_signal_wait_until`` returns the contents of the signal data object, **sig_addr**, at the calling PE that satisfies the wait condition.

Callable from the **host** and **device**.

**Description:**
``ishmem_signal_wait_until`` operation blocks until the value contained in the
signal data object, **sig_addr**, at the calling PE satisfies the wait
condition.

.. In an Intel® SHMEM program with single-threaded or multithreaded PEs,

The **sig_addr** object at the calling PE is expected only to be updated using
the APIs defined in :ref:`Signaling Operations<signaling>` that perform a write
to a signal variable.
This routine can be used to implement point-to-point synchronization
between PEs or between threads within the same PE. A call to this routine
blocks until the value of **sig_addr** at the calling PE satisfies the wait
condition specified by the comparison operator, **cmp**, and comparison value,
**cmp_value**. Implementations must ensure that ``ishmem_signal_wait_until`` do
not return before the update of the memory indicated by **sig_addr** is fully
complete.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_SIGNAL_WAIT_UNTIL_WORK_GROUP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wait for a variable on the local PE to change from a signaling operation.

.. cpp:function:: uint64_t ishmemx_signal_wait_until_work_group(uint64_t* sig_addr, int cmp, uint64_t cmp_value, const Group& group)

  :param sig_addr: Local, symmetric address of the source signal variable.
  :param cmp: A comparison operator from Table :ref:`Point-to-point Comparison Constants<p2p_consts>` that compares **sig_addr** with **cmp_value**.
  :param cmp_value: The value against which the object pointed to by **sig_addr** will be compared.
  :param group: The SYCL ``group`` or ``sub_group`` with which to collectively perform the wait operation.
  :returns: ``ishmemx_signal_wait_until_work_group`` returns the contents of the signal data object, **sig_addr**, at the calling PE that satisfies the wait condition.

Callable from the **device**.

**Description:**
``ishmemx_signal_wait_until_work_group`` operation blocks until the value
contained in the signal data object, **sig_addr**, at the calling PE
satisfies the wait condition.

.. In an Intel® SHMEM program with single-threaded or multithreaded PEs,

The **sig_addr** object at the calling PE is expected only to be updated using
the APIs defined in :ref:`Signaling Operations<signaling>` that perform a write
to a signal variable.
This routine can be used to implement point-to-point synchronization
between PEs or between threads within the same PE. A call to this routine
blocks until the value of **sig_addr** at the calling PE satisfies the wait
condition specified by the comparison operator, **cmp**, and comparison value,
**cmp_value**. Implementations must ensure that
``ishmemx_signal_wait_until_work_group`` do not return before the update of the
memory indicated by **sig_addr** is fully complete.
