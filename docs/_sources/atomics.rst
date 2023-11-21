.. _atomics: 

------------------------------
Atomic Memory Operations (AMO)
------------------------------

An AMO is a one-sided communication mechanism that
combines memory read, update, or write operations with
atomicity guarantees described in :ref:`Atomicity
Guarantees<amo_guarantees>`.
Similar to the RMA routines, described in :ref:`Remote Memory
Access (RMA)<rma>`, the AMOs are performed only on
symmetric objects.
Intel® SHMEM defines two types of AMO
routines:

#. The `fetching` routines return the original value of, and optionally update, the remote data object in a single atomic operation.  The routines return after the data has been fetched from the target PE and delivered to the calling PE. The data type of the returned value is the same as the type of the remote data object.

  The fetching routines include:
  ``ishmem_atomic_{fetch,compare_swap,swap}[_nbi]`` and
  ``ishmem_atomic_fetch_{inc,add,and,or,xor}[_nbi]``.

#. The `non-fetching` routines update the remote data object in a single atomic operation.  A call to a non-fetching atomic routine issues the atomic operation and may return before the operation executes on the target PE. The ``ishmem_quiet``, ``ishmem_barrier``, or ``ishmem_barrier_all`` routines can be used to force completion for these non-fetching atomic routines.

  The non-fetching routines include:
  ``ishmem_atomic_{set,inc,add,and,or,xor}[_nbi]``.

.. Intel® SHMEM AMO routines specified in this section have two
.. variants. In one of the variants, the context handle, `ctx`, is explicitly
.. passed as an argument. In this variant, the operation is performed on the
.. specified context. If the context handle `ctx` does not correspond to a
.. valid context, the behavior is undefined. In the other variant, the context
.. handle is not explicitly passed and thus, the operations are performed on the
.. default context.

.. Where appropriate compiler support is available, Intel® SHMEM
.. provides type-generic AMO interfaces via \Cstd[11] generic selection.  The
.. type-generic support for the AMO routines is as follows:

.. #. ``ishmem_atomic_{compare_swap,fetch_inc,inc,fetch_add,add}[_nbi]`` support
..    the ``standard AMO types'' listed in Table~\ref{stdamotypes},
.. #. ``ishmem_atomic_{fetch,set,swap}`` support
..   the ``extended AMO types'' listed in Table~\ref{extamotypes}, and
.. #. ``ishmem_atomic_{fetch_and,and,fetch_or,or,fetch_xor,xor}[_nbi]``
..   support the ``bitwise AMO types'' listed in Table~\ref{bitamotypes}.

The standard, extended, and bitwise AMO types include some of the exact-width
integer types defined in the C++11 language library header ``<cstdint>``.

.. _stdamotypes:

**Standard AMO Types:**

==================   =========
TYPE                 TYPENAME 
==================   =========
int                  int      
long                 long     
long long            longlong 
unsigned int         uint     
unsigned long        ulong    
unsigned long long   ulonglong
int32_t              int32    
int64_t              int64    
uint32_t             uint32   
uint64_t             uint64   
size_t               size     
ptrdiff_t            ptrdiff  
==================   =========

.. _extamotypes:

**Extended AMO Types:**

==================   =========
TYPE                 TYPENAME  
==================   =========
float                float    
double               double   
int                  int      
long                 long     
long long            longlong 
unsigned int         uint     
unsigned long        ulong    
unsigned long long   ulonglong
int32_t              int32    
int64_t              int64    
uint32_t             uint32   
uint64_t             uint64   
size_t               size     
ptrdiff_t            ptrdiff  
==================   =========

.. _bitamotypes:

**Bitwise AMO Types:**

==================   =========
TYPE                 TYPENAME
==================   =========
unsigned int         uint      
unsigned long        ulong     
unsigned long long   ulonglong 
int32_t              int32     
int64_t              int64     
uint32_t             uint32    
uint64_t             uint64    
==================   =========

^^^^^^^^^^^^^^^^^^^
ISHMEM_ATOMIC_FETCH
^^^^^^^^^^^^^^^^^^^

Atomically fetches the value of a remote data object.

Below, TYPE is one of the extended AMO types and has a corresponding TYPENAME
specified by the :ref:`Extended AMO Types table<extamotypes>`.

.. cpp:function:: template<typename TYPE> TYPE ishmem_atomic_fetch(const TYPE* source, int pe)

.. cpp:function:: TYPE ishmem_TYPENAME_atomic_fetch(const TYPE* source, int pe)

  :param source: Symmetric address of the source data object. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Extended AMO Types<extamotypes>`.
  :param pe: An integer that indicates the PE number from which **source** is to be fetched.
  :returns: The contents at the **source** address on the remote PE. The data type of the return value is the same as the type of the remote data object.

Callable from the **device**.

**Description:**
``ishmem_atomic_fetch`` performs an atomic fetch operation.  It returns the
contents of the **source** as an atomic operation.

^^^^^^^^^^^^^^^^^
ISHMEM_ATOMIC_SET
^^^^^^^^^^^^^^^^^

Atomically sets the value of a remote data object.

Below, TYPE is one of the extended AMO types and has a corresponding TYPENAME
specified by the :ref:`Extended AMO Types table<extamotypes>`.

.. cpp:function:: template<typename TYPE> void ishmem_atomic_set(TYPE* dest, TYPE value, int pe)

.. cpp:function:: void ishmem_TYPENAME_atomic_set(TYPE* dest, TYPE value, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Extended AMO Types<extamotypes>`.
  :param value: The operand to the atomic set operation. The type of **value** should match TYPE and TYPENAME according to the table of :ref:`Extended AMO Types<extamotypes>`.
  :param pe: An integer that indicates the PE number on which **dest** is to be updated.

Callable from the **device**.

**Description:**
``ishmem_atomic_set`` performs an atomic set operation.
It writes the **value** into **dest** on **pe** as an atomic operation.

^^^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEM_ATOMIC_COMPARE_SWAP
^^^^^^^^^^^^^^^^^^^^^^^^^^

Performs an atomic conditional swap on a remote data object.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by the :ref:`Standard AMO Types table<stdamotypes>`.

.. cpp:function:: template<typename TYPE> TYPE ishmem_atomic_compare_swap(TYPE* dest, TYPE cond, TYPE value, int pe)

.. cpp:function:: TYPE ishmem_TYPENAME_atomic_compare_swap(TYPE* dest, TYPE cond, TYPE value, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param cond: **cond** is compared to the remote **dest** value. If **cond** and the remote **dest** are equal, then **value** is swapped into the remote **dest**; otherwise, the remote **dest** is unchanged.  In either case, the old value of the remote **dest** is returned as the routine return value. **cond** must be of the same data type as **dest**.
  :param value: The value to be atomically written to the remote PE. The type of **value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param pe: An integer that indicates the PE number from which **dest** is to be fetched.
  :returns:  The contents that had been in the **dest** data object on the remote PE prior to the conditional swap. Data type is the same as the **dest** data type.

Callable from the **device**.

**Description:**
The conditional swap routines conditionally update a **dest** data object on
the specified PE and return the prior contents of the data object in one
atomic operation.

^^^^^^^^^^^^^^^^^^
ISHMEM_ATOMIC_SWAP
^^^^^^^^^^^^^^^^^^

Performs an atomic swap to a remote data object.

Below, TYPE is one of the extended AMO types and has a corresponding TYPENAME
specified by the :ref:`Extended AMO Types table<extamotypes>`.

.. cpp:function:: template<typename TYPE> TYPE ishmem_atomic_swap(TYPE* dest, TYPE value, int pe)

.. cpp:function:: TYPE ishmem_TYPENAME_atomic_swap(TYPE* dest, TYPE value, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Extended AMO Types<extamotypes>`.
  :param value: The value to be atomically written to the remote PE. The type of **value** should match the TYPE and TYPENAME according to the table of :ref:`Extended AMO Types<extamotypes>`.
  :param pe: An integer that indicates the PE number from which **dest** is to be fetched.
  :returns: The content that had been at the **dest** address on the remote PE prior to the swap.

Callable from the **device**.

**Description:**
``ishmem_atomic_swap`` performs an atomic swap operation.
It writes **value** into **dest** on **pe** and returns the previous contents
of **dest** as an atomic operation.


^^^^^^^^^^^^^^^^^^^^^^^
ISHMEM_ATOMIC_FETCH_INC
^^^^^^^^^^^^^^^^^^^^^^^

Performs an atomic fetch-and-increment operation on a remote data object.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by the :ref:`Standard AMO Types table<stdamotypes>`.

.. cpp:function:: template<typename TYPE> TYPE ishmem_atomic_fetch_inc(TYPE* dest, int pe)

.. cpp:function:: TYPE ishmem_TYPENAME_atomic_fetch_inc(TYPE* dest, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param pe: An integer that indicates the PE number from which **dest** is to be fetched.
  :returns: The content that had been at the **dest** address on the remote PE prior to the increment.  The datatype of the return value is the same as **dest**.

Callable from the **device**.

**Description:**
These routines perform a fetch-and-increment operation.
The **dest** on PE **pe** is increased by one and the routine returns the
previous contents of **dest** as an atomic operation.

^^^^^^^^^^^^^^^^^
ISHMEM_ATOMIC_INC
^^^^^^^^^^^^^^^^^

Performs an atomic increment operation on a remote data object.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by the :ref:`Standard AMO Types table<stdamotypes>`.

.. cpp:function:: template<typename TYPE> void ishmem_atomic_inc(TYPE* dest, int pe)

.. cpp:function:: void ishmem_TYPENAME_atomic_inc(TYPE* dest, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param pe: An integer that indicates the PE number from which **dest** is to be updated.

Callable from the **device**.

**Description:**
These routines perform an atomic increment operation on the **dest** data
object on PE **pe**.

^^^^^^^^^^^^^^^^^^^^^^^
ISHMEM_ATOMIC_FETCH_ADD
^^^^^^^^^^^^^^^^^^^^^^^

Performs an atomic fetch-and-add operation on a remote data object.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by the :ref:`Standard AMO Types table<stdamotypes>`.

.. cpp:function:: template<typename TYPE> TYPE ishmem_atomic_fetch_add(TYPE* dest, TYPE value, int pe)

.. cpp:function:: TYPE ishmem_TYPENAME_atomic_fetch_add(TYPE* dest, TYPE value, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param value: The operand to the atomic fetch-and-add operation. The type of **value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param pe: An integer that indicates the PE number from which **dest** is to be updated.
  :returns: The contents that had been at the **dest** address on the remote PE prior to the atomic addition operation.  The data type of the return value is the same as **dest**.

Callable from the **device**.

**Description:**
``ishmem_atomic_fetch_add`` routines perform an atomic fetch-and-add
operation.
An atomic fetch-and-add operation fetches the old **dest** and adds **value**
to **dest** without the possibility of another atomic operation on the
**dest** between the time of the fetch and the update.
These routines add **value** to **dest** on **pe** and return the previous
contents of **dest** as an atomic operation.

^^^^^^^^^^^^^^^^^
ISHMEM_ATOMIC_ADD
^^^^^^^^^^^^^^^^^

Performs an atomic add operation on a remote symmetric data object.

Below, TYPE is one of the standard AMO types and has a corresponding TYPENAME
specified by the :ref:`Standard AMO Types table<stdamotypes>`.

.. cpp:function:: template<typename TYPE> void ishmem_atomic_add(TYPE* dest, TYPE value, int pe)

.. cpp:function:: void ishmem_TYPENAME_atomic_add(TYPE* dest, TYPE value, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param value: The operand to the atomic add operation. The type of **value** should match the TYPE and TYPENAME according to the table of :ref:`Standard AMO Types<stdamotypes>`.
  :param pe: An integer that indicates the PE number from which **dest** is to be updated.

Callable from the **device**.

**Description:**
The ``ishmem_atomic_add`` routine performs an atomic add operation.
It adds **value** to **dest** on PE **pe** and atomically updates the **dest**
without returning the value.

^^^^^^^^^^^^^^^^^^^^^^^
ISHMEM_ATOMIC_FETCH_AND
^^^^^^^^^^^^^^^^^^^^^^^

Atomically perform a fetching bitwise AND operation on a remote data object.

Below, TYPE is one of the bitwise AMO types and has a corresponding TYPENAME
specified by the :ref:`Bitwise AMO Types table<bitamotypes>`.

.. cpp:function:: template<typename TYPE> TYPE ishmem_atomic_fetch_and(TYPE* dest, TYPE value, int pe)

.. cpp:function:: TYPE ishmem_TYPENAME_atomic_fetch_and(TYPE* dest, TYPE value, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Bitwise AMO Types<bitamotypes>`.
  :param value: The operand to the atomic add operation. The type of **value** should match the TYPE and TYPENAME according to the table of :ref:`Bitwise AMO Types<bitamotypes>`.
  :param pe: An integer that indicates the PE number from which **dest** is to be updated.
  :returns: The value pointed to by **dest** on PE **pe** immediately before the operation is performed.

Callable from the **device**.

**Description:**
``ishmem_atomic_fetch_and`` atomically performs a fetching bitwise AND on the
remotely accessible data object pointed to by **dest** at PE **pe** with the
operand **value**.

^^^^^^^^^^^^^^^^^
ISHMEM_ATOMIC_AND
^^^^^^^^^^^^^^^^^

Atomically perform a non-fetching bitwise AND operation on a remote data
object.

Below, TYPE is one of the bitwise AMO types and has a corresponding TYPENAME
specified by the :ref:`Bitwise AMO Types table<bitamotypes>`.

.. cpp:function:: template<typename TYPE> void ishmem_atomic_and(TYPE* dest, TYPE value, int pe)

.. cpp:function:: void ishmem_TYPENAME_atomic_and(TYPE* dest, TYPE value, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Bitwise AMO Types<bitamotypes>`.
  :param value: The operand to the atomic AND operation. The type of **value** should match the TYPE and TYPENAME according to the table of :ref:`Bitwise AMO Types<bitamotypes>`.
  :param pe: An integer that indicates the PE number from which **dest** is to be updated.

Callable from the **device**.

**Description:**
``ishmem_atomic_and`` atomically performs a non-fetching bitwise AND on the
remotely accessible data object pointed to by **dest** at PE **pe** with the
operand **value**.

^^^^^^^^^^^^^^^^^^^^^^
ISHMEM_ATOMIC_FETCH_OR
^^^^^^^^^^^^^^^^^^^^^^

Atomically perform a fetching bitwise OR operation on a remote data object.

Below, TYPE is one of the bitwise AMO types and has a corresponding TYPENAME
specified by the :ref:`Bitwise AMO Types table<bitamotypes>`.

.. cpp:function:: template<typename TYPE> TYPE ishmem_atomic_fetch_or(TYPE* dest, TYPE value, int pe)

.. cpp:function:: TYPE ishmem_TYPENAME_atomic_fetch_or(TYPE* dest, TYPE value, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Bitwise AMO Types<bitamotypes>`.
  :param value: The operand to the atomic OR operation. The type of **value** should match the TYPE and TYPENAME according to the table of :ref:`Bitwise AMO Types<bitamotypes>`.
  :param pe: An integer that indicates the PE number from which **dest** is to be updated.
  :returns: The value pointed to by **dest** on PE **pe** immediately before the operation is performed.

Callable from the **device**.

**Description:**
``ishmem_atomic_fetch_or`` atomically performs a fetching bitwise OR on the
remotely accessible data object pointed to by **dest** at PE **pe** with the
operand **value**.

^^^^^^^^^^^^^^^^
ISHMEM_ATOMIC_OR
^^^^^^^^^^^^^^^^

Atomically perform a non-fetching bitwise OR operation on a remote data
object.

Below, TYPE is one of the bitwise AMO types and has a corresponding TYPENAME
specified by the :ref:`Bitwise AMO Types table<bitamotypes>`.

.. cpp:function:: template<typename TYPE> void ishmem_atomic_or(TYPE* dest, TYPE value, int pe)

.. cpp:function:: void ishmem_TYPENAME_atomic_or(TYPE* dest, TYPE value, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Bitwise AMO Types<bitamotypes>`.
  :param value: The operand to the bitwise OR operation. The type of **value** should match the TYPE and TYPENAME according to the table of :ref:`Bitwise AMO Types<bitamotypes>`.
  :param pe: An integer that indicates the PE number from which **dest** is to be updated.

Callable from the **device**.

**Description:**
``ishmem_atomic_or`` atomically performs a non-fetching bitwise OR on the
remotely accessible data object pointed to by **dest** at PE **pe** with the
operand **value**.

^^^^^^^^^^^^^^^^^^^^^^^
ISHMEM_ATOMIC_FETCH_XOR
^^^^^^^^^^^^^^^^^^^^^^^

Atomically perform a fetching bitwise exclusive OR (XOR) operation on a
remote data object.

Below, TYPE is one of the bitwise AMO types and has a corresponding TYPENAME
specified by the :ref:`Bitwise AMO Types table<bitamotypes>`.

.. cpp:function:: template<typename TYPE> TYPE ishmem_atomic_fetch_xor(TYPE* dest, TYPE value, int pe)

.. cpp:function:: TYPE ishmem_TYPENAME_atomic_fetch_xor(TYPE* dest, TYPE value, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Bitwise AMO Types<bitamotypes>`.
  :param value: The operand to the atomic XOR operation. The type of **value** should match the TYPE and TYPENAME according to the table of :ref:`Bitwise AMO Types<bitamotypes>`.
  :param pe: An integer that indicates the PE number from which **dest** is to be updated.
  :returns: The value pointed to by **dest** on PE **pe** immediately before the operation is performed.

Callable from the **device**.

**Description:**
``ishmem_atomic_fetch_xor`` atomically performs a fetching bitwise XOR on the
remotely accessible data object pointed to by **dest** at PE **pe** with the
operand **value**.

Callable from the **device**.

**Description:**

^^^^^^^^^^^^^^^^^
ISHMEM_ATOMIC_XOR
^^^^^^^^^^^^^^^^^

Atomically perform a non-fetching bitwise exclusive OR (XOR) operation on a
remote data object.

Below, TYPE is one of the bitwise AMO types and has a corresponding TYPENAME
specified by the :ref:`Bitwise AMO Types table<bitamotypes>`.

.. cpp:function:: template<typename TYPE> void ishmem_atomic_xor(TYPE* dest, TYPE value, int pe)

.. cpp:function:: void ishmem_TYPENAME_atomic_xor(TYPE* dest, TYPE value, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Bitwise AMO Types<bitamotypes>`.
  :param value: The operand to the bitwise XOR operation. The type of **value** should match the TYPE and TYPENAME according to the table of :ref:`Bitwise AMO Types<bitamotypes>`.
  :param pe: An integer that indicates the PE number from which **dest** is to be updated.

Callable from the **device**.

**Description:**
``ishmem_atomic_XOR`` atomically performs a non-fetching bitwise XOR on the
remotely accessible data object pointed to by **dest** at PE **pe** with the
operand **value**.

