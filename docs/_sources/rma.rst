.. _rma:

--------------------------
Remote Memory Access (RMA)
--------------------------

The RMA routines described in this section can be used to perform reads from and writes to symmetric data objects. These operations are one-sided, meaning that the PE invoking an operation provides all communication parameters and the targeted PE is passive. A characteristic of one-sided communication is that it decouples communication from synchronization. One-sided communication mechanisms transfer data; however, they do not synchronize the sender of the data with the receiver of the data.

Intel® SHMEM RMA routines are performed on symmetric data objects. The initiator PE of a call is designated as the `origin` PE and the PE targeted by an operation is designated as the `destination` PE. The `source` and `dest` designators refer to the data objects that an operation reads from and writes to. In the case of the remote update routine, `Put`, the origin PE provides the `source` data object and the destination PE provides the `dest` data object. In the case of the remote read routine, `Get`, the origin PE provides the `dest` data object and the destination PE provides the `source` data object.

The destination PE is specified as an integer representing the PE number. If the PE number passed to the routine is invalid, being negative or greater than or equal to the size of the launched job, then the behavior is undefined.

.. The destination PE is specified as an integer representing the PE number. This PE number is relative to the team associated with the communication context being using for the operation. If no context argument is passed to the routine, then the routine operates on the default context, which implies that the PE number is relative to the world team. If the PE number passed to the routine is invalid, being negative or greater than or equal to the size of the Intel® SHMEM team, then the behavior is undefined.

.. Intel® SHMEM RMA routines specified in this section have two variants. In one of the variants, the context handle, ctx, is explicitly passed as an argument. In this variant, the operation is performed on the specified context. If the context handle ctx does not correspond to a valid context, the behavior is undefined. In the other variant, the context handle is not explicitly passed and thus, the operations are performed on the default context.

Intel® SHMEM provides type-generic one-sided communication interfaces via C11 generic selection (C11 §6.5.1.1) for block, scalar, and block-strided put and get communication. Such type-generic routines are supported for the “standard RMA types” listed in Table :ref:`Standard RMA Types<stdrmatypes>`.

The standard RMA types include the exact-width integer types defined in ``stdint.h`` by C996 §7.18.1.1 and C11 §7.20.1.1. When the C translation environment does not provide exact-width integer types with ``stdint.h``, an Intel® SHMEM implemementation is not required to provide support for these types.

.. _stdrmatypes:

**Standard RMA Types:**

===================  ========
TYPE                 TYPENAME
===================  ========
float                float
double               double
char                 char
signed char          schar
short                short
int                  int
long                 long
long long            longlong
unsigned char        uchar
unsigned short       ushort
unsigned int         uint
unsigned long        ulong
unsigned long long   ulonglong
int8_t               int8
int16_t              int16
int32_t              int32
int64_t              int64
uint8_t              uint8
uint16_t             uint16
uint32_t             uint32
uint64_t             uint64
size_t               size
ptrdiff_t            ptrdiff
===================  ========

.. long double       longdouble

^^^^^^^^^^^^
Blocking RMA
^^^^^^^^^^^^

""""""""""
ISHMEM_PUT
""""""""""
The `put` routines provide a method for copying data from a contiguous local
data object to a data object on a specified PE.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename TYPE> void ishmem_put(TYPE* dest, const TYPE* source, size_t nelems, int pe)

.. cpp:function:: void ishmem_TYPENAME_put(TYPE* dest, const TYPE* source, size_t nelems, int pe)

.. cpp:function:: void ishmem_putmem(void* dest, const void* source, size_t nelems, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param source: Local address of the data object containing the data to be copied. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param nelems: Number of elements in the **dest** and **source** arrays. For ``ishmem_putmem`` and ``ishmemx_putmem_work_group``, elements are bytes.
  :param pe: PE number of the remote PE.

Callable from the **host** and **device**.

**Description:**
The `put` routines return after the data has been copied out of the **source**
array on the local PE.
The delivery of data words into the data object on the destination PE may occur
in any order.
Furthermore, two successive `put` routines may deliver data out of order unless
a call to ``ishmem_fence`` or ``ishmemx_fence_work_group`` is introduced
between the two calls.

""""""""""""""""""""""
ISHMEMX_PUT_WORK_GROUP
""""""""""""""""""""""
The `put` routines provide a method for copying data from a contiguous local
data object to a data object on a specified PE.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename TYPE, typename Group> void ishmemx_put_work_group(TYPE* dest, const TYPE* source, size_t nelems, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_TYPENAME_put_work_group(TYPE* dest, const TYPE* source, size_t nelems, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_putmem_work_group(void* dest, const void* source, size_t nelems, int pe, const Group& group)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param source: Local address of the data object containing the data to be copied. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param nelems: Number of elements in the **dest** and **source** arrays. For ``ishmem_putmem`` and ``ishmemx_putmem_work_group``, elements are bytes.
  :param pe: PE number of the remote PE.
  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the `Put` operation.

Callable from the **device**.

**Description:**
The `put` routines return after the data has been copied out of the **source**
array on the local PE.
The delivery of data words into the data object on the destination PE may occur
in any order.
Furthermore, two successive `put_work_group` routines may deliver data out of
order unless a call to ``ishmem_fence`` or ``ishmemx_fence_work_group`` is
introduced between the two calls.

""""""""
ISHMEM_P
""""""""
Copies one data item to a remote PE.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename TYPE> void ishmem_p(TYPE* dest, TYPE value, int pe)

.. cpp:function:: void ishmem_TYPENAME_p(TYPE* dest, TYPE value, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param value: The value to be transferred to **dest** . The type of **value** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param pe: PE number of the remote PE.

Callable from the **host** and **device**.

**Description:**
These routines provide a very low latency put capability for single elements of
the standard RMA types.

As with ``ishmem_put``, these routines start the remote transfer and may
return before the data is delivered to the remote PE.
Use ``ishmem_quiet`` or ``ishmemx_quiet_work_group`` to force completion of
all remote `Put` transfers.

"""""""""""
ISHMEM_IPUT
"""""""""""
Copies strided data to a specified PE.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename TYPE> void ishmem_iput(TYPE* dest, const TYPE* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)

.. cpp:function:: void ishmem_TYPENAME_iput(TYPE* dest, const TYPE* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)

    :param dest: Symmetric address of the destination array data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
    :param source: Local address of the array containing the data to be copied. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
    :param dst: The stride between consecutive elements of the **dest** array. The stride is scaled by the element size of the **dest** array. A value of 1 indicates contiguous data.
    :param sst: The stride between consecutive elements of the **source** array. The stride is scaled by the element size of the **source** array. A value of 1 indicates contiguous data.
    :param nelems: Number of elements in the **dest** and **source** arrays.
    :param pe: PE number of the remote PE.

Callable from the **host** and **device**.

**Description:**
The `iput` routines provide a method for copying strided data elements
(specified by **sst**) of an array from a **source** array on the local PE to
locations specified by stride **dst** on a **dest** array on specified remote
PE.
Both strides, **dst** and **sst**, must be greater than or equal to 1.
The routines return when the data has been copied out of the **source** array on the local PE but not necessarily before the data has been delivered to the remote data object.

"""""""""""""""""""""""
ISHMEMX_IPUT_WORK_GROUP
"""""""""""""""""""""""
Copies strided data to a specified PE.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename TYPE, typename Group> void ishmemx_iput_work_group(TYPE* dest, const TYPE* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_TYPENAME_iput_work_group(TYPE* dest, const TYPE* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const Group& group)

    :param dest: Symmetric address of the destination array data object. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
    :param source: Local address of the array containing the data to be copied. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
    :param dst: The stride between consecutive elements of the **dest** array. The stride is scaled by the element size of the **dest** array. A value of 1 indicates contiguous data.
    :param sst: The stride between consecutive elements of the **source** array. The stride is scaled by the element size of the **source** array. A value of 1 indicates contiguous data.
    :param nelems: Number of elements in the **dest** and **source** arrays.
    :param pe: PE number of the remote PE.
    :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the `Put` operation.

Callable from the **device**.

**Description:**
The `iput` routines provide a method for copying strided data
elements (specified by **sst**) of an array from a **source** array on the
local PE to locations specified by stride **dst** on a **dest** array on
specified remote PE.
Both strides, **dst** and **sst**, must be greater than or equal to 1.
The routines return when the data has been copied out of the **source** array on the local PE but not necessarily before the data has been delivered to the remote data object.

""""""""""""
ISHMEM_GET
""""""""""""
Copies data from a specified PE.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename TYPE> void ishmem_get(TYPE* dest, const TYPE* source, size_t nelems, int pe)

.. cpp:function:: void ishmem_TYPENAME_get(TYPE* dest, const TYPE* source, size_t nelems, int pe)

.. cpp:function:: void ishmem_getmem(void* dest, const void* source, size_t nelems, int pe)

  :param dest: Local address of the data object containing the data to be updated. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param source: Symmetric address of the source data object. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`RMA types<stdrmatypes>`.
  :param nelems: Number of elements in the **dest** and **source** arrays. For ``ishmem_getmem`` and ``ishmemx_getmem_work_group``, elements are bytes.
  :param pe: PE number of the remote PE.

Callable from the **host** and **device**.

**Description:**
The `get` routines provide a method for copying a contiguous symmetric data
object from a different PE to a contiguous data object on the local PE.
The routines return after the data has been delivered to the **dest** array on
the local PE.

""""""""""""""""""""""
ISHMEMX_GET_WORK_GROUP
""""""""""""""""""""""
Copies data from a specified PE.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename TYPE, typename Group> void ishmemx_get_work_group(TYPE* dest, const TYPE* source, size_t nelems, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_TYPENAME_get_work_group(TYPE* dest, const TYPE* source, size_t nelems, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_getmem_work_group(void* dest, const void* source, size_t nelems, int pe)

  :param dest: Local address of the data object containing the data to be updated. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param source: Symmetric address of the source data object. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`RMA types<stdrmatypes>`.
  :param nelems: Number of elements in the **dest** and **source** arrays. For ``ishmem_getmem`` and ``ishmemx_getmem_work_group``, elements are bytes.
  :param pe: PE number of the remote PE.
  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the `Put` operation.

Callable from the **device**.

**Description:**
The `get` routines provide a method for copying a contiguous symmetric data
object from a different PE to a contiguous data object on the local
PE.  The routines return after the data has been delivered to the **dest**
array on the local PE.

""""""""""
ISHMEM_G
""""""""""
Copies one data item from a remote PE.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename TYPE> TYPE ishmem_g(const TYPE* source, int pe)

.. cpp:function:: TYPE ishmem_TYPENAME_g(const TYPE* source, int pe)

  :param source: Symmetric address of the source data object. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param pe: PE number of the remote PE on which **source** resides.
  :returns: Returns a single element of type TYPE.

Callable from the **host** and **device**.

**Description:**
These routines provide a very low latency get capability for single elements of
the standard RMA types.

"""""""""""""
ISHMEM_IGET
"""""""""""""
Copies strided data from a specified PE.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename TYPE> void ishmem_iget(TYPE* dest, const TYPE* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)

.. cpp:function:: void ishmem_TYPENAME_iget(TYPE* dest, const TYPE* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe)

    :param dest: Local address of the array to be updated. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
    :param source: Symmetric address of the source array data object. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
    :param dst: The stride between consecutive elements of the **dest** array. The stride is scaled by the element size of the **dest** array. A value of 1 indicates contiguous data.
    :param sst: The stride between consecutive elements of the **source** array. The stride is scaled by the element size of the **source** array. A value of 1 indicates contiguous data.
    :param nelems: Number of elements in the **dest** and **source** arrays.
    :param pe: PE number of the remote PE.

Callable from the **host** and **device**.

**Description:**
The `iget` routines provide a method for copying strided data elements from a
symmetric array from a specified remote PE to strided locations on a local
array.
The routines return when the data has been copied into the local **dest**
array.

"""""""""""""""""""""""
ISHMEMX_IGET_WORK_GROUP
"""""""""""""""""""""""
Copies strided data from a specified PE.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename TYPE, typename Group> void ishmemx_iget_work_group(TYPE* dest, const TYPE* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_TYPENAME_iget_work_group(TYPE* dest, const TYPE* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, const Group& group)

    :param dest: Local address of the array to be updated. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
    :param source: Symmetric address of the source array data object. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
    :param dst: The stride between consecutive elements of the **dest** array. The stride is scaled by the element size of the **dest** array. A value of 1 indicates contiguous data.
    :param sst: The stride between consecutive elements of the **source** array. The stride is scaled by the element size of the **source** array. A value of 1 indicates contiguous data.
    :param nelems: Number of elements in the **dest** and **source** arrays.
    :param pe: PE number of the remote PE.
    :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the `Put` operation.

Callable from the **device**.

**Description:**
The `iget` routines provide a method for copying strided data
elements from a symmetric array from a specified remote PE to strided
locations on a local array.
The routines return when the data has been copied into the local **dest**
array.

^^^^^^^^^^^^^^^
Nonblocking RMA
^^^^^^^^^^^^^^^

""""""""""""""""
ISHMEM_PUT_NBI
""""""""""""""""
The `nonblocking put` routines provide a method for copying data from a
contiguous local data object to a data object on a specified PE.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename TYPE> void ishmem_put_nbi(TYPE* dest, const TYPE* source, size_t nelems, int pe)

.. cpp:function:: void ishmem_TYPENAME_put_nbi(TYPE* dest, const TYPE* source, size_t nelems, int pe)

.. cpp:function:: void ishmem_putmem_nbi(void* dest, const void* source, size_t nelems, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE of TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param source: Local address of the data object containing the data to be copied. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param nelems: Number of elements in the **dest** and **source** arrays. For ``ishmem_putmem_nbi`` and ``ishmemx_putmem_nbi_work_group``, elements are bytes.
  :param pe: PE number of the remote PE.

Callable from the **host** and **device**.

**Description:**
The `nonblocking put` routines return after initiating the operation.
The operation is considered complete after a subsequent call to
``ishmem_quiet`` or ``ishmemx_quiet_work_group``.
At the completion of the quiet operation, the data has been copied into the
**dest** array on the destination PE.
The delivery of data words into the data object on the destination PE may occur
in any order.
Furthermore, two successive put routines may deliver data out of order unless a
call to ``ishmem_fence`` or ``ishmemx_fence_work_group`` is introduced
between the two calls.

""""""""""""""""""""""""""
ISHMEMX_PUT_NBI_WORK_GROUP
""""""""""""""""""""""""""
The `nonblocking put` routines provide a method for copying data
from a contiguous local data object to a data object on a specified PE.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename TYPE, typename Group> void ishmemx_put_nbi_work_group(TYPE* dest, const TYPE* source, size_t nelems, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_TYPENAME_put_nbi_work_group(TYPE* dest, const TYPE* source, size_t nelems, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_putmem_nbi_work_group(void* dest, const void* source, size_t nelems, int pe)

  :param dest: Symmetric address of the destination data object. The type of **dest** should match the TYPE of TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`. 
  :param source: Local address of the data object containing the data to be copied. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param nelems: Number of elements in the **dest** and **source** arrays. For ``ishmem_putmem_nbi`` and ``ishmemx_putmem_nbi_work_group``, elements are bytes.
  :param pe: PE number of the remote PE.
  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the `Put` operation.

Callable from the **device**.

**Description:**
The `nonblocking put` routines return after initiating the
operation.
The operation is considered complete after a subsequent call to
``ishmem_quiet`` or ``ishmemx_quiet_work_group``.
At the completion of the quiet operation, the data has been copied into the
**dest** array on the destination PE.
The delivery of data words into the data object on the destination PE may occur
in any order.
Furthermore, two successive put routines may deliver data out of order unless a
call to ``ishmem_fence`` or ``ishmemx_fence_work_group`` is introduced
between the two calls.

""""""""""""""""
ISHMEM_GET_NBI
""""""""""""""""
The `nonblocking get` routines provide a method for copying data from a
contiguous remote data object on the specified PE to the local data object.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename TYPE> void ishmem_get_nbi(TYPE* dest, const TYPE* source, size_t nelems, int pe)

.. cpp:function:: void ishmem_TYPENAME_get_nbi(TYPE* dest, const TYPE* source, size_t nelems, int pe)

.. cpp:function:: void ishmem_getmem_nbi(void* dest, const void* source, size_t nelems, int pe)

  :param dest: Local address of the data object containing the data to be updated. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param source: Symmetric address of the source data object. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param nelems: Number of elements in the **dest** and **source** arrays. For ``ishmem_getmem_nbi`` and ``ishmemx_getmem_nbi_work_group``, elements are bytes.
  :param pe: PE number of the remote PE.

Callable from the **host** and **device**.

**Description:**
The `nonblocking get` routines provide a method for copying a contiguous symmetric data
object from a different PE to a contiguous data object on the local PE.
The routines return after initiating the operation.
The operation is considered complete after a subsequent call to
``ishmem_quiet`` or ``ishmemx_quiet_work_group``.
At the completion of the quiet operation, the data has been delivered to the
**dest** array on the local PE.

""""""""""""""""""""""""""
ISHMEMX_GET_NBI_WORK_GROUP
""""""""""""""""""""""""""
The `nonblocking get` routines provide a method for copying data from a
contiguous remote data object on the specified PE to the local data object.

In the functions below, TYPE is one of the standard RMA types and has a
corresponding TYPENAME specified by Table :ref:`Standard RMA
Types<stdrmatypes>`.

.. cpp:function:: template<typename TYPE, typename Group> void ishmemx_get_nbi_work_group(TYPE* dest, const TYPE* source, size_t nelems, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_TYPENAME_get_nbi_work_group(TYPE* dest, const TYPE* source, size_t nelems, int pe, const Group& group)

.. cpp:function:: template<typename Group> void ishmemx_getmem_nbi_work_group(void* dest, const void* source, size_t nelems, int pe)

  :param dest: Local address of the data object containing the data to be updated. The type of **dest** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param source: Symmetric address of the source data object. The type of **source** should match the TYPE and TYPENAME according to the table of :ref:`Standard RMA types<stdrmatypes>`.
  :param nelems: Number of elements in the **dest** and **source** arrays. For ``ishmem_getmem_nbi`` and ``ishmemx_getmem_nbi_work_group``, elements are bytes.
  :param pe: PE number of the remote PE.
  :param group: The SYCL ``group`` or ``sub_group`` on which to collectively perform the `Put` operation.

Callable from the **device**.

**Description:**
The `nonblocking get` routines provide a method for copying a contiguous symmetric data
object from a different PE to a contiguous data object on the local PE.
The routines return after initiating the operation.
The operation is considered complete after a subsequent call to
``ishmem_quiet`` or ``ishmemx_quiet_work_group``.
At the completion of the quiet operation, the data has been delivered to the
**dest** array on the local PE.

