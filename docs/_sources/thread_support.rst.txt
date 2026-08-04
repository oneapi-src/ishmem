.. _thread_support_routines:

--------------
Thread Support
--------------

This section specifies the interaction between the Intel® SHMEM interfaces 
and user threads. It also describes the routines that can be used for 
initializing and querying the thread environment. There are four levels 
of threading defined by the specification.

.. _thread_level_constants:

**Thread Level Constants**

========================     ==============================================
Thread Levels                Description
========================     ==============================================
ISHMEM_THREAD_SINGLE         The Intel® SHMEM program must not be 
                             multithreaded.
ISHMEM_THREAD_FUNNELED       The Intel® SHMEM program may be multithreaded.
                             However, the program must ensure that only the 
                             main thread invokes the Intel® SHMEM 
                             interfaces. The main thread is the thread that 
                             invokes either :ref:`ishmem_init<ishmem_init>` 
                             or 
                             :ref:`ishmem_init_thread<ishmem_init_thread>`.
ISHMEM_THREAD_SERIALIZED     The Intel® SHMEM program may be multithreaded. 
                             However, the program must ensure that the 
                             Intel® SHMEM interfaces are not invoked 
                             concurrently by multiple threads.
ISHMEM_THREAD_MULTIPLE       The Intel® SHMEM program may be multithreaded 
                             and any thread may invoke the Intel® SHMEM 
                             interfaces.
========================     ==============================================


The thread level constants must have 
increasing integer values; i.e., ISHMEM_THREAD_SINGLE <
ISHMEM_THREAD_FUNNELED < ISHMEM_THREAD_SERIALIZED < ISHMEM_THREAD_MULTIPLE. 

The following semantics apply to the usage of these models:

#. In the ISHMEM_THREAD_FUNNELED, ISHMEM_THREAD_SERIALIZED, and
   ISHMEM_THREAD_MULTIPLE thread levels, the 
   :ref:`ishmem_init_thread<ishmem_init_thread>` and 
   :ref:`ishmem_finalize<ishmem_finalize>` calls must be invoked by the 
   same thread.

#. Any Intel® SHMEM operation initiated by a thread is considered an action of 
   the PE as a whole. The symmetric heap and symmetric variables scope are 
   not impacted by multiple threads invoking the Intel® SHMEM interfaces. 
   Each PE has a single symmetric heap that is shared by all threads within 
   that PE. For example, a CPU thread invoking a memory allocation routine 
   such as :ref:`ishmem_malloc<ishmem_malloc>` allocates memory that is
   accessible by all threads of the PE. The requirement that the same 
   symmetric heap operations must be executed by all PEs in the same order 
   also applies in a threaded environment. Similarly, the completion of 
   collective operations is not impacted by multiple threads. For example, 
   :ref:`ishmem_barrier_all<ishmem_barrier_all>` is completed when all PEs
   enter and exit the call, even though only one thread in the PE is 
   participating in the collective call.

#. Blocking Intel® SHMEM calls will only block the calling thread, allowing 
   other threads, if available, to continue executing. The calling thread 
   will be blocked until the event on which it is waiting occurs. Once the 
   blocking call is completed, the thread is ready to continue execution. 
   A blocked thread will not prevent progress of other threads on the same PE 
   and will not prevent them from executing other Intel® SHMEM calls when the 
   thread level permits. In addition, a blocked thread will not prevent the 
   progress of Intel® SHMEM calls performed on other PEs.

#. In the ISHMEM_THREAD_MULTIPLE thread level, all Intel® SHMEM calls are 
   thread-safe. That is, any two concurrently running threads may make 
   Intel® SHMEM calls.

#. In the ISHMEM_THREAD_SERIALIZED and ISHMEM_THREAD_MULTIPLE thread levels, 
   if multiple threads call collective routines, including the symmetric 
   heap management routines, it is the programmer’s responsibility to 
   ensure the correct ordering of collective calls.

.. _ishmem_init_thread:

^^^^^^^^^^^^^^^^^^
ISHMEM_INIT_THREAD
^^^^^^^^^^^^^^^^^^

.. cpp:function:: int ishmem_init_thread(int requested, int * provided)

  :param requested: The thread level support requested by the user.
  :param provided: The thread level support provided by the Intel® SHMEM implementation.
  :returns: 0 upon success; otherwise, a nonzero value.

Callable from the **host**.

**Description:**
Initializes the ``ishmem`` library, similar to 
:ref:`ishmem_init<ishmem_init>` and
:ref:`ishmemx_init_attr<ishmemx_init_attr>`. In addition, it also performs the 
initialization required for supporting the provided thread level. 
The argument ``requested`` is used to specify the desired level of thread support. 
The argument ``provided`` returns the support level provided by the library. 
The allowed values for provided and requested are
ISHMEM_THREAD_SINGLE, ISHMEM_THREAD_FUNNELED, ISHMEM_THREAD_SERIALIZED, and
ISHMEM_THREAD_MULTIPLE.

.. important:: As of Intel® SHMEM |version|, only ISHMEM_THREAD_MULTIPLE is supported for device, host, and on_queue APIs.

At the end of the ``ishmem`` program which it initialized, the call to
``ishmem_init_thread`` must be matched with a call to ``ishmem_finalize``.
After the first call to ``ishmem_init_thread``, a subsequent call to
``ishmem_init`` or ``ishmemx_init_attr`` or ``ishmem_init_thread`` 
in the same program results in
undefined behavior.


.. _ishmem_query_thread:

^^^^^^^^^^^^^^^^^^^
ISHMEM_QUERY_THREAD
^^^^^^^^^^^^^^^^^^^

.. cpp:function:: void ishmem_query_thread(int * provided)

  :param provided: The thread level support provided by the Intel® SHMEM implementation.
  :returns: None.

Callable from the **host**.

**Description:**
The ``ishmem_query_thread`` call returns the level of thread support currently 
being provided. The value returned will be same as was returned in provided by 
a call to ``ishmem_init_thread`` during initialization. If the library was 
initialized by ``ishmem_init`` or ``ishmemx_init_attr``, the implementation can 
choose to provide any one of the defined thread levels, and ``ishmem_query_thread`` 
returns this thread level.

This function may be called at any time, regardless of the thread safety level of the library.
