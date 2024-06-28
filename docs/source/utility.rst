.. _utility:

----------------
Utility Routines
----------------

^^^^^^^^^^^^^^^^^^^^^^^^
ISHMEMX_PRINT_MSG_TYPE_T
^^^^^^^^^^^^^^^^^^^^^^^^
An enumeration of different message types.

.. cpp:enum:: ishmemx_print_msg_type_t

  .. cpp:type:: DEBUG

        Debug message type
  .. cpp:type:: WARNING

        Warning message type
  .. cpp:type:: ERROR

        Error message type
  .. cpp:type:: STDOUT

        Any standard output message type
  .. cpp:type:: STDERR

        Any standard error message type

^^^^^^^^^^^^^
ISHMEMX_PRINT
^^^^^^^^^^^^^
Writes the C string pointed by `out` to the standard error.

.. cpp:function:: void ishmemx_print(const char* out)

.. cpp:function:: void ishmemx_print(const char* out, ishmemx_print_msg_type_t msg_type)

.. cpp:function:: void ishmemx_print(const char* file, long int line, const char* func, const char* out, ishmemx_print_msg_type_t msg_type)

  :param out: C string that contains the text to be written.
  :param msg_type: A message type of type ishmemx_print_msg_type_t.
  :param file: C string that can take __FILE__ as input, representing the full path to the current file.
  :param line: A long int that can take __LINE__ as input for the current line number.
  :param func: C string that can take __FUNC__ as input, representing the current function name.  
  :returns: None.

Callable from the **host** and **device**.

**Description:**
This routine prints the C string to the standard error (stderr), unless
**STDOUT** message type is passed as `msg_type`. For user convenience, all
prints include the file, line number, function name, PE ID, and message type,
when ``ISHMEM_ENABLE_VERBOSE_PRINT`` environment variable is used. For
**DEBUG** messages, in addition to the `msg_type`, user should also enable
``ISHMEM_DEBUG``.
