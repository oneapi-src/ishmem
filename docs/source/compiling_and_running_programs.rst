.. _compiling_and_running_programs:

==============================
Compiling and Running Programs
==============================

Let's consider the simple example program from Section :ref:`Writing Intel® 
SHMEM Programs<writing_programs>` and assume the code is in a file called 
``ishmem_example.cpp``.

To compile the program, we must pass the necessary flags to the Intel®
oneAPI DPC++/C++ Compiler.
For example::

$ icpx -I${ISHMEM_INSTALL_DIR}/include -L${ISHMEM_INSTALL_DIR}/lib -fsycl -std=gnu++1z ishmem_example.cpp -o ishmem_example -lsma -lpmi -lze_loader -ldl

where ``ISHMEM_INSTALL_DIR`` is the path to the Intel® SHMEM
installation directory.

While building Intel® SHMEM with the ``ENABLE_OPENSHMEM`` CMake option enabled, it
may be convenient to use the ``oshc++`` compiler wrapper (instead of ``icpx``
directly) to easily include the necessary compilation flags that enable the host
OpenSHMEM back-end.

Intel® SHMEM provides a launcher script, ``ishmrun`` that
assigns the environment variable **ZE_AFFINITY_MASK** so that each PE is
assigned a single SYCL device.
To invoke the ``ishmrun`` script, pass it as the first argument to your
process launcher.
The following example assumes the ``ISHMEM_INSTALL_DIR/bin`` directory is
on your user path and use of the Portable Batch System launcher::

$ aprun -N 12 -n 6 ishmrun ishmem_example

This will launch the example program on 12 PEs with 6 PEs per compute node.

As described in section :ref:`Building Intel® SHMEM<building_ishmem>`, the
following environment variables may be required for execution, depending on the
Intel® SHMEM build configuration::

    ISHMEM_RUNTIME
    ISHMEM_SHMEM_LIB_NAME
    ISHMEM_RUNTIME_USE_OSHMPI

See section :ref:`Library Constants<library_constants>` for more information
about these variables.

Selecting SPIR-V Compilation Targets
------------------------------------

On some systems, you may encounter an error in which the correct SPIR-V targets
are not successfully selected when linking with Intel® SHMEM.  This may result in
problems when using device-initiated communication including compilation
warnings: ::

    icpx: warning: linked binaries do not contain expected 'spir64-unknown-unknown' target; found targets: 'spir64_gen-unknown-unknown' [-Wsycl-target]

as well as runtime errors: ::

    terminate called after throwing an instance of 'sycl::_V1::compile_program_error'
      what():  The program was built for 1 devices
    Build program log for 'Intel(R) Data Center GPU Max 1550':
    Module <0x29941d0>:  Unresolved Symbol <_Z13ishmem_putmemPvPKvmi>
    Module <0x29941d0>:  Unresolved Symbol <_Z13ishmem_putmemPvPKvmi>
    Module <0x29941d0>:  Unresolved Symbol <_Z13ishmem_putmemPvPKvmi>
    Module <0x29941d0>:  Unresolved Symbol <_Z13ishmem_putmemPvPKvmi> -11 (PI_ERROR_BUILD_PROGRAM_FAILURE)

This error can be resolved by indicating the desired target at compile time. To
compile with the appropriate target for a Intel® Data Center GPU Max 1550 (PVC)
GPU, add the following flags when linking: ::

    -fsycl-targets=spir64_gen --start-no-unused-arguments -Xs "-device pvc" --end-no-unused-arguments --start-no-unused-arguments -Xsycl-target-backend "-q" --end-no-unused-arguments
