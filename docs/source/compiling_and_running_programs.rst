.. _compiling_and_running_programs:

==============================
Compiling and Running Programs
==============================

Consider the simple example program from Section :ref:`Writing Intel® SHMEM
Programs<writing_programs>` and assume the code is in a file called
``ishmem_example.cpp``.

To compile the program with the default static host and static device
libraries, the necessary flags must be passed to the Intel® oneAPI DPC++/C++
Compiler. For example::

$ icpx -I${ISHMEM_INSTALL_DIR}/include -fsycl -std=gnu++1z ishmem_example.cpp ${ISHMEM_INSTALL_DIR}/lib/libishmem_host.a ${ISHMEM_INSTALL_DIR}/lib/libishmem_device.a -o ishmem_example -lpthread -lze_loader

To link the host library dynamically while keeping the device library static::

$ icpx -I${ISHMEM_INSTALL_DIR}/include -fsycl -std=gnu++1z ishmem_example.cpp ${ISHMEM_INSTALL_DIR}/lib/libishmem_host.so ${ISHMEM_INSTALL_DIR}/lib/libishmem_device.a -o ishmem_example -lpthread -lze_loader

where ``ISHMEM_INSTALL_DIR`` is the path to the Intel® SHMEM installation
directory.

Alternatively, when building with CMake, the ``find_package`` command may be
used to define all necessary compiler flags. For example::

    find_package(ISHMEM REQUIRED)
    add_executable(ishmem_example ishmem_example.cpp)
    target_link_libraries(ishmem_example PRIVATE ISHMEM::ISHMEM)

``ISHMEM::ISHMEM`` uses the static host library by default. Configure the
consumer with ``-DISHMEM_HOST_LINK_TYPE=SHARED`` to select
``libishmem_host.so``. Explicit targets are also provided as
``ISHMEM::ISHMEM_STATIC`` and ``ISHMEM::ISHMEM_SHARED``.

If Intel® SHMEM is not sourced via the installed environment script, it may
be necessary to prepend the installation path to ``CMAKE_PREFIX_PATH``.

Intel® SHMEM provides a launcher script, ``ishmrun``, that sets CPU and GPU
affinity so that each PE is assigned a single SYCL device and a corresponding
set of CPU cores with close affinity. To invoke the ``ishmrun`` script, pass it
as the first argument to your process launcher.
The following example assumes the ``ISHMEM_INSTALL_DIR/bin`` directory is
on your user path and use of the Portable Batch System launcher::

$ aprun -N 12 -n 6 ishmrun ishmem_example

This will launch the example program on 12 PEs with 6 PEs per compute node.

As described in section :ref:`Building Intel® SHMEM<building_ishmem>`, the
following environment variables may be required for execution, depending on the
Intel® SHMEM build configuration::

    ISHMEM_RUNTIME
    ISHMEM_MPI_LIB_NAME
    ISHMEM_SHMEM_LIB_NAME
    ISHMEM_RUNTIME_USE_OSHMPI

See section :ref:`Library Constants<library_constants>` for more information
about these variables.

Launcher-Independent MPI Initialization
----------------------------------------------

Intel® SHMEM also supports launcher-independent MPI initialization with
``ishmemx_attr_t.use_uid`` for workflows such as manual process launch or
``torchrun``-style launchers.

This mode requires:

* ``ISHMEM_RUNTIME=MPI``
* ``ishmemx_attr_t.use_uid = true``
* ``ishmemx_attr_t.nranks`` and ``ishmemx_attr_t.rank``
* ``ishmemx_attr_t.uid`` set from ``ishmemx_get_uniqueid``

The following environment variables must be set by the launcher or run script:

* ``WORLD_SIZE``
* ``RANK``
* ``I_MPI_MPCP_SERVER_NAME`` or ``MASTER_ADDR``

Optional variables:

* ``I_MPI_MPCP_SERVER_PORT`` (defaults to ``35555`` when unset)
* ``MASTER_PORT`` (if set, it must be different from ``I_MPI_MPCP_SERVER_PORT``)

Example (manual launch on one node)::

    export WORLD_SIZE=4
    export MASTER_ADDR=localhost
    export ISHMEM_RUNTIME=MPI
    RANK=0 ./app &
    RANK=1 ./app &
    RANK=2 ./app &
    RANK=3 ./app &
    wait

See ``examples/7_uid_mpcp.cpp`` for a complete example using this flow.

Selecting SPIR-V Compilation Targets
------------------------------------

On some systems, you may encounter an error in which the correct SPIR-V targets
are not successfully selected when linking with Intel® SHMEM. This may result in
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

These errors can be resolved by ensuring the desired target(s) match the device
IR bundled into the Intel® SHMEM device library. The target(s) are specified at
Intel® SHMEM's configure time using ``-DISHMEM_AOT_DEVICE_TYPES``. The default
value is ``xe-hpc,xe2`` to target Intel® Data Center Max and Intel® Arc™
B-Series GPUs, respectively. Below is an example set of flags to add to the
linking process for adding these target devices::

    -fsycl-targets=spir64_gen --start-no-unused-arguments -Xs "-device xe-hpc,xe2" --end-no-unused-arguments --start-no-unused-arguments -Xsycl-target-backend "-q" --end-no-unused-arguments

When building with CMake, the ``ISHMEM::ISHMEM`` interface automatically adds
the corresponding target devices to the compilation command.
