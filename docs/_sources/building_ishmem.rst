.. _building_ishmem:

=====================
Building Intel® SHMEM
=====================

To download Intel® SHMEM, either clone the repository::

    git clone https://github.com/oneapi-src/ishmem.git

or download a release tarball from: https://github.com/oneapi-src/ishmem/releases.

The ``README.md`` file describes how to build the dependencies of Intel® SHMEM.

The :ref:`CMake Build Options<cmake_options>` section lists all the available
CMake build options for customizing Intel® SHMEM.

Intel® SHMEM requires enabling a host back-end that is either:

#. a suitable OpenSHMEM v1.5 library
#. Intel® MPI Library (Message Passing Interface)

.. #. PMI (Process Manager Interface)

These options are covered in sections :ref:`Enabling
OpenSHMEM<enabling_openshmem>` and :ref:`Enabling MPI<enabling_mpi>`,
respectively.

.. and :ref:`Enabling PMI<enabling_pmi>`, respectively.

.. _enabling_openshmem:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Enabling the OpenSHMEM back-end
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As of version |release|, Intel® SHMEM supports the following OpenSHMEM
libraries:

- |sos_url| - please use the ``v1.5.3`` release.
- |oshmpi_url| - please use this |oshmpi_branch|.

The OpenSHMEM back-end is enabled by default and is controlled via the
``ENABLE_OPENSHMEM`` CMake option. The installation path is discovered via
``pkg-config``. This can be specified either by adding OpenSHMEM to the
``PKG_CONFIG_PATH`` environment variable, or by using the ``SHMEM_DIR``
CMake option. For example::

    CC=icx CXX=icpx cmake .. -DSHMEM_DIR=<shmem_dir> -DCMAKE_INSTALL_PREFIX=<ishmem_install_dir>

where ``<shmem_dir>`` is the path to the Sandia OpenSHMEM installation
directory, and ``<ishmem_install_dir>`` is the path to the desired Intel®
SHMEM installation directory.

After successfully running the ``cmake`` command, run ``make`` to build the
library and ``make install`` to install the library to
``<ishmem_install_dir>``.

To enable OSHMPI, the build process is similar::

    CC=icx CXX=icpx cmake .. -DSHMEM_DIR=<oshmpi_dir> -DCMAKE_INSTALL_PREFIX=<ishmem_install_dir>

where ``<oshmpi_dir>`` is the path to the OSHMPI installation. Additionally,
setting the following environment variables are required for execution::

    export ISHMEM_SHMEM_LIB_NAME=liboshmpi.so
    export ISHMEM_RUNTIME_USE_OSHMPI=true

.. note:: Enabling the OSHMPI back-end in Intel® SHMEM version |release| may
   require setting environment variable ``OSHMPI_TEAM_SHARED_ONLY_SELF=1`` if the
   ``ISHMEM_TEAM_SHARED`` team does not properly match ``SHMEM_TEAM_SHARED``. This
   is likely the case if the following warning is encountered:
   ``ishmemi_runtime_team_predefined_set SHARED failed. Runtime ISHMEM_TEAM_SHARED
   unable to use SHMEMX_TEAM_NODE``

.. |sos_url| raw:: html

   <a href="https://github.com/Sandia-OpenSHMEM/SOS.git" target="_blank">Sandia OpenSHMEM</a>

.. |oshmpi_url| raw:: html

   <a href="https://github.com/pmodels/oshmpi.git" target="_blank">OSHMPI</a>

.. |oshmpi_branch| raw:: html

   <a href="https://github.com/davidozog/oshmpi/tree/wip/ishmem" target="_blank">experimental branch</a>

.. _enabling_mpi:

^^^^^^^^^^^^^^^^^^^^^^^^^
Enabling the MPI back-end
^^^^^^^^^^^^^^^^^^^^^^^^^

As of version |release|, Intel® SHMEM supports the following MPI libraries:

- |impi_url| - please use the ``2021.14.0`` release or newer.

.. |impi_url| raw:: html

   <a href="https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library-download.html" target="_blank">Intel® MPI Library</a>

The MPI installation path is discovered via CMake's ``find_package``. Depending on the
CMake version, it is typically not necessary to include any extra CMake options. However,
``MPI_DIR`` may be provided to hint at the MPI installation path. For example, to enable
the MPI back-end and disable the OpenSHMEM back-end::

    CC=icx CXX=icpx cmake .. -DENABLE_OPENSHMEM=OFF -DENABLE_MPI=ON -DMPI_DIR=<impi_install_dir> -DCMAKE_INSTALL_PREFIX=<ishmem_install_dir>

where ``<impi_install_dir>`` is the path to the Intel® MPI Library installation.

After successfully running the ``cmake`` command, run ``make`` to build the
library and ``make install`` to install the library to
``<ishmem_install_dir>``.

Note that enabling *both* the OpenSHMEM and MPI back-ends is also supported.
In this case, the desired backend can be selected via the environment variable,
``ISHMEM_RUNTIME``, which can be set to either "OpenSHMEM" or "MPI". For
example building with both runtimes enabled::

    CC=icx CXX=icpx cmake .. -DSHMEM_DIR=<shmem_dir> -DENABLE_MPI=ON -DMPI_DIR=<impi_install_dir>

And running with either ``ISHMEM_RUNTIME=OPENSHMEM`` or ``ISHMEM_RUNTIME=MPI``.

.. TODO: add "PMI" as a possible ISHMEM_RUNTIME option above
.. .. _enabling_pmi:
.. 
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. Enabling the PMI back-end (experimental)
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. 
.. The Process Management Interface (PMI) is planned to be supported as an Intel®
.. SHMEM back-end in a future release.

.. _cmake_options:

^^^^^^^^^^^^^^^^^^^
CMake Build Options
^^^^^^^^^^^^^^^^^^^

+---------------------------------+------------------------------------------------------------+---------+
| **CMake Variable**              | Description                                                | Default |
+=================================+============================================================+=========+
| ``ENABLE_OPENSHMEM``            | Enable OpenSHMEM back-end support                          | ON      |
+---------------------------------+------------------------------------------------------------+---------+
| ``ENABLE_MPI``                  | Enable MPI back-end support                                | OFF     |
+---------------------------------+------------------------------------------------------------+---------+
| ``BUILD_UNIT_TESTS``            | Build unit tests                                           | OFF     |
+---------------------------------+------------------------------------------------------------+---------+
| ``BUILD_PERF_TESTS``            | Build performance tests                                    | OFF     |
+---------------------------------+------------------------------------------------------------+---------+
| ``BUILD_EXAMPLES``              | Build examples                                             | OFF     |
+---------------------------------+------------------------------------------------------------+---------+
| ``BUILD_APPS``                  | Build apps                                                 | OFF     |
+---------------------------------+------------------------------------------------------------+---------+
| ``BUILD_CMAKE_CONFIG``          | Build CMake config files                                   | ON      |
+---------------------------------+------------------------------------------------------------+---------+
| ``ENABLE_ERROR_CHECKING``       | Validate API inputs                                        | OFF     |
+---------------------------------+------------------------------------------------------------+---------+
| ``ENABLE_DLMALLOC``             | Enable dlmalloc for shared heap                            | ON      |
+---------------------------------+------------------------------------------------------------+---------+
| ``ENABLE_REDUCED_LINK_ENGINES`` | Enable reduced link engines (i.e. for single tile devices) | OFF     |
+---------------------------------+------------------------------------------------------------+---------+
| ``ENABLE_AOT_COMPILATION``      | Enables Ahead-Of-Time compilation for GPU kernels          | ON      |
+---------------------------------+------------------------------------------------------------+---------+
| ``SKIP_COMPILER_CHECK``         | Skips compiler validation (NOT RECOMMENDED)                | OFF     |
+---------------------------------+------------------------------------------------------------+---------+
