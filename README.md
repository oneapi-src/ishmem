# Intel® SHMEM <!-- omit in toc --> <img align="right" width="100" height="100" src="https://spec.oneapi.io/oneapi-logo-white-scaled.jpg">

[Installation](#installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Usage](#usage)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Release Notes](RELEASE_NOTES.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](https://oneapi-src.github.io/ishmem/intro.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[How to Contribute](CONTRIBUTING.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[License](LICENSE)

Intel® SHMEM provides an efficient implementation of GPU-initiated communication on systems with Intel GPUs.

## Table of Contents <!-- omit in toc -->

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Launching Example Application](#launching-example-application)
- [Additional Resources](#additional-resources)
  - [OpenSHMEM Specification](#openshmem-spec)
  - [Specification](#ishmem-spec)

## Prerequisites

- Linux OS
- Intel® oneAPI DPC++/C++ Compiler 2024.0 or higher.

### SYCL support <!-- omit in toc -->
Intel® oneAPI DPC++/C++ Compiler with Level Zero support.


## Installation

### Building Level Zero
For detailed information on Level Zero, refer to the [Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver repository](https://github.com/intel/compute-runtime/releases) or to the [installation guide](https://dgpu-docs.intel.com/installation-guides/index.html) for oneAPI users.

To install, download the oneAPI Level Zero from the repository.

```
git clone https://github.com/oneapi-src/level-zero.git
```

Build Level Zero following instructions below. 

```
cd level-zero
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=<level_zero_dir> ..
make -j
make install
```
### The Host Back-End Library
Intel® SHMEM requires a host OpenSHMEM or MPI back-end to be used for host-sided operations support. In particular, the OpenSHMEM back-end relies on a collection of extension APIs (`shmemx_heap_create`, `shmemx_heap_preinit`, and `shmemx_heap_postinit`) to coordinate the Intel® SHMEM and OpenSHMEM heaps. We recommend [Sandia OpenSHMEM v1.5.3rc1](https://github.com/Sandia-OpenSHMEM/SOS/releases/tag/v1.5.3rc1) or newer for this purpose. A [work-in-progress branch](https://github.com/davidozog/oshmpi/tree/wip/ishmem) of [OSHMPI](https://github.com/pmodels/oshmpi.git) is also supported but is currently considered experimental.  See the [Building OSHMPI](#building-oshmpi-optional-and-experimental) section before for more details.

We recommend the Intel® MPI Library as the MPI back-end option for the current version of Intel® SHMEM. See the [Building Intel® SHMEM](#building-intel-shmem) section below for more details.

### Building Sandia OpenSHMEM (SOS)
Download the SOS repo to be configured as a back-end for Intel® SHMEM.

```
git clone --recurse-submodules https://github.com/Sandia-OpenSHMEM/SOS.git SOS
```

Build SOS following instructions below. `FI_HMEM` support in the provider is required for use with Intel® SHMEM. To enable `FI_HMEM` with a supported provider, we recommend a specific set of config flags. Below are two examples for configuring and building SOS with two providers supporting `FI_HMEM`. To configure SOS with the `verbs;ofi_rxm` provider, use the following instructions:

```
cd SOS
./autogen.sh
CC=icx CXX=icpx ./configure --prefix=<shmem_dir> --with-ofi=<ofi_installation> --enable-pmi-simple --enable-ofi-mr=basic --disable-ofi-inject --enable-ofi-hmem --disable-bounce-buffers --enable-hard-polling
make -j
make install
```
To configure SOS with the HPE Slingshot provider `cxi`, please use the following instructions:
```
cd SOS
./autogen.sh
CC=icx CXX=icpx ./configure --prefix=<shmem_dir> --with-ofi=<ofi_installation> --enable-pmi-simple --enable-ofi-mr=basic --disable-ofi-inject --enable-ofi-hmem --disable-bounce-buffers --enable-ofi-manual-progress --enable-mr-endpoint --disable-nonfetch-amo --enable-manual-progress
make -j
make install
```
To configure SOS with the `psm3` provider, please use the following instructions:
```
cd SOS
./autogen.sh
CC=icx CXX=icpx ./configure --prefix=<shmem_dir> --with-ofi=<ofi_installation> --enable-pmi-simple --enable-manual-progress --enable-ofi-hmem --disable-bounce-buffers --enable-ofi-mr=basic --enable-mr-endpoint
make -j
make install
```

Please choose an appropriate PMI configure flag based on the available PMI client library in the system. Please check for further instructions on [SOS Wiki pages](https://github.com/Sandia-OpenSHMEM/SOS/wiki). Optionally, users may also choose to add `--disable-fortran` since fortran interfaces will not be used.

### Building OSHMPI (Optional and experimental)
Intel® SHMEM has experimental support for OSHMPI when built using the Intel® MPI Library.
Here is information on how to [Get Started with Intel® MPI Library on Linux](https://www.intel.com/content/www/us/en/docs/mpi-library/get-started-guide-linux/2021-11/overview.html).

To download the OSHMPI repository:

```
git clone -b wip/ishmem --recurse-submodules https://github.com/davidozog/oshmpi.git oshmpi
```
After ensuring Intel® MPI Library is enabled (for example, by sourcing the `/opt/intel/oneapi/setvars.sh` script),
please build OSHMPI following the instructions below.

```
cd oshmpi
./autogen.sh
CC=mpiicx CXX=mpiicpx ./configure --prefix=<shmem_dir> --disable-fortran --enable-rma=direct --enable-amo=direct --enable-async-thread=yes
make -j
make install
```

### Building Intel® SHMEM
Check that the SOS build process has successfully created a `<shmem_dir>` directory with `include` and `lib` as subdirectories. Please find `shmem.h` and `shmemx.h` in `include`.

Build Intel® SHMEM with an OpenSHMEM back-end using the following instructions:

```
cd ishmem
mkdir build
cd build
CC=icx CXX=icpx cmake .. -DENABLE_OPENSHMEM=ON -DSHMEM_DIR=<shmem_dir> -DCMAKE_INSTALL_PREFIX=<ishmem_install_dir>
make -j
```
Alternatively, Intel® SHMEM can be built by enabling an Intel® MPI Library back-end.
Here is information on how to [Get Started with Intel® MPI Library on Linux](https://www.intel.com/content/www/us/en/docs/mpi-library/get-started-guide-linux/2021-11/overview.html).

```
CC=icx CXX=icpx cmake .. -DENABLE_OPENSHMEM=OFF -DENABLE_MPI=ON -DMPI_DIR=<impi_dir> -DCMAKE_INSTALL_PREFIX=<ishmem_install_dir>
```
where `<impi_dir>` is the path to the Intel® MPI Library installation.

Enabling both the OpenSHMEM and MPI back-ends is also supported.  In this case,
the desired backend can be selected via the environment variable,
`ISHMEM_RUNTIME`, which can be set to either "OpenSHMEM" or "MPI".
The default value for `ISHMEM_RUNTIME` is "OpenSHMEM".

## Usage

### Launching Example Application

Validate that Intel® SHMEM was built correctly by running an example program.

1. Add the path for the back-end library to the environment, for example:

```
export LD_LIBRARY_PATH=<shmem_dir>/lib:$LD_LIBRARY_PATH
```

When enabling only the Intel® MPI Library back-end, simply source the appropriate
`setvars.sh` script. When enabling both OpenSHMEM and MPI back-ends, first
source the `setvars.sh` script, then configure the dynamic linker to load the
OpenSHMEM library (for example by prepending `<shmem_dir>/lib` to
`LD_LIBRARY_PATH`).

2. Run the example program or test on an allocated node using a process launcher:

```
ISHMEM_RUNTIME=<back-end> mpiexec.hydra -n 2 -hosts <allocated_node_id> ./scripts/ishmrun ./test/unit/int_get_device
```
where `<back-end>` is the selected host back-end library.

- *Note:* Current supported launchers include: MPI process launchers (i.e. `mpiexec`, `mpiexec.hydra`, `mpirun`, etc.), Slurm (i.e. `srun`, `salloc`, etc.), and PBS (i.e. `qsub`).

- *Note:* Intel® SHMEM execution model requires applications to use a 1:1 mapping between PEs and GPU devices. Attempting to run an application without the ishmrun launch script may result in undefined behavior if this mapping is not maintained.
  - For further details on the device selection, please see [the ONEAPI_DEVICE_SELECTOR](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#oneapi_device_selector).

3. Validate the application ran successfully; example output:

```
Selected device: Intel(R) Data Center GPU Max 1550
Selected vendor: Intel(R) Corporation
Selected device: Intel(R) Data Center GPU Max 1550
Selected vendor: Intel(R) Corporation
No errors
No errors
```

### Launching Example Application w/ CTest

`ctest` can be used to run Intel® SHMEM tests that are generated at compile-time. To see a list of tests available via `ctest`, run:

```
ctest -N
```

To launch a single test, execute:

```
ctest -R <test_name>
```

Alternatively, all the tests in a directory (such as `test/unit/`) can be run with the following command:

```
ctest --test-dir <directory_name>
```

By default, a passed or failed test can be detected by the output:
```
    Start 69: sync-2-gpu
1/1 Test #69: sync-2-gpu .......................   Passed    2.29 sec

100% tests passed, 0 tests failed out of 1
```

To have a test's output printed to the console, add either the `--verbose` or `--output-on-failure` flag to the `ctest` command

### Available Scheduler Wrappers for Jobs Run via CTest
The following values may be assigned to `CTEST_LAUNCHER` at configure-time (ex. `-DCTEST_LAUNCHER=mpi`) to set which scheduler will be used to run tests launched through a call to `ctest`:
 - srun (default)
   - Launches CTest jobs on a single node using Slurm's `srun`.
 - mpi
   - Uses `mpirun` to launch CTest jobs with the appropriate number of processes.
 - qsub
   - Launches CTest jobs on a single node using `qsub`. If this option is being used on a system where a reservation must be made (i.e. via `pbsresnode`) prior to running a test, assign the `JOB_QUEUE` environment variable to the queue associated with your reservation:
   ```
   export JOB_QUEUE=<queue>
   ```

## Additional Resources

### OpenSHMEM Specification

- [OpenSHMEM](http://openshmem.org/site/)
- [Specification](http://openshmem.org/site/sites/default/site_files/OpenSHMEM-1.5.pdf)

### Intel® SHMEM Specification

- [Intel® SHMEM Specification](https://oneapi-src.github.io/ishmem/intro.html)
