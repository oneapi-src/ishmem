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

To install Level Zero, refer to the instructions in [Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver repository](https://github.com/intel/compute-runtime/releases) or to the [installation guide](https://dgpu-docs.intel.com/installation-guides/index.html) for oneAPI users.

## Installation
Intel® SHMEM requires a host OpenSHMEM back-end to be used for host-sided operations support. In particular, it relies on a collection of extension APIs (`shmemx_heap_create`, `shmemx_heap_preinit`, and `shmemx_heap_postinit`) to coordinate the Intel® SHMEM and OpenSHMEM heaps. We recommend [Sandia OpenSHMEM v1.5.3rc1](https://github.com/Sandia-OpenSHMEM/SOS/releases/tag/v1.5.3rc1) or newer for this purpose.

### Building Sandia OpenSHMEM (SOS)
Download the SOS repo to be configured as a back-end for Intel® SHMEM.

```
git clone --recurse-submodules https://github.com/Sandia-OpenSHMEM/SOS.git SOS
```

Build SOS following instructions below. `FI_HMEM` support in the provider is required for use with Intel® SHMEM. To enable `FI_HMEM` with a supported provider, we recommend a specific set of config flags. Below are two examples for configuring and building SOS with two providers supporting `FI_HMEM`. To configure SOS with the `verbs;ofi_rxm` provider, use the following instructions:

```
cd SOS
./autogen.sh
./configure --prefix=<sos_dir> --with-ofi=<ofi_installation> --enable-pmi-simple --enable-ofi-mr=basic --disable-ofi-inject --enable-ofi-hmem --disable-bounce-buffers --enable-hard-polling
make -j
make install
```
To configure SOS with the HPE Slingshot provider `cxi`, please use the following instructions:
```
cd SOS
./autogen.sh
./configure --prefix=<sos_dir> --with-ofi=<ofi_installation> --enable-pmi-simple --enable-ofi-mr=basic --disable-ofi-inject --enable-ofi-hmem --disable-bounce-buffers --enable-ofi-manual-progress --enable-mr-endpoint --disable-nonfetch-amo --enable-manual-progress
make -j
make install
```
To configure SOS with the `psm3` provider, please use the following instructions:
```
cd SOS
./autogen.sh
./configure --prefix=<sos_dir> --with-ofi=<ofi_installation> --enable-pmi-simple --enable-manual-progress --enable-ofi-hmem --disable-bounce-buffers --enable-ofi-mr=basic --enable-mr-endpoint
make -j
make install
```
 
Please choose an appropriate PMI configure flag based on the available PMI client library in the system. Please check for further instructions on [SOS Wiki pages](https://github.com/Sandia-OpenSHMEM/SOS/wiki). Optionally, users may also choose to add `--disable-fortran` since fortran interfaces will not be used.


### Building Intel® SHMEM
Check that the SOS build process has successfully created an `<sos_dir>` directory with `include` and `lib` as subdirectories. Please find `shmem.h` and `shmemx.h` in `include`. 

Build Intel® SHMEM using the following instructions:

```
cd ishmem
mkdir build
cd build
cmake .. -DSHMEM_INSTALL_PREFIX=<sos_dir> -DCMAKE_INSTALL_PREFIX=<ishmem_install_dir>
make -j
```

## Usage

### Launching Example Application

Validate that Intel® SHMEM was built correctly by running an example program.

1. Add the library path for SOS to the environment:

```
export LD_LIBRARY_PATH=<sos_dir>/lib:$LD_LIBRARY_PATH
```

2. Run the example program or test on an allocated node using a process launcher:

```
mpiexec.hydra -n 2 -hosts <allocated_node_id> ./scripts/ishmrun ./test/unit/SHMEM/int_get_device
```

- *Note:* Current supported launchers include: MPI process launchers (i.e. `mpiexec`, `mpiexec.hydra`, `mpirun`, etc.), Slurm (i.e. `srun`, `salloc`, etc.), and PBS (i.e. `qsub`).

- *Note:* Intel® SHMEM execution model requires applications to use a 1:1 mapping between PEs and GPU devices. Attempting to run an application without the ishmrun launch script may result in undefined behavior if this mapping is not maintained.
  - For further details on the device selection, please see [the ONEAPI_DEVICE_SELECTOR](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#oneapi_device_selector).

3. Validate the application ran succesfully; example output:

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

Alternatively, all the tests in a directory (such as `test/unit/SHMEM/`) can be run with the following command:

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
The following values may be assigned to `CTEST_SCHEDULER` at configure-time (ex. `-DCTEST_SCHEDULER=mpi`) to set which scheduler will be used to run tests launched through a call to `ctest`:
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
