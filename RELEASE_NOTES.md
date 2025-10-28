# Release Notes <!-- omit in toc -->
This document contains a list of new features and known limitations of Intel® SHMEM releases.

## Release 1.5.1

### New Features and Enhancements
- Fix a compilation issue due to missing files

### Known Limitations
- Only [Sandia OpenSHMEM](https://github.com/Sandia-OpenSHMEM/SOS) and [Intel® MPI Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html) are currently supported as the host back-end.
- Not all APIs from OpenSHMEM standard are supported. Please refer to [Supported/Unsupported Features](https://oneapi-src.github.io/ishmem/supported_features.html) to get a complete view.
- Intel® SHMEM requires a one-to-one mapping of PEs to SYCL devices. This implies that Intel® SHMEM executions must launch with a number of processes on each compute node that is no more than the number of available SYCL devices on each one of those nodes. By default, the Intel® SHMEM runtime considers each individual device tile to make up a single SYCL device and assigns a tile per PE.
- All collective operations within a kernel must complete before invoking subsequent kernel-initiated collective operation.
- To run Intel® SHMEM with SOS enabling the Slingshot provider in OFI, environment variable `FI_CXI_OPTIMIZED_MRS=0` must be used. It is also recommended to use `FI_CXI_DEFAULT_CQ_SIZE=131072`.
- To run Intel® SHMEM with SOS enabling the verbs provider, environment variable `MLX5_SCATTER_TO_CQE=0` must be used.
- To run Intel® SHMEM with Intel® MPI Library, environment variable `I_MPI_OFFLOAD=1` must be used. Additionally, `I_MPI_OFFLOAD_RDMA=1` may be necessary for GPU RDMA depending on the OFI provider. Please refer to the [reference guide](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-linux/2021-16/gpu-buffers-support.html) for further details.
- Inter-node communication in Intel® SHMEM requires [dma-buf](https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html) support in the Linux kernel. Inter-node functionality in Intel® SHMEM Release 1.5.0 is tested with SUSE Linux Enterprise Server 15 SP4.
- Support for Intel® Arc™ B-Series GPUs is preliminary. As such, not all APIs are currently supported.
- When using Intel® Arc™ B-Series GPUs, environment variable `RenderCompressedBuffersEnabled=0` is required. This is automatically set when running with the launcher script `ishmrun`.

## Release 1.5.0

### New Features and Enhancements
- Support for new collectives: inclusive and exclusive scan.
- Improved affinity assignment through launcher script `ishmrun`.
- Preliminary support for Intel® Arc™ B-Series GPUs.
- Bug fixes improving functionality.

### Known Limitations
- Only [Sandia OpenSHMEM](https://github.com/Sandia-OpenSHMEM/SOS) and [Intel® MPI Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html) are currently supported as the host back-end.
- Not all APIs from OpenSHMEM standard are supported. Please refer to [Supported/Unsupported Features](https://oneapi-src.github.io/ishmem/supported_features.html) to get a complete view.
- Intel® SHMEM requires a one-to-one mapping of PEs to SYCL devices. This implies that Intel® SHMEM executions must launch with a number of processes on each compute node that is no more than the number of available SYCL devices on each one of those nodes. By default, the Intel® SHMEM runtime considers each individual device tile to make up a single SYCL device and assigns a tile per PE.
- All collective operations within a kernel must complete before invoking subsequent kernel-initiated collective operation.
- To run Intel® SHMEM with SOS enabling the Slingshot provider in OFI, environment variable `FI_CXI_OPTIMIZED_MRS=0` must be used. It is also recommended to use `FI_CXI_DEFAULT_CQ_SIZE=131072`.
- To run Intel® SHMEM with SOS enabling the verbs provider, environment variable `MLX5_SCATTER_TO_CQE=0` must be used.
- To run Intel® SHMEM with Intel® MPI Library, environment variable `I_MPI_OFFLOAD=1` must be used. Additionally, `I_MPI_OFFLOAD_RDMA=1` may be necessary for GPU RDMA depending on the OFI provider. Please refer to the [reference guide](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-linux/2021-16/gpu-buffers-support.html) for further details.
- Inter-node communication in Intel® SHMEM requires [dma-buf](https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html) support in the Linux kernel. Inter-node functionality in Intel® SHMEM Release 1.5.0 is tested with SUSE Linux Enterprise Server 15 SP4.
- Support for Intel® Arc™ B-Series GPUs is preliminary. As such, not all APIs are currently supported.
- When using Intel® Arc™ B-Series GPUs, environment variable `RenderCompressedBuffersEnabled=0` is required. This is automatically set when running with the launcher script `ishmrun`.

## Release 1.4.0

### New Features and Enhancements
- Dependency check improvements during CMake configuration.
- CMake config file for adding Intel® SHMEM as a project dependency.
- Bug fixes improving functionality.

### Known Limitations
- Only [Sandia OpenSHMEM](https://github.com/Sandia-OpenSHMEM/SOS) and [Intel® MPI Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html) are currently supported as the host back-end.
- Not all APIs from OpenSHMEM standard are supported. Please refer to [Supported/Unsupported Features](https://oneapi-src.github.io/ishmem/supported_features.html) to get a complete view.
- Intel® SHMEM requires a one-to-one mapping of PEs to SYCL devices. This implies that Intel® SHMEM executions must launch with a number of processes on each compute node that is no more than the number of available SYCL devices on each one of those nodes. By default, the Intel® SHMEM runtime considers each individual device tile to make up a single SYCL device and assigns a tile per PE.
- All collective operations within a kernel must complete before invoking subsequent kernel-initiated collective operation.
- To run Intel® SHMEM with SOS enabling the Slingshot provider in OFI, environment variable `FI_CXI_OPTIMIZED_MRS=0` must be used. It is also recommended to use `FI_CXI_DEFAULT_CQ_SIZE=131072`.
- To run Intel® SHMEM with SOS enabling the verbs provider, environment variable `MLX5_SCATTER_TO_CQE=0` must be used.
- To run Intel® SHMEM with Intel® MPI Library, environment variable `I_MPI_OFFLOAD=1` must be used. Additionally, `I_MPI_OFFLOAD_RDMA=1` may be necessary for GPU RDMA depending on the OFI provider. Please refer to the [reference guide](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-linux/2021-14/gpu-buffers-support.html) for further details.
- Inter-node communication in Intel® SHMEM requires [dma-buf](https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html) support in the Linux kernel. Inter-node functionality in Intel® SHMEM Release 1.4.0 is tested with SUSE Linux Enterprise Server 15 SP4.

## Release 1.3.0

### New Features and Enhancements
- Support for C++ templated APIs for non-Debug build types.
- Bug fixes improving functionality.

### Known Limitations
- Only [Sandia OpenSHMEM](https://github.com/Sandia-OpenSHMEM/SOS) and [Intel® MPI Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html) are currently supported as the host back-end.
- Not all APIs from OpenSHMEM standard are supported. Please refer to [Supported/Unsupported Features](https://oneapi-src.github.io/ishmem/supported_features.html) to get a complete view.
- Intel® SHMEM requires a one-to-one mapping of PEs to SYCL devices. This implies that Intel® SHMEM executions must launch with a number of processes on each compute node that is no more than the number of available SYCL devices on each one of those nodes. By default, the Intel® SHMEM runtime considers each individual device tile to make up a single SYCL device and assigns a tile per PE.
- All collective operations within a kernel must complete before invoking subsequent kernel-initiated collective operation.
- To run Intel® SHMEM with SOS enabling the Slingshot provider in OFI, environment variable `FI_CXI_OPTIMIZED_MRS=0` must be used. It is also recommended to use `FI_CXI_DEFAULT_CQ_SIZE=131072`.
- To run Intel® SHMEM with SOS enabling the verbs provider, environment variable `MLX5_SCATTER_TO_CQE=0` must be used.
- To run Intel® SHMEM with Intel® MPI Library, environment variable `I_MPI_OFFLOAD=1` must be used. Additionally, `I_MPI_OFFLOAD_RDMA=1` may be necessary for GPU RDMA depending on the OFI provider. Please refer to the [reference guide](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-linux/2021-14/gpu-buffers-support.html) for further details.
- Inter-node communication in Intel® SHMEM requires [dma-buf](https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html) support in the Linux kernel. Inter-node functionality in Intel® SHMEM Release 1.3.0 is tested with SUSE Linux Enterprise Server 15 SP4.

## Release 1.2.0

### New Features and Enhancements
- Support for [Intel® MPI Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html) as a host back-end for Intel® SHMEM. Please follow instructions on [Building Intel® SHMEM](https://oneapi-src.github.io/ishmem/building_ishmem.html). 
- Support for `on_queue` API extensions allowing OpenSHMEM operations to be queued on SYCL devices from host. These APIs also allow users option to provide a list of SYCL events as a dependency vector.  
- Experimental support for [OSHMPI](https://github.com/pmodels/oshmpi). Intel® SHMEM can now be configured to run over OSHMPI with suitable MPI back-end. More details are available at [Building Intel® SHMEM](https://oneapi-src.github.io/ishmem/building_ishmem.html).
- Support for Intel® SHMEM on [Intel® Tiber™ AI Cloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/services.html). Please follow instructions [here](https://oneapi-src.github.io/ishmem/ishmem_in_devcloud.html).
- Limited support for OpenSHMEM thread models. Host API support for thread initialization and query routines. 
- Device and host API support for vector point-to-point synchronization operations.
- Support for [OFI Libfabric](https://github.com/ofiwg/libfabric) MLX provider-enabled networks via Intel® MPI Library.
- Bug fixes improving functionality and performance.
- Updated [specification](https://oneapi-src.github.io/ishmem/intro.html) with new feature descriptions and APIs.
- An improved and additional set of [unit tests](test/unit) covering functionality of the new APIs.

### Known Limitations
- Only [Sandia OpenSHMEM](https://github.com/Sandia-OpenSHMEM/SOS) and [Intel® MPI Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html) are currently supported as the host back-end.
- Not all APIs from OpenSHMEM standard are supported. Please refer to [Supported/Unsupported Features](https://oneapi-src.github.io/ishmem/supported_features.html) to get a complete view.
- Intel® SHMEM requires a one-to-one mapping of PEs to SYCL devices. This implies that Intel® SHMEM executions must launch with a number of processes on each compute node that is no more than the number of available SYCL devices on each one of those nodes. By default, the Intel® SHMEM runtime considers each individual device tile to make up a single SYCL device and assigns a tile per PE.
- All collective operations within a kernel must complete before invoking subsequent kernel-initiated collective operation.
- To run Intel® SHMEM with SOS enabling the Slingshot provider in OFI, environment variable `FI_CXI_OPTIMIZED_MRS=0` must be used. It is also recommended to use `FI_CXI_DEFAULT_CQ_SIZE=131072`.
- To run Intel® SHMEM with SOS enabling the verbs provider, environment variable `MLX5_SCATTER_TO_CQE=0` must be used.
- To run Intel® SHMEM with Intel® MPI Library, environment variable `I_MPI_OFFLOAD=1` must be used. Additionally, `I_MPI_OFFLOAD_RDMA=1` may be necessary for GPU RDMA depending on the OFI provider. Please refer to the [reference guide](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-linux/2021-14/gpu-buffers-support.html) for further details.
- The C++ templated APIs are currently available only with a Debug build (using `-DCMAKE_BUILD_TYPE=Debug` during configure).
- Inter-node communication in Intel® SHMEM requires [dma-buf](https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html) support in the Linux kernel. Inter-node functionality in Intel® SHMEM Release 1.2.0 is tested with SUSE Linux Enterprise Server 15 SP4.

## Release 1.1.0

### New Features and Enhancements
- Support for OpenSHMEM 1.5 teams and team-based collective operations.
- Device and host API support for strided RMA operations - ibput and ibget, from OpenSHMEM 1.6.
- Device and host API support for non-blocking atomic operations.
- Device and host API support for size-based RMA and signaling operations.
- Device and host API support for all/any/some versions of point-to-point synchronization operations.
- Device and host API support for signal set, add, and wait-until operations.
- Fixed implementation of `ishmem_free`.
- Compatible with [Sandia OpenSHMEM (SOS)](https://github.com/Sandia-OpenSHMEM/SOS) v1.5.3rc1 and newer releases.
- Support for [OFI](https://github.com/ofiwg/libfabric) PSM3 provider enabled networks via SOS. 
- Updated [specification](https://oneapi-src.github.io/ishmem/intro.html) with the teams API, size-based RMA, non-blocking AMO, team-based collectives, all/any/some flavors of synchronization operations, utility extensions for print messages, etc.
- An improved and additional set of [unit tests](test/unit/SHMEM) covering functionality of the new APIs.
- New [examples](examples/SHMEM) illustrating use cases of Intel® SHMEM functionalities including the Teams APIs.
- Updated [launcher script](scripts/ishmrun) to launch Intel® SHMEM applications on the available SYCL devices in the system.

### Known Limitations
- Only [Sandia OpenSHMEM](https://github.com/Sandia-OpenSHMEM/SOS) is currently supported as the host back-end.
- Not all APIs from OpenSHMEM standard are supported. Please refer to [Supported/Unsupported Features](https://oneapi-src.github.io/ishmem/supported_features.html) to get a complete view.
- Intel® SHMEM requires a one-to-one mapping of PEs to SYCL devices. This implies that Intel® SHMEM executions must launch with a number of processes on each compute node that is no more than the number of available SYCL devices on each one of those nodes. By default, the Intel® SHMEM runtime considers each individual device tile to make up a single SYCL device and assigns a tile per PE.
- All collective operations within a kernel must complete before invoking subsequent kernel-initiated collective operation.
- To run Intel® SHMEM with SOS enabling the Slingshot provider in OFI, environment variable `FI_CXI_OPTIMIZED_MRS=0` must be used. It is also recommended to use `FI_CXI_DEFAULT_CQ_SIZE=131072`.
- To run Intel® SHMEM with SOS enabling the verbs provider, environment variable `MLX5_SCATTER_TO_CQE=0` must be used.
- Inter-node communication in Intel® SHMEM requires [dma-buf](https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html) support in the Linux kernel. Inter-node functionality in Intel® SHMEM Release 1.1.0 is tested with SUSE Linux Enterprise Server 15 SP4.

## Release 1.0.0

### New Features
- OpenSHMEM programming on Intel® GPUs.
- A complete [specification](https://oneapi-src.github.io/ishmem/intro.html) detailing the programming model, supported API, example programs, build and run instructions, etc.
- Device and host API support for OpenSHMEM 1.5 compliant point-to-point Remote Memory Access, Atomic Memory Operations, Signaling, Memory Ordering, and Synchronization Operations.
- Device and host API support for OpenSHMEM collective operations across all PEs.
- Device API support for SYCL work-group and sub-group level extensions of Remote Memory Access, Signaling, Collective, Memory Ordering, and Synchronization Operations.
- Support of C++ template function routines replacing the C11 Generic selection routines from OpenSHMEM specification.
- GPU RDMA support when configured with [Sandia OpenSHMEM](https://github.com/Sandia-OpenSHMEM/SOS) with suitable [OFI](https://github.com/ofiwg/libfabric) providers as host back-end.
- Support of bypassing device-to-device communication and using SYCL USM host memory as symmetric heap via environment variables.
- A comprehensive set of [unit tests](test/unit/SHMEM) to test out functionality of core operations.
- A suite of [performance benchmarks](test/performance) covering device-initiated operation performance for a subset of the operations.
- An implementation of [2D stencil kernel](test/apps/SHMEM/jacobi.cpp) utilizing Intel® SHMEM RMA operations.
- [Examples](examples/SHMEM) to illustrate different use cases of Intel® SHMEM functionalities.
- A [launcher script](scripts/ishmrun) to launch Intel® SHMEM applications on the available SYCL devices in the system with the correct mapping.

### Known Limitations
- Only [Sandia OpenSHMEM](https://github.com/Sandia-OpenSHMEM/SOS) as the host back-end is currently supported.
- Not all APIs from OpenSHMEM standard are supported. Please refer to [Supported/Unsupported Features](https://oneapi-src.github.io/ishmem/supported_features.html) to get a complete view.
- Intel® SHMEM requires a one-to-one mapping of PEs to SYCL devices. This implies that Intel® SHMEM executions must launch with a number of processes on each compute node that is no more than the number of available SYCL devices on each one of those nodes. By default, the Intel® SHMEM runtime considers each individual device tile to make up a single SYCL device.
- Current implementation of `ishmem_free` does not release memory for use in subsequent allocations.
- Intel® SHMEM does not yet support teams-based collectives. All collectives must operate on the world team.
- All collective operations must complete before another kernel calls collective operations.
- Intel® SHMEM forces assigning a single tile per PE when using `ZE_FLAT_DEVICE_HIERARCHY` in `COMBINED` or `COMPOSITE` mode.
- To run Intel® SHMEM with SOS enabling the Slingshot provider in OFI, environment variable `FI_CXI_OPTIMIZED_MRS=0` must be used. It is also recommended to use `FI_CXI_DEFAULT_CQ_SIZE=131072`.
- To run Intel® SHMEM with SOS enabling the verbs provider, environment variable `MLX5_SCATTER_TO_CQE=0` must be used.
- Inter-node communication in Intel® SHMEM requires [dma-buf](https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html) support in the Linux kernel. Inter-node functionality in Intel® SHMEM Release 1.0.0 is tested with SUSE Linux Enterprise Server 15 SP4.
