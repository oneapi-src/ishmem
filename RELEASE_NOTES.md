# Release Notes <!-- omit in toc -->
This document contains a list of new features and known limitations of Intel® SHMEM in the most recent release.

## Release 1.0.0
### Table of Contents <!-- omit in toc -->
- [New Features](#new-features)
- [Known Limitations](#known-limitations)

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
