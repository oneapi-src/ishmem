# ishmem4py Python Package

`ishmem4py` supports two install modes. The base package provides host-side Intel SHMEM
allocation, RMA, collectives, and team APIs. An optional Torch/XPU extension adds symmetric
`torch.Tensor` allocation plus queue-aware XPU RMA.

## Standard Install

Build `ishmem` with `-DBUILD_PYTHON_BINDINGS=ON`, then install from the build tree:

```bash
pip install /path/to/ishmem-build/ishmem4py/python
```

That package directory includes `_ishmem4py_runtime.so`, so `ISHMEM4PY_RUNTIME_LIBRARY` is not
needed.

To include the optional Torch/XPU interop module in the package, build with
`-DISHMEM4PY_BUILD_TORCH_INTEROP=ON` and use a Python executable from an environment where
PyTorch XPU is already installed:

```bash
cmake -S /path/to/ishmem-src -B /path/to/ishmem-build \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DISHMEM4PY_BUILD_TORCH_INTEROP=ON \
  -DPython3_EXECUTABLE=/path/to/python \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build /path/to/ishmem-build --target ishmem4py -j4
```

## Editable / Dev Install

For active development:

```bash
pip install -e /path/to/ishmem-src/ishmem4py/python
```

Point the editable package at the CMake-built runtime:

```bash
export ISHMEM4PY_RUNTIME_LIBRARY=/path/to/ishmem-build/ishmem4py/python/ishmem4py/_ishmem4py_runtime.so
```

If Torch/XPU interop is enabled in the build tree, the editable package will also find
`_ishmem4py_torch.so` from that package directory through `PYTHONPATH`.

## Runtime Environment

The package still depends on the surrounding Intel SHMEM runtime setup:

```bash
source /opt/intel/oneapi/setvars.sh
unset ISHMEM_DIR
export ISHMEM_RUNTIME=OPENSHMEM
export LD_LIBRARY_PATH=/path/to/openshmem/lib:$LD_LIBRARY_PATH
```

For source-tree testing:

```bash
export PYTHONPATH=/path/to/ishmem-src/ishmem4py/python:/path/to/ishmem-src/ishmem4py/test
```

## Notes

- `import ishmem4py` and `import ishmem4py.core` both expose the full public API.
- Torch/XPU interop is imported lazily through `ishmem4py.tensor(...)`,
  `ishmem4py.free_tensor(...)`, `ishmem4py.get_peer_tensor(...)`,
  `ishmem4py.tensor_base(...)`, and
  `ishmem4py.is_symmetric_tensor(...)` so that plain package import does not eagerly import
  PyTorch.
- `ishmem4py.get_peer_tensor(tensor, pe=...)` mirrors `nvshmem4py`'s peer-tensor helper for
  directly addressable XPU aliases returned by `ishmem_ptr`. The returned tensor is a non-owning
  alias and must not be passed to `ishmem4py.free_tensor(...)`.
- The runtime library is loaded lazily, so importing the package for documentation or static
  inspection does not require a live Intel SHMEM runtime.
- `ishmem.init(device_id=...)` selects an explicit visible XPU ordinal for multi-device launch
  modes. `device_id` may be an integer, an XPU device string such as `xpu:1`, or a
  `torch.device("xpu", 1)`. Leaving it unset preserves the single-visible-device behavior.
- `put`, `get`, and `quiet` accept `queue=` for queue-based host-initiated XPU RMA. Supported
  queue objects are `torch.xpu.Stream`, `ctypes.c_void_p`, and raw integer SYCL queue pointers.
- The current Torch/XPU MVP is focused on XPU device memory and XPU-to-XPU one-sided transfers.
- The exported API is limited to functionality that has been validated in the current
  OpenSHMEM-backed test environment.
