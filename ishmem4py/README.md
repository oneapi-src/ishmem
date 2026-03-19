# ishmem4py

`ishmem4py` is the in-tree Python binding layer for Intel SHMEM.

The binding stays close to Intel SHMEM/OpenSHMEM naming where that produces a clear Python API,
but it also mirrors the high-level shape of `nvshmem4py` where practical:

- synchronous host-side `put` / `get`
- symmetric-memory handles managed from Python
- world/team queries and team objects
- host collectives over symmetric buffers

The current implementation is intentionally host-driven. Stream-based and device-initiated APIs
remain future work.

## Public API

Main imports:

```python
import ishmem4py as ishmem
# or
import ishmem4py.core as ishmem
```

Primary entry points:

- setup and queries
  - `init`, `finalize`, `init_status`, `is_initialized`
  - `my_pe`, `n_pes`
  - `info_get_version`, `info_get_name`, `get_version`
- symmetric memory
  - `malloc`, `calloc`, `buffer`, `free`
  - `SymmetricMemory`
  - `ptr`, `ishmem_ptr`
- RMA
  - `put`, `get`, `quiet`, `fence`
  - `putmem`, `getmem` compatibility aliases
- collectives
  - `barrier_all`, `sync_all`
  - `barrier`, `sync`
  - `broadcast`, `collect`, `fcollect`, `alltoall`, `reduce`, `reducescatter`
  - `reducescatter` is implemented as a software fallback on top of `reduce`
- teams
  - `TEAM_WORLD`, `TEAM_SHARED`, `TEAM_INVALID`
  - `Team`
  - `team_my_pe`, `team_n_pes`, `team_translate_pe`, `team_sync`

Supported reduction dtypes:

- `int32`
- `int64`
- `uint32`
- `uint64`
- `float32` and `float64` for arithmetic reductions

## Design Notes

`ishmem4py` keeps symmetric allocations as explicit Python objects instead of exposing the
symmetric heap as a direct Python buffer. That matches Intel SHMEM more closely than pretending
every symmetric allocation is safely CPU-dereferenceable.

Data movement is therefore centered on ordinary Python buffers and explicit SHMEM operations:

```python
buf = ishmem.malloc(16)
buf.write(b"abcd")

host = bytearray(16)
ishmem.get(host, buf, pe=0)
```

## Build

Typical build:

```bash
source /opt/intel/oneapi/setvars.sh

cmake -S /docker-mount/ishmem \
  -B /docker-mount/ishmem/build-ishmem4py-icpx-noaot \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DENABLE_OPENSHMEM=ON \
  -DSHMEM_DIR=/home/xiii/pkg/SOS-2026-03-06 \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_APPS=OFF \
  -DENABLE_AOT_COMPILATION=OFF

cmake --build /docker-mount/ishmem/build-ishmem4py-icpx-noaot --target ishmem4py -j4
```

## Run / Test Environment

Source-tree development:

```bash
unset ISHMEM_DIR
export PYTHONPATH=/docker-mount/ishmem/ishmem4py/python:/docker-mount/ishmem/ishmem4py/test
export ISHMEM4PY_RUNTIME_LIBRARY=/docker-mount/ishmem/build-ishmem4py-icpx-noaot/ishmem4py/python/ishmem4py/_ishmem4py_runtime.so
export LD_LIBRARY_PATH=/home/xiii/pkg/SOS-2026-03-06/lib:$LD_LIBRARY_PATH
export ISHMEM_RUNTIME=OPENSHMEM
```

Example test runs:

```bash
mpiexec -n 1 /docker-mount/ishmem/scripts/ishmrun \
  python3 /docker-mount/ishmem/ishmem4py/test/smoke_test.py

mpiexec -n 2 /docker-mount/ishmem/scripts/ishmrun \
  python3 /docker-mount/ishmem/ishmem4py/test/ring_test.py

mpiexec -n 2 /docker-mount/ishmem/scripts/ishmrun \
  python3 /docker-mount/ishmem/ishmem4py/test/collective_test.py
```

## Current Scope

The public Python surface is limited to functionality that has been validated in this environment.
Host atomics and dynamic team-split helpers were intentionally removed from the exported API after
backend-level failures in the current OpenSHMEM runtime.
