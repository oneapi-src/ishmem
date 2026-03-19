# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregated public ``ishmem4py`` API."""

from . import collective, init_fini, memory, rma, teams
from ._common import (
    Comparison,
    InitStatus,
    IshmemError,
    IshmemStateError,
    MemoryPointer,
    SignalOp,
    SymmetricMemory,
    TEAM_INVALID,
    TEAM_SHARED,
    TEAM_WORLD,
    Team,
    Version,
)
from .collective import *
from .init_fini import *
from .memory import *
from .rma import *
from .teams import *

__all__ = [
    "Comparison",
    "InitStatus",
    "IshmemError",
    "IshmemStateError",
    "MemoryPointer",
    "SignalOp",
    "SymmetricMemory",
    "TEAM_INVALID",
    "TEAM_SHARED",
    "TEAM_WORLD",
    "Team",
    "Version",
]

__all__ += collective.__all__
__all__ += init_fini.__all__
__all__ += memory.__all__
__all__ += rma.__all__
__all__ += teams.__all__


def tensor(*args, **kwargs):
    from . import torch as torch_interop

    return torch_interop.tensor(*args, **kwargs)


def free_tensor(*args, **kwargs):
    from . import torch as torch_interop

    return torch_interop.free_tensor(*args, **kwargs)


def get_peer_tensor(*args, **kwargs):
    from . import torch as torch_interop

    return torch_interop.get_peer_tensor(*args, **kwargs)


def tensor_base(*args, **kwargs):
    from . import torch as torch_interop

    return torch_interop.tensor_base(*args, **kwargs)


def is_symmetric_tensor(*args, **kwargs):
    from . import torch as torch_interop

    return torch_interop.is_symmetric_tensor(*args, **kwargs)


__all__ += [
    "free_tensor",
    "get_peer_tensor",
    "is_symmetric_tensor",
    "tensor",
    "tensor_base",
]
