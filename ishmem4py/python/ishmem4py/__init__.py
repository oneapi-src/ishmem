# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""Public top-level imports for the Intel SHMEM Python bindings."""

from . import core
from .core import *
from .version import __version__

__all__ = list(core.__all__) + ["__version__"]
