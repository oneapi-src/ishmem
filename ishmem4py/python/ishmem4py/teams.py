# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""Team queries and synchronization helpers."""

from __future__ import annotations

from ._common import (
    TEAM_SHARED,
    TEAM_WORLD,
    Team,
    _check_status,
    _normalize_team,
    _require_initialized,
)
from ._lib import RUNTIME

__all__ = [
    "TEAM_SHARED",
    "TEAM_WORLD",
    "Team",
    "team_my_pe",
    "team_n_pes",
    "team_sync",
    "team_translate_pe",
]


def team_my_pe(team: Team | int | None = None) -> int:
    """Return the calling PE's rank within ``team``."""
    _require_initialized()
    return int(RUNTIME.ishmem4py_team_my_pe(_normalize_team(team)))


def team_n_pes(team: Team | int | None = None) -> int:
    """Return the number of PEs in ``team``."""
    _require_initialized()
    return int(RUNTIME.ishmem4py_team_n_pes(_normalize_team(team)))


def team_translate_pe(src_team: Team | int | None, src_pe: int, dest_team: Team | int | None) -> int:
    """Translate ``src_pe`` from ``src_team`` numbering into ``dest_team`` numbering."""
    _require_initialized()
    return int(
        RUNTIME.ishmem4py_team_translate_pe(
            _normalize_team(src_team),
            src_pe,
            _normalize_team(dest_team),
        )
    )


def team_sync(team: Team | int | None = None) -> None:
    """Synchronize all PEs in ``team``."""
    _require_initialized()
    _check_status(RUNTIME.ishmem4py_team_sync(_normalize_team(team)), "ishmem_team_sync")
