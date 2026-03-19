# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import struct


def expect_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def expect_true(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)


def pack_int32(*values: int) -> bytes:
    return struct.pack(f"={len(values)}i", *values)


def pack_uint32(*values: int) -> bytes:
    return struct.pack(f"={len(values)}I", *values)


def unpack_int32(data: bytes) -> int:
    return struct.unpack("=i", data)[0]


def unpack_uint32(data: bytes) -> int:
    return struct.unpack("=I", data)[0]


def unpack_int32_list(data: bytes) -> list[int]:
    if len(data) % 4 != 0:
        raise ValueError("int32 payload size must be a multiple of 4 bytes")
    count = len(data) // 4
    return list(struct.unpack(f"={count}i", data))


def unpack_uint32_list(data: bytes) -> list[int]:
    if len(data) % 4 != 0:
        raise ValueError("uint32 payload size must be a multiple of 4 bytes")
    count = len(data) // 4
    return list(struct.unpack(f"={count}I", data))
