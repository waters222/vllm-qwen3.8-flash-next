# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental lossless, tile-aligned partition of 640 expert channels."""


def expert_partition(layer_index: int, rank: int) -> tuple[int, int]:
    """Alternate wide ranks so each rank stores 7680 channels over 48 layers."""
    if type(layer_index) is not int or not 0 <= layer_index < 48:
        raise ValueError("Expected target layer index 0..47")
    if type(rank) is not int or not 0 <= rank < 4:
        raise ValueError("Expected TP4 rank 0..3")
    widths = (128, 192, 128, 192) if layer_index % 2 == 0 else (192, 128, 192, 128)
    return sum(widths[:rank]), widths[rank]


def checkpoint_partition(
    layer_index: int, rank: int, axis_extent: int
) -> tuple[int, int]:
    """Return the same channel interval in dense, packed, or group-scale units."""
    if type(axis_extent) is not int or axis_extent not in (640, 80, 20):
        raise ValueError("Expected Flash-Next dense/packed/group32 checkpoint axis")
    start, width = expert_partition(layer_index, rank)
    divisor = 640 // axis_extent
    if start % divisor or width % divisor:
        raise ValueError("Partition crosses a packed value or quantization group")
    return start // divisor, width // divisor
