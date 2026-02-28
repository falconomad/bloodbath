from __future__ import annotations


def cap_position_size(size: float, min_size: float = 0.0, max_size: float = 1.0) -> float:
    return max(float(min_size), min(float(max_size), float(size)))


def can_open_new_position(current_positions: int, max_positions: int) -> bool:
    return int(current_positions) < int(max_positions)
