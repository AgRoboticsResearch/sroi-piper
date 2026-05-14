"""Time translation utilities for multi-process controller.

UMI pattern: inference uses time.time() (wall clock), controller uses
time.monotonic() (never goes backward). These helpers translate between them.
"""

import time


def wall_to_monotonic(wall_t: float) -> float:
    return wall_t - time.time() + time.monotonic()


def monotonic_to_wall(mono_t: float) -> float:
    return mono_t - time.monotonic() + time.time()
