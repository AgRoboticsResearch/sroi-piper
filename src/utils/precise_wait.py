"""Precise timing utilities for real-time control loops.

Port from UMI (diffusion_policy/common/precise_sleep.py).
"""

import time


def precise_sleep(dt: float, slack: float = 0.001, time_func=time.monotonic) -> None:
    t_start = time_func()
    if dt > slack:
        time.sleep(dt - slack)
    t_end = t_start + dt
    while time_func() < t_end:
        pass


def precise_wait(t_end: float, slack: float = 0.001, time_func=time.monotonic) -> None:
    t_wait = t_end - time_func()
    if t_wait > 0:
        t_sleep = t_wait - slack
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
