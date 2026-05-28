#!/usr/bin/env python3
"""Calibrate SROI gripper open/close limits.

Reads current position with zero force (kp=0), prompts user to
manually move gripper to closed and open positions, then saves
the calibration values.

Output format matches piper_env.py defaults:
    gripper_closed_rad: float = X.XXX
    gripper_open_rad:  float = X.XXX

Usage:
  python scripts/calibrate_gripper.py
  python scripts/calibrate_gripper.py --port /dev/ttyACM0
"""

import argparse
import time

from modules.gripper import Gripper


def read_stable(g: Gripper, samples: int = 10, hz: float = 10.0) -> float:
    """Read position over N samples and return the average."""
    positions = []
    dt = 1.0 / hz
    for _ in range(samples):
        state = g.send_command(kp=0.0, kd=0.0, position=g.position)
        positions.append(state.position)
        time.sleep(dt)
    return sum(positions) / len(positions)


def main():
    parser = argparse.ArgumentParser(description="Calibrate gripper open/close limits")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    args = parser.parse_args()

    with Gripper(args.port) as g:
        print("=== SROI Gripper Calibration ===")
        print(f"Port: {args.port}")
        print(f"Current position: {g.position:+.4f} rad\n")

        # Step 1: closed position
        print("Step 1: Hold the gripper FULLY CLOSED, then press Enter")
        input("  >>> ")
        closed = read_stable(g)
        print(f"  Closed position: {closed:+.4f} rad\n")

        # Step 2: open position
        print("Step 2: Hold the gripper FULLY OPEN, then press Enter")
        input("  >>> ")
        opened = read_stable(g)
        print(f"  Open position: {opened:+.4f} rad\n")

        # Results
        sweep = abs(closed - opened)
        print("=" * 40)
        print(f"  Closed: {closed:+.4f} rad")
        print(f"  Open:   {opened:+.4f} rad")
        print(f"  Range:  {sweep:.4f} rad ({sweep * 180 / 3.14159:.1f} deg)")
        print("=" * 40)
        print("\nUpdate piper_env.py defaults:")
        print(f"    gripper_closed_rad: float = {closed:.3f}")
        print(f"    gripper_open_rad:  float = {opened:.3f}")
        print("\nAlso update scripts/test_gripper_slow_cycle.py:")
        print(f"    CLOSED_RAD = {closed:.3f}")
        print(f"    OPEN_RAD = {opened:.3f}")


if __name__ == "__main__":
    main()
