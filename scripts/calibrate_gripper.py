#!/usr/bin/env python3
"""Calibrate SROI gripper open/close limits.

Reads current position with zero force (kp=0), prompts user to
manually move gripper to closed and open positions, then auto-updates
calibration values in all pipeline files.

Files updated:
  - src/modules/piper_env.py
  - scripts/test_gripper_slow_cycle.py
  - scripts/test_gripper_slow_cycle_mp.py

Usage:
  python scripts/calibrate_gripper.py
  python scripts/calibrate_gripper.py --port /dev/ttyACM0
"""

import argparse
import re
import time
from pathlib import Path

from modules.gripper import Gripper

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES_TO_UPDATE = {
    "piper_env.py": (
        PROJECT_ROOT / "src" / "modules" / "piper_env.py",
        [
            (r"gripper_closed_rad: float = [+-]?\d+\.\d+", f"gripper_closed_rad: float = {{closed}}"),
            (r"gripper_open_rad: float = [+-]?\d+\.\d+", f"gripper_open_rad: float = {{open}}"),
        ],
    ),
    "test_gripper_slow_cycle.py": (
        PROJECT_ROOT / "scripts" / "test_gripper_slow_cycle.py",
        [
            (r"CLOSED_RAD = [+-]?\d+\.\d+", "CLOSED_RAD = {closed}"),
            (r"OPEN_RAD = [+-]?\d+\.\d+", "OPEN_RAD = {open}"),
        ],
    ),
    "test_gripper_slow_cycle_mp.py": (
        PROJECT_ROOT / "scripts" / "test_gripper_slow_cycle_mp.py",
        [
            (r"CLOSED_RAD = [+-]?\d+\.\d+", "CLOSED_RAD = {closed}"),
            (r"OPEN_RAD = [+-]?\d+\.\d+", "OPEN_RAD = {open}"),
        ],
    ),
}


def read_stable(g: Gripper, samples: int = 10, hz: float = 10.0) -> float:
    """Read position over N samples and return the average."""
    positions = []
    dt = 1.0 / hz
    for _ in range(samples):
        state = g.send_command(kp=0.0, kd=0.0, position=g.position)
        positions.append(state.position)
        time.sleep(dt)
    return sum(positions) / len(positions)


def update_files(closed: float, opened: float) -> list[str]:
    """Update calibration values in all pipeline files. Returns list of updated files."""
    updated = []
    for label, (path, patterns) in FILES_TO_UPDATE.items():
        if not path.exists():
            print(f"  SKIP {label} (not found)")
            continue
        text = path.read_text()
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement.format(closed=f"{closed:.3f}", open=f"{opened:.3f}"), text)
        path.write_text(text)
        updated.append(label)
    return updated


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

    # Auto-update files
    print("\nUpdating calibration in:")
    updated = update_files(closed, opened)
    for f in updated:
        print(f"  UPDATED {f}")
    print("\nDone")


if __name__ == "__main__":
    main()
