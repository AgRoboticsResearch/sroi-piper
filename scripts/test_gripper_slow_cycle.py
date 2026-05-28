#!/usr/bin/env python3
"""Slow open/close cycle test for SROI gripper.

Sweeps smoothly between calibrated open and closed positions.
Does NOT auto-home — uses the known calibration range.

Usage:
  python scripts/test_gripper_slow_cycle.py
  python scripts/test_gripper_slow_cycle.py --cycles 3 --speed 0.5
"""

import argparse
import time

from modules.gripper import Gripper

CLOSED_RAD = 0.450
OPEN_RAD = -0.158


def main():
    parser = argparse.ArgumentParser(description="Slow gripper open/close cycle")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--kp", type=float, default=10.0)
    parser.add_argument("--kd", type=float, default=1.0)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--speed", type=float, default=0.05,
                        help="Radians per second (lower = slower)")
    args = parser.parse_args()

    sweep_range = CLOSED_RAD - OPEN_RAD
    step_dt = 0.02  # 50 Hz command rate
    step_rad = args.speed * step_dt

    with Gripper(args.port) as g:
        print(f"Range: closed={CLOSED_RAD:+.3f}  open={OPEN_RAD:+.3f}  "
              f"sweep={sweep_range:.3f} rad  speed={args.speed} rad/s")
        print(f"Starting at: {g.position:+.4f} rad\n")

        for cycle in range(1, args.cycles + 1):
            # Close
            pos = g.position
            print(f"[{cycle}/{args.cycles}] Closing ... {OPEN_RAD:+.3f} -> {CLOSED_RAD:+.3f}")
            while pos < CLOSED_RAD:
                pos = min(pos + step_rad, CLOSED_RAD)
                state = g.send_command(args.kp, args.kd, pos)
                print(f"\r  pos={state.position:+.4f}  torque={state.torque:+.3f} Nm", end="", flush=True)
                time.sleep(step_dt)
            state = g.read_state()
            print(f"\n  Settled: pos={state.position:+.4f}  torque={state.torque:+.3f} Nm")

            # Open
            pos = g.position
            print(f"[{cycle}/{args.cycles}] Opening  ... {CLOSED_RAD:+.3f} -> {OPEN_RAD:+.3f}")
            while pos > OPEN_RAD:
                pos = max(pos - step_rad, OPEN_RAD)
                state = g.send_command(args.kp, args.kd, pos)
                print(f"\r  pos={state.position:+.4f}  torque={state.torque:+.3f} Nm", end="", flush=True)
                time.sleep(step_dt)
            state = g.read_state()
            print(f"\n  Settled: pos={state.position:+.4f}  torque={state.torque:+.3f} Nm")

        print(f"\nDone — final position: {g.position:+.4f} rad")


if __name__ == "__main__":
    main()
