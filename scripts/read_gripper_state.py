#!/usr/bin/env python3
"""Read-only gripper state monitor.

Sends kp=0 MIT commands (no force) to trigger motor response.
Does NOT move the gripper.

Usage:
  python scripts/read_gripper_state.py
  python scripts/read_gripper_state.py --port /dev/ttyACM0 --hz 10
"""

import argparse
import time

from modules.gripper import Gripper


def main():
    parser = argparse.ArgumentParser(description="Gripper state monitor (read-only)")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--hz", type=float, default=10, help="Read rate (Hz)")
    args = parser.parse_args()

    dt = 1.0 / args.hz

    with Gripper(args.port) as g:
        print(f"Connected on {args.port}  |  rate={args.hz}Hz  |  Ctrl-C to stop")
        print(f"{'time':>6s}  {'pos_rad':>8s}  {'vel_rad_s':>9s}  {'torque_Nm':>9s}  {'temp_C':>5s}  status")
        print("-" * 60)
        t0 = time.monotonic()
        try:
            while True:
                # kp=0 → no force applied, just reads state
                state = g.send_command(kp=0.0, kd=0.0, position=g.position)
                t = time.monotonic() - t0
                print(f"{t:6.1f}  {state.position:+8.4f}  {state.velocity:+9.4f}  "
                      f"{state.torque:+9.3f}  {state.temp_mos:5d}  {state.status}")
                time.sleep(dt)
        except KeyboardInterrupt:
            pass

    print("\nDone")


if __name__ == "__main__":
    main()
