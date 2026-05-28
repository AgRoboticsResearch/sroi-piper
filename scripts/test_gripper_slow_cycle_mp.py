#!/usr/bin/env python3
"""Slow open/close cycle test using GripperProcess (mp.Process).

Same as test_gripper_slow_cycle.py but through the shared-memory
queue + ring buffer path used by PiperEnv in production.

Usage:
  python scripts/test_gripper_slow_cycle_mp.py
  python scripts/test_gripper_slow_cycle_mp.py --cycles 3 --speed 0.1
"""

import argparse
import time
from multiprocessing.managers import SharedMemoryManager

from modules.gripper import GripperProcess

CLOSED_RAD = 0.450
OPEN_RAD = -0.158


def main():
    parser = argparse.ArgumentParser(description="Slow gripper cycle via GripperProcess")
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

    shm = SharedMemoryManager()
    shm.start()

    gripper = GripperProcess(
        shm_manager=shm,
        port=args.port,
        frequency=50.0,
        position_min=OPEN_RAD,
        position_max=CLOSED_RAD,
    )

    gripper.start()
    gripper.start_wait()

    try:
        time.sleep(0.3)
        state = gripper.get_state()
        print(f"Range: closed={CLOSED_RAD:+.3f}  open={OPEN_RAD:+.3f}  "
              f"sweep={sweep_range:.3f} rad  speed={args.speed} rad/s")
        print(f"Starting at: {float(state['position']):+.4f} rad\n")

        for cycle in range(1, args.cycles + 1):
            # Close
            pos = float(state["position"])
            print(f"[{cycle}/{args.cycles}] Closing ... {OPEN_RAD:+.3f} -> {CLOSED_RAD:+.3f}")
            while pos < CLOSED_RAD:
                pos = min(pos + step_rad, CLOSED_RAD)
                gripper.send_command(kp=args.kp, kd=args.kd, position=pos)
                state = gripper.get_state()
                print(f"\r  pos={float(state['position']):+.4f}  "
                      f"torque={float(state['torque']):+.3f} Nm  "
                      f"safety={int(state['safety_flag'])}", end="", flush=True)
                time.sleep(step_dt)
            time.sleep(0.3)
            state = gripper.get_state()
            print(f"\n  Settled: pos={float(state['position']):+.4f}  "
                  f"torque={float(state['torque']):+.3f} Nm")

            # Open
            pos = float(state["position"])
            print(f"[{cycle}/{args.cycles}] Opening  ... {CLOSED_RAD:+.3f} -> {OPEN_RAD:+.3f}")
            while pos > OPEN_RAD:
                pos = max(pos - step_rad, OPEN_RAD)
                gripper.send_command(kp=args.kp, kd=args.kd, position=pos)
                state = gripper.get_state()
                print(f"\r  pos={float(state['position']):+.4f}  "
                      f"torque={float(state['torque']):+.3f} Nm  "
                      f"safety={int(state['safety_flag'])}", end="", flush=True)
                time.sleep(step_dt)
            time.sleep(0.3)
            state = gripper.get_state()
            print(f"\n  Settled: pos={float(state['position']):+.4f}  "
                  f"torque={float(state['torque']):+.3f} Nm")

        print(f"\nDone — final position: {float(state['position']):+.4f} rad")

    finally:
        gripper.stop()
        shm.shutdown()


if __name__ == "__main__":
    main()
