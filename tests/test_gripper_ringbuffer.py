#!/usr/bin/env python3
"""Test DM4310 gripper via GripperProcess — ring buffer + queue (non-blocking).

GripperProcess runs serial I/O in a subprocess. The main process sends
commands via SharedMemoryQueue and reads state from SharedMemoryRingBuffer
— zero blocking, same pattern as RealSenseCamera.

Usage:
  conda activate lerobot_piper_sroi
  PYTHONPATH=src:$PYTHONPATH python tests/test_gripper_ringbuffer.py
  PYTHONPATH=src:$PYTHONPATH python tests/test_gripper_ringbuffer.py --cycles 5 --open_rad -0.139 --closed_rad 0.734
"""

import argparse
import logging
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Test GripperProcess (ring buffer + queue, non-blocking)"
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--kp", type=float, default=10.0)
    parser.add_argument("--kd", type=float, default=1.0)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--closed_rad", type=float, default=0.734)
    parser.add_argument("--open_rad", type=float, default=-0.139)
    parser.add_argument("--step_time", type=float, default=1.0,
                        help="Seconds between open and close commands")
    args = parser.parse_args()

    from multiprocessing.managers import SharedMemoryManager
    from modules.gripper import GripperProcess

    shm = SharedMemoryManager()
    shm.start()

    gripper = GripperProcess(
        shm_manager=shm,
        port=args.port,
        frequency=50.0,
    )
    gripper.start()
    gripper.start_wait()
    logger.info("GripperProcess ready — command via queue, state via ring buffer")

    # ── Initial state ─────────────────────────────────────────
    time.sleep(0.3)
    state = gripper.get_state()
    logger.info(
        "Initial: pos=%.3f rad  vel=%.3f rad/s  torque=%.3f Nm  temp=%d C",
        float(state["position"]), float(state["velocity"]),
        float(state["torque"]), int(state["temp_mos"]),
    )

    # ── Open/Close cycles ─────────────────────────────────────
    for cycle in range(1, args.cycles + 1):
        # OPEN
        gripper.send_command(kp=args.kp, kd=args.kd, position=args.open_rad)
        time.sleep(args.step_time)
        state = gripper.get_state()
        logger.info(
            "[%d/%d] OPEN:  pos=%+.3f rad  torque=%+.3f Nm  temp=%d C",
            cycle, args.cycles,
            float(state["position"]), float(state["torque"]),
            int(state["temp_mos"]),
        )

        # CLOSE
        gripper.send_command(kp=args.kp, kd=args.kd, position=args.closed_rad)
        time.sleep(args.step_time)
        state = gripper.get_state()
        logger.info(
            "[%d/%d] CLOSE: pos=%+.3f rad  torque=%+.3f Nm  temp=%d C",
            cycle, args.cycles,
            float(state["position"]), float(state["torque"]),
            int(state["temp_mos"]),
        )

    # ── Shutdown ──────────────────────────────────────────────
    gripper.stop()
    shm.shutdown()
    logger.info("Done — GripperProcess stopped, ring buffer + queue verified")


if __name__ == "__main__":
    main()
