#!/usr/bin/env python3
"""Read Piper arm state via ring buffer (dry_run — motors disabled).

Starts PiperController in dry_run mode. Arm state is read from the
SharedMemoryRingBuffer with zero blocking I/O. No motor movement.

Usage:
  conda activate lerobot_piper_sroi
  PYTHONPATH=src:$PYTHONPATH python tests/test_piper_state.py
  PYTHONPATH=src:$PYTHONPATH python tests/test_piper_state.py --can_port can0 --duration 30
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = str(PROJECT_ROOT / "src" / "utils" / "piper_urdf" / "piper.urdf")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Read Piper arm state from ring buffer (dry_run, motors disabled)"
    )
    parser.add_argument("--can_port", type=str, default="can0")
    parser.add_argument("--urdf_path", type=str, default=URDF_PATH)
    parser.add_argument("--control_hz", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=30.0,
                        help="How long to read state (seconds)")
    args = parser.parse_args()

    from multiprocessing.managers import SharedMemoryManager
    from modules.piper_controller import PiperController

    shm = SharedMemoryManager()
    shm.start()

    controller = PiperController(
        shm_manager=shm,
        can_port=args.can_port,
        urdf_path=args.urdf_path,
        target_frame="ee_link",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        frequency=args.control_hz,
        dry_run=True,
        verbose=True,
    )
    controller.start()
    controller.start_wait()
    logger.info("PiperController ready (dry_run=True, motors DISABLED)")
    logger.info("Move the arm by hand to see state changes in real time")

    t_start = time.monotonic()
    last_log = t_start

    try:
        while time.monotonic() - t_start < args.duration:
            state = controller.get_state()

            joints = state["ActualJointState"]
            ee = state["ActualEEPose"]
            grip = float(state["gripper"])

            now = time.monotonic()
            if now - last_log >= 0.5:
                joint_str = ", ".join(f"{j:+.1f}°" for j in joints)
                ee_pos = ee[:3] * 1000  # m → mm
                ee_rot = np.rad2deg(ee[3:])
                logger.info(
                    "Joints: [%s]  |  EE: [%+.0f %+.0f %+.0f] mm [%+.0f %+.0f %+.0f]°  |  grip=%.2f",
                    joint_str,
                    ee_pos[0], ee_pos[1], ee_pos[2],
                    ee_rot[0], ee_rot[1], ee_rot[2],
                    grip,
                )
                last_log = now

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        controller.stop()
        shm.shutdown()
        logger.info("Done")


if __name__ == "__main__":
    main()
