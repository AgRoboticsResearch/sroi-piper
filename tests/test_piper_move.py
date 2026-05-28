#!/usr/bin/env python3
"""Test Piper arm movement with motors enabled.

Moves the arm from its current position to a safe home pose using
MOVE_JOINTS (linear joint-space interpolation at the controller level).

Usage:
  python tests/test_piper_move.py --can_port can0
  python tests/test_piper_move.py --can_port can0 --home_pose 0,30,-30,0,10,0 --duration 3.0
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = str(PROJECT_ROOT / "src" / "utils" / "piper_urdf" / "piper.urdf")

SAFE_HOME_DEG = np.array([0.0, 50.60, -50.40, -1.21, 10.00, 0.0])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_joints(s: str) -> np.ndarray:
    return np.array([float(x.strip()) for x in s.split(",")])


def main():
    parser = argparse.ArgumentParser(description="Move Piper arm (motors ENABLED)")
    parser.add_argument("--can_port", type=str, default="can0")
    parser.add_argument("--urdf_path", type=str, default=URDF_PATH)
    parser.add_argument("--frequency", type=float, default=50.0)
    parser.add_argument("--home_pose", type=str, default="0.0,50.60,-50.40,-1.21,10.00,0.0",
                        help="Target joint angles (deg), comma-separated")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Movement duration (seconds)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    home_pose = parse_joints(args.home_pose)
    if len(home_pose) != 6:
        logger.error("home_pose must have exactly 6 joint values")
        sys.exit(1)

    logger.info("Target home pose: %s deg", np.round(home_pose, 1))
    logger.info("WARNING: Motors will be ENABLED. Ensure robot is clear of obstacles.")
    logger.info("Press Ctrl+C to abort, Enter to continue...")
    input()

    from multiprocessing.managers import SharedMemoryManager
    from modules.piper_controller import PiperController

    shm_manager = SharedMemoryManager()
    shm_manager.start()

    controller = PiperController(
        shm_manager=shm_manager,
        can_port=args.can_port,
        urdf_path=args.urdf_path,
        frequency=args.frequency,
        dry_run=False,
        max_vel_deg_s=30.0,
        verbose=args.verbose,
    )

    try:
        logger.info("Starting controller (motors ENABLED)...")
        controller.start(wait=True)
        time.sleep(0.5)

        # Read initial state
        state = controller.get_state()
        start_joints = state["ActualJointState"]
        logger.info("Start joints: %s deg", np.round(start_joints, 1))
        logger.info("Start EE pose: %s", np.round(state["ActualEEPose"], 3))

        # Move to home
        logger.info("Moving to home pose in %.1fs...", args.duration)
        controller.move_to_joints(home_pose, duration=args.duration)

        # Monitor progress
        t_start = time.monotonic()
        while time.monotonic() - t_start < args.duration + 0.2:
            state = controller.get_state()
            joints = state["ActualJointState"]
            ee = state["ActualEEPose"]
            error = np.max(np.abs(joints - home_pose))
            logger.info(
                "Joints: [%s] | max_err=%.2f deg | EE pos=[%.3f,%.3f,%.3f]",
                " ".join(f"{j:7.2f}" for j in joints),
                error,
                ee[0], ee[1], ee[2],
            )
            time.sleep(0.3)

        # Wait for settle
        time.sleep(0.5)
        final_state = controller.get_state()
        logger.info("Final joints: %s deg", np.round(final_state["ActualJointState"], 1))
        logger.info("Final EE pose: %s", np.round(final_state["ActualEEPose"], 3))

        logger.info("Movement test complete. Holding position. Press Ctrl-C to release.")
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Interrupted — disabling motors")
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)
    finally:
        logger.info("Stopping controller...")
        controller.stop()
        shm_manager.shutdown()
        logger.info("Done")


if __name__ == "__main__":
    main()
