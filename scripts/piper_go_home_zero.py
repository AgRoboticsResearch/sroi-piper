#!/usr/bin/env python3
"""Move Piper arm: home pose → zero pose, then hold.

First moves to a safe home position, then to all-zeros.
The controller keeps running and holding position until Ctrl-C.

Usage:
  python scripts/piper_go_home_zero.py --can_port can0
  python scripts/piper_go_home_zero.py --can_port can0 --home_duration 5.0 --zero_duration 5.0
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = str(PROJECT_ROOT / "src" / "utils" / "piper_urdf" / "piper.urdf")

HOME_DEG = np.array([0.0, 50.60, -50.40, -1.21, 10.00, 0.0])
ZERO_DEG = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Move Piper: home → zero, then hold")
    parser.add_argument("--can_port", type=str, default="can0")
    parser.add_argument("--urdf_path", type=str, default=URDF_PATH)
    parser.add_argument("--frequency", type=float, default=50.0)
    parser.add_argument("--max_vel", type=float, default=30.0,
                        help="Max velocity (deg/s)")
    parser.add_argument("--home_duration", type=float, default=4.0)
    parser.add_argument("--zero_duration", type=float, default=5.0)
    args = parser.parse_args()

    from multiprocessing.managers import SharedMemoryManager
    from modules.piper_controller import PiperController

    shm = SharedMemoryManager()
    shm.start()

    controller = PiperController(
        shm_manager=shm,
        can_port=args.can_port,
        urdf_path=args.urdf_path,
        frequency=args.frequency,
        dry_run=False,
        max_vel_deg_s=args.max_vel,
        verbose=True,
    )

    try:
        logger.info("Starting controller (motors ENABLED)...")
        controller.start(wait=True)
        time.sleep(0.5)

        state = controller.get_state()
        logger.info("Current joints: %s deg", np.round(state["ActualJointState"], 1))

        # Step 1: go to home
        logger.info("Moving to HOME %s deg over %.1fs ...", np.round(HOME_DEG, 1), args.home_duration)
        controller.move_to_joints(HOME_DEG, duration=args.home_duration)
        time.sleep(args.home_duration + 1.0)
        state = controller.get_state()
        logger.info("Home reached: %s deg", np.round(state["ActualJointState"], 1))

        # Step 2: go to zero
        logger.info("Moving to ZERO [0, 0, 0, 0, 0, 0] deg over %.1fs ...", args.zero_duration)
        controller.move_to_joints(ZERO_DEG, duration=args.zero_duration)
        time.sleep(args.zero_duration + 1.0)
        state = controller.get_state()
        logger.info("Zero reached: %s deg", np.round(state["ActualJointState"], 1))

        # Hold
        logger.info("Holding zero position. Press Ctrl-C to release.")
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Interrupted — releasing motors")
    except Exception as e:
        logger.error("Error: %s", e)
        import sys; sys.exit(1)
    finally:
        logger.info("Stopping controller...")
        controller.stop()
        shm.shutdown()
        logger.info("Done")


if __name__ == "__main__":
    main()
