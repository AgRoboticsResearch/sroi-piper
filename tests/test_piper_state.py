#!/usr/bin/env python3
"""Test reading Piper joint states via controller ring buffer (dry_run, no movement).

Usage:
  python tests/test_piper_state.py --can_port can0
  python tests/test_piper_state.py --can_port can0 --duration 30 --urdf_path ...
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = str(
    PROJECT_ROOT
    / "third_party" / "lerobot_robot_piper" / "lerobot_robot_piper"
    / "urdf" / "piper_description.urdf"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Read Piper joint states (dry_run)")
    parser.add_argument("--can_port", type=str, default="can0")
    parser.add_argument("--urdf_path", type=str, default=URDF_PATH)
    parser.add_argument("--frequency", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Seconds to read state")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    from multiprocessing.managers import SharedMemoryManager
    from modules.piper_controller import PiperController

    shm_manager = SharedMemoryManager()
    shm_manager.start()

    controller = PiperController(
        shm_manager=shm_manager,
        can_port=args.can_port,
        urdf_path=args.urdf_path,
        frequency=args.frequency,
        dry_run=True,
        verbose=args.verbose,
    )

    try:
        logger.info("Starting controller (dry_run=True, motors DISABLED)...")
        controller.start(wait=True)
        logger.info("Controller ready. Reading state for %.0fs...", args.duration)

        start = time.monotonic()
        while time.monotonic() - start < args.duration:
            state = controller.get_state()
            joints = state["ActualJointState"]
            ee_pose = state["ActualEEPose"]

            logger.info(
                "Joints(deg): [%s] | EE: pos=[%s] rotvec=[%s]",
                " ".join(f"{j:7.2f}" for j in joints),
                " ".join(f"{v:7.3f}" for v in ee_pose[:3]),
                " ".join(f"{v:7.3f}" for v in ee_pose[3:]),
            )
            time.sleep(1.0)

        logger.info("State read complete.")

    except KeyboardInterrupt:
        logger.info("Interrupted")
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
