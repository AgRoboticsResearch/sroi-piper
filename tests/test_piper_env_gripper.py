#!/usr/bin/env python3
"""Test PiperEnv gripper integration patterns in isolation.

Replicates what PiperEnv.start/exec_actions/get_obs do with the
gripper, without requiring camera or arm controller hardware.

Usage:
  python tests/test_piper_env_gripper.py --port /dev/ttyACM0
"""

import argparse
import logging
import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _norm_to_rad(norm: float, closed_rad: float, open_rad: float) -> float:
    """PiperEnv._gripper_norm_to_rad: 0=closed, 1=open."""
    return closed_rad + norm * (open_rad - closed_rad)


def _rad_to_norm(rad: float, closed_rad: float, open_rad: float) -> float:
    """PiperEnv._gripper_rad_to_norm."""
    denom = open_rad - closed_rad
    if abs(denom) <= 0:
        return 1.0
    return (rad - closed_rad) / denom


def main():
    parser = argparse.ArgumentParser(
        description="PiperEnv gripper integration pattern test"
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--kp", type=float, default=10.0)
    parser.add_argument("--kd", type=float, default=1.0)
    args = parser.parse_args()

    shm = SharedMemoryManager()
    shm.start()

    from modules.gripper import GripperProcess

    # ── Simulate PiperEnv.start() ──────────────────────────────────
    g = GripperProcess(
        shm_manager=shm,
        port=args.port,
        frequency=50.0,
        verbose=True,
    )

    logger.info("Starting GripperProcess (simulates PiperEnv.start)...")
    g.start()
    logger.info("GripperProcess pid=%d", g.pid)

    try:
        # ── Simulate PiperEnv.is_calibrated ────────────────────────
        state = g.get_state()
        logger.info("is_calibrated=%d (expect 0)", state["is_calibrated"])

        # ── Manual calibration via GripperProcess ──────────────────
        logger.info("Manual calibration:")
        logger.info("  1. Move gripper to FULLY CLOSED, press Enter")
        input("  >>> ")
        g.calibrate_set_closed()
        time.sleep(0.1)
        state = g.get_state()
        closed_raw = state["position"]
        logger.info("  Recorded closed at raw position: %.3f rad", closed_raw)

        logger.info("  2. Move gripper to FULLY OPEN, press Enter")
        input("  >>> ")
        g.calibrate_set_open()
        time.sleep(0.1)
        state = g.get_state()
        open_raw = state["position"]
        logger.info("  Recorded open at raw position: %.3f rad", open_raw)

        g.calibrate_confirm()
        time.sleep(0.2)

        state = g.get_state()
        assert state["is_calibrated"] == 1, "Calibration failed!"
        closed_rad = float(state["closed_angle"])
        open_rad = float(state["open_angle"])
        grip_range = abs(closed_rad - open_rad)
        assert grip_range > 0, f"Invalid range: closed={closed_rad}, open={open_rad}"
        logger.info("Calibrated: closed_angle=%.3f, open_angle=%.3f, range=%.3f rad",
                     closed_rad, open_rad, grip_range)

        # Verify semantic values preserved (no swap corrupting meaning)
        # closed_angle is what user recorded as closed, open_angle as open
        logger.info("Semantic check: closed_angle=%.3f (raw closed=%.3f), "
                     "open_angle=%.3f (raw open=%.3f)",
                     closed_rad, closed_raw, open_rad, open_raw)

        # ── Simulate PiperEnv.exec_actions ─────────────────────────
        test_actions = [
            (0.8, "OPEN (torque mode)"),
            (0.5, "MID (position mode)"),
            (0.2, "CLOSE (torque mode)"),
            (0.8, "OPEN (torque mode)"),
        ]

        for norm, desc in test_actions:
            if norm < 0.3:
                g.send_torque(args.kd, -2.0)
                logger.info("  %s: send_torque(kd=%.1f, torque=-2.0)", desc, args.kd)
            elif norm > 0.7:
                g.send_torque(args.kd, 2.0)
                logger.info("  %s: send_torque(kd=%.1f, torque=+2.0)", desc, args.kd)
            else:
                target = _norm_to_rad(norm, closed_rad, open_rad)
                g.send_command(args.kp, args.kd, target)
                logger.info("  %s: send_command(kp=%.1f, kd=%.1f, pos=%.3f)",
                             desc, args.kp, args.kd, target)

            time.sleep(0.5)
            state = g.get_state()
            norm_actual = _rad_to_norm(
                float(state["position"]), closed_rad, open_rad
            )
            logger.info("    → pos=%.3f rad, norm=%.2f, torque=%.2f Nm, safety=%d",
                         state["position"], norm_actual,
                         state["torque"], state["safety_flag"])

        # ── Simulate PiperEnv.get_obs gripper part ─────────────────
        state = g.get_state()
        grip_norm = _rad_to_norm(
            float(state["position"]), closed_rad, open_rad
        )
        logger.info("get_obs gripper_width: %.3f (norm)", grip_norm)
        logger.info("get_obs is_calibrated: %d", state["is_calibrated"])
        logger.info("get_obs safety_flag:   %d", state["safety_flag"])
        logger.info("get_obs temp_mos:       %d C", state["temp_mos"])

        logger.info("PiperEnv integration pattern test PASSED")

    finally:
        logger.info("Stopping (simulates PiperEnv.stop)...")
        g.stop()
        shm.shutdown()
        logger.info("Done")


if __name__ == "__main__":
    main()
