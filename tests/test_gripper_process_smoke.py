#!/usr/bin/env python3
"""Smoke test for GripperProcess with real hardware.

Verifies the mp.Process-based gripper starts correctly, processes
commands, and reports state via the ring buffer.

Usage:
  python tests/test_gripper_process_smoke.py --port /dev/ttyACM0
  python tests/test_gripper_process_smoke.py --port /dev/ttyACM0 --calibrate
"""

import argparse
import logging
import time
from multiprocessing.managers import SharedMemoryManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GripperProcess hardware smoke test")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run manual calibration via GripperProcess")
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()

    shm = SharedMemoryManager()
    shm.start()

    from modules.gripper import GripperProcess

    g = GripperProcess(
        shm_manager=shm,
        port=args.port,
        frequency=50.0,
        verbose=True,
    )

    logger.info("Starting GripperProcess...")
    g.start()
    logger.info("GripperProcess pid=%d", g.pid)

    try:
        # 1. Verify startup
        state = g.get_state()
        logger.info("Startup state: pos=%.3f rad, torque=%.3f Nm, "
                     "calibrated=%d, safety=%d, mode=%d",
                     state["position"], state["torque"],
                     state["is_calibrated"], state["safety_flag"],
                     state["mode"])

        # 2. Hold position for 1s, verify no drift/jump
        logger.info("Holding position for 1s (should NOT move)...")
        for i in range(5):
            time.sleep(0.2)
            state = g.get_state()
            logger.info("  t=%.1fs: pos=%.3f rad, torque=%.2f Nm",
                         (i + 1) * 0.2, state["position"], state["torque"])

        # 3. Optional: manual calibration via GripperProcess
        if args.calibrate:
            logger.info("Manual calibration via GripperProcess...")
            logger.info("  Move gripper to FULLY CLOSED, then press Enter")
            input("  >>> ")
            g.calibrate_set_closed()
            time.sleep(0.1)
            state = g.get_state()
            closed = state["position"]
            logger.info("  Closed limit recorded: %.3f rad", closed)

            logger.info("  Move gripper to FULLY OPEN, then press Enter")
            input("  >>> ")
            g.calibrate_set_open()
            time.sleep(0.1)
            state = g.get_state()
            open_pos = state["position"]
            logger.info("  Open limit recorded: %.3f rad", open_pos)

            g.calibrate_confirm()
            time.sleep(0.1)
            state = g.get_state()
            logger.info("  Calibration confirmed: is_calibrated=%d, "
                         "closed=%.3f, open=%.3f",
                         state["is_calibrated"],
                         state["closed_angle"], state["open_angle"])

            # 4. Test position cycling after calibration
            logger.info("Cycling %d times with calibrated limits...", args.cycles)
            for cycle in range(args.cycles):
                # Open
                g.send_command(kp=10.0, kd=1.0, position=open_pos)
                time.sleep(0.6)
                state = g.get_state()
                logger.info("  [%d/%d] OPEN:  target=%.3f  actual=%.3f  err=%.3f",
                             cycle + 1, args.cycles, open_pos,
                             state["position"], abs(state["position"] - open_pos))

                # Close
                g.send_command(kp=10.0, kd=1.0, position=closed)
                time.sleep(0.6)
                state = g.get_state()
                logger.info("  [%d/%d] CLOSE: target=%.3f  actual=%.3f  err=%.3f",
                             cycle + 1, args.cycles, closed,
                             state["position"], abs(state["position"] - closed))

        # 5. Final state
        state = g.get_state()
        logger.info("Final state: pos=%.3f rad, safety=%d, temp=%dC",
                     state["position"], state["safety_flag"],
                     state["temp_mos"])

        logger.info("Smoke test PASSED")

    finally:
        logger.info("Stopping GripperProcess...")
        g.stop()
        shm.shutdown()
        logger.info("Done")


if __name__ == "__main__":
    main()
