#!/usr/bin/env python3
"""Test SROI gripper: open/close cycles via DAMIAO DM-FDCAN serial bridge.

Usage:
  python tests/test_gripper.py --port /dev/ttyACM0
  python tests/test_gripper.py --port /dev/ttyACM0 --kp 5.0 --kd 0.5 --cycles 5
"""

import argparse
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test SROI gripper open/close")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--kp", type=float, default=10.0,
                        help="Impedance stiffness (Nm/rad)")
    parser.add_argument("--kd", type=float, default=1.0,
                        help="Impedance damping (Nm·s/rad)")
    parser.add_argument("--cycles", type=int, default=3,
                        help="Number of open/close cycles")
    parser.add_argument("--closed_rad", type=float, default=0.734,
                        help="Encoder position when fully closed")
    parser.add_argument("--open_rad", type=float, default=-0.139,
                        help="Encoder position when fully open")
    parser.add_argument("--step_time", type=float, default=0.5,
                        help="Time between open/close steps")
    args = parser.parse_args()

    from modules.gripper import Gripper

    g = Gripper(port=args.port)
    logger.info("Opening serial port...")
    g.connect()
    logger.info("Connected. Setting zero position...")
    g.set_zero()
    time.sleep(0.5)

    logger.info("Cycling gripper %d times (kp=%.1f, kd=%.1f)", args.cycles, args.kp, args.kd)

    for cycle in range(args.cycles):
        # Open
        g.send_command(kp=args.kp, kd=args.kd, position=args.open_rad)
        time.sleep(args.step_time)
        state = g.state
        logger.info("  [%d/%d] OPEN:  pos=%.3f rad, torque=%.3f Nm, temp=%d C",
                     cycle + 1, args.cycles, state.position, state.torque, state.temp_mos)

        # Close
        g.send_command(kp=args.kp, kd=args.kd, position=args.closed_rad)
        time.sleep(args.step_time)
        state = g.state
        logger.info("  [%d/%d] CLOSE: pos=%.3f rad, torque=%.3f Nm, temp=%d C",
                     cycle + 1, args.cycles, state.position, state.torque, state.temp_mos)

    logger.info("Disabling and disconnecting...")
    g.disconnect()
    logger.info("Done")


if __name__ == "__main__":
    main()
