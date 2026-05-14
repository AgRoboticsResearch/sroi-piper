"""Piper arm hardware interface over CAN bus.

Wraps piper_sdk.C_PiperInterface_V2 with a clean, LeRobot-free API.
Equivalent to UMI's Arx5Client (ZMQ) — created inside PiperController.run().

All angles are in degrees. All positions are the raw signed values
matching the robot's configured joint orientation.
"""

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    C_PiperInterface_V2 = None

# SDK uses thousandths of degrees for joints, 0.01mm for gripper
_JOINT_SCALE = 1000.0
_GRIPPER_SCALE = 10000.0


class PiperInterface:
    """Thin wrapper over piper_sdk for real-time joint control.

    Usage (inside controller subprocess):
        iface = PiperInterface("can0")
        iface.connect()
        joints = iface.read_joints()          # np.ndarray(6) degrees
        iface.write_joints(target_joints)      # send joint targets
        iface.disconnect()
    """

    def __init__(self, can_port: str = "can0", enable_timeout: float = 5.0):
        self._can_port = can_port
        self._enable_timeout = enable_timeout
        self._piper: Any = None
        self._min_pos: list[float] = []
        self._max_pos: list[float] = []

    def connect(self) -> None:
        if C_PiperInterface_V2 is None:
            raise ImportError("piper_sdk is not installed")

        self._piper = C_PiperInterface_V2(self._can_port)
        self._piper.ConnectPort()
        time.sleep(0.1)

        self._resume_if_needed()
        self._enable()
        self._set_joint_mode()
        self._read_limits()

        logger.info("PiperInterface connected on %s", self._can_port)

    def disconnect(self) -> None:
        if self._piper is not None:
            try:
                self._piper.JointCtrl(0, 0, 0, 0, 25000, 0)
            except Exception:
                pass
            try:
                self._piper.DisablePiper()
            except Exception:
                pass
            self._piper = None

    def read_joints(self) -> np.ndarray:
        """Read current joint angles in degrees. Returns shape (6,)."""
        js = self._piper.GetArmJointMsgs().joint_state
        return np.array([
            js.joint_1 / _JOINT_SCALE,
            js.joint_2 / _JOINT_SCALE,
            js.joint_3 / _JOINT_SCALE,
            js.joint_4 / _JOINT_SCALE,
            js.joint_5 / _JOINT_SCALE,
            js.joint_6 / _JOINT_SCALE,
        ])

    def write_joints(self, joints_deg: np.ndarray) -> None:
        """Send joint target angles in degrees."""
        ints = [int(round(d * _JOINT_SCALE)) for d in joints_deg[:6]]
        self._piper.JointCtrl(*ints)

    def read_gripper(self) -> float:
        """Read gripper position in mm."""
        g = self._piper.GetArmGripperMsgs()
        return g.gripper_state.grippers_angle / _GRIPPER_SCALE

    def write_gripper(self, position_mm: float, speed: int = 1000) -> None:
        """Send gripper target in mm."""
        val = int(round(position_mm * _GRIPPER_SCALE))
        self._piper.GripperCtrl(abs(val), speed, 0x01, 0)

    def disable(self) -> None:
        try:
            self._piper.DisablePiper()
        except Exception:
            pass

    @property
    def min_pos(self) -> list[float]:
        return list(self._min_pos[:6])

    @property
    def max_pos(self) -> list[float]:
        return list(self._max_pos[:6])

    # ---- internal ----

    def _resume_if_needed(self) -> None:
        try:
            status = self._piper.GetArmStatus().arm_status
            if status.motion_status != 0:
                self._piper.EmergencyStop(0x02)
            if getattr(status, "ctrl_mode", 0) == 2:
                self._piper.EmergencyStop(0x02)
        except Exception:
            pass

    def _enable(self) -> None:
        t0 = time.time()
        while True:
            try:
                if self._piper.EnablePiper():
                    return
            except Exception:
                pass
            if time.time() - t0 > self._enable_timeout:
                raise TimeoutError(
                    f"EnablePiper timed out after {self._enable_timeout}s"
                )
            time.sleep(0.01)

    def _set_joint_mode(self) -> None:
        try:
            self._piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        except Exception as e:
            logger.warning("MotionCtrl_2 failed: %s", e)

    def _read_limits(self) -> None:
        try:
            info = self._piper.GetAllMotorAngleLimitMaxSpd()
            motors = info.all_motor_angle_limit_max_spd.motor[1:7]
            self._min_pos = [m.min_angle_limit / 10.0 for m in motors]
            self._max_pos = [m.max_angle_limit / 10.0 for m in motors]
        except Exception as e:
            logger.warning("Could not read joint limits: %s", e)
            self._min_pos = [-180.0] * 6
            self._max_pos = [180.0] * 6
