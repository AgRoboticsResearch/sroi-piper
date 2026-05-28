"""Piper environment: orchestrates camera, controller (arm), and gripper.

UMI-ARX pattern: the env is the top-level orchestrator. It manages all
hardware subsystems, handles timestamp alignment for observations, and
dispatches actions. The controller focuses on arm servo only — the env
owns the gripper directly.

Usage:
    from modules.piper_env import PiperEnv

    env = PiperEnv(
        shm_manager=shm_manager,
        can_port="can0",
        urdf_path="...",
        gripper_port="/dev/ttyACM0",
        dry_run=True,
    )
    with env:
        obs = env.get_obs()
        env.exec_actions(actions, timestamps)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PiperEnv:
    """Top-level orchestrator: camera + arm controller + gripper.

    Gripper is managed here (not in the controller) because it's a
    separate hardware device on its own serial port.
    """

    def __init__(
        self,
        # Controller (arm)
        shm_manager,
        can_port: str = "can0",
        urdf_path: str = "",
        target_frame: str = "ee_link",
        joint_names: list[str] | None = None,
        joint_signs: list[int] | None = None,
        control_frequency: float = 50.0,
        max_vel_deg_s: float = 60.0,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        max_pos_speed: float = float("inf"),
        max_rot_speed: float = float("inf"),
        launch_timeout: float = 10.0,
        dry_run: bool = False,
        # Camera
        dev_video_path: str = "",
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        # Gripper (mp.Process — managed by env, not controller)
        gripper_port: str = "",
        gripper_baudrate: int = 921600,
        gripper_can_id: int = 0x08,
        gripper_recv_id: int = 0x18,
        gripper_kp: float = 10.0,
        gripper_kd: float = 1.0,
        gripper_closed_rad: float = 0.734,
        gripper_open_rad: float = -0.139,
        gripper_frequency: float = 50.0,
        # Observation
        camera_obs_latency: float = 0.125,
        camera_obs_horizon: int = 2,
        robot_obs_horizon: int = 2,
        frequency: float = 20.0,
        # Verbose
        verbose: bool = False,
    ):
        self.shm_manager = shm_manager

        # ── Controller ─────────────────────────────────────────────
        from modules.piper_controller import PiperController

        self.controller = PiperController(
            shm_manager=shm_manager,
            can_port=can_port,
            urdf_path=urdf_path,
            target_frame=target_frame,
            joint_names=joint_names,
            joint_signs=joint_signs,
            frequency=control_frequency,
            max_vel_deg_s=max_vel_deg_s,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            launch_timeout=launch_timeout,
            dry_run=dry_run,
            verbose=verbose,
        )

        # ── Camera ─────────────────────────────────────────────────
        from modules.rs_camera import RealSenseCamera

        self.camera = RealSenseCamera(
            shm_manager=shm_manager,
            dev_video_path=dev_video_path,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
            camera_name="color",
        )

        # ── Gripper ─────────────────────────────────────────────────
        self._gripper = None
        self.gripper_port = gripper_port
        self._gripper_baudrate = gripper_baudrate
        self._gripper_can_id = gripper_can_id
        self._gripper_recv_id = gripper_recv_id
        self.gripper_kp = gripper_kp
        self.gripper_kd = gripper_kd
        self._gripper_closed_rad = gripper_closed_rad
        self._gripper_open_rad = gripper_open_rad
        self._gripper_range = abs(gripper_closed_rad - gripper_open_rad)
        self._gripper_frequency = gripper_frequency

        # ── Timing ─────────────────────────────────────────────────
        self.frequency = frequency
        self.camera_obs_latency = camera_obs_latency
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon

        self._last_camera_data = None

    # ==================================================================
    # Gripper calibration
    # ==================================================================

    def calibrate_gripper(
        self,
        torque_threshold: float = 2.0,
        timeout: float = 5.0,
    ) -> float:
        """Home gripper by closing against the mechanical stop.

        Uses the synchronous Gripper API — must be called **before**
        ``start()`` (which spawns GripperProcess).  Updates
        ``gripper_closed_rad`` and ``gripper_open_rad`` so the
        GripperProcess starts pre-calibrated.

        Returns the measured range in radians.
        """
        from modules.gripper import Gripper

        with Gripper(
            self.gripper_port, self._gripper_baudrate,
            self._gripper_can_id, self._gripper_recv_id,
        ) as g:
            state = g.home(
                close_direction=1,
                torque_threshold=torque_threshold,
                timeout=timeout,
            )

        range_rad = abs(state.position)
        self.gripper_closed_rad = 0.0
        self.gripper_open_rad = -range_rad
        self._gripper_range = range_rad

        logger.info(
            "Gripper calibrated: closed=%.3f, open=%.3f, range=%.3f rad",
            self.gripper_closed_rad, self.gripper_open_rad, self._gripper_range,
        )
        return range_rad

    @property
    def is_calibrated(self) -> bool:
        """Whether the gripper has been calibrated."""
        if self._gripper is None:
            return False
        try:
            state = self._gripper.get_state()
            return bool(state.get("is_calibrated", 0))
        except Exception:
            return False

    def _gripper_update_calib_from_ring(self) -> None:
        """Pull closed/open angles from the ring buffer (called after calibration)."""
        if self._gripper is None:
            return
        state = self._gripper.get_state()
        if state.get("is_calibrated", 0):
            self._gripper_closed_rad = float(state["closed_angle"])
            self._gripper_open_rad = float(state["open_angle"])
            self._gripper_range = abs(self._gripper_closed_rad - self._gripper_open_rad)

    # ==================================================================
    # Gripper helpers
    # ==================================================================

    def _gripper_norm_to_rad(self, norm: float) -> float:
        """Map normalized gripper (0=closed, 1=open) to radians."""
        return self._gripper_closed_rad + norm * (self._gripper_open_rad - self._gripper_closed_rad)

    def _gripper_rad_to_norm(self, rad: float) -> float:
        """Map gripper radians to normalized (0=closed, 1=open)."""
        if self._gripper_range <= 0:
            return 1.0
        return (rad - self._gripper_closed_rad) / (self._gripper_open_rad - self._gripper_closed_rad)

    @property
    def gripper_position(self) -> float:
        """Current gripper position in normalized [0, 1]."""
        if self._gripper is None:
            return 1.0
        return self._gripper_rad_to_norm(self._gripper.position)

    # ==================================================================
    # Lifecycle
    # ==================================================================

    @property
    def is_ready(self) -> bool:
        ready = self.camera.is_ready and self.controller.is_alive()
        if self._gripper is not None:
            ready = ready and self._gripper.is_alive()
        return ready

    def start(self, wait: bool = True):
        self.camera.start(wait=False)
        self.controller.start(wait=False)

        if self.gripper_port:
            from modules.gripper import GripperProcess
            self._gripper = GripperProcess(
                shm_manager=self.shm_manager,
                port=self.gripper_port,
                baudrate=self._gripper_baudrate,
                can_id=self._gripper_can_id,
                recv_id=self._gripper_recv_id,
                frequency=self._gripper_frequency,
                position_min=self._gripper_open_rad,
                position_max=self._gripper_closed_rad,
            )
            self._gripper.start(wait=False)

        if wait:
            self.start_wait()
            if self._gripper is not None:
                self._gripper_update_calib_from_ring()

    def start_wait(self):
        self.camera.start_wait()
        self.controller.start_wait()
        if self._gripper is not None:
            self._gripper.start_wait()
            logger.info(
                "Gripper process ready on %s, pos=%.3f (norm)",
                self.gripper_port, self.gripper_position,
            )

    def stop(self, wait: bool = True):
        self.controller.stop(wait=False)
        self.camera.stop(wait=False)
        if self._gripper is not None:
            try:
                self._gripper.stop(wait=False)
            except Exception:
                pass
        if wait:
            self.stop_wait()

    def stop_wait(self):
        self.controller.stop_wait()
        self.camera.stop_wait()
        if self._gripper is not None:
            self._gripper.stop_wait()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ==================================================================
    # Observation (UMI pattern: timestamp-aligned camera + robot state)
    # ==================================================================

    def get_obs(self) -> dict:
        """Return timestamp-aligned observation dict.

        Reads N latest frames from camera ring buffer and robot ring
        buffer, aligns them to a common timestamp horizon, and returns
        a dict compatible with the policy input format.
        """
        assert self.is_ready

        dt = 1.0 / self.frequency

        # Camera: read last K frames
        k_cam = max(self.camera_obs_horizon + 1, 2)
        self._last_camera_data = self.camera.get(k=k_cam)

        # Robot: read last K states
        k_robot = max(self.robot_obs_horizon + 1, 2)
        last_robot_data = self.controller.get_state(k=k_robot)

        # Latest timestamp from camera
        cam_ts = float(self._last_camera_data["timestamp"])
        if hasattr(cam_ts, '__len__'):
            last_ts = float(cam_ts) if np.isscalar(cam_ts) else float(cam_ts[-1] if hasattr(cam_ts, '__getitem__') else cam_ts)
        else:
            last_ts = cam_ts

        # Camera observation horizon
        cam_obs_ts = last_ts - np.arange(self.camera_obs_horizon)[::-1] * dt
        camera_obs = {}
        for key in ["color"]:
            if key in self._last_camera_data:
                camera_obs[key] = self._last_camera_data[key]

        # Robot observation
        robot_obs = {
            "robot_eef_pos": last_robot_data["ActualEEPose"][:3],
            "robot_eef_rot_axis_angle": last_robot_data["ActualEEPose"][3:],
            "joint_pos": last_robot_data["ActualJointState"],
        }

        # Gripper state (read directly from hardware)
        grip_value = self.gripper_position
        robot_obs["gripper_width"] = np.float64(grip_value)

        obs_data = {
            **camera_obs,
            **robot_obs,
            "timestamp": cam_obs_ts,
        }
        return obs_data

    # ==================================================================
    # Action dispatch
    # ==================================================================

    def exec_actions(
        self,
        actions: np.ndarray,
        obs_timestamps: float | None = None,
        dt: float = 1.0 / 30.0,
        compensate_latency: bool = False,
    ) -> int:
        """Dispatch predicted actions to controller (arm) and gripper.

        Actions shape: (N, 7) = [dx, dy, dz, drx, dry, drz, gripper]
        """
        assert self.is_ready

        if obs_timestamps is None:
            obs_timestamps = time.time()
        elif obs_timestamps < 1e9:
            obs_timestamps = obs_timestamps - time.monotonic() + time.time()

        action_timestamps = np.arange(len(actions), dtype=np.float64) * dt + obs_timestamps
        receive_time = time.time()
        is_new = action_timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = action_timestamps[is_new]

        for i in range(len(new_actions)):
            # Arm: schedule waypoint via controller
            self.controller.schedule_waypoint(
                new_actions[i, :6],
                new_timestamps[i],
                gripper=float(new_actions[i, 6]),
            )
            # Gripper: send command directly (env owns the hardware)
            if self._gripper is not None:
                grip_norm = float(new_actions[i, 6])
                if self.is_calibrated:
                    # Torque-mode grasping when calibrated
                    if grip_norm < 0.3:
                        self._gripper.send_torque(
                            self.gripper_kd, -2.0,
                        )
                    elif grip_norm > 0.7:
                        self._gripper.send_torque(
                            self.gripper_kd, 2.0,
                        )
                    else:
                        grip_target = self._gripper_norm_to_rad(grip_norm)
                        self._gripper.send_command(
                            self.gripper_kp, self.gripper_kd, grip_target,
                        )
                else:
                    grip_target = self._gripper_norm_to_rad(grip_norm)
                    self._gripper.send_command(
                        self.gripper_kp, self.gripper_kd, grip_target,
                    )

        return len(new_actions)

    def get_robot_state(self):
        return self.controller.get_state()
