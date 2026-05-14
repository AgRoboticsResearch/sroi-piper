"""Piper controller: separate process running IK at fixed frequency.

Mirrors UMI's RTDEInterpolationController:
  - Runs as mp.Process (GIL-free, process isolation)
  - Commands via SharedMemoryQueue (lock-free)
  - State via SharedMemoryRingBuffer (lock-free)
  - IK inside run() (Piper needs it — ARX does IK server-side)
"""

import enum
import logging
import multiprocessing as mp
import time
from queue import Empty

import numpy as np
from multiprocessing.managers import SharedMemoryManager
from scipy.spatial.transform import Rotation

from shared_memory import SharedMemoryQueue, SharedMemoryRingBuffer
from utils.precise_wait import precise_wait
from utils.time_utils import wall_to_monotonic
from utils.kinematics import RobotKinematics
from .pose_trajectory_interpolator import PoseTrajectoryInterpolator

logger = logging.getLogger(__name__)

JOINT_LIMITS_DEG = {
    "min": np.array([-150.0, 0.0, -170.0, -100.0, -70.0, -120.0]),
    "max": np.array([150.0, 180.0, 0.0, 100.0, 70.0, 120.0]),
}
JOINT_LIMIT_TOLERANCE_DEG = 0.5


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    MOVE_JOINTS = 3


class PiperController(mp.Process):
    """Background IK control process.

    Usage:
        with SharedMemoryManager() as shm:
            ctrl = PiperController(shm_manager=shm, ...)
            ctrl.start()
            ctrl.start_wait()
            ctrl.schedule_waypoint(pose_6d, target_time)
            ...
            ctrl.stop()
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        can_port: str = "can0",
        urdf_path: str = "",
        target_frame: str = "ee_link",
        joint_names: list[str] | None = None,
        joint_signs: list[int] | None = None,
        frequency: float = 50.0,
        max_vel_deg_s: float = 60.0,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        max_pos_speed: float = float("inf"),
        max_rot_speed: float = float("inf"),
        launch_timeout: float = 10.0,
        verbose: bool = False,
        dry_run: bool = False,
        # Gripper (UMI pattern: same process, same queue)
        gripper_port: str = "",
        gripper_kp: float = 10.0,
        gripper_kd: float = 1.0,
        gripper_closed_rad: float = 0.734,
        gripper_open_rad: float = -0.139,
    ):
        super().__init__(name="PiperController")
        self.can_port = can_port
        self.urdf_path = urdf_path
        self.target_frame = target_frame
        self.joint_names = joint_names or [f"joint_{i+1}" for i in range(6)]
        self.joint_signs = np.array(joint_signs or [1, 1, 1, 1, 1, 1])
        self.frequency = frequency
        self.max_vel_deg_s = max_vel_deg_s
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.dry_run = dry_run

        # Gripper settings
        self.gripper_port = gripper_port
        self.gripper_kp = gripper_kp
        self.gripper_kd = gripper_kd
        self.gripper_closed_rad = gripper_closed_rad
        self.gripper_open_rad = gripper_open_rad
        self._gripper_range = gripper_closed_rad - gripper_open_rad  # >0

        # Command queue: inference → controller
        cmd_example = {
            "cmd": np.int64(0),
            "target_pose": np.zeros(6, dtype=np.float64),
            "target_time": np.float64(0.0),
            "duration": np.float64(0.0),
            "gripper": np.float64(1.0),
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=cmd_example,
            buffer_size=256,
        )

        # State ring buffer: controller → inference/viz
        state_example = {
            "ActualJointState": np.zeros(6, dtype=np.float64),
            "ActualEEPose": np.zeros(6, dtype=np.float64),
            "gripper": np.float64(1.0),
            "robot_timestamp": np.float64(0.0),
            "robot_timestamp_mono": np.float64(0.0),
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=state_example,
            get_max_k=int(frequency * 5),
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()

    # ========== Lifecycle (same as UMI) ==========

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop(self, wait=True):
        self.input_queue.put({"cmd": np.int64(Command.STOP.value)})
        if wait:
            self.stop_wait()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========== Command interface ==========

    def move_to_pose(self, pose_6d, duration: float = 0.1) -> None:
        self.input_queue.put({
            "cmd": np.int64(Command.SERVOL.value),
            "target_pose": np.asarray(pose_6d, dtype=np.float64),
            "duration": np.float64(duration),
        })

    def move_to_joints(
        self, joint_targets_deg: np.ndarray, duration: float = 3.0, gripper: float | None = None
    ) -> None:
        self.input_queue.put({
            "cmd": np.int64(Command.MOVE_JOINTS.value),
            "target_pose": np.asarray(joint_targets_deg, dtype=np.float64),
            "duration": np.float64(duration),
            "gripper": np.float64(gripper if gripper is not None else 1.0),
        })

    def schedule_waypoint(
        self, pose_6d: np.ndarray, target_time: float, gripper: float | None = None
    ) -> None:
        self.input_queue.put({
            "cmd": np.int64(Command.SCHEDULE_WAYPOINT.value),
            "target_pose": np.asarray(pose_6d, dtype=np.float64),
            "target_time": np.float64(target_time),
            "gripper": np.float64(gripper if gripper is not None else 1.0),
        })

    def exec_actions(
        self,
        actions: np.ndarray,
        obs_timestamps: float | None = None,
        dt: float = 1.0 / 30.0,
    ) -> int:
        """Execute predicted actions as schedule_waypoint (UMI pattern)."""
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
            self.schedule_waypoint(new_actions[i, :6], new_timestamps[i], gripper=float(new_actions[i, 6]))

        return len(new_actions)

    # ========== State feedback ==========

    def get_state(self, k=None):
        if k is None:
            return self.ring_buffer.get()
        return self.ring_buffer.get_last_k(k=k)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def remaining(self) -> int:
        return self.input_queue.qsize()

    # ========== Main loop ==========

    def run(self):
        from .piper_interface import PiperInterface

        iface = PiperInterface(can_port=self.can_port)
        iface.connect()

        if self.dry_run:
            try:
                iface.disable()
                if self.verbose:
                    logger.info("Controller: dry_run — motors disabled")
            except Exception:
                pass

        # Load kinematics (Placo) — optional, skip if no URDF
        kin = None
        if self.urdf_path:
            try:
                kin = RobotKinematics(
                    urdf_path=self.urdf_path,
                    target_frame_name=self.target_frame,
                    joint_names=self.joint_names,
                )
            except Exception as e:
                logger.warning("Kinematics not available (URDF=%s): %s", self.urdf_path, e)

        dt = 1.0 / self.frequency

        # Read initial state
        last_joints = iface.read_joints()
        grip_value = 1.0

        # ── Gripper (UMI pattern: same process, same loop) ─────────
        gripper = None
        if self.gripper_port:
            from .gripper import Gripper
            gripper = Gripper(port=self.gripper_port)
            gripper.connect()
            grip_value = self._gripper_rad_to_norm(gripper.position)
            if self.verbose:
                logger.info("Gripper connected on %s, pos=%.3f (norm)", self.gripper_port, grip_value)

        # Initialize interpolator at current EE pose
        curr_pose_6d = self._joints_to_ee(kin, last_joints)
        curr_t = time.monotonic()
        last_waypoint_time = curr_t
        pose_interp = PoseTrajectoryInterpolator(times=[curr_t], poses=[curr_pose_6d])

        t_start = time.monotonic()
        iter_idx = 0
        keep_running = True
        ik_errors = 0
        ik_total = 0

        # Joint-space move state
        joint_move_active = False
        joint_move_start = None
        joint_move_end = None
        joint_move_t_start = 0.0
        joint_move_t_end = 0.0
        joint_move_gripper = 1.0

        try:
            while keep_running:
                t_now = time.monotonic()

                if joint_move_active:
                    alpha = min((t_now - joint_move_t_start) / max(joint_move_t_end - joint_move_t_start, 1e-6), 1.0)
                    joints_cmd = joint_move_start * (1 - alpha) + joint_move_end * alpha
                    grip_value = joint_move_gripper

                    if not self.dry_run:
                        iface.write_joints(joints_cmd)
                        if gripper is not None:
                            gripper.send_command(
                                self.gripper_kp, self.gripper_kd,
                                self._gripper_norm_to_rad(grip_value),
                            )

                    if alpha >= 1.0:
                        joint_move_active = False
                        last_joints = joint_move_end.copy()
                        curr_pose_6d = self._joints_to_ee(kin, last_joints)
                        curr_t = time.monotonic()
                        last_waypoint_time = curr_t
                        pose_interp = PoseTrajectoryInterpolator(times=[curr_t], poses=[curr_pose_6d])

                    try:
                        last_joints = iface.read_joints()
                    except Exception:
                        last_joints = joints_cmd.copy()

                else:
                    # Step 1: interpolate (UMI pattern)
                    pose_command_6d = pose_interp(t_now)

                    # Step 2: IK + safety + send
                    ik_total += 1
                    try:
                        if kin is not None:
                            T_target = np.eye(4)
                            T_target[:3, 3] = pose_command_6d[:3]
                            T_target[:3, :3] = Rotation.from_rotvec(pose_command_6d[3:]).as_matrix()

                            joints_target = kin.inverse_kinematics(
                                last_joints, T_target,
                                position_weight=self.position_weight,
                                orientation_weight=self.orientation_weight,
                            )
                        else:
                            # No kinematics — use joints directly (joint-space commands only)
                            joints_target = last_joints.copy()

                        joints_safe = np.clip(
                            joints_target,
                            JOINT_LIMITS_DEG["min"] - JOINT_LIMIT_TOLERANCE_DEG,
                            JOINT_LIMITS_DEG["max"] + JOINT_LIMIT_TOLERANCE_DEG,
                        )
                        max_step = self.max_vel_deg_s * dt
                        error = joints_safe - last_joints
                        max_error = np.max(np.abs(error))
                        if max_error > max_step:
                            joints_cmd = last_joints + error * (max_step / max_error)
                        else:
                            joints_cmd = joints_safe

                        if not self.dry_run:
                            iface.write_joints(joints_cmd)
                            if gripper is not None:
                                gripper.send_command(
                                    self.gripper_kp, self.gripper_kd,
                                    self._gripper_norm_to_rad(grip_value),
                                )

                        try:
                            last_joints = iface.read_joints()
                        except Exception:
                            last_joints = joints_cmd.copy()

                    except Exception as e:
                        ik_errors += 1
                        if ik_errors <= 5:
                            logger.warning("Controller: IK failed: %s", e)

                # Step 3: update state ring buffer
                ee_pose_6d = self._joints_to_ee(kin, last_joints)
                if gripper is not None:
                    grip_value = self._gripper_rad_to_norm(gripper.position)
                try:
                    self.ring_buffer.put({
                        "ActualJointState": last_joints,
                        "ActualEEPose": ee_pose_6d,
                        "gripper": np.float64(grip_value),
                        "robot_timestamp": np.float64(time.time()),
                        "robot_timestamp_mono": np.float64(time.monotonic()),
                    }, wait=False)
                except TimeoutError:
                    pass

                # Step 4: fetch command (UMI: max 1 per cycle)
                try:
                    command = self.input_queue.get()
                    n_cmd = 1
                except Empty:
                    n_cmd = 0

                # Step 5: execute command
                if n_cmd > 0:
                    cmd_val = int(command["cmd"])

                    if cmd_val == Command.STOP.value:
                        keep_running = False

                    elif cmd_val == Command.SERVOL.value:
                        target_pose = command["target_pose"]
                        duration = float(command["duration"])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose, time=t_insert, curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed, max_rot_speed=self.max_rot_speed,
                        )
                        last_waypoint_time = t_insert

                    elif cmd_val == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command["target_pose"]
                        target_time = float(command["target_time"])
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose, time=target_time,
                            max_pos_speed=self.max_pos_speed, max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time, last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = target_time
                        grip_value = float(command["gripper"])

                    elif cmd_val == Command.MOVE_JOINTS.value:
                        joint_move_start = last_joints.copy()
                        joint_move_end = np.asarray(command["target_pose"], dtype=np.float64)
                        duration = float(command["duration"])
                        joint_move_t_start = t_now + dt
                        joint_move_t_end = joint_move_t_start + duration
                        joint_move_gripper = float(command["gripper"])
                        joint_move_active = True
                        if self.verbose:
                            logger.info("MOVE_JOINTS → %s over %.1fs", np.round(joint_move_end, 1), duration)

                # Step 6: regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)
                iter_idx += 1

                if iter_idx == 1:
                    self.ready_event.set()

                if self.verbose and iter_idx % 100 == 0:
                    logger.info("tick=%d IK_errors=%d/%d", iter_idx, ik_errors, ik_total)

        finally:
            iface.disconnect()
            if gripper is not None:
                try:
                    gripper.disconnect()
                except Exception:
                    pass
            self.ready_event.set()
            logger.info("Controller process terminated")

    # ── Gripper calibration helpers ──────────────────────────────

    def _gripper_norm_to_rad(self, norm: float) -> float:
        """Map normalized gripper (0=closed, 1=open) to radians."""
        return self.gripper_closed_rad - norm * self._gripper_range

    def _gripper_rad_to_norm(self, rad: float) -> float:
        """Map gripper radians to normalized (0=closed, 1=open)."""
        return (self.gripper_closed_rad - rad) / self._gripper_range

    @staticmethod
    def _joints_to_ee(kin, joints: np.ndarray) -> np.ndarray:
        """Convert joint angles to 6D EE pose [x,y,z,rx,ry,rz]."""
        if kin is None:
            return np.zeros(6)
        T = kin.forward_kinematics(joints)
        return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])
