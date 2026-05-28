#!/usr/bin/env python3
"""Real-hardware Piper eval with keyboard control + meshcat waypoint visualization.

Architecture (UMI pattern):
  pyrealsense2 (main process) ──→ color frames (direct read)
  GripperProcess (mp.Process) ──→ SharedMemoryRingBuffer (position)
  PiperController (mp.Process) ──→ SharedMemoryRingBuffer (joints, EE pose)
                                   ← SharedMemoryQueue (scheduled waypoints)
  Main process: reads camera → ZMQ/local inference → exec_actions → viz

Keyboard controls:
  S — Start / Stop (toggle between IDLE and RUNNING)
  P — Pause (hold position, resume with S)
  Q — Quit

Usage:
  # ZMQ mode (policy server on remote GPU):
  python scripts/eval_piper_remote.py \
      --camera_serial 230322273077 --can_port can0 --gripper_port /dev/ttyACM0

  # Local mode (model loaded in-process):
  python scripts/eval_piper_remote.py \
      --local --pretrained_path outputs/.../pretrained_model \
      --camera_serial 230322273077 --can_port can0 --gripper_port /dev/ttyACM0

  # Dry run (motors disabled, for testing):
  python scripts/eval_piper_remote.py --dry_run \
      --camera_serial 230322273077 --can_port can0
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = str(PROJECT_ROOT / "src" / "utils" / "piper_urdf" / "piper.urdf")
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
ALL_LINKS = [
    "base_link", "link1", "link2", "link3", "link4", "link5", "link6",
    "ee_link", "camera_link",
]
HOME_POSE_DEG = np.array([0.0, 50.60, -50.40, -1.21, 10.00, 0.00])

logger = logging.getLogger(__name__)


# ===========================================================================
# SE(3) helpers
# ===========================================================================

def _pose6d_to_matrix(pose_6d: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, 3] = pose_6d[:3]
    T[:3, :3] = Rotation.from_rotvec(pose_6d[3:]).as_matrix()
    return T


def _pose_to_state(T: np.ndarray, gripper: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    pos = T[:3, 3]
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.array([*pos, *rotvec, gripper], dtype=np.float32)


def build_state_2step(controller, grip_pos: float | None = None):
    try:
        count = controller.ring_buffer.count
        states = controller.get_state(k=2) if count >= 2 else controller.get_state()
    except Exception:
        states = controller.get_state()

    ee_poses = states["ActualEEPose"]

    if ee_poses.ndim == 1:
        ee_now = ee_prev = ee_poses
        grip_now = grip_prev = float(states["gripper"])
    else:
        ee_prev, ee_now = ee_poses[0], ee_poses[-1]
        grippers = states["gripper"]
        grip_prev, grip_now = float(grippers[0]), float(grippers[-1])

    if grip_pos is not None:
        grip_prev = grip_now = grip_pos

    T_now = _pose6d_to_matrix(ee_now)
    T_prev = _pose6d_to_matrix(ee_prev)
    T_base = T_now
    state_prev = _pose_to_state(np.linalg.inv(T_base) @ T_prev, grip_prev)
    state_curr = _pose_to_state(np.linalg.inv(T_base) @ T_now, grip_now)
    return np.stack([state_prev, state_curr]), T_base


def world_from_ee_deltas(pred_ee: np.ndarray, T_base: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    pred_world = np.zeros_like(pred_ee)
    for t in range(len(pred_ee)):
        T_delta = np.eye(4)
        T_delta[:3, 3] = pred_ee[t, :3]
        rotvec = pred_ee[t, 3:6]
        angle = np.linalg.norm(rotvec)
        if angle > 1e-10:
            axis = rotvec / angle
            c, s = np.cos(angle), np.sin(angle)
            v = 1 - c
            x, y, z = axis
            T_delta[:3, :3] = np.array([
                [x * x * v + c,     x * y * v - z * s, x * z * v + y * s],
                [y * x * v + z * s, y * y * v + c,     y * z * v - x * s],
                [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
            ])
        T_world = T_base @ T_delta
        pos = T_world[:3, 3]
        R = T_world[:3, :3]
        angle_w = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle_w < 1e-10:
            rv = np.zeros(3)
        else:
            axis_w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            rv = axis_w / (2 * np.sin(angle_w)) * angle_w
        pred_world[t] = [*pos, *rv, pred_ee[t, 6]]
    return pred_world


# ===========================================================================
# Meshcat visualization (placo)
# ===========================================================================

def init_placo_viz():
    import placo
    from placo_utils.visualization import robot_viz, frame_viz

    robot = placo.RobotWrapper(URDF_PATH)
    for jn, val in zip(ARM_JOINTS, HOME_POSE_DEG):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()

    viz = robot_viz(robot)
    viz.display(robot.state.q)
    time.sleep(1.0)

    for name in ALL_LINKS:
        try:
            T = robot.get_T_world_frame(name)
            frame_viz(name, T)
        except Exception:
            pass

    logger.info("meshcat viewer: http://127.0.0.1:7001/static/")
    return robot, viz


def update_robot_viz(robot, viz, joint_deg: np.ndarray):
    from placo_utils.visualization import frame_viz
    for jn, val in zip(ARM_JOINTS, joint_deg):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()
    viz.display(robot.state.q)
    for name in ALL_LINKS:
        try:
            T = robot.get_T_world_frame(name)
            frame_viz(name, T)
        except Exception:
            pass


def viz_waypoints(controller) -> None:
    from queue import Empty
    from placo_utils.visualization import frame_viz, points_viz

    try:
        commands = controller.input_queue.peek_all()
    except Empty:
        for s in range(5):
            points_viz(f"wp_seg{s}", np.zeros((1, 3)), radius=0.001, color=0x444444)
        frame_viz("wp_start", np.eye(4))
        frame_viz("wp_end", np.eye(4))
        return

    n = len(commands["cmd"])
    poses = commands["target_pose"]
    points = poses[:, :3]

    n_segments = min(5, n)
    seg_len = max(1, n // n_segments)
    colors = [0x00FF00, 0x88FF00, 0xFFFF00, 0xFF8800, 0xFF0000]
    for s in range(n_segments):
        start = s * seg_len
        end = min((s + 1) * seg_len + 1, n)
        seg = points[start:end]
        if len(seg) > 0:
            points_viz(
                f"wp_seg{s}", seg, radius=0.004,
                color=colors[min(s, len(colors) - 1)],
            )
    for s in range(n_segments, 5):
        points_viz(f"wp_seg{s}", np.zeros((1, 3)), radius=0.001, color=0x444444)

    frame_viz("wp_start", _pose6d_to_matrix(poses[0]))
    if n > 1:
        frame_viz("wp_end", _pose6d_to_matrix(poses[-1]))


# ===========================================================================
# Inference backends
# ===========================================================================

def create_zmq_client(host: str, port: int):
    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{host}:{port}")
    logger.info("ZMQ client connected to %s:%d", host, port)
    return sock, ctx


def zmq_infer(socket, state_2step: np.ndarray, images: dict, task: str) -> np.ndarray:
    raw_obs = {"observation.state": state_2step, "task": task}
    for cam_name, img in images.items():
        raw_obs[f"observation.images.{cam_name}"] = img
    socket.send_pyobj(raw_obs)
    result = socket.recv_pyobj()
    if isinstance(result, str):
        raise RuntimeError(f"Policy server error: {result}")
    return result


def create_local_pipeline(pretrained_path: str, device: str = "cuda",
                          dataset_root: str | None = None):
    import json, os
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    pretrained_path = str(pretrained_path)
    config_path = Path(pretrained_path) / "train_config.json"
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        policy_config = train_config.get("policy", {})
        ds_cfg = train_config.get("dataset", {})
    else:
        policy_config = {}
        ds_cfg = {}

    ds_repo_id = ds_cfg.get("repo_id", "sroi_piper_strawberry_picking")
    ds_root = dataset_root or ds_cfg.get("root", str(PROJECT_ROOT / "Datasets" / ds_repo_id))

    logger.info("Loading stats from: %s at %s", ds_repo_id, ds_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    ds_meta = LeRobotDatasetMetadata(ds_repo_id, root=ds_root)

    cfg = SmolVLAConfig(
        derive_state_from_action=policy_config.get("derive_state_from_action", True),
        use_relative_actions=policy_config.get("use_relative_actions", True),
        relative_exclude_joints=policy_config.get("relative_exclude_joints", ["gripper"]),
        relative_exclude_state_joints=policy_config.get("relative_exclude_state_joints", ["gripper"]),
        device=device,
        resize_imgs_with_padding=tuple(policy_config.get("resize_imgs_with_padding", (512, 512))),
        freeze_vision_encoder=policy_config.get("freeze_vision_encoder", True),
        train_expert_only=policy_config.get("train_expert_only", True),
        train_state_proj=policy_config.get("train_state_proj", True),
        load_vlm_weights=False, push_to_hub=False,
        pretrained_path=pretrained_path,
    )
    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=pretrained_path, dataset_stats=ds_meta.stats,
    )
    task_prompt = None
    if hasattr(ds_meta, "tasks") and len(ds_meta.tasks) > 0:
        task_prompt = ds_meta.tasks.index[0]
    if not task_prompt:
        task_prompt = "perform the task"
    return policy, preprocessor, postprocessor, task_prompt, device


def local_infer(pipeline, state_2step: np.ndarray, images: dict) -> np.ndarray:
    import torch
    policy, preprocessor, postprocessor, task_prompt, device = pipeline
    batch = {
        "observation.state": torch.from_numpy(state_2step).unsqueeze(0).to(device),
        "task": [task_prompt],
    }
    for cam_name, img in images.items():
        img_float = img.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1))
        batch[f"observation.images.{cam_name}"] = (
            torch.from_numpy(img_chw).unsqueeze(0).unsqueeze(0).to(device).float()
        )
    with torch.inference_mode():
        processed = preprocessor(batch)
        pred_actions = policy.predict_action_chunk(processed)
        pred_abs = postprocessor(pred_actions)
    pred_ee = pred_abs[0]
    if hasattr(pred_ee, "cpu"):
        return pred_ee.cpu().numpy()
    return pred_ee.numpy()


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Piper eval with keyboard control + meshcat viz")
    # Robot
    parser.add_argument("--can_port", type=str, default="can0")
    parser.add_argument("--urdf_path", type=str, default=URDF_PATH)
    parser.add_argument("--control_hz", type=float, default=50.0)
    parser.add_argument("--max_vel_deg_s", type=float, default=60.0)
    parser.add_argument("--max_pos_speed", type=float, default=0.01)
    parser.add_argument("--max_rot_speed", type=float, default=0.5)
    # Gripper
    parser.add_argument("--gripper_port", type=str, default="",
                        help="Serial port for gripper (e.g. /dev/ttyACM0)")
    parser.add_argument("--gripper_kp", type=float, default=5.0,  help="Gripper kp (lower=less heat)")
    parser.add_argument("--gripper_kd", type=float, default=0.5,  help="Gripper kd")
    # Camera
    parser.add_argument("--camera_serial", type=str, required=True,
                        help="RealSense serial number (pyrealsense2 SDK)")
    parser.add_argument("--cam_width", type=int, default=640)
    parser.add_argument("--cam_height", type=int, default=480)
    parser.add_argument("--cam_fps", type=int, default=30)
    # Inference
    parser.add_argument("--local", action="store_true",
                        help="Load SmolVLA in-process instead of ZMQ")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Required for --local mode")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--policy_host", type=str, default="localhost")
    parser.add_argument("--policy_port", type=int, default=8766)
    parser.add_argument("--device", type=str, default="cuda")
    # Task
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--infer_every", type=int, default=1,
                        help="Run inference every N loop iterations")
    # Safety / debug
    parser.add_argument("--dry_run", action="store_true",
                        help="Disable motors (visualization only)")
    parser.add_argument("--no_viz", action="store_true",
                        help="Disable meshcat visualization")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # ── 0. Keyboard listener ──────────────────────────────────────────
    from pynput.keyboard import KeyCode, Listener
    from collections import defaultdict
    from threading import Lock

    class KeyWatch(Listener):
        def __init__(self):
            self._press_list = []
            self._counts = defaultdict(lambda: 0)
            self._lock = Lock()
            super().__init__(on_press=self._on_press)

        def _on_press(self, key):
            with self._lock:
                self._counts[key] += 1
                self._press_list.append(key)

        def drain(self) -> list:
            with self._lock:
                events = self._press_list
                self._press_list = []
                self._counts.clear()
                return events

        def was_pressed(self, key) -> bool:
            with self._lock:
                return self._counts[key] > 0

    keywatch = KeyWatch()
    keywatch.start()

    # ── 1. Inference backend ──────────────────────────────────────────
    if args.local:
        if not args.pretrained_path:
            logger.error("--pretrained_path required for --local mode")
            sys.exit(1)
        logger.info("Loading SmolVLA locally from %s", args.pretrained_path)
        pipeline = create_local_pipeline(
            args.pretrained_path, args.device, dataset_root=args.dataset_root,
        )
        task_prompt = args.task or pipeline[3]
        zmq_sock = None
    else:
        logger.info("Connecting to ZMQ policy server at %s:%d",
                     args.policy_host, args.policy_port)
        zmq_sock, zmq_ctx = create_zmq_client(args.policy_host, args.policy_port)
        task_prompt = args.task or "pick the red strawberry"
        pipeline = None

    logger.info("Task prompt: '%s'", task_prompt)

    # ── 2. Camera (pyrealsense2 SDK) ──────────────────────────────────
    import pyrealsense2 as rs
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(args.camera_serial)
    rs_config.enable_stream(rs.stream.color, args.cam_width, args.cam_height,
                            rs.format.rgb8, args.cam_fps)
    rs_pipeline.start(rs_config)
    logger.info("pyrealsense2 camera started (serial=%s, %dx%d@%d)",
                 args.camera_serial, args.cam_width, args.cam_height, args.cam_fps)

    # ── 3. SharedMemoryManager ────────────────────────────────────────
    from multiprocessing.managers import SharedMemoryManager
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    # ── 4. Gripper subprocess (optional) ──────────────────────────────
    gripper = None
    if args.gripper_port:
        from modules.gripper import GripperProcess
        gripper = GripperProcess(
            shm_manager=shm_manager,
            port=args.gripper_port,
            frequency=50.0,
            verbose=True,
        )
        gripper.start()
        gripper.start_wait()
        logger.info("GripperProcess started (holding current position)")

    # ── 5. PiperController subprocess ─────────────────────────────────
    from modules.piper_controller import PiperController

    controller = PiperController(
        shm_manager=shm_manager,
        can_port=args.can_port,
        urdf_path=args.urdf_path,
        target_frame="ee_link",
        joint_names=ARM_JOINTS,
        frequency=args.control_hz,
        max_vel_deg_s=args.max_vel_deg_s,
        max_pos_speed=args.max_pos_speed,
        max_rot_speed=args.max_rot_speed,
        dry_run=args.dry_run,
        verbose=True,
    )
    controller.start(wait=True)
    logger.info("Controller ready (dry_run=%s)", args.dry_run)

    # ── 6. Move to home ───────────────────────────────────────────────
    logger.info("Moving to home pose...")
    controller.move_to_joints(HOME_POSE_DEG, duration=3.0, gripper=0.0)
    time.sleep(3.5)
    logger.info("Home pose reached")

    # ── 7. Init meshcat visualization ─────────────────────────────────
    robot, viz = None, None
    if not args.no_viz:
        robot, viz = init_placo_viz()

    # ── 8. Warm-up inference ──────────────────────────────────────────
    logger.info("Warming up inference...")
    try:
        frames = rs_pipeline.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data())
        images = {"color": img}
        state_2step, T_base = build_state_2step(controller)
        if args.local:
            _ = local_infer(pipeline, state_2step, images)
        else:
            _ = zmq_infer(zmq_sock, state_2step, images, task_prompt)
        logger.info("Warm-up inference OK")
    except Exception as e:
        logger.error("Warm-up failed: %s", e)
        logger.error("Check policy server is running and model is loaded")
        sys.exit(1)

    # ── 9. Main loop ──────────────────────────────────────────────────
    dt = 1.0 / args.fps
    state = "IDLE"  # IDLE | RUNNING | PAUSED
    step_count = 0
    infer_count = 0
    n_sent = 0
    last_infer_ms = 0.0
    last_log_time = time.monotonic()

    def print_status(extra: str = ""):
        motor = "DRY_RUN" if args.dry_run else "LIVE"
        logger.warning("[%s] [%s] S=start/stop  P=pause  Q=quit  %s",
                       state, motor, extra)

    print_status()
    logger.warning("Press S to begin.")

    try:
        while True:
            loop_start = time.monotonic()

            # ── Handle keyboard events ────────────────────────────
            for key in keywatch.drain():
                if key == KeyCode(char='q'):
                    logger.info("Quit requested — stopping")
                    raise KeyboardInterrupt
                elif key == KeyCode(char='s'):
                    if state == "IDLE":
                        state = "RUNNING"
                    elif state == "RUNNING":
                        state = "IDLE"
                    elif state == "PAUSED":
                        state = "RUNNING"
                    print_status()
                elif key == KeyCode(char='p'):
                    if state == "RUNNING":
                        state = "PAUSED"
                        print_status()

            # ── Update robot visualization (always, even when idle) ─
            if robot is not None:
                try:
                    joints = controller.get_state()["ActualJointState"]
                    update_robot_viz(robot, viz, joints)
                except Exception:
                    pass
                try:
                    viz_waypoints(controller)
                except Exception:
                    pass

            # ── Read gripper position ──────────────────────────────
            grip_pos = None
            if gripper is not None:
                try:
                    grip_pos = float(gripper.get_state()["position"])
                except Exception:
                    pass

            # ── Inference + execution (RUNNING only) ───────────────
            if state == "RUNNING":
                should_infer = (args.infer_every <= 0
                                or step_count % args.infer_every == 0)
                if should_infer and controller.remaining() >= 30:
                    should_infer = False

                if should_infer:
                    t_infer_start = time.perf_counter()
                    try:
                        # Capture
                        frames = rs_pipeline.wait_for_frames()
                        img = np.asanyarray(frames.get_color_frame().get_data())
                        images = {"color": img}
                        t_obs = time.time()

                        # Build state
                        state_2step, T_base = build_state_2step(controller,
                                                               grip_pos=grip_pos)

                        # Infer
                        if args.local:
                            pred_ee = local_infer(pipeline, state_2step, images)
                        else:
                            pred_ee = zmq_infer(zmq_sock, state_2step, images,
                                               task_prompt)

                        # Convert & execute
                        pred_world = world_from_ee_deltas(pred_ee, T_base)
                        n_sent = controller.exec_actions(
                            pred_world, obs_timestamps=t_obs, dt=dt,
                        )
                        infer_count += 1
                        last_infer_ms = (time.perf_counter() - t_infer_start) * 1000
                        step_count += 1
                    except Exception as e:
                        import traceback
                        from queue import Full
                        if isinstance(e, Full):
                            logger.warning("Inference skipped: controller queue full (%d)",
                                           controller.remaining())
                        else:
                            logger.error("Inference failed [%s]: %s",
                                         type(e).__name__, e)
                            logger.error(traceback.format_exc())

            # ── Periodic stats ─────────────────────────────────────
            now = time.monotonic()
            if now - last_log_time >= 1.0:
                last_log_time = now
                queue_n = controller.remaining()
                ee_str = "xyz=?"
                if robot is not None:
                    try:
                        T_ee = robot.get_T_world_frame("ee_link")
                        ee_str = (f"xyz=[{T_ee[0,3]*1000:.0f},"
                                  f"{T_ee[1,3]*1000:.0f},"
                                  f"{T_ee[2,3]*1000:.0f}]mm")
                    except Exception:
                        pass
                grip_str = f" grip={grip_pos:.2f}" if grip_pos is not None else ""
                if state == "RUNNING":
                    logger.info(
                        "step=%d infer=%d dt=%.0fms queue=%d sent=%d %s%s",
                        step_count, infer_count, last_infer_ms, queue_n,
                        n_sent, ee_str, grip_str,
                    )
                else:
                    logger.info("[%s] queue=%d %s%s", state, queue_n, ee_str, grip_str)

            # ── Loop timing ────────────────────────────────────────
            elapsed = time.monotonic() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        logger.info("Shutting down...")
        keywatch.stop()
        controller.stop()
        rs_pipeline.stop()
        if gripper is not None:
            gripper.stop()
        shm_manager.shutdown()
        if zmq_sock:
            zmq_sock.close()
        if not args.local:
            zmq_ctx.term()
        logger.info("Done. Steps: %d, Inferences: %d", step_count, infer_count)


if __name__ == "__main__":
    main()
