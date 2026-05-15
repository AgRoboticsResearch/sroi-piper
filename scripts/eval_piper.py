#!/usr/bin/env python3
"""Real-hardware autonomous SmolVLA control (UMI-EE architecture).

Mirrors UMI eval_arx5.py pattern:
  Camera → get_obs() → policy.predict() → exec_actions() → PiperController
  Controller runs as separate mp.Process at 50Hz with IK interpolation.

Usage (pyrealsense2 — recommended):
  PYTHONPATH=src:$PYTHONPATH python scripts/eval_piper.py \
      --pretrained_path policy/lerobot/outputs/sroi_piper_26050602_50k/checkpoints/020000/pretrained_model \
      --dataset_root policy/lerobot/Datasets/sroi_piper_26050602 \
      --can_port can0 \
      --camera_serial 230322273077 \
      --gripper_port /dev/ttyACM0

Usage (lerobot cameras):
  PYTHONPATH=src:$PYTHONPATH python scripts/eval_piper.py \
      --pretrained_path ... --dataset_root ... \
      --can_port can0 \
      --cameras "{ color: {type: intelrealsense, serial_number_or_name: '230322273077', width: 640, height: 480, fps: 30} }"
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = str(PROJECT_ROOT / "src" / "utils" / "piper_urdf" / "piper.urdf")
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
ALL_LINKS = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6",
             "ee_link", "camera_link"]
HOME_POSE_DEG = np.array([0.0, 50.60, -50.40, -1.21, 10.00, 0.00])

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pose conversion utilities
# ---------------------------------------------------------------------------

def _pose6d_to_matrix(pose_6d: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, 3] = pose_6d[:3]
    T[:3, :3] = Rotation.from_rotvec(pose_6d[3:]).as_matrix()
    return T


def _pose_to_ee_state(T: np.ndarray, gripper: float = 1.0) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    pos = T[:3, 3]
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.array([*pos, *rotvec, gripper], dtype=np.float32)


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
                [x*x*v+c,   x*y*v-z*s, x*z*v+y*s],
                [y*x*v+z*s, y*y*v+c,   y*z*v-x*s],
                [z*x*v-y*s, z*y*v+x*s, z*z*v+c],
            ])
        T_world = T_base @ T_delta
        pos = T_world[:3, 3]
        R = T_world[:3, :3]
        angle_w = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle_w < 1e-10:
            rv = np.zeros(3)
        else:
            axis_w = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
            rv = axis_w / (2 * np.sin(angle_w)) * angle_w
        pred_world[t] = [*pos, *rv, pred_ee[t, 6]]
    return pred_world


# ---------------------------------------------------------------------------
# Pipeline loading (lerobot)
# ---------------------------------------------------------------------------

def load_smolvla_pipeline(pretrained_path: str, device: str = "cuda",
                          dataset_root: str | None = None):
    import os
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
    ds_root = (dataset_root or
               ds_cfg.get("root", str(PROJECT_ROOT / "Datasets" / ds_repo_id)))

    logger.info("Loading stats: %s at %s", ds_repo_id, ds_root)
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
    logger.info("Task prompt: '%s'", task_prompt)

    return policy, preprocessor, postprocessor, task_prompt


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

def build_state_2step(controller, grip_pos: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build (2,7) canonical UMI state in current EE frame."""
    try:
        count = controller.ring_buffer.count
        states = controller.get_state(k=2) if count >= 2 else controller.get_state()
    except Exception:
        states = controller.get_state()

    ee_poses = states["ActualEEPose"]
    grippers = states["gripper"]

    if ee_poses.ndim == 1:
        ee_now = ee_prev = ee_poses
        grip_now = grip_prev = float(grippers)
    else:
        ee_prev, ee_now = ee_poses[0], ee_poses[-1]
        grip_prev, grip_now = float(grippers[0]), float(grippers[-1])

    if grip_pos is not None:
        grip_prev = grip_now = grip_pos

    T_now = _pose6d_to_matrix(ee_now)
    T_prev = _pose6d_to_matrix(ee_prev)
    T_base = T_now
    state_prev = _pose_to_ee_state(np.linalg.inv(T_base) @ T_prev, grip_prev)
    state_curr = _pose_to_ee_state(np.linalg.inv(T_base) @ T_now, grip_now)
    return np.stack([state_prev, state_curr]), T_base


# ---------------------------------------------------------------------------
# Placo meshcat visualization (optional, for debugging)
# ---------------------------------------------------------------------------

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
    return {"robot": robot, "viz": viz}


def update_robot_viz(viz_state, joint_deg):
    from placo_utils.visualization import frame_viz
    robot = viz_state["robot"]
    for jn, val in zip(ARM_JOINTS, joint_deg):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()
    viz_state["viz"].display(robot.state.q)
    for name in ALL_LINKS:
        try:
            T = robot.get_T_world_frame(name)
            frame_viz(name, T)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SROI Piper eval (UMI-EE architecture)")
    # Robot
    parser.add_argument("--can_port", type=str, default="can0")
    parser.add_argument("--urdf_path", type=str, default=URDF_PATH)
    parser.add_argument("--control_hz", type=float, default=50.0)
    parser.add_argument("--max_vel_deg_s", type=float, default=60.0)
    parser.add_argument("--max_pos_speed", type=float, default=float("inf"))
    parser.add_argument("--max_rot_speed", type=float, default=float("inf"))
    # Gripper
    parser.add_argument("--gripper_port", type=str, default="",
                        help="Serial port for gripper (e.g. /dev/ttyACM0)")
    # Camera (two modes)
    parser.add_argument("--camera_serial", type=str, default="",
                        help="RealSense serial. Uses pyrealsense2 in main process (recommended).")
    parser.add_argument("--cameras", type=str, default="",
                        help="Lerobot cameras YAML. Falls back if --camera_serial not set.")
    parser.add_argument("--cam_width", type=int, default=640)
    parser.add_argument("--cam_height", type=int, default=480)
    parser.add_argument("--cam_fps", type=int, default=30)
    # Inference
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=float, default=30.0)
    # Limits
    parser.add_argument("--n_steps", type=int, default=0,
                        help="Max inference steps (0=infinite)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Disable motors")
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

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ── 1. Load policy ─────────────────────────────────────────────────
    logger.info("Loading SmolVLA pipeline...")
    policy, preprocessor, postprocessor, task_prompt = load_smolvla_pipeline(
        args.pretrained_path, device, dataset_root=args.dataset_root,
    )
    if args.task:
        task_prompt = args.task
    logger.info("Task prompt: '%s'", task_prompt)

    # ── 2. Camera setup ─────────────────────────────────────────────────
    rs_pipeline = None
    cameras = None

    if args.camera_serial:
        import pyrealsense2 as rs
        rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(args.camera_serial)
        rs_config.enable_stream(rs.stream.color, args.cam_width, args.cam_height,
                                rs.format.rgb8, args.cam_fps)
        rs_pipeline.start(rs_config)
        logger.info("RealSense pyrealsense2 camera (serial=%s)", args.camera_serial)
    elif args.cameras:
        import yaml
        from lerobot.cameras import CameraConfig
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot.cameras import make_cameras_from_configs

        cameras_dict = yaml.safe_load(args.cameras)
        configs = {}
        for name, config in cameras_dict.items():
            cam_type = config.pop("type")
            if cam_type == "opencv":
                config["index_or_path"] = str(config.get("index_or_path", ""))
                configs[name] = OpenCVCameraConfig(**config)
            elif cam_type == "intelrealsense":
                configs[name] = RealSenseCameraConfig(**config)
            else:
                configs[name] = CameraConfig.get_choice_class(cam_type)(**config)
        cameras = make_cameras_from_configs(configs)
        for cam_name, camera in cameras.items():
            camera.connect()
            logger.info("Camera connected: %s", cam_name)
    else:
        logger.error("No camera specified. Use --camera_serial or --cameras.")
        sys.exit(1)

    # ── 3. Start controller subprocess ──────────────────────────────────
    from multiprocessing.managers import SharedMemoryManager
    from modules.piper_controller import PiperController

    shm_manager = SharedMemoryManager()
    shm_manager.start()

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

    # ── 4. Start gripper subprocess ──────────────────────────────────────
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
        gripper.send_command(kp=10.0, kd=1.0, position=0.0)
        logger.info("Gripper started")

    # ── 5. Init visualization ────────────────────────────────────────────
    placo_viz = None
    if not args.no_viz:
        placo_viz = init_placo_viz()

    # ── 6. Move to home ─────────────────────────────────────────────────
    logger.info("Moving to home pose...")
    controller.move_to_joints(HOME_POSE_DEG, duration=3.0, gripper=1.0)
    time.sleep(3.5)
    logger.info("Home pose reached")

    # ── 7. Wait for user ─────────────────────────────────────────────────
    logger.warning("=" * 55)
    logger.warning("  Robot at HOME. Motors %s. Press Enter to START.",
                   "DISABLED" if args.dry_run else "ENABLED")
    logger.warning("  Press Ctrl+C to quit.")
    logger.warning("=" * 55)
    input()

    # ── 8. Main inference loop ──────────────────────────────────────────
    policy.reset()
    for step in preprocessor.steps:
        if hasattr(step, "reset"):
            step.reset()

    image_history: dict[str, list] = {}
    step_count = 0
    dt = 1.0 / args.fps
    last_log_time = time.monotonic()

    try:
        while True:
            loop_start = time.monotonic()

            # A. Get camera image
            if rs_pipeline is not None:
                frames = rs_pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                img = np.asanyarray(color_frame.get_data())
                t_obs = time.time()
                images = {"color": img}
            else:
                t_before = time.monotonic()
                images = {name: cam.read() for name, cam in cameras.items()}
                t_obs = (t_before + time.monotonic()) / 2.0

            # B. Get gripper position
            grip_pos = None
            if gripper is not None:
                try:
                    grip_pos = float(gripper.get_state()["position"])
                except Exception:
                    pass

            # C. Build state
            state_2step, T_base = build_state_2step(controller, grip_pos=grip_pos)

            # D. Update visualization
            if placo_viz is not None:
                try:
                    state = controller.get_state()
                    update_robot_viz(placo_viz, state["ActualJointState"])
                except Exception:
                    pass

            # E. Build batch
            batch = {
                "observation.state": torch.from_numpy(state_2step).unsqueeze(0).to(device),
                "task": [task_prompt],
            }
            for cam_name, img in images.items():
                img_float = img.astype(np.float32) / 255.0
                img_chw = np.transpose(img_float, (2, 0, 1))
                hist = image_history.get(cam_name, [])
                hist.append(img_chw.copy())
                hist = hist[-1:]
                while len(hist) < 1:
                    hist.insert(0, img_chw.copy())
                image_history[cam_name] = hist
                batch[f"observation.images.{cam_name}"] = (
                    torch.from_numpy(np.stack(hist, axis=0))
                    .unsqueeze(0).to(device).float()
                )

            # F. Inference
            with torch.no_grad():
                processed = preprocessor(batch)
                pred_actions = policy.predict_action_chunk(processed)
                pred_abs = postprocessor(pred_actions)

            pred_ee = pred_abs[0]
            if hasattr(pred_ee, "cpu"):
                pred_ee = pred_ee.cpu().numpy()
            elif hasattr(pred_ee, "numpy"):
                pred_ee = pred_ee.numpy()
            else:
                pred_ee = np.asarray(pred_ee)

            # G. Convert to world frame & execute
            pred_world = world_from_ee_deltas(pred_ee, T_base)
            n_sent = controller.exec_actions(pred_world, obs_timestamps=t_obs, dt=dt)
            step_count += 1

            # H. Periodic stats
            now = time.monotonic()
            if now - last_log_time >= 1.0:
                last_log_time = now
                queue_n = controller.remaining()
                grip_str = f" grip={grip_pos:.2f}" if grip_pos is not None else ""
                ee_str = "xyz=?"
                if placo_viz is not None:
                    try:
                        T_ee = placo_viz["robot"].get_T_world_frame("ee_link")
                        ee_str = (f"xyz=[{T_ee[0,3]*1000:.0f},"
                                   f"{T_ee[1,3]*1000:.0f},"
                                   f"{T_ee[2,3]*1000:.0f}]mm")
                    except Exception:
                        pass
                logger.info(
                    "step=%d sent=%d/%d queue=%d %s%s",
                    step_count, n_sent, len(pred_world), queue_n, ee_str, grip_str,
                )

            if args.n_steps > 0 and step_count >= args.n_steps:
                logger.info("Reached n_steps limit (%d)", args.n_steps)
                break

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        logger.info("Shutting down...")
        controller.stop()
        if gripper is not None:
            gripper.stop()
        shm_manager.shutdown()
        if rs_pipeline is not None:
            rs_pipeline.stop()
        if cameras is not None:
            for cam in cameras.values():
                try:
                    cam.disconnect()
                except Exception:
                    pass
        logger.info("Shutdown complete. Steps: %d", step_count)


if __name__ == "__main__":
    main()
