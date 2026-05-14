#!/usr/bin/env python3
"""Real-hardware autonomous SmolVLA control (UMI-EE architecture).

Mirrors UMI scripts/eval_arx5.py:
  Camera → get_obs() → policy.predict() → exec_actions() → PiperController
  All in main loop. Controller runs as separate mp.Process.

Usage:
  python scripts/eval_piper.py \
      --pretrained_path outputs/smolvla_umi_strawberry_50k/checkpoints/050000/pretrained_model \
      --cameras "{ color: {type: intelrealsense, serial_number_or_name: '230322274337', width: 640, height: 480, fps: 30} }" \
      --can_port can0
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

HOME_POSE_DEG = np.array([0.0, 50.60, -50.40, -1.21, 10.00, 0.00])


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def parse_cameras_config(cameras_str: str) -> dict:
    from lerobot.cameras import CameraConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    cameras_dict = yaml.safe_load(cameras_str)
    cameras = {}
    for name, config in cameras_dict.items():
        camera_type = config.pop("type")
        if camera_type == "opencv":
            if isinstance(config.get("index_or_path"), int):
                config["index_or_path"] = str(config["index_or_path"])
            cameras[name] = OpenCVCameraConfig(**config)
        elif camera_type == "intelrealsense":
            cameras[name] = RealSenseCameraConfig(**config)
        else:
            cameras[name] = CameraConfig.get_choice_class(camera_type)(**config)
    return cameras


def load_smolvla_pipeline(pretrained_path: str, device: str = "cuda", dataset_root: str | None = None):
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
    if dataset_root:
        ds_root = dataset_root
    else:
        ds_root = ds_cfg.get("root", str(Path(__file__).resolve().parents[1] / "Datasets" / ds_repo_id))

    logger.info("Loading stats from: %s at %s", ds_repo_id, ds_root)

    import os
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
        load_vlm_weights=False,
        push_to_hub=False,
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
# Observation (≡ UMI Arx5Env.get_obs, simplified)
# ---------------------------------------------------------------------------

def get_obs(controller, cameras: dict) -> dict:
    """Read cameras + controller state. Returns obs dict."""
    t_before = time.monotonic()
    images = {name: cam.read() for name, cam in cameras.items()}
    t_after = time.monotonic()

    state = controller.get_state()
    return {
        "images": images,
        "joints": state["ActualJointState"],
        "ee_pose": state["ActualEEPose"],
        "gripper": float(state["gripper"]),
        "t_obs": (t_before + t_after) / 2.0,
    }


def build_state_2step(controller) -> tuple[np.ndarray, np.ndarray]:
    """Build (2,7) EE state from ring buffer for policy input."""
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

    T_now = pose_6d_to_matrix(ee_now)
    T_prev = pose_6d_to_matrix(ee_prev)
    T_base = T_now

    state_prev = pose_to_ee_state(np.linalg.inv(T_base) @ T_prev, grip_prev)
    state_curr = pose_to_ee_state(np.linalg.inv(T_base) @ T_now, grip_now)
    return np.stack([state_prev, state_curr]), T_base


def pose_6d_to_matrix(pose_6d: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = pose_6d[:3]
    T[:3, :3] = Rotation.from_rotvec(pose_6d[3:]).as_matrix()
    return T


def action_to_pose(action_7d: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = action_7d[:3]
    rotvec = action_7d[3:6]
    angle = np.linalg.norm(rotvec)
    if angle > 1e-10:
        axis = rotvec / angle
        c, s = np.cos(angle), np.sin(angle)
        v = 1 - c
        x, y, z = axis
        T[:3, :3] = np.array([
            [x*x*v+c,     x*y*v-z*s, x*z*v+y*s],
            [y*x*v+z*s,   y*y*v+c,   y*z*v-x*s],
            [z*x*v-y*s,   z*y*v+x*s, z*z*v+c],
        ])
    return T


def pose_to_ee_state(T: np.ndarray, gripper: float = 1.0) -> np.ndarray:
    pos = T[:3, 3]
    R = T[:3, :3]
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-10:
        rotvec = np.zeros(3)
    else:
        axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
        rotvec = axis / (2 * np.sin(angle)) * angle
    return np.array([*pos, *rotvec, gripper], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main (≡ UMI eval_arx5.py)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SROI Piper eval (UMI-EE architecture)")
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--cameras", type=str, required=True)
    parser.add_argument("--can_port", type=str, default="can0")
    parser.add_argument("--urdf_path", type=str, default="")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--control_hz", type=float, default=50.0)
    parser.add_argument("--max_vel_deg_s", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_pos_speed", type=float, default=0.01)
    parser.add_argument("--max_rot_speed", type=float, default=0.5)
    parser.add_argument("--n_steps", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    args = parse_args()

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ── 1. Load policy ─────────────────────────────────────────────
    logger.info("Loading SmolVLA pipeline...")
    policy, preprocessor, postprocessor, task_prompt = load_smolvla_pipeline(
        args.pretrained_path, device, dataset_root=args.dataset_root,
    )
    if args.task:
        task_prompt = args.task

    # ── 2. Connect cameras ─────────────────────────────────────────
    from lerobot.cameras import make_cameras_from_configs
    cameras_config = parse_cameras_config(args.cameras)
    cameras = make_cameras_from_configs(cameras_config)
    for cam_name, camera in cameras.items():
        camera.connect()
        logger.info("Camera connected: %s", cam_name)

    # ── 3. Start controller subprocess ─────────────────────────────
    from multiprocessing.managers import SharedMemoryManager
    from modules.piper_controller import PiperController

    shm_manager = SharedMemoryManager()
    shm_manager.start()

    controller = PiperController(
        shm_manager=shm_manager,
        can_port=args.can_port,
        urdf_path=args.urdf_path,
        frequency=args.control_hz,
        max_vel_deg_s=args.max_vel_deg_s,
        max_pos_speed=args.max_pos_speed,
        max_rot_speed=args.max_rot_speed,
        dry_run=args.dry_run,
        verbose=True,
    )

    logger.info("Starting controller subprocess...")
    controller.start(wait=True)
    logger.info("Controller subprocess ready")

    # ── 4. Move to home ────────────────────────────────────────────
    logger.info("Moving to home pose...")
    controller.move_to_joints(HOME_POSE_DEG, duration=3.0, gripper=1.0)
    time.sleep(3.5)
    logger.info("Home pose reached")

    # ── 5. Wait for user ───────────────────────────────────────────
    logger.warning("=" * 55)
    logger.warning("  Robot at HOME. Motors ENABLED. Press Enter to START.")
    logger.warning("  Press Ctrl+C to quit.")
    logger.warning("=" * 55)
    input()

    # ── 6. Main inference loop (≡ UMI eval_arx5.py) ────────────────
    policy.reset()
    for step in preprocessor.steps:
        if hasattr(step, "reset"):
            step.reset()

    image_history: dict[str, list] = {}
    step_count = 0
    dt = 1.0 / args.fps

    try:
        while True:
            # A. Get observation
            obs = get_obs(controller, cameras)
            state_2step, T_base = build_state_2step(controller)

            # B. Build batch
            batch = {
                "observation.state": torch.from_numpy(state_2step).unsqueeze(0).to(device),
                "task": [task_prompt],
            }
            for cam_name, img in obs["images"].items():
                img_float = img.astype(np.float32) / 255.0
                img_chw = np.transpose(img_float, (2, 0, 1))
                hist = image_history.get(cam_name, [])
                hist.append(img_chw.copy())
                hist = hist[-1:]
                while len(hist) < 1:
                    hist.insert(0, img_chw.copy())
                image_history[cam_name] = hist
                batch[f"observation.images.{cam_name}"] = (
                    torch.from_numpy(np.stack(hist, axis=0)).unsqueeze(0).to(device).float()
                )

            # C. Inference
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

            # D. Convert to world frame
            pred_world = np.zeros_like(pred_ee)
            for t in range(len(pred_ee)):
                T_delta = action_to_pose(pred_ee[t])
                T_world = T_base @ T_delta
                pred_world[t] = pose_to_ee_state(T_world, pred_ee[t, 6])

            # E. Execute (≡ UMI env.exec_actions)
            n_sent = controller.exec_actions(pred_world, obs_timestamps=obs["t_obs"], dt=dt)
            step_count += 1

            logger.info(
                "step=%d sent=%d/%d queue=%d",
                step_count, n_sent, len(pred_world), controller.remaining(),
            )

            if args.n_steps > 0 and step_count >= args.n_steps:
                logger.info("Reached n_steps limit (%d)", args.n_steps)
                break

    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        controller.stop()
        shm_manager.shutdown()
        for cam in cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
