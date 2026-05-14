#!/usr/bin/env python3
"""Visualize Piper state + SmolVLA predicted waypoints in meshcat (dry_run).

Camera runs as independent mp.Process writing to SharedMemoryRingBuffer
(UMI pattern). Controller runs as separate process. No blocking I/O in
the main viz loop.

Visualizes:
  - URDF at live sensor positions (green)
  - Pending waypoints as colored dots (green→red)
  - Current EE position

Usage:
  # Terminal 1: start ZMQ server
  python scripts/policy_server_zmq.py --pretrained_path ... --host 0.0.0.0 --port 8766

  # Terminal 2: run visualization
  python tests/viz_waypoints.py \
      --camera_type intelrealsense --camera_serial 230322273077 \
      --cam_width 640 --cam_height 480 --can_port can0
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
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
ALL_LINKS = [
    "base_link", "link1", "link2", "link3", "link4", "link5", "link6",
    "ee_link", "camera_link",
]
HOME_POSE_DEG = np.array([0.0, 50.60, -50.40, -1.21, 10.00, 0.00])

logger = logging.getLogger(__name__)


# ===========================================================================
# Placo meshcat visualization
# ===========================================================================

def init_placo_viz(urdf_path: str, home_deg: np.ndarray) -> dict:
    import placo
    from placo_utils.visualization import robot_viz, frame_viz

    robot = placo.RobotWrapper(urdf_path)
    for jn, val in zip(ARM_JOINTS, home_deg):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()

    T_world_ee = robot.get_T_world_frame("ee_link")

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
    logger.info(
        "Initial EE: pos=(%.0f, %.0f, %.0f) mm",
        T_world_ee[0, 3] * 1000, T_world_ee[1, 3] * 1000, T_world_ee[2, 3] * 1000,
    )
    return {"robot": robot, "viz": viz}


def update_robot_viz(viz_state: dict, joint_deg: np.ndarray):
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


def viz_waypoints(viz_state: dict, controller) -> None:
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


def _pose6d_to_matrix(pose_6d: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, 3] = pose_6d[:3]
    T[:3, :3] = Rotation.from_rotvec(pose_6d[3:]).as_matrix()
    return T


# ===========================================================================
# Observation (reads from ring buffers — no blocking camera.read())
# ===========================================================================

def get_obs(controller, cam_rb: dict) -> dict:
    """Read camera frames from ring buffers + robot state from controller ring buffer."""
    images = {}
    t_obs = 0.0
    for name, cam in cam_rb.items():
        data = cam.get()  # latest frame from camera ring buffer
        images[name] = data["color"]
        t_obs = float(data["timestamp"])

    state = controller.get_state()
    return {
        "images": images,
        "joints": state["ActualJointState"],
        "ee_pose": state["ActualEEPose"],
        "gripper": float(state["gripper"]),
        "t_obs": t_obs,
    }


def build_state_2step(controller):
    from scipy.spatial.transform import Rotation

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

    T_now = _pose6d_to_matrix(ee_now)
    T_prev = _pose6d_to_matrix(ee_prev)

    pos = T_now[:3, 3]
    R = T_now[:3, :3]
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-10:
        rotvec = np.zeros(3)
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        rotvec = axis / (2 * np.sin(angle)) * angle
    state_curr = np.array([*pos, *rotvec, grip_now], dtype=np.float32)

    T_rel = np.linalg.inv(T_now) @ T_prev
    pos_p = T_rel[:3, 3]
    R_p = T_rel[:3, :3]
    angle_p = np.arccos(np.clip((np.trace(R_p) - 1) / 2, -1, 1))
    if angle_p < 1e-10:
        rotvec_p = np.zeros(3)
    else:
        axis_p = np.array([R_p[2, 1] - R_p[1, 2], R_p[0, 2] - R_p[2, 0], R_p[1, 0] - R_p[0, 1]])
        rotvec_p = axis_p / (2 * np.sin(angle_p)) * angle_p
    state_prev = np.array([*pos_p, *rotvec_p, grip_prev], dtype=np.float32)

    return np.stack([state_prev, state_curr]), T_now


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
    parser = argparse.ArgumentParser(description="Visualize Piper + SmolVLA waypoints (dry_run)")
    # Robot
    parser.add_argument("--can_port", type=str, default="can0")
    parser.add_argument("--urdf_path", type=str, default=URDF_PATH)
    parser.add_argument("--control_hz", type=float, default=50.0)
    # Camera (RealSense subprocess — writes to ring buffer)
    parser.add_argument("--camera_serial", type=str, default="",
                        help="RealSense serial number")
    parser.add_argument("--cam_width", type=int, default=640)
    parser.add_argument("--cam_height", type=int, default=480)
    parser.add_argument("--cam_fps", type=int, default=30)
    # Inference
    parser.add_argument("--local", action="store_true",
                        help="Load SmolVLA in-process instead of ZMQ")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Only needed for --local mode")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--policy_host", type=str, default="localhost")
    parser.add_argument("--policy_port", type=int, default=8766)
    parser.add_argument("--device", type=str, default="cuda")
    # Task
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--infer_every", type=int, default=6,
                        help="Run inference every N viz frames")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # ── 1. SharedMemoryManager (all shared memory lives here) ─────────
    from multiprocessing.managers import SharedMemoryManager
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    # ── 2. Setup inference backend ───────────────────────────────────
    if args.local:
        if not args.pretrained_path:
            logger.error("--pretrained_path required for --local mode")
            sys.exit(1)
        logger.info("Loading SmolVLA locally from %s", args.pretrained_path)
        pipeline = create_local_pipeline(
            args.pretrained_path, args.device, dataset_root=args.dataset_root,
        )
        task_prompt = args.task or pipeline[3]
        logger.info("Task prompt: '%s'", task_prompt)
        zmq_sock = None
    else:
        logger.info("Connecting to ZMQ policy server at %s:%d",
                     args.policy_host, args.policy_port)
        zmq_sock, zmq_ctx = create_zmq_client(args.policy_host, args.policy_port)
        task_prompt = args.task or "pick the red strawberry"
        pipeline = None

    # ── 3. Start camera subprocess (UMI pattern) ─────────────────────
    from modules.rs_camera import RealSenseCamera

    cam_rb = {}
    cam_process = RealSenseCamera(
        shm_manager=shm_manager,
        serial_number=args.camera_serial,
        width=args.cam_width,
        height=args.cam_height,
        fps=args.cam_fps,
        camera_name="color",
    )
    cam_process.start()
    cam_process.start_wait()
    cam_rb["color"] = cam_process
    logger.info("RealSense camera subprocess started — writing to ring buffer @ %d FPS", args.cam_fps)

    # ── 4. Start PiperController subprocess (dry_run) ─────────────────
    from modules.piper_controller import PiperController

    controller = PiperController(
        shm_manager=shm_manager,
        can_port=args.can_port,
        urdf_path=args.urdf_path,
        frequency=args.control_hz,
        dry_run=True,
        verbose=True,
    )

    logger.info("Starting controller subprocess (dry_run=True)...")
    controller.start(wait=True)
    logger.info("Controller process started — motors DISABLED")

    # ── 5. Init placo visualization ─────────────────────────────────
    placo_viz = init_placo_viz(args.urdf_path, HOME_POSE_DEG)

    # ── 6. Main loop ────────────────────────────────────────────────
    dt = 1.0 / args.fps
    frame_count = 0
    infer_count = 0
    last_infer_ms = 0.0
    last_log_time = time.monotonic()

    logger.info("Visualization running. Move arm by hand to see predicted waypoints.")
    logger.info("Press Ctrl+C to quit")
    logger.info("Open http://127.0.0.1:7001/static/ in browser")

    try:
        while True:
            loop_start = time.monotonic()

            # A. Update URDF from controller ring buffer
            try:
                state = controller.get_state()
                joints_now = state["ActualJointState"]
                update_robot_viz(placo_viz, joints_now)
            except Exception:
                pass

            # B. Run inference periodically
            should_infer = (args.infer_every <= 0
                            or frame_count % args.infer_every == 0)
            if should_infer:
                t_infer_start = time.perf_counter()
                try:
                    # Read observations from ring buffers (non-blocking)
                    obs = get_obs(controller, cam_rb)
                    state_2step, T_base = build_state_2step(controller)

                    if args.local:
                        pred_ee = local_infer(pipeline, state_2step, obs["images"])
                    else:
                        pred_ee = zmq_infer(zmq_sock, state_2step, obs["images"], task_prompt)

                    pred_world = world_from_ee_deltas(pred_ee, T_base)
                    n_sent = controller.exec_actions(
                        pred_world, obs_timestamps=obs["t_obs"], dt=dt,
                    )
                    infer_count += 1
                    last_infer_ms = (time.perf_counter() - t_infer_start) * 1000
                except Exception as e:
                    logger.error("Inference failed: %s", e)

            # C. Visualize pending waypoints
            try:
                viz_waypoints(placo_viz, controller)
            except Exception:
                pass

            # D. Periodic stats
            now = time.monotonic()
            if now - last_log_time >= 1.0:
                last_log_time = now
                queue_n = controller.remaining()
                try:
                    T_ee = placo_viz["robot"].get_T_world_frame("ee_link")
                    ee_pos = T_ee[:3, 3] * 1000
                    ee_str = f"xyz=[{ee_pos[0]:.0f},{ee_pos[1]:.0f},{ee_pos[2]:.0f}]mm"
                except Exception:
                    ee_str = "xyz=?"
                logger.info(
                    "infer=%d dt=%.0fms queue=%d %s",
                    infer_count, last_infer_ms, queue_n, ee_str,
                )

            frame_count += 1

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        logger.info("Shutting down...")
        controller.stop()
        cam_process.stop()
        shm_manager.shutdown()
        if zmq_sock:
            zmq_sock.close()
        if not args.local:
            zmq_ctx.term()
        logger.info("Done")


if __name__ == "__main__":
    main()
