#!/usr/bin/env python3
"""Remote inference eval — PiperController talks to ZMQ policy server.

Architecture (≡ UMI eval_arx5.py):
  Camera → get_obs() → ZMQ REQ send_pyobj(obs) → PolicyServer (remote GPU)
  PolicyServer → recv_pyobj() → preprocessor → SmolVLA → postprocessor → send_pyobj(action)
  Client → recv_pyobj(action_chunk) → controller.exec_actions()

No LeRobot transport/protobuf/grpc on the robot side. Just pyzmq.

Usage:
  # Terminal 1: start ZMQ inference server (on GPU machine)
  python scripts/policy_server_zmq.py \
      --pretrained_path outputs/smolvla_umi_strawberry_50k/checkpoints/050000/pretrained_model \
      --host 0.0.0.0 --port 8766

  # Terminal 2: start robot client (on robot machine)
  python scripts/eval_piper_remote.py \
      --policy_host 192.168.1.100 --policy_port 8766 \
      --cameras "{ color: {type: intelrealsense, serial_number_or_name: '230322274337', width: 640, height: 480, fps: 30} }" \
      --can_port can0
"""

import argparse
import logging
import time

import numpy as np
import yaml
import zmq
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

HOME_POSE_DEG = np.array([0.0, 50.60, -50.40, -1.21, 10.00, 0.00])


# ---------------------------------------------------------------------------
# Camera config (same as eval_piper.py — lerobot cameras are OK as drivers)
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


# ---------------------------------------------------------------------------
# Observation: read cameras + controller state
# ---------------------------------------------------------------------------

def get_obs(controller, cameras: dict) -> dict:
    """Read cameras and controller state. Returns raw obs dict."""
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


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------

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
# ZMQ client helpers
# ---------------------------------------------------------------------------

def connect_policy_server(host: str, port: int) -> zmq.Socket:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{host}:{port}")
    logger.info("Connected to policy server at %s:%d", host, port)
    return socket, context


def request_action(socket: zmq.Socket, state_2step: np.ndarray,
                   images: dict, task: str) -> np.ndarray:
    """Send observation to server, receive action chunk. Returns (N, 7) array."""
    raw_obs = {
        "observation.state": state_2step,
        "task": task,
    }
    for cam_name, img in images.items():
        raw_obs[f"observation.images.{cam_name}"] = img  # uint8 HWC

    socket.send_pyobj(raw_obs)
    result = socket.recv_pyobj()

    if isinstance(result, str):
        raise RuntimeError(f"Policy server error: {result}")

    return result  # (N, 7) float32 EE deltas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SROI Piper remote eval via ZMQ")
    # Robot
    parser.add_argument("--can_port", type=str, default="can0")
    parser.add_argument("--urdf_path", type=str, default="")
    parser.add_argument("--cameras", type=str, required=True)
    parser.add_argument("--control_hz", type=float, default=50.0)
    parser.add_argument("--max_vel_deg_s", type=float, default=60.0)
    parser.add_argument("--max_pos_speed", type=float, default=0.01)
    parser.add_argument("--max_rot_speed", type=float, default=0.5)
    # ZMQ
    parser.add_argument("--policy_host", type=str, default="localhost")
    parser.add_argument("--policy_port", type=int, default=8766)
    # Task
    parser.add_argument("--task", type=str, default="pick the strawberry")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--n_steps", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # ── 1. Connect cameras ─────────────────────────────────────────
    from lerobot.cameras import make_cameras_from_configs
    cameras_config = parse_cameras_config(args.cameras)
    cameras = make_cameras_from_configs(cameras_config)
    for cam_name, camera in cameras.items():
        camera.connect()
        logger.info("Camera connected: %s", cam_name)

    # ── 2. Start controller subprocess ─────────────────────────────
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

    # ── 3. Move to home ────────────────────────────────────────────
    logger.info("Moving to home pose...")
    controller.move_to_joints(HOME_POSE_DEG, duration=3.0, gripper=1.0)
    time.sleep(3.5)
    logger.info("Home pose reached")

    # ── 4. Connect to ZMQ policy server ────────────────────────────
    socket, zmq_context = connect_policy_server(args.policy_host, args.policy_port)

    # ── 5. Warm-up inference ───────────────────────────────────────
    logger.info("Warming up policy server...")
    obs = get_obs(controller, cameras)
    state_2step, _ = build_state_2step(controller)
    try:
        _ = request_action(socket, state_2step, obs["images"], args.task)
        logger.info("Warm-up complete")
    except Exception as e:
        logger.error("Warm-up failed: %s", e)

    # ── 6. Wait for user ───────────────────────────────────────────
    logger.warning("=" * 55)
    logger.warning("  Robot at HOME. Motors ENABLED.")
    logger.warning("  Policy server: %s:%d", args.policy_host, args.policy_port)
    logger.warning("  Press Enter to START.")
    logger.warning("  Press Ctrl+C to quit.")
    logger.warning("=" * 55)
    input()

    # ── 7. Main loop (≡ UMI eval_arx5.py) ──────────────────────────
    dt = 1.0 / args.fps
    step_count = 0

    try:
        while True:
            loop_start = time.monotonic()

            # A. Capture observation
            obs = get_obs(controller, cameras)
            state_2step, T_base = build_state_2step(controller)

            # B. Send to server, get action chunk
            try:
                pred_ee = request_action(socket, state_2step, obs["images"], args.task)
            except Exception as e:
                logger.error("Inference request failed: %s", e)
                time.sleep(max(0, dt - (time.monotonic() - loop_start)))
                continue

            # C. Convert EE deltas → world frame
            pred_world = np.zeros_like(pred_ee)
            for t in range(len(pred_ee)):
                T_delta = action_to_pose(pred_ee[t])
                T_world = T_base @ T_delta
                pred_world[t] = pose_to_ee_state(T_world, pred_ee[t, 6])

            # D. Execute (≡ UMI env.exec_actions)
            n_sent = controller.exec_actions(
                pred_world, obs_timestamps=obs["t_obs"], dt=dt,
            )
            step_count += 1

            logger.info(
                "step=%d sent=%d/%d queue=%d",
                step_count, n_sent, len(pred_world), controller.remaining(),
            )

            if args.n_steps > 0 and step_count >= args.n_steps:
                logger.info("Reached n_steps limit (%d)", args.n_steps)
                break

            # Regulate loop frequency
            elapsed = time.monotonic() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        controller.stop()
        shm_manager.shutdown()
        socket.close()
        zmq_context.term()
        for cam in cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
