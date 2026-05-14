#!/usr/bin/env python3
"""Full pipeline dry-run test: camera + controller + ZMQ inference + Placo viz + gripper.

Runs the complete UMI-style pipeline with motors DISABLED. Move the arm by hand
to see predicted waypoints in meshcat. Gripper state is read but not commanded.

Architecture (UMI pattern):
  RealSenseCamera (mp.Process) ──→ SharedMemoryRingBuffer (color frames)
  PiperController (mp.Process) ──→ SharedMemoryRingBuffer (joints, EE pose, gripper)
                                  ← SharedMemoryQueue (scheduled waypoints)
  Main process: reads ring buffers → ZMQ inference → schedules waypoints → meshcat viz

Usage:
  # Terminal 1: start ZMQ policy server
  python scripts/policy_server_zmq.py --pretrained_path ... --host 0.0.0.0 --port 8766

  # Terminal 2: run pipeline dry-run
  python tests/test_full_pipeline.py \
      --dev_video_path /dev/video4 --can_port can0 --gripper_port /dev/ttyACM0

  # Without gripper:
  python tests/test_full_pipeline.py --dev_video_path /dev/video4 --can_port can0
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

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
HOME_POSE_DEG = np.array([0.0, 50.60, -50.40, -1.21, 10.00, 0.0])

logger = logging.getLogger(__name__)


# ===========================================================================
# Placo meshcat visualization
# ===========================================================================

def init_placo_viz(urdf_path: str) -> dict:
    import placo
    from placo_utils.visualization import robot_viz, frame_viz

    robot = placo.RobotWrapper(urdf_path)
    for jn, val in zip(ARM_JOINTS, HOME_POSE_DEG):
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
    T = np.eye(4)
    T[:3, 3] = pose_6d[:3]
    T[:3, :3] = Rotation.from_rotvec(pose_6d[3:]).as_matrix()
    return T


# ===========================================================================
# Observation (reads from ring buffers — non-blocking)
# ===========================================================================

def get_obs(controller, cameras: dict) -> dict:
    images = {}
    t_obs = 0.0
    for name, cam in cameras.items():
        data = cam.get()
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
# ZMQ inference
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


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full pipeline dry-run: camera + controller + ZMQ + Placo viz + gripper"
    )
    # Robot
    parser.add_argument("--can_port", type=str, default="can0")
    parser.add_argument("--urdf_path", type=str, default=URDF_PATH)
    parser.add_argument("--control_hz", type=float, default=50.0)
    # Gripper
    parser.add_argument("--gripper_port", type=str, default="",
                        help="Serial port for gripper (e.g. /dev/ttyACM0). Omit to skip gripper.")
    parser.add_argument("--gripper_kp", type=float, default=10.0)
    parser.add_argument("--gripper_kd", type=float, default=1.0)
    # Camera (RealSense via OpenCV V4L2 subprocess)
    parser.add_argument("--dev_video_path", type=str, default="",
                        help="V4L2 device path (e.g. /dev/video4). Auto-detected if omitted.")
    parser.add_argument("--cam_width", type=int, default=640)
    parser.add_argument("--cam_height", type=int, default=480)
    parser.add_argument("--cam_fps", type=int, default=30)
    # Inference
    parser.add_argument("--policy_host", type=str, default="localhost")
    parser.add_argument("--policy_port", type=int, default=8766)
    parser.add_argument("--task", type=str, default="pick the red strawberry")
    # Loop
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--infer_every", type=int, default=6,
                        help="Run inference every N frames")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # ── 1. SharedMemoryManager ───────────────────────────────────────
    from multiprocessing.managers import SharedMemoryManager
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    # ── 2. ZMQ inference client ──────────────────────────────────────
    logger.info("Connecting to ZMQ policy server at %s:%d",
                args.policy_host, args.policy_port)
    zmq_sock, zmq_ctx = create_zmq_client(args.policy_host, args.policy_port)

    # ── 3. Start camera subprocess (UMI pattern) ─────────────────────
    from modules.rs_camera import RealSenseCamera

    cam_rb = {}
    cam_process = RealSenseCamera(
        shm_manager=shm_manager,
        dev_video_path=args.dev_video_path,
        width=args.cam_width,
        height=args.cam_height,
        fps=args.cam_fps,
        camera_name="color",
    )
    cam_process.start()
    cam_process.start_wait()
    cam_rb["color"] = cam_process
    logger.info("RealSense camera subprocess started @ %d FPS", args.cam_fps)

    # ── 4. Start PiperController subprocess (dry_run) ────────────────
    from modules.piper_controller import PiperController

    controller_kwargs = dict(
        shm_manager=shm_manager,
        can_port=args.can_port,
        urdf_path=args.urdf_path,
        frequency=args.control_hz,
        dry_run=True,
        verbose=True,
    )
    if args.gripper_port:
        controller_kwargs.update(
            gripper_port=args.gripper_port,
            gripper_kp=args.gripper_kp,
            gripper_kd=args.gripper_kd,
        )
        logger.info("Gripper enabled on %s", args.gripper_port)

    controller = PiperController(**controller_kwargs)
    logger.info("Starting controller subprocess (dry_run=True)...")
    controller.start(wait=True)
    logger.info("Controller process started — motors DISABLED")

    # ── 5. Init Placo meshcat ───────────────────────────────────────
    placo_viz = init_placo_viz(args.urdf_path)

    # ── 6. Main loop ────────────────────────────────────────────────
    dt = 1.0 / args.fps
    frame_count = 0
    infer_count = 0
    last_infer_ms = 0.0
    last_log_time = time.monotonic()

    logger.info("Pipeline running. Move arm by hand to see predicted waypoints.")
    logger.info("Press Ctrl+C to quit. Open http://127.0.0.1:7001/static/ in browser.")

    try:
        while True:
            loop_start = time.monotonic()

            # A. Update URDF viz from controller ring buffer
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
                    obs = get_obs(controller, cam_rb)
                    state_2step, T_base = build_state_2step(controller)
                    pred_ee = zmq_infer(zmq_sock, state_2step, obs["images"], args.task)
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
                grip_str = ""
                try:
                    grip_val = float(controller.get_state()["gripper"])
                    grip_str = f" grip={grip_val:.2f}"
                except Exception:
                    pass
                try:
                    T_ee = placo_viz["robot"].get_T_world_frame("ee_link")
                    ee_pos = T_ee[:3, 3] * 1000
                    ee_str = f"xyz=[{ee_pos[0]:.0f},{ee_pos[1]:.0f},{ee_pos[2]:.0f}]mm"
                except Exception:
                    ee_str = "xyz=?"
                logger.info(
                    "infer=%d dt=%.0fms queue=%d %s%s",
                    infer_count, last_infer_ms, queue_n, ee_str, grip_str,
                )

            frame_count += 1

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        logger.info("Shutting down...")
        controller.stop()
        cam_process.stop()
        shm_manager.shutdown()
        zmq_sock.close()
        zmq_ctx.term()
        logger.info("Done")


if __name__ == "__main__":
    main()
