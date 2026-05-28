#!/usr/bin/env python3
"""Quick test: start RealSense camera subprocess, read frames from ring buffer.

Verifies the OpenCV V4L2 → ring buffer pipeline works before integrating
with the full viz/inference stack.

Usage:
  conda activate lerobot_piper_sroi
  PYTHONPATH=src:$PYTHONPATH python tests/test_rs_camera.py
  PYTHONPATH=src:$PYTHONPATH python tests/test_rs_camera.py --dev_video_path /dev/video4
  PYTHONPATH=src:$PYTHONPATH python tests/test_rs_camera.py --width 1280 --height 720 --fps 15
"""

import argparse
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Test RealSense camera subprocess")
    parser.add_argument("--dev_video_path", type=str, default="",
                        help="V4L2 device path (auto-detect if empty)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Test duration in seconds")
    parser.add_argument("--get_k", type=int, default=0,
                        help="Read last K frames each iteration (0 = single frame)")
    args = parser.parse_args()

    from multiprocessing.managers import SharedMemoryManager
    from modules.rs_camera import RealSenseCamera

    # ── 1. Start camera subprocess ─────────────────────────────────
    shm = SharedMemoryManager()
    shm.start()

    cam = RealSenseCamera(
        shm_manager=shm,
        dev_video_path=args.dev_video_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_name="color",
    )
    cam.start()
    cam.start_wait()
    logger.info("Camera ready — reading frames for %.0fs...", args.duration)

    # ── 2. Read loop ───────────────────────────────────────────────
    frame_count = 0
    t_start = time.monotonic()
    last_log = t_start

    try:
        while time.monotonic() - t_start < args.duration:
            if args.get_k > 0:
                data = cam.get(k=args.get_k)
                color = data["color"]
                frame_count += args.get_k
            else:
                data = cam.get()
                color = data["color"]
                frame_count += 1

            step = int(data["step_idx"])

            now = time.monotonic()
            if now - last_log >= 1.0:
                elapsed = now - t_start
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                logger.info(
                    "step=%d frames=%d fps=%.1f shape=%s dtype=%s",
                    step, frame_count, fps_actual, color.shape, color.dtype,
                )
                last_log = now

    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.monotonic() - t_start
        actual_fps = frame_count / elapsed if elapsed > 0 else 0
        logger.info(
            "Stopped. %d frames in %.1fs = %.1f FPS (target: %d)",
            frame_count, elapsed, actual_fps, args.fps,
        )
        cam.stop()
        shm.shutdown()
        logger.info("Done.")


if __name__ == "__main__":
    main()
