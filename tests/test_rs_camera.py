#!/usr/bin/env python3
"""Quick test: start RealSense camera subprocess, read frames from ring buffer.

Verifies the pyrealsense2 → ring buffer pipeline works before integrating
with the full viz/inference stack.

Usage:
  python tests/test_rs_camera.py --serial 230322273077
  python tests/test_rs_camera.py --serial 230322273077 --width 1280 --height 720 --fps 15
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
    parser.add_argument("--serial", type=str, default="")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Test duration in seconds")
    parser.add_argument("--get_k", type=int, default=1,
                        help="Read last K frames each iteration")
    args = parser.parse_args()

    from multiprocessing.managers import SharedMemoryManager
    from modules.rs_camera import RealSenseCamera

    # ── 1. Start camera subprocess ─────────────────────────────────
    shm = SharedMemoryManager()
    shm.start()

    cam = RealSenseCamera(
        shm_manager=shm,
        serial_number=args.serial,
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
            data = cam.get(k=args.get_k)

            color = data["color"]
            ts = float(data["timestamp"])
            step = int(data["step_idx"])

            frame_count += args.get_k if args.get_k > 0 else 1

            now = time.monotonic()
            if now - last_log >= 1.0:
                elapsed = now - t_start
                fps_actual = frame_count / elapsed
                if args.get_k > 1:
                    logger.info(
                        "step=%d frames=%d fps=%.1f shape=%s ts=%.3f (last %d frames)",
                        step, frame_count, fps_actual, color.shape, ts, args.get_k,
                    )
                else:
                    logger.info(
                        "step=%d frames=%d fps=%.1f shape=%s dtype=%s ts=%.3f",
                        step, frame_count, fps_actual, color.shape, color.dtype, ts,
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
