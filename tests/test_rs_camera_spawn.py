#!/usr/bin/env python3
"""Standalone test: PyRealSenseCamera spawn subprocess.

Verifies the camera subprocess starts, captures frames, and writes to the
ring buffer. Saves a sample frame and prints frame statistics.

Usage:
  python tests/test_rs_camera_spawn.py --camera_serial 230322273077
  python tests/test_rs_camera_spawn.py --camera_serial 230322273077 --duration 10
"""

import argparse
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "output"

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test PyRealSenseCamera spawn subprocess"
    )
    parser.add_argument("--camera_serial", type=str, default="",
                        help="RealSense serial number")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Test duration in seconds")
    parser.add_argument("--save_frame", action="store_true", default=True,
                        help="Save sample frame to tests/output/")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # ── 0. spawn required for pyrealsense2 ──────────────────────────────
    mp.set_start_method("spawn")

    # ── 1. SharedMemoryManager ─────────────────────────────────────────
    from multiprocessing.managers import SharedMemoryManager
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    # ── 2. Import after spawn is set ───────────────────────────────────
    from modules.rs_camera_pysdk import PyRealSenseCamera

    # ── 3. Start camera subprocess ─────────────────────────────────────
    cam = PyRealSenseCamera(
        shm_manager=shm_manager,
        serial_number=args.camera_serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_name="test",
    )

    logger.info("Starting PyRealSenseCamera subprocess (spawn)...")
    cam.start()
    cam.start_wait(timeout=15.0)

    if not cam.is_alive():
        logger.error("Camera subprocess DIED during start_wait — check errors above")
        logger.error("(pyrealsense2 may have failed to open device %s)", args.camera_serial)
        cam.stop(wait=False)
        shm_manager.shutdown()
        sys.exit(1)

    logger.info("Camera ready (pid=%d)", cam.pid)

    # ── 4. Collect frames for N seconds ────────────────────────────────
    logger.info("Capturing frames for %.1fs...", args.duration)
    frame_count = 0
    t_start = time.monotonic()
    last_log = t_start
    last_step = 0
    first_frame = None

    while (time.monotonic() - t_start) < args.duration:
        data = cam.get()
        img = data["color"]
        step = int(data["step_idx"])
        frame_count += 1

        if first_frame is None:
            first_frame = img.copy()
            first_step = step
            if float(img.mean()) == 0.0:
                logger.error("First frame is all zeros — camera subprocess may have crashed")
                break

        now = time.monotonic()
        if now - last_log >= 1.0:
            actual_fps = (step - last_step) / (now - last_log)
            frame_age_ms = (time.time() - float(data["timestamp"])) * 1000
            logger.info(
                "read=%d step=%d cam_fps=%.1f shape=%s mean=%.1f min=%d max=%d age=%.1fms",
                frame_count, step, actual_fps,
                img.shape, float(img.mean()),
                int(img.min()), int(img.max()), frame_age_ms,
            )
            last_log = now
            last_step = step

    elapsed = time.monotonic() - t_start
    cam_fps = (step - first_step) / elapsed if step > first_step else 0
    logger.info(
        "Ring buffer reads: %d in %.1fs. Camera: %d frames in %.1fs (%.1f fps)",
        frame_count, elapsed, step - first_step, elapsed, cam_fps,
    )

    # ── 5. Save sample frame ───────────────────────────────────────────
    if args.save_frame and first_frame is not None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = OUTPUT_DIR / "test_rs_camera_spawn.jpg"
        # pyrealsense2 gives RGB, cv2 expects BGR
        cv2.imwrite(str(save_path), cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        logger.info("Saved sample frame: %s", save_path)
        logger.info(
            "Frame stats: shape=%s mean=%.1f min=%d max=%d",
            first_frame.shape,
            float(first_frame.mean()),
            int(first_frame.min()),
            int(first_frame.max()),
        )

    # ── 6. Shutdown ────────────────────────────────────────────────────
    logger.info("Stopping camera...")
    cam.stop()
    shm_manager.shutdown()
    logger.info("Done")


if __name__ == "__main__":
    main()
