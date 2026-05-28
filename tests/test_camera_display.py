#!/usr/bin/env python3
"""Display RealSense ring buffer frames — save snapshots + print info.

Usage:
  conda activate lerobot_piper_sroi
  PYTHONPATH=src:$PYTHONPATH python tests/test_camera_display.py
  PYTHONPATH=src:$PYTHONPATH python tests/test_camera_display.py --dev_video_path /dev/video4 --save_frames 5
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "camera_snapshots"


def main():
    parser = argparse.ArgumentParser(description="Display RealSense ring buffer frames")
    parser.add_argument("--dev_video_path", type=str, default="",
                        help="V4L2 device path (auto-detect if empty)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=10.0,
                        help="How long to run (seconds)")
    parser.add_argument("--save_frames", type=int, default=3,
                        help="Number of frames to save as PNG")
    args = parser.parse_args()

    from multiprocessing.managers import SharedMemoryManager
    from modules.rs_camera import RealSenseCamera

    # ── 1. Start camera ─────────────────────────────────────────
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
    logger.info("Camera ready — %s %dx%d@%d",
                 cam.dev_video_path, args.width, args.height, args.fps)

    # ── 2. Read loop ───────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    frame_count = 0
    t_start = time.monotonic()
    last_log = t_start
    last_step = -1

    try:
        while time.monotonic() - t_start < args.duration:
            data = cam.get()
            color = data["color"]
            step = int(data["step_idx"])
            frame_count += 1

            # Save frames (one per distinct step)
            if saved < args.save_frames and step != last_step:
                last_step = step
                path = OUTPUT_DIR / f"frame_{saved:03d}_step{step:04d}.png"
                cv2.imwrite(str(path), color)
                logger.info("Saved: %s  shape=%s  min=%d max=%d",
                             path.name, color.shape,
                             int(color.min()), int(color.max()))
                saved += 1

            # Periodic stats
            now = time.monotonic()
            if now - last_log >= 1.0:
                fps_actual = frame_count / (now - t_start)
                logger.info("step=%d ring_reads=%d fps=%.1f",
                             step, frame_count, fps_actual)
                last_log = now

    except KeyboardInterrupt:
        logger.info("Interrupted")

    finally:
        elapsed = time.monotonic() - t_start
        cam.stop()
        shm.shutdown()

        logger.info("Saved %d frames to %s", saved, OUTPUT_DIR)
        logger.info("Total: %d ring reads in %.1fs", frame_count, elapsed)

        # Print summary of saved files
        pngs = sorted(OUTPUT_DIR.glob("frame_*.png"))
        if pngs:
            print(f"\n{'='*55}")
            print(f"  Saved {len(pngs)} images:")
            for p in pngs:
                size_kb = p.stat().st_size / 1024
                print(f"    {p.name}  ({size_kb:.0f} KB)")
            print(f"  View with:  xdg-open {OUTPUT_DIR}")
            print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
