#!/usr/bin/env python3
"""Live RealSense D405 feed via ring buffer → OpenCV window.

Usage:
  conda activate lerobot_piper_sroi
  PYTHONPATH=src:$PYTHONPATH python tests/test_camera_live.py
  PYTHONPATH=src:$PYTHONPATH python tests/test_camera_live.py --dev_video_path /dev/video4
"""

import argparse
import logging
import time

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Live RealSense feed from ring buffer")
    parser.add_argument("--dev_video_path", type=str, default="")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    from multiprocessing.managers import SharedMemoryManager
    from modules.rs_camera import RealSenseCamera

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

    logger.info("LIVE — press Q to quit")

    try:
        while True:
            data = cam.get()
            frame = data["color"]
            step = int(data["step_idx"])

            cv2.putText(
                frame, f"step: {step}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("RealSense RingBuffer LIVE", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        cam.stop()
        shm.shutdown()
        logger.info("Done")


if __name__ == "__main__":
    main()
