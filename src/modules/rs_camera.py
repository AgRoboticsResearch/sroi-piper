"""RealSense camera subprocess: OpenCV V4L2 capture → SharedMemoryRingBuffer.

Uses cv2.VideoCapture on the RealSense V4L2 color stream (video-index4).
Follows the UMI UvcCamera pattern exactly — mp.Process (fork) + OpenCV.
No pyrealsense2 dependency.

RealSense D405 V4L2 node layout:
  video-index0 → depth (Z16)
  video-index2 → IR (GREY)
  video-index4 → color (YUYV)  ← this is what we use

Stable device path (preferred):
  /dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_405_*-video-index4

Or direct:
  /dev/video4

Usage:
    from multiprocessing.managers import SharedMemoryManager
    from modules.rs_camera import RealSenseCamera

    shm = SharedMemoryManager()
    shm.start()

    cam = RealSenseCamera(shm_manager=shm, dev_video_path="/dev/video4",
                          width=640, height=480, fps=30, camera_name="color")
    cam.start()
    cam.start_wait()

    data = cam.get()  # {"color": (H,W,3) uint8 BGR, "timestamp": float, ...}
    cam.stop()
"""

import logging
import multiprocessing as mp
import time

import cv2
import numpy as np

from shared_memory import SharedMemoryRingBuffer

logger = logging.getLogger(__name__)


def find_realsense_color_path() -> str | None:
    """Find the RealSense D405 color V4L2 node (video-index4).

    Scans /dev/v4l/by-id/ for a RealSense Depth Camera 405 entry,
    resolves the symlink to the real /dev/videoN path (OpenCV handles
    short paths more reliably than long by-id symlinks).
    Returns the resolved device path, or None.
    """
    import glob, os
    pattern = "/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_405*-video-index4"
    matches = sorted(glob.glob(pattern))
    if matches:
        return os.path.realpath(matches[0])
    return None


class RealSenseCamera(mp.Process):
    """RealSense camera subprocess using OpenCV V4L2 → SharedMemoryRingBuffer.

    Inherits mp.Process and uses fork (Linux default). All camera init
    happens in run() so nothing is pickled. cap.release() on V4L2
    properly releases the USB device via the kernel, so repeated
    start/stop cycles work correctly.
    """

    def __init__(
        self,
        shm_manager,
        dev_video_path: str = "",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        camera_name: str = "color",
        transform=None,
        get_max_k: int = 30,
        get_time_budget: float = 0.2,
        cap_buffer_size: int = 1,
    ):
        super().__init__(name=f"RS-{camera_name}")

        if not dev_video_path:
            dev_video_path = find_realsense_color_path()
            if not dev_video_path:
                raise ValueError(
                    "No dev_video_path given and no RealSense D405 found. "
                    "Specify --dev_video_path /dev/video4 or the full by-id path."
                )

        self.dev_video_path = dev_video_path
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_name = camera_name
        self.transform = transform
        self.put_fps = fps
        self.cap_buffer_size = cap_buffer_size

        # Ring buffer: stores frames + timestamps
        examples = {
            "color": np.empty(shape=(height, width, 3), dtype=np.uint8),
            "camera_receive_timestamp": np.float64(0.0),
            "timestamp": np.float64(0.0),
            "step_idx": np.int64(0),
        }

        if transform is not None:
            examples = transform(examples)

        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            get_max_k=get_max_k,
            get_time_budget=get_time_budget,
            put_desired_frequency=fps,
        )

        self.ready_event = mp.Event()
        self.stop_event = mp.Event()

    # ========== Lifecycle ==========

    def start(self, wait: bool = True):
        super().start()
        if wait:
            self.start_wait()

    def start_wait(self, timeout: float = 10.0):
        self.ready_event.wait(timeout)
        if not self.ready_event.is_set():
            raise TimeoutError(
                f"RealSenseCamera {self.camera_name} not ready within {timeout}s"
            )

    def stop(self, wait: bool = True):
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def stop_wait(self):
        self.join()

    # ========== Data access (main process) ==========

    def get(self, k: int | None = None) -> dict[str, np.ndarray]:
        if k is None:
            return self.ring_buffer.get()
        return self.ring_buffer.get_last_k(k=k)

    # ========== Main loop (subprocess) ==========

    def run(self):
        # Open V4L2 device
        cap = cv2.VideoCapture(self.dev_video_path, cv2.CAP_V4L2)
        try:
            w, h = self.width, self.height
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cap_buffer_size)

            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logger.info(
                "RealSense %s started: %s %dx%d@%d V4L2",
                self.camera_name, self.dev_video_path,
                int(actual_w), int(actual_h), self.fps,
            )

            dt = 1.0 / self.fps
            step_idx = 0
            frame_count = 0

            while not self.stop_event.is_set():
                loop_start = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    logger.warning("RealSense %s: cap.read() failed", self.camera_name)
                    time.sleep(max(0, dt - (time.monotonic() - loop_start)))
                    continue

                t_recv = time.time()

                data = {
                    "color": frame,
                    "camera_receive_timestamp": np.float64(t_recv),
                    "timestamp": np.float64(t_recv),
                    "step_idx": np.int64(step_idx),
                }

                if self.transform is not None:
                    data = self.transform(data)

                try:
                    self.ring_buffer.put(data, wait=False)
                except TimeoutError:
                    pass  # consumer too slow, drop frame

                step_idx += 1
                frame_count += 1

                if frame_count == 1:
                    self.ready_event.set()

                # Regulate FPS
                elapsed = time.monotonic() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        finally:
            cap.release()
            self.ready_event.set()
            logger.info(
                "RealSense %s stopped, %d frames captured",
                self.camera_name, frame_count,
            )
