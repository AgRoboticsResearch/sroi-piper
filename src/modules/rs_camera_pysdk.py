"""RealSense camera subprocess using pyrealsense2 SDK → SharedMemoryRingBuffer.

Uses the official pyrealsense2 SDK (not V4L2/cv2) to match training data quality.
pyrealsense2 does white balance, exposure control, and color correction that V4L2
does not provide, so images match what the model was trained on.

Must use mp.set_start_method('spawn') BEFORE creating any instances.  Fork
copies the parent's USB/librealsense handles into the child, causing
"Device or resource busy" / "failed to open USB" errors.

Usage:
    import multiprocessing as mp
    mp.set_start_method("spawn")

    from multiprocessing.managers import SharedMemoryManager
    from modules.rs_camera_pysdk import PyRealSenseCamera

    shm = SharedMemoryManager()
    shm.start()

    cam = PyRealSenseCamera(shm_manager=shm, serial_number="230322273077",
                            width=640, height=480, fps=30)
    cam.start()
    cam.start_wait()

    data = cam.get()  # {"color": (H,W,3) uint8 RGB, "timestamp": float, ...}
    cam.stop()
"""

import logging
import multiprocessing as mp
import time

import numpy as np

from shared_memory import SharedMemoryRingBuffer

logger = logging.getLogger(__name__)


class PyRealSenseCamera(mp.Process):
    """Camera subprocess using pyrealsense2 → SharedMemoryRingBuffer.

    All pyrealsense2 init happens in run() (child process). The parent
    stores only serializable config; nothing from librealsense is
    inherited via fork.

    Must be used with mp.set_start_method('spawn').
    """

    def __init__(
        self,
        shm_manager,
        serial_number: str = "",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        camera_name: str = "color",
        get_max_k: int = 30,
        get_time_budget: float = 0.2,
    ):
        super().__init__(name=f"RS-{camera_name}")
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_name = camera_name

        examples = {
            "color": np.empty(shape=(height, width, 3), dtype=np.uint8),
            "camera_receive_timestamp": np.float64(0.0),
            "timestamp": np.float64(0.0),
            "step_idx": np.int64(0),
        }
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
                f"PyRealSenseCamera {self.camera_name} not ready within {timeout}s"
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
        import traceback

        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number:
            config.enable_device(self.serial_number)
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps,
        )

        profile = pipeline.start(config)
        logger.info(
            "PyRealSense %s: pipeline started (serial=%s, %dx%d@%d)",
            self.camera_name, self.serial_number or "any",
            self.width, self.height, self.fps,
        )

        # Warm-up: let auto-exposure / white balance settle.
        # In spawn mode the USB subsystem may need extra time for first frame.
        time.sleep(1.0)
        for i in range(30):
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                if frames:
                    break
            except RuntimeError:
                pass
            time.sleep(0.1)
        else:
            logger.error("PyRealSense %s: no frames during warm-up", self.camera_name)
            pipeline.stop()
            self.ready_event.set()  # signal failure so start_wait doesn't hang
            return

        dt = 1.0 / self.fps
        step_idx = 0
        frame_count = 0
        ok = False

        try:
            while not self.stop_event.is_set():
                loop_start = time.monotonic()

                frames = pipeline.wait_for_frames(timeout_ms=int(dt * 1000 + 500))
                color_frame = frames.get_color_frame()
                if not color_frame:
                    time.sleep(0.001)
                    continue

                img = np.asanyarray(color_frame.get_data())  # (H, W, 3) uint8 RGB
                t_recv = time.time()

                data = {
                    "color": img,
                    "camera_receive_timestamp": np.float64(t_recv),
                    "timestamp": np.float64(t_recv),
                    "step_idx": np.int64(step_idx),
                }

                try:
                    self.ring_buffer.put(data, wait=False)
                except TimeoutError:
                    pass

                step_idx += 1
                frame_count += 1

                if frame_count == 1:
                    ok = True
                    self.ready_event.set()

                elapsed = time.monotonic() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        except Exception:
            logger.error(
                "PyRealSense %s crashed:\n%s",
                self.camera_name, traceback.format_exc(),
            )
        finally:
            pipeline.stop()
            if not ok:
                self.ready_event.set()  # unblock start_wait even on failure
            logger.info(
                "PyRealSense %s stopped, %d frames captured",
                self.camera_name, frame_count,
            )
