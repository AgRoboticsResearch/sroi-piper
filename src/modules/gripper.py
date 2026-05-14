"""SROI gripper — DM4310 motor over DAMIAO DM-FDCAN USB-CAN serial bridge.

Zero LeRobot dependency. Uses pyserial, designed to run inside
PiperController.run() alongside the arm (UMI pattern: gripper and
arm share the same process, same command queue, same ring buffer).

Usage (standalone test):
    from modules.gripper import Gripper
    with Gripper("/dev/ttyACM0") as g:
        g.set_zero()
        g.send_command(kp=5.0, kd=0.5, position=0.0)
        state = g.read_state()
        print(f"pos={state.position:.3f} rad")
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import struct
import threading
import time
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# Motor types & limits
# ===========================================================================

class MotorType(IntEnum):
    DM3507 = 0
    DM4310 = 1
    DM4310_48V = 2
    DM4340 = 3
    DM4340_48V = 4
    DM6006 = 5
    DM8006 = 6
    DM8009 = 7
    DM10010L = 8
    DM10010 = 9
    DMH3510 = 10
    DMH6215 = 11
    DMG6220 = 12


@dataclass(frozen=True)
class MotorLimits:
    p_max: float
    v_max: float
    t_max: float


_MOTOR_LIMITS: dict[MotorType, MotorLimits] = {
    MotorType.DM3507:     MotorLimits(12.566, 50, 5),
    MotorType.DM4310:     MotorLimits(12.5, 30, 10),
    MotorType.DM4310_48V: MotorLimits(12.5, 50, 10),
    MotorType.DM4340:     MotorLimits(12.5, 10, 28),
    MotorType.DM4340_48V: MotorLimits(12.5, 20, 28),
    MotorType.DM6006:     MotorLimits(12.5, 45, 12),
    MotorType.DM8006:     MotorLimits(12.5, 45, 20),
    MotorType.DM8009:     MotorLimits(12.5, 45, 54),
    MotorType.DM10010L:   MotorLimits(12.5, 25, 200),
    MotorType.DM10010:    MotorLimits(12.5, 20, 200),
    MotorType.DMH3510:    MotorLimits(12.5, 280, 1),
    MotorType.DMH6215:    MotorLimits(12.5, 45, 10),
    MotorType.DMG6220:    MotorLimits(12.5, 45, 10),
}


# ===========================================================================
# DM CAN protocol helpers
# ===========================================================================

MIT_KP_RANGE = (0.0, 500.0)
MIT_KD_RANGE = (0.0, 5.0)

CMD_ENABLE  = 0xFC
CMD_DISABLE = 0xFD
CMD_SET_ZERO = 0xFE

_STATUS_NAMES: dict[int, str] = {
    0x0: "DISABLED",  0x1: "ENABLED",
    0x8: "OVER_VOLTAGE",  0x9: "UNDER_VOLTAGE",
    0xA: "OVER_CURRENT",  0xB: "MOS_OVER_TEMP",
    0xC: "ROTOR_OVER_TEMP", 0xD: "LOST_COMM",  0xE: "OVERLOAD",
}


@dataclass
class MotorState:
    position: float
    velocity: float
    torque: float
    temp_mos: int = 0
    temp_rotor: int = 0
    status: int = 0


def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    x = max(x_min, min(x_max, x))
    return int((x - x_min) / (x_max - x_min) * ((1 << bits) - 1))


def _uint_to_float(x: int, x_min: float, x_max: float, bits: int) -> float:
    return float(x) / ((1 << bits) - 1) * (x_max - x_min) + x_min


def encode_mit_cmd(kp: float, kd: float, position: float,
                   velocity: float, torque: float, limits: MotorLimits) -> list[int]:
    q_u  = _float_to_uint(position, -limits.p_max, limits.p_max, 16)
    dq_u = _float_to_uint(velocity, -limits.v_max, limits.v_max, 12)
    kp_u = _float_to_uint(kp, *MIT_KP_RANGE, 12)
    kd_u = _float_to_uint(kd, *MIT_KD_RANGE, 12)
    tau_u = _float_to_uint(torque, -limits.t_max, limits.t_max, 12)

    return [
        (q_u >> 8) & 0xFF, q_u & 0xFF,
        dq_u >> 4,
        ((dq_u & 0x0F) << 4) | ((kp_u >> 8) & 0x0F),
        kp_u & 0xFF,
        kd_u >> 4,
        ((kd_u & 0x0F) << 4) | ((tau_u >> 8) & 0x0F),
        tau_u & 0xFF,
    ]


def encode_simple_cmd(cmd_byte: int) -> list[int]:
    return [0xFF] * 7 + [cmd_byte]


def decode_motor_state(data: bytes, limits: MotorLimits) -> MotorState:
    status = (data[0] >> 4) & 0x0F
    q_u  = (data[1] << 8) | data[2]
    dq_u = (data[3] << 4) | (data[4] >> 4)
    tau_u = ((data[4] & 0x0F) << 8) | data[5]
    return MotorState(
        position=_uint_to_float(q_u, -limits.p_max, limits.p_max, 16),
        velocity=_uint_to_float(dq_u, -limits.v_max, limits.v_max, 12),
        torque=_uint_to_float(tau_u, -limits.t_max, limits.t_max, 12),
        temp_mos=data[6] if len(data) > 6 else 0,
        temp_rotor=data[7] if len(data) > 7 else 0,
        status=status,
    )


# ===========================================================================
# DAMIAO DM-FDCAN serial bridge transport
# ===========================================================================

_TX_HEADER = bytes([0x55, 0xAA, 0x1E])
_TX_CMD_SEND = 0x03
_RX_SOF = 0xAA
_RX_EOF = 0x55
_RX_CMD_DATA = 0x11
_RX_FRAME_SIZE = 16


class _Transport:
    """DAMIAO DM-FDCAN USB-CAN serial bridge (internal)."""

    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 921600):
        self._port = port
        self._baudrate = baudrate
        self._ser = None
        self._rx_buffer = bytearray()

    @property
    def is_open(self) -> bool:
        return self._ser is not None

    def open(self) -> None:
        import serial
        if self._ser is not None:
            return
        self._ser = serial.Serial(port=self._port, baudrate=self._baudrate, timeout=0.01)
        time.sleep(0.1)

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def send(self, can_id: int, data: list[int] | bytes) -> None:
        payload = bytes(data).ljust(8, b'\x00')[:8]
        frame = bytearray()
        frame.extend(_TX_HEADER)
        frame.append(_TX_CMD_SEND)
        frame.extend(struct.pack('<I', 1))
        frame.extend(struct.pack('<I', 10))
        frame.append(0x00)
        frame.extend(struct.pack('<I', can_id))
        frame.append(0x00)
        frame.append(len(payload))
        frame.append(0x00)
        frame.append(0x00)
        frame.extend(payload)
        frame.append(0x00)
        self._ser.write(bytes(frame))

    def recv(self, timeout: float = 0.01) -> tuple[int, bytes] | None:
        deadline = time.monotonic() + timeout
        while True:
            while len(self._rx_buffer) >= _RX_FRAME_SIZE:
                start = self._rx_buffer.find(_RX_SOF)
                if start < 0:
                    self._rx_buffer.clear()
                    break
                if start > 0:
                    del self._rx_buffer[:start]
                if len(self._rx_buffer) < _RX_FRAME_SIZE:
                    break
                frame = bytes(self._rx_buffer[:_RX_FRAME_SIZE])
                del self._rx_buffer[:_RX_FRAME_SIZE]
                if frame[1] != _RX_CMD_DATA or frame[15] != _RX_EOF:
                    continue
                can_id = struct.unpack('<I', frame[3:7])[0]
                return (can_id, frame[7:15])

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._ser.timeout = min(remaining, 0.005)
            chunk = self._ser.read(64)
            if chunk:
                self._rx_buffer.extend(chunk)
            else:
                return None

    def drain(self) -> None:
        self._rx_buffer.clear()
        if self._ser is not None:
            self._ser.reset_input_buffer()


# ===========================================================================
# Gripper — public API
# ===========================================================================

_HANDSHAKE_TIMEOUT = 0.1


class GripperError(RuntimeError):
    pass


class Gripper:
    """DM4310 gripper via DAMIAO DM-FDCAN serial dongle.

    Usage:
        with Gripper("/dev/ttyACM0") as g:
            g.send_command(kp=5.0, kd=0.5, position=0.0)
            state = g.read_state()
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 921600,
        can_id: int = 0x08,
        recv_id: int = 0x18,
        motor_type: MotorType = MotorType.DM4310,
    ):
        self._transport = _Transport(port=port, baudrate=baudrate)
        self._can_id = can_id
        self._recv_id = recv_id
        self._motor_type = motor_type
        self._limits: MotorLimits = _MOTOR_LIMITS[motor_type]
        self._state: MotorState | None = None
        self._lock = threading.Lock()

    # -- Lifecycle ----------------------------------------------------

    def connect(self) -> None:
        self._transport.open()
        self._transport.drain()
        self._transport.send(self._can_id, encode_simple_cmd(CMD_ENABLE))
        resp = self._recv_response(timeout=_HANDSHAKE_TIMEOUT)
        if resp is None:
            self._transport.close()
            raise ConnectionError(
                f"Motor 0x{self._can_id:02X} did not respond. "
                "Check power (24V), CAN wiring, and serial port."
            )
        self._update_state(resp)

    def disconnect(self) -> None:
        if self._transport.is_open:
            try:
                self.disable()
            except Exception:
                pass
            self._transport.close()

    def __enter__(self) -> Gripper:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    # -- Motor control -----------------------------------------------

    def enable(self) -> MotorState:
        self._send_command_and_read(CMD_ENABLE)
        return self._get_state()

    def disable(self) -> None:
        self._transport.send(self._can_id, encode_simple_cmd(CMD_DISABLE))
        time.sleep(0.002)

    def set_zero(self) -> MotorState:
        self._send_command_and_read(CMD_SET_ZERO)
        return self._get_state()

    # -- Impedance control -------------------------------------------

    def send_command(
        self, kp: float, kd: float, position: float,
        velocity: float = 0.0, torque: float = 0.0,
    ) -> MotorState:
        with self._lock:
            data = encode_mit_cmd(kp, kd, position, velocity, torque, self._limits)
            self._transport.send(self._can_id, data)
            resp = self._recv_response(timeout=0.005)
            if resp is not None:
                self._update_state(resp)
            return self._get_state()

    # -- State reading -----------------------------------------------

    def read_state(self) -> MotorState:
        with self._lock:
            resp = self._recv_response(timeout=0.01)
            if resp is not None:
                self._update_state(resp)
            return self._get_state()

    @property
    def state(self) -> MotorState:
        return self._get_state()

    @property
    def position(self) -> float:
        return self._get_state().position

    # -- Internal ----------------------------------------------------

    def _send_command_and_read(self, cmd: int) -> None:
        with self._lock:
            self._transport.send(self._can_id, encode_simple_cmd(cmd))
            resp = self._recv_response(timeout=0.01)
            if resp is not None:
                self._update_state(resp)

    def _recv_response(self, timeout: float) -> tuple[int, bytes] | None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            frame = self._transport.recv(timeout=remaining)
            if frame is None:
                return None
            can_id, _data = frame
            if can_id == self._recv_id:
                return frame

    def _update_state(self, frame: tuple[int, bytes]) -> None:
        _can_id, data = frame
        self._state = decode_motor_state(data, self._limits)
        if self._state.status > 0x1:
            raise GripperError(
                f"Motor fault: {_STATUS_NAMES.get(self._state.status, f'0x{self._state.status:X}')}"
            )

    def _get_state(self) -> MotorState:
        if self._state is None:
            return MotorState(position=0.0, velocity=0.0, torque=0.0)
        return self._state


# ===========================================================================
# GripperProcess — mp.Process for UMI-style non-blocking access
# ===========================================================================

_DEFAULT_KP = 10.0
_DEFAULT_KD = 1.0
_DEFAULT_POSITION = 0.0


class GripperProcess(mp.Process):
    """DM4310 gripper as mp.Process: serial I/O in subprocess, state via ring buffer.

    UMI pattern — the main process never blocks on serial. It writes
    target commands to a queue and reads state from a ring buffer.

    Usage:
        from multiprocessing.managers import SharedMemoryManager
        shm = SharedMemoryManager()
        shm.start()

        g = GripperProcess(shm_manager=shm, port="/dev/ttyACM0")
        g.start()
        g.start_wait()

        g.send_command(kp=10.0, kd=1.0, position=0.5)
        state = g.get_state()  # non-blocking ring buffer read
        print(f"pos={state['position']:.3f} rad")

        g.stop()
    """

    def __init__(
        self,
        shm_manager,
        port: str = "/dev/ttyACM0",
        baudrate: int = 921600,
        can_id: int = 0x08,
        recv_id: int = 0x18,
        motor_type: MotorType = MotorType.DM4310,
        frequency: float = 50.0,
        launch_timeout: float = 10.0,
        verbose: bool = False,
    ):
        super().__init__(name="GripperProcess")
        self._port = port
        self._baudrate = baudrate
        self._can_id = can_id
        self._recv_id = recv_id
        self._motor_type = motor_type
        self._frequency = frequency
        self._launch_timeout = launch_timeout
        self._verbose = verbose

        # Command queue: main process → gripper process
        cmd_example = {
            "kp": np.float64(_DEFAULT_KP),
            "kd": np.float64(_DEFAULT_KD),
            "position": np.float64(_DEFAULT_POSITION),
            "velocity": np.float64(0.0),
            "torque": np.float64(0.0),
        }
        from shared_memory import SharedMemoryQueue
        self._cmd_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=cmd_example,
            buffer_size=64,
        )

        # State ring buffer: gripper process → main process
        state_example = {
            "position": np.float64(0.0),
            "velocity": np.float64(0.0),
            "torque": np.float64(0.0),
            "temp_mos": np.int64(0),
            "temp_rotor": np.int64(0),
            "status": np.int64(0),
        }
        from shared_memory import SharedMemoryRingBuffer
        self._ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=state_example,
            get_max_k=int(frequency * 2),
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()

    # ========== Lifecycle ==========

    def start(self, wait: bool = True):
        super().start()
        if wait:
            self.start_wait()

    def start_wait(self):
        self.ready_event.wait(self._launch_timeout)
        if not self.ready_event.is_set():
            raise TimeoutError("GripperProcess did not become ready")

    def stop(self, wait: bool = True):
        # Send disable command
        try:
            self._cmd_queue.put({
                "kp": np.float64(0.0),
                "kd": np.float64(0.0),
                "position": np.float64(0.0),
                "velocity": np.float64(0.0),
                "torque": np.float64(0.0),
            })
        except Exception:
            pass
        if wait:
            self.stop_wait()

    def stop_wait(self):
        self.join(timeout=5.0)
        if self.is_alive():
            self.terminate()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========== Command (non-blocking) ==========

    def send_command(
        self, kp: float, kd: float, position: float,
        velocity: float = 0.0, torque: float = 0.0,
    ) -> None:
        """Queue a command. Non-blocking — returns immediately."""
        self._cmd_queue.put({
            "kp": np.float64(kp),
            "kd": np.float64(kd),
            "position": np.float64(position),
            "velocity": np.float64(velocity),
            "torque": np.float64(torque),
        })

    # ========== State (non-blocking) ==========

    def get_state(self, k: int | None = None) -> dict:
        """Read latest state from ring buffer. Non-blocking."""
        if k is None:
            return self._ring_buffer.get()
        return self._ring_buffer.get_last_k(k=k)

    @property
    def position(self) -> float:
        return float(self._ring_buffer.get()["position"])

    # ========== Main loop (subprocess) ==========

    def run(self):
        limits = _MOTOR_LIMITS[self._motor_type]

        transport = _Transport(port=self._port, baudrate=self._baudrate)
        transport.open()
        transport.drain()

        # Enable motor
        transport.send(self._can_id, encode_simple_cmd(CMD_ENABLE))
        resp = self._recv_response(transport, self._recv_id, timeout=_HANDSHAKE_TIMEOUT)
        if resp is None:
            transport.close()
            self.ready_event.set()
            raise ConnectionError(
                f"Motor 0x{self._can_id:02X} did not respond. "
                "Check power (24V), CAN wiring, and serial port."
            )
        _can_id, data = resp
        state = decode_motor_state(data, limits)
        if state.status > 0x1:
            transport.close()
            self.ready_event.set()
            raise GripperError(
                f"Motor fault: {_STATUS_NAMES.get(state.status, f'0x{state.status:X}')}"
            )

        dt = 1.0 / self._frequency
        last_kp = _DEFAULT_KP
        last_kd = _DEFAULT_KD
        last_position = state.position
        iter_idx = 0
        t_start = time.monotonic()

        try:
            while True:
                # Step 1: read latest command (non-blocking drain)
                from queue import Empty
                cmd = None
                try:
                    while True:
                        cmd = self._cmd_queue.get()
                except Empty:
                    pass

                if cmd is not None:
                    last_kp = float(cmd["kp"])
                    last_kd = float(cmd["kd"])
                    last_position = float(cmd["position"])

                # Step 2: send MIT command
                data = encode_mit_cmd(
                    last_kp, last_kd, last_position,
                    float(cmd["velocity"]) if cmd else 0.0,
                    float(cmd["torque"]) if cmd else 0.0,
                    limits,
                )
                transport.send(self._can_id, data)

                # Step 3: read response
                resp = self._recv_response(transport, self._recv_id, timeout=0.005)
                if resp is not None:
                    _can_id, resp_data = resp
                    state = decode_motor_state(resp_data, limits)
                    if state.status > 0x1:
                        raise GripperError(
                            f"Motor fault: {_STATUS_NAMES.get(state.status, f'0x{state.status:X}')}"
                        )

                # Step 4: write state to ring buffer
                try:
                    self._ring_buffer.put({
                        "position": np.float64(state.position),
                        "velocity": np.float64(state.velocity),
                        "torque": np.float64(state.torque),
                        "temp_mos": np.int64(state.temp_mos),
                        "temp_rotor": np.int64(state.temp_rotor),
                        "status": np.int64(state.status),
                    }, wait=False)
                except TimeoutError:
                    pass

                if iter_idx == 0:
                    self.ready_event.set()
                    if self._verbose:
                        logger.info("GripperProcess ready on %s, pos=%.3f rad",
                                    self._port, state.position)

                # Step 5: regulate frequency
                t_wait = t_start + (iter_idx + 1) * dt
                now = time.monotonic()
                if t_wait > now:
                    time.sleep(t_wait - now)
                iter_idx += 1

        except Exception as e:
            logger.error("GripperProcess error: %s", e)
        finally:
            try:
                transport.send(self._can_id, encode_simple_cmd(CMD_DISABLE))
                time.sleep(0.002)
            except Exception:
                pass
            transport.close()
            self.ready_event.set()
            if self._verbose:
                logger.info("GripperProcess terminated")

    @staticmethod
    def _recv_response(
        transport: _Transport, recv_id: int, timeout: float,
    ) -> tuple[int, bytes] | None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            frame = transport.recv(timeout=remaining)
            if frame is None:
                return None
            can_id, _data = frame
            if can_id == recv_id:
                return frame
