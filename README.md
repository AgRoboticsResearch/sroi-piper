# 🥧 sroi-piper

SROI Piper robot control stack — UMI-ARX architecture, zero LeRobot dependency on the client.

```
🎥 Camera (mp.Process)  ──→  Ring Buffer
🦾 Arm (mp.Process)      ──→  Ring Buffer  ──→  🧠 ZMQ Inference  ──→  🌍 Placo Meshcat Viz
🤏 Gripper (mp.Process)  ──→  Ring Buffer
```

## 🏗️ Architecture

```
src/
├── modules/
│   ├── piper_env.py          🎪 Orchestrator (cameras + arm + gripper)
│   ├── piper_controller.py   🦾 Arm control loop (mp.Process, IK @ 50Hz)
│   ├── piper_interface.py    🔌 CAN bus driver (piper_sdk wrapper)
│   ├── gripper.py            🤏 DM4310 gripper + GripperProcess (mp.Process)
│   ├── rs_camera.py          🎥 RealSense D405 camera (mp.Process)
│   └── pose_trajectory_interpolator.py  📐 SE(3) spline interpolation
├── shared_memory/
│   ├── ring_buffer.py        📡 Lock-free ring buffer (state → main)
│   └── queue.py              📨 Lock-free queue (commands → worker)
└── utils/
    ├── kinematics.py         🧮 FK/IK via Placo
    └── precise_wait.py       ⏱️ Busy-wait timer
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install numpy scipy pyserial zmq placo placo-utils

# 2. Arm only (optional, for real hardware)
pip install piper_sdk

# 3. Test gripper open/close
python tests/test_gripper.py --port /dev/ttyACM0 --cycles 5

# 4. Read arm state (motors disabled, safe!)
python tests/test_piper_state.py --can_port can0

# 5. Move arm to home pose (motors enabled ⚠️)
python tests/test_piper_move.py --can_port can0

# 6. Full pipeline dry-run (arm disabled, visualizes waypoints)
#    Start policy server first:
python scripts/policy_server_zmq.py --pretrained_path ./checkpoint --port 8766
#    Then run pipeline:
python tests/test_full_pipeline.py --dev_video_path /dev/video4 --can_port can0
```

## 🧪 Tests

| Test | What it does | Motors |
|---|---|---|
| `test_gripper.py` | Open/close gripper cycles | Enabled |
| `test_piper_state.py` | Read joint states from ring buffer | Disabled ✅ |
| `test_piper_move.py` | Move arm to home pose | Enabled ⚠️ |
| `test_full_pipeline.py` | Camera + controller + ZMQ + Placo viz + gripper | Disabled ✅ |

## 🎮 Inference

```bash
# Server (GPU machine)
python scripts/policy_server_zmq.py --pretrained_path ./checkpoints/smolvla --host 0.0.0.0 --port 8766

# Client (robot)
python scripts/eval_piper_remote.py --policy_host <server_ip> --policy_port 8766 --can_port can0
```

## 🤏 Gripper

DM4310 motor via DAMIAO DM-FDCAN USB-CAN serial bridge. Two modes:

- **`Gripper`** — synchronous serial (standalone testing)
- **`GripperProcess(mp.Process)`** — non-blocking shared memory (env integration)

```python
from modules.gripper import Gripper
with Gripper("/dev/ttyACM0") as g:
    g.set_zero()
    g.send_command(kp=10.0, kd=1.0, position=0.0)
```

## 🦾 Arm

Piper 6-DOF arm via CAN bus. Controller runs IK at 50 Hz in a background process:

```python
from modules.piper_env import PiperEnv
env = PiperEnv(shm_manager=shm, can_port="can0", dry_run=True)
with env:
    obs = env.get_obs()           # non-blocking ring buffer read
    env.exec_actions(pred_ee, dt)  # schedule waypoints + gripper
```

## 🔑 Key Design Decisions

- **All hardware runs as `mp.Process`** — no blocking I/O in the inference loop
- **Shared memory** — lock-free ring buffers for state, queues for commands
- **Fork** not spawn — pyrealsense2 / mp.Event compatibility
- **Gripper at env level** — separate hardware, not part of the arm controller
- **No LeRobot on the client** — own kinematics (Placo), own camera driver, own gripper driver

## 📄 License

MIT
