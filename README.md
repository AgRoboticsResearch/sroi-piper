# sroi-piper

SROI Piper robot control stack — UMI-ARX architecture, zero LeRobot dependency on the client.

```
Camera (mp.Process)  -->  Ring Buffer
Arm (mp.Process)     -->  Ring Buffer  -->  ZMQ Inference  -->  Placo Meshcat Viz
Gripper (mp.Process) -->  Ring Buffer
```

## Quick Start

```bash
conda activate lerobot_piper_sroi

# Gripper — calibrate, then test
python scripts/calibrate_gripper.py
python scripts/test_gripper_slow_cycle.py

# Gripper — read state (no movement, safe)
python scripts/read_gripper_state.py

# Arm — go home then zero (motors enabled, holds until Ctrl-C)
python scripts/piper_go_home_zero.py --can_port can0
```

## Arm Scripts

| Script | What it does | Motors |
|---|---|---|
| `scripts/piper_go_home_zero.py` | Home pose -> zero pose, then hold | Enabled |

## Gripper Scripts

| Script | What it does | Moves? |
|---|---|---|
| `scripts/calibrate_gripper.py` | Interactive calibration: set closed/open limits | No (kp=0) |
| `scripts/read_gripper_state.py` | Live state monitor (pos, vel, torque, temp) | No (kp=0) |
| `scripts/test_gripper_slow_cycle.py` | Slow open/close sweep with calibrated range | Yes |
| `scripts/test_gripper_slow_cycle_mp.py` | Same but via GripperProcess (mp.Process) | Yes |

### Calibration workflow

```bash
# 1. Calibrate
python scripts/calibrate_gripper.py

# 2. Update defaults with printed values:
#    src/modules/piper_env.py  -> gripper_closed_rad, gripper_open_rad
#    scripts/test_gripper_slow_cycle.py -> CLOSED_RAD, OPEN_RAD

# 3. Test the calibrated range
python scripts/test_gripper_slow_cycle.py
python scripts/test_gripper_slow_cycle.py --speed 0.15  # faster
```

## Tests

```bash
# Unit tests (no hardware)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_gripper_unit.py -v -p anyio
```

## Architecture

```
src/
  modules/
    piper_env.py          Orchestrator (cameras + arm + gripper)
    piper_controller.py   Arm control loop (mp.Process, IK @ 50Hz)
    piper_interface.py    CAN bus driver (piper_sdk wrapper)
    gripper.py            DM4310 gripper + GripperProcess (mp.Process)
    rs_camera.py          RealSense D405 camera (mp.Process)
  shared_memory/
    ring_buffer.py        Lock-free ring buffer (state -> main)
    queue.py              Lock-free queue (commands -> worker)
  utils/
    kinematics.py         FK/IK via Placo
    piper_urdf/           URDF model + gripper mesh
```

## Key Design Decisions

- All hardware runs as `mp.Process` — no blocking I/O in the inference loop
- Shared memory — lock-free ring buffers for state, queues for commands
- Gripper at env level — separate hardware, not part of the arm controller
- No LeRobot on the client — own kinematics, own camera driver, own gripper driver

## License

MIT
