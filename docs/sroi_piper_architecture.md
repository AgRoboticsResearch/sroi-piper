# SROI Piper Architecture: UMI-ARX Style Refactoring

## Overview

Refactor the Piper robot deployment to match UMI-ARX's architecture.
Key principle: each major component is an `mp.Process` communicating via
lock-free shared memory. No LeRobot dependency at runtime — we wrap
`piper_sdk` (CAN bus) directly.

## Structure

```
sroi_piper/
├── setup.py
├── pyproject.toml
│
├── src/
│   └── sroi_piper/
│       ├── __init__.py
│       │
│       ├── shared_memory/              # IPC primitives (1:1 port from UMI)
│       │   ├── __init__.py
│       │   ├── ndarray.py              # SharedNDArray
│       │   ├── queue.py                # SharedMemoryQueue (FIFO, commands)
│       │   ├── ring_buffer.py          # SharedMemoryRingBuffer (FILO, state)
│       │   └── util.py                 # ArraySpec, SharedAtomicCounter
│       │
│       ├── modules/                    # process-based components
│       │   ├── __init__.py
│       │   ├── piper_interface.py      # CAN bus wrapper (≡ Arx5Client)
│       │   ├── piper_controller.py     # mp.Process + IK (≡ Arx5Controller)
│       │   ├── piper_env.py            # orchestrator (≡ Arx5Env) — TO BUILD
│       │   └── pose_trajectory_interpolator.py
│       │
│       ├── inference/                  # policy inference pipeline
│       │   ├── __init__.py
│       │   ├── inference_thread.py     # background inference loop
│       │   ├── observation_synchronizer.py
│       │   └── action_processor.py     # relative→absolute pose conversion
│       │
│       └── utils/
│           ├── __init__.py
│           ├── precise_wait.py
│           └── time_utils.py
│
├── scripts/
│   └── run_autonomous.py              # entry point (≡ eval_arx5.py)
│
└── tests/
    ├── test_shared_memory.py
    ├── test_controller.py
    └── test_inference.py
```

## File-by-file mapping to UMI-ARX

### shared_memory/ — identical to UMI

| UMI-ARX                    | sroi_piper              | Notes                   |
|----------------------------|-------------------------|-------------------------|
| shared_ndarray.py          | ndarray.py              | SharedNDArray wrapper   |
| shared_memory_queue.py     | queue.py                | FIFO command queue      |
| shared_memory_ring_buffer.py | ring_buffer.py        | FILO state buffer       |
| shared_memory_util.py      | util.py                 | ArraySpec, atomic ctr   |

### modules/ — direct match

| UMI-ARX                    | sroi_piper              | Notes                   |
|----------------------------|-------------------------|-------------------------|
| arx5_zmq_client.py         | piper_interface.py      | HW client. ZMQ → CAN    |
| arx5_controller.py         | piper_controller.py     | mp.Process control loop |
| arx5_env.py                | piper_env.py            | Orchestrator — TO BUILD |
| pose_trajectory_interpolator.py | pose_trajectory_interpolator.py | 1:1 port   |
| replay_buffer.py           | (later)                 | Zarr recording          |
| timestamp_accumulator.py   | (later)                 | Obs/action accumulation |

### inference/ — Piper-unique

| File                       | Why Piper needs it                                    |
|----------------------------|-------------------------------------------------------|
| inference_thread.py        | UMI runs inference in main thread; SmolVLA is slow so we thread it |
| observation_synchronizer.py| UMI has camera process; we read cameras synchronously |
| action_processor.py        | SmolVLA predicts relative EE deltas → convert to absolute world frame |

### utils/

| UMI-ARX                    | sroi_piper              | Notes                   |
|----------------------------|-------------------------|-------------------------|
| other_util.py (precise_wait)| precise_wait.py        | Busy-wait for RT loops  |
| —                          | time_utils.py           | wall↔monotonic conversion |

## Piper-unique design decisions

### 1. piper_interface.py (replaces arx5_zmq_client.py)

ARX robot: external ZMQ server accepts EE pose commands, does IK server-side.
Piper robot: CAN bus, only accepts joint angle commands. No server-side IK.

So `piper_interface.py` wraps `piper_sdk.C_PiperInterface_V2`:
- `connect()` / `disconnect()` — CAN bus + EnablePiper
- `read_joints() → np.ndarray(6)` — joint angles in degrees
- `write_joints(joints: np.ndarray)` — send joint targets
- `read_gripper() → float` — gripper state
- `write_gripper(position: float)` — gripper command

### 2. piper_controller.py (adds IK layer)

UMI's `Arx5Controller.run()` loop:
```
interpolate(t) → EE pose → send to robot (robot does IK)
```

Piper's `PiperController.run()` loop:
```
interpolate(t) → EE pose → IK(EE, seed_joints) → joints → send to robot
```

IK is done inside the controller subprocess using Placo.
This is the only structural addition vs UMI.

### 3. piper_env.py (orchestrator — TO BUILD)

Mirrors `Arx5Env`:
- Creates `SharedMemoryManager`
- Spawns `PiperController` (mp.Process)
- Provides `get_obs()` → reads state from controller ring_buffer
- Provides `exec_actions(actions, timestamps)` → schedule_waypoint
- Context manager: `with PiperEnv(...) as env:`
- Later: camera integration, recording, teleop

## Dependency graph

```
scripts/run_autonomous.py
        │
        ▼
  modules/piper_env.py              ← orchestrator
   │             │
   ▼             ▼
modules/       inference/
piper_controller.py  inference_thread.py
   │             │
   ▼             ▼
modules/      shared_memory/        ← both use IPC primitives
piper_interface.py
   │
   ▼
 utils/
```

- `shared_memory/` and `utils/` have zero internal imports
- Nothing imports upward
- `piper_interface.py` only depends on `piper_sdk` (third party)

## Lifecycle (matches UMI)

```python
shm_manager = SharedMemoryManager()
shm_manager.start()

controller = PiperController(shm_manager=shm_manager, ...)
controller.start(wait=True)      # spawns process, blocks until ready

# ... send commands via controller.schedule_waypoint(), exec_actions() ...
# ... read state via controller.get_state(), get_all_state() ...

controller.stop(wait=True)       # sends STOP, joins process
shm_manager.shutdown()
```

## What's deferred (not needed now)

- Camera as separate mp.Process (synchronous reads are fine)
- ReplayBuffer / Zarr recording
- Teleoperation (spacemouse/keyboard demos)
- VideoRecorder (GPU H.264 encoding)
- Multi-camera visualizer
- Timestamp alignment / multi-sensor interpolation
