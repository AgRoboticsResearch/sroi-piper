#!/usr/bin/env python3
"""ZMQ inference server wrapping a LeRobot SmolVLA policy.

Listens on ZMQ REP socket, receives raw observation dicts (images as uint8
numpy arrays, state as float32), runs inference, returns action chunks.

Matches UMI-ARX's PolicyInferenceNode pattern: load model once, then
recv_pyobj → predict → send_pyobj in a tight loop.

Usage:
  python scripts/policy_server_zmq.py \
      --pretrained_path outputs/smolvla_umi_strawberry_50k/checkpoints/050000/pretrained_model \
      --host 0.0.0.0 --port 8766
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import zmq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline loading (same as eval_piper.py)
# ---------------------------------------------------------------------------

def load_smolvla_pipeline(pretrained_path: str, device: str = "cuda",
                          dataset_root: str | None = None):
    import os

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    pretrained_path = str(pretrained_path)
    config_path = Path(pretrained_path) / "train_config.json"
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        policy_config = train_config.get("policy", {})
        ds_cfg = train_config.get("dataset", {})
    else:
        policy_config = {}
        ds_cfg = {}

    ds_repo_id = ds_cfg.get("repo_id", "sroi_piper_strawberry_picking")
    if dataset_root:
        ds_root = dataset_root
    else:
        ds_root = ds_cfg.get("root", str(Path(__file__).resolve().parents[1] / "Datasets" / ds_repo_id))

    logger.info("Loading stats from: %s at %s", ds_repo_id, ds_root)

    os.environ["HF_HUB_OFFLINE"] = "1"
    ds_meta = LeRobotDatasetMetadata(ds_repo_id, root=ds_root)

    cfg = SmolVLAConfig(
        derive_state_from_action=policy_config.get("derive_state_from_action", True),
        use_relative_actions=policy_config.get("use_relative_actions", True),
        relative_exclude_joints=policy_config.get("relative_exclude_joints", ["gripper"]),
        relative_exclude_state_joints=policy_config.get("relative_exclude_state_joints", ["gripper"]),
        device=device,
        resize_imgs_with_padding=tuple(policy_config.get("resize_imgs_with_padding", (512, 512))),
        freeze_vision_encoder=policy_config.get("freeze_vision_encoder", True),
        train_expert_only=policy_config.get("train_expert_only", True),
        train_state_proj=policy_config.get("train_state_proj", True),
        load_vlm_weights=False,
        push_to_hub=False,
        pretrained_path=pretrained_path,
    )

    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=pretrained_path, dataset_stats=ds_meta.stats,
    )

    task_prompt = None
    if hasattr(ds_meta, "tasks") and len(ds_meta.tasks) > 0:
        task_prompt = ds_meta.tasks.index[0]
    if not task_prompt:
        task_prompt = "perform the task"
    logger.info("Task prompt: '%s'", task_prompt)

    return policy, preprocessor, postprocessor, task_prompt


# ---------------------------------------------------------------------------
# Observation → batch conversion
# ---------------------------------------------------------------------------

def build_batch(raw_obs: dict, device: str, task_prompt: str) -> dict:
    """Convert raw numpy observation dict to batch dict for preprocessor."""
    import torch

    state = raw_obs["observation.state"]  # (2, 7) float32
    task = raw_obs.get("task", task_prompt)

    batch = {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "task": [task],
    }

    for key, img in raw_obs.items():
        if not key.startswith("observation.images."):
            continue
        img_float = img.astype(np.float32) / 255.0          # uint8 → [0,1]
        img_chw = np.transpose(img_float, (2, 0, 1))        # HWC → CHW
        batch[key] = (
            torch.from_numpy(img_chw).unsqueeze(0).unsqueeze(0).to(device).float()
        )  # (1, 1, C, H, W)

    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ZMQ SmolVLA inference server")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to pretrained SmolVLA checkpoint")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Dataset root for normalization stats")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # ── 1. Load pipeline ─────────────────────────────────────────────
    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info("Loading SmolVLA pipeline from %s", args.pretrained_path)
    policy, preprocessor, postprocessor, task_prompt = load_smolvla_pipeline(
        args.pretrained_path, device, dataset_root=args.dataset_root,
    )
    logger.info("Model loaded on %s", device)

    # ── 2. Bind ZMQ REP socket ───────────────────────────────────────
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    bind_addr = f"tcp://{args.host}:{args.port}"
    socket.bind(bind_addr)
    logger.info("ZMQ REP socket bound to %s", bind_addr)
    logger.info("Ready for requests")

    # ── 3. Serve loop ────────────────────────────────────────────────
    request_count = 0
    try:
        while True:
            try:
                raw_obs = socket.recv_pyobj()
                t0 = time.perf_counter()

                batch = build_batch(raw_obs, device, task_prompt)

                with torch.inference_mode():
                    processed = preprocessor(batch)
                    pred_actions = policy.predict_action_chunk(processed)
                    pred_abs = postprocessor(pred_actions)

                pred_ee = pred_abs[0]
                if hasattr(pred_ee, "cpu"):
                    action = pred_ee.cpu().numpy()
                elif hasattr(pred_ee, "numpy"):
                    action = pred_ee.numpy()
                else:
                    action = np.asarray(pred_ee)

                socket.send_pyobj(action)
                dt = (time.perf_counter() - t0) * 1000
                request_count += 1
                logger.info(
                    "request=%d dt=%.1fms shape=%s", request_count, dt, action.shape,
                )

            except Exception as e:
                logger.exception("Inference failed")
                socket.send_pyobj(f"ERROR: {e}")

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        socket.close()
        context.term()
        logger.info("Server stopped")


if __name__ == "__main__":
    main()
