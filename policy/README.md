# Policy

This folder contains trained policy models and the training/evaluation codebase.

## Structure

```
policy/
├── lerobot/   # LeRobot fork with SmolVLA UMI training & eval (not tracked in git)
└── README.md  # This file
```

## Setting up LeRobot

The `lerobot/` directory is a customized fork of [LeRobot](https://github.com/huggingface/lerobot) with SmolVLA UMI support. It is **not tracked in git** due to its size (~13 GB with datasets and outputs).

To set up:

```bash
cd policy/lerobot
pip install -e .
```

See `lerobot/SMOLVLA_UMI_EE_TRAINING.md` for end-effector pose training and `lerobot/UMI_EE_POSE_PIPELINE.md` for the full data pipeline.

## Training

```bash
cd policy/lerobot

# SmolVLA UMI training
python train_smolvla_umi.py

# End-effector pose variant
python train_smolvla_umi_ee.py

# State-based variant
python train_smolvla_umi_state.py
```

## Evaluation

```bash
# Eval with trained checkpoint
python eval_smolvla_umi.py --policy-path <checkpoint_dir>

# Remote eval (connects to robot)
python eval_smolvla_umi_strawberry.py
```
