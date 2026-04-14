# MPO Experiments – Robust Ant Quickstart

This README explains **only** how to start the **Robust Ant experiments**.
The **maze part is intentionally ignored for now**, because it is still under development.

## Goal

The goal is to make it easy to run the project on a fresh machine.
For that reason, this repository also includes:

- `requirements-robust.txt` – Python packages needed to run the Robust Ant experiments

## Recommended Python Version

So the safest choice is to use **Python 3.10**, especially to avoid friction with MuJoCo, Gymnasium, and the existing code.

## Quick Setup on a New Machine

### 1) Clone or unpack the repository

Then move into the project directory:

```bash
cd mpo_experiments-mpo_multitask
```

### 2) Create a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 3) Install Python packages

```bash
pip install -r requirements-robust-ant.txt
```

## Optional system packages (Linux)

Depending on the machine, you may need additional system packages for MuJoCo rendering or video recording.
Typical Ubuntu/Debian packages are:

```bash
sudo apt update
sudo apt install -y ffmpeg libgl1 libosmesa6 libglfw3
```

Notes:

- `ffmpeg` is useful for video logging.
- If you start **without video**, you often do not need this immediately.
- MuJoCo is used through the Python package `mujoco`, and Gymnasium's MuJoCo environments rely on it.

## Start training

Important: run the script **from the project root**, meaning the directory that contains `main_mpo.py`.

### Default Robust Ant run

```bash
python main_mpo.py robust_ant
```

This is the main entry point.
`main_mpo.py` uses Tyro subcommands, and `robust_ant` is the relevant subcommand for the Ant experiments.

## Recommended first smoke test

For a quick first test on a new machine, start with a small run:

```bash
WANDB_MODE=offline python main_mpo.py robust_ant \
  --device cpu \
  --num-envs 1 \
  --max-training-steps 10000 \
  --sample-steps-per-iter 256 \
  --warm-up-steps 1000 \
  --no-capture-video
```

Why this setup?

- `WANDB_MODE=offline` keeps Weights & Biases local and avoids login/network issues.
- `--device cpu` is usually the most robust first test on a fresh machine.
- `--num-envs 1` reduces complexity.
- `--no-capture-video` avoids extra rendering/video dependencies on the first run.
- `--max-training-steps 10000` is only meant as a smoke test, not as a real training run.

## Typical GPU run

If CUDA is available and working:

```bash
WANDB_MODE=offline python main_mpo.py robust_ant \
  --device cuda \
  --num-envs 4
```

## Useful variants

### Inverted task without task hint

```bash
python main_mpo.py robust_ant --task-mode inverted_without_task_hint
```

### Target-goal task

```bash
python main_mpo.py robust_ant --task-mode target_goal
```

### Domain randomization / robustness modes

```bash
python main_mpo.py robust_ant --rand-mode RFI
python main_mpo.py robust_ant --rand-mode RAO
```

## Important directories

After starting a run, these folders are relevant:

- `checkpoints/` – saved models
- `runs/` – TensorBoard logs
- `videos/` – recorded episodes
- `mpo_logs/` – additional logs

## View TensorBoard

```bash
tensorboard --logdir runs
```

Then open the local URL shown in the terminal.

## Common issues

### 1) Python version

Use **Python 3.10** whenever possible.
Newer Python versions can make RL and MuJoCo setups unnecessarily fragile.

### 2) MuJoCo / Gymnasium

`Ant-v5` is a Gymnasium MuJoCo environment, so MuJoCo support must be installed correctly.

### 3) W&B

The current code directly calls `wandb.init(...)`.
On a fresh machine, `WANDB_MODE=offline` is usually the easiest option.

### 4) Video / rendering

If video logging causes problems, start with `--no-capture-video`.

### 5) Always start from the project root

The command should be run from the root of the repository:

```bash
python main_mpo.py robust_ant
```

## Minimal quickstart checklist

If you just want the fastest possible way to start:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-robust-ant.txt
WANDB_MODE=offline python main_mpo.py robust_ant --device cpu --num-envs 1 --max-training-steps 10000 --sample-steps-per-iter 256 --warm-up-steps 1000 --no-capture-video
```

## What this README intentionally does not cover

- Ant Maze
- benchmark comparisons
- hyperparameter tuning
- cluster/Slurm setup
- evaluation of old checkpoints

This README is intentionally written as a **quickstart for Robust Ant only**.
