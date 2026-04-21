# MPO Experiments – Quickstart

This repository contains the current MPO experiment setup.
The README is intentionally minimal and only covers what is needed to get the project running.
For more information on implementation details and results, please see here: https://grittojakob.github.io/jakobs_experiments_blog/

## Requirements

Recommended:
- Python 3.10
- Linux machine
- optional: CUDA-capable GPU

## Setup

Clone the repository and enter the project root:

```bash
git clone <repo-url>
cd mpo_experiments-main
```

Create and activate a conda environment:

```bash
conda create -n mpo_env python=3.10
conda activate mpo_env
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Optional system packages (Linux)

If MuJoCo rendering or video logging causes issues, install:

```bash
sudo apt update
sudo apt install -y ffmpeg libgl1 libosmesa6 libglfw3
```

## Run training

Always run commands from the project root (the directory containing `main_mpo.py`).

Default run:

```bash
python main_mpo.py
```

## Recommended smoke test

For a quick first check on a new machine:

```bash
WANDB_MODE=offline python main_mpo.py \
  --device cpu \
  --num-envs 1 \
  --max-training-steps 10000 \
  --sample-steps-per-iter 256 \
  --warm-up-steps 1000 \
  --no-capture-video
```

This keeps the setup simple:
- CPU only
- one environment
- no online W\&B dependency
- no video requirement

## GPU run

If CUDA is available:

```bash
WANDB_MODE=offline python main_mpo.py \
  --device cuda \
  --num-envs 4
```

## Outputs

Relevant directories created during training:
- `checkpoints/` – saved models
- `runs/` – TensorBoard logs
- `videos/` – recorded episodes
- `mpo_logs/` – additional logs

## TensorBoard

```bash
tensorboard --logdir runs
```

## Notes

- Use Python 3.10 if possible.
- If W\&B causes issues, use `WANDB_MODE=offline`.
- If rendering/video causes issues, use `--no-capture-video`.
