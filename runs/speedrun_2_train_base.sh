#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 1.5 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~1.5 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/workspace/.cache/nanochat"  # /workspaceに外部ストレージをマウントしてデータを保存
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install the repo dependencies
uv sync --extra gpu --cache-dir /tmp/uv-cache  # /workspaceにuvキャッシュが保存されてエラーが出ないように修正
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Base model (pretraining)
DEPTH=3
WANDB_RUN="d${DEPTH}"

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=$DEPTH --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --device-batch-size=16

