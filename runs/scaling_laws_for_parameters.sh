#!/bin/bash
# 完了済み d12 と総学習トークン数をそろえ、d3・6・9・15 を事前学習する。
# d12 のログで確認した学習設定を直接指定する。
# 実行方法: bash runs/speedrun_2_train_base_scaling.sh
# トークナイザとデータファイル一覧は d12 の学習時と同じものを使用する。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/workspace/.cache/nanochat}"
export PYTHONUNBUFFERED=1

# 2026-09-22 の d12 ログに合わせる。
# 総学習トークン数は 1,680 × 524,288 = 880,803,840。
NUM_ITERATIONS=1680
TOTAL_BATCH_SIZE=524288
MAX_SEQ_LEN=2048
WINDOW_PATTERN="SSSL"

DEPTHS=(3 6 9 15)
DEVICE_BATCH_SIZE=16
NPROC_PER_NODE=1
EVAL_TOKENS=41943040  # base_train の既定値と同じ（80 × 524288）。

uv sync --extra gpu --cache-dir /tmp/uv-cache
source .venv/bin/activate

# チェックポイントは標準の base_checkpoints/d{DEPTH} に保存する。
for DEPTH in "${DEPTHS[@]}"; do
    # WANDB_RUN=scaling などを指定すると、各 depth の結果を Wandb に記録する。
    RUN=dummy
    if [[ "${WANDB_RUN:-dummy}" != dummy ]]; then
        RUN="${WANDB_RUN}_d${DEPTH}"
    fi
    echo "Training d${DEPTH}: $((NUM_ITERATIONS * TOTAL_BATCH_SIZE)) tokens"
    # ステップ数と総バッチを明示して学習トークン数を固定する。
    # 比率は depth に応じた既存の最適化補正にも使うため、正の値を維持する。
    # 検証は開始時と最終ステップのみ。途中の検証・CORE 評価・文章生成は行わない。
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
        --depth="$DEPTH" --aspect-ratio=64 --head-dim=128 \
        --max-seq-len="$MAX_SEQ_LEN" --window-pattern="$WINDOW_PATTERN" \
        --num-iterations="$NUM_ITERATIONS" --total-batch-size="$TOTAL_BATCH_SIZE" \
        --target-param-data-ratio=8 --device-batch-size="$DEVICE_BATCH_SIZE" \
        --fp8 --run="$RUN" \
        --eval-every="$NUM_ITERATIONS" --eval-tokens="$EVAL_TOKENS" \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1

done

echo "学習完了。保存先: $NANOCHAT_BASE_DIR/base_checkpoints/d{3,6,9,15}"
