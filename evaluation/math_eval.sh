#!/bin/bash
#SBATCH --job-name=eval_math
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$REPO_ROOT/logs"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

BACKEND="${BACKEND:-lrt}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
REASONING_NET_PATH="${REASONING_NET_PATH:-Qwen/Qwen3-Embedding-0.6B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/checkpoints/DSR1-Qwen-1.5B-LRT-Math}"
MODEL_NAME="${MODEL_NAME:-lrt-math}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/eval/math}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval_outputs/math}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-1024}"
TASKS="${TASKS:-math500 gsm8k}"
EXTRA_ARGS=()

if [[ "${USE_CHAT_TEMPLATE:-false}" == "true" ]]; then
    EXTRA_ARGS+=(--use_chat_template)
fi

python "$SCRIPT_DIR/math_eval.py" \
    --backend "$BACKEND" \
    --model_path "$MODEL_PATH" \
    --reasoning_net_path "$REASONING_NET_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --model_name "$MODEL_NAME" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --tasks $TASKS \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --prompt_max_length "$PROMPT_MAX_LENGTH" \
    "${EXTRA_ARGS[@]}"
