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
mkdir -p "$SCRIPT_DIR/logs"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-Distill-Qwen-1.5B}"
DATA_DIR="${DATA_DIR:-/ShorterBetter/eval_data/math}"
OUTPUT_DIR="${OUTPUT_DIR:-/ShorterBetter/eval_data/outputs/math}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TASKS="${TASKS:-all}"

python "$SCRIPT_DIR/math_eval.py" \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --tasks $TASKS \
    --batch_size "$BATCH_SIZE"
