#!/bin/bash
#SBATCH --job-name=llm_verify
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-Distill-Qwen-1.5B}"
DATASET_DIR="${DATASET_DIR:-/ShorterBetter/eval_data/outputs/math/$MODEL_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATASET_DIR/verified}"
VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen2.5-7B}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TASKS="${TASKS:-all}"

python "$SCRIPT_DIR/verifier.py" \
    --model "$VERIFIER_MODEL" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --tasks $TASKS \
    --batch_size "$BATCH_SIZE"
