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
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$REPO_ROOT/logs"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

MODEL_NAME="${MODEL_NAME:-lrt-math}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/eval_outputs/math/$MODEL_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATASET_DIR/verified}"
VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen2.5-7B}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TASKS="${TASKS:-math500 gsm8k}"

python "$SCRIPT_DIR/verifier.py" \
    --model "$VERIFIER_MODEL" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --tasks $TASKS \
    --batch_size "$BATCH_SIZE"
