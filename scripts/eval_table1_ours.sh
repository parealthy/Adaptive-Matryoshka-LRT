#!/bin/bash
set -euo pipefail

# ============================================================
# Table 1 - Ours-only evaluation
#
# Runs the latent-reasoning method on MATH-500 and GSM8K.
#
# Usage:
#   CHECKPOINT_PATH=checkpoints/DSR1-Qwen-1.5B-LRT-Math \
#     bash scripts/eval_table1_ours.sh
#
# Optional overrides:
#   SLOW_THINKING_MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
#   REASONING_NET_PATH=Qwen/Qwen3-Embedding-0.6B
#   BATCH_SIZE=4
#   MAX_NEW_TOKENS=4096
#   DEVICE=cuda:0
# ============================================================

SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
REASONING_NET_PATH="${REASONING_NET_PATH:-Qwen/Qwen3-Embedding-0.6B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/DSR1-Qwen-1.5B-LRT-Math}"

BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-1024}"
LATENT_TRAJECTORY_LENGTH="${LATENT_TRAJECTORY_LENGTH:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/table1_ours}"

python evaluation/evaluate_table1_ours.py \
    --model_path "$SLOW_THINKING_MODEL_PATH" \
    --reasoning_net_path "$REASONING_NET_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --datasets math500 gsm8k \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --prompt_max_length "$PROMPT_MAX_LENGTH" \
    --latent_trajectory_length "$LATENT_TRAJECTORY_LENGTH" \
    --temperature "$TEMPERATURE" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite
