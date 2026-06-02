#!/bin/bash
set -euo pipefail

# ============================================================
# ALR Stage1 Length-Elastic — Random-Length Interactive Inference
#
# Usage:
#   bash inference/run_alr_stage1_inference.sh
#
# Optional overrides:
#   CHECKPOINT_PATH=checkpoints/ALR-Stage1-DSR1-Qwen-1.5B \
#   LATENT_TRAJECTORY_LENGTHS=64,128,192,256 \
#     bash inference/run_alr_stage1_inference.sh
# ============================================================

SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
REASONING_NET_PATH="${REASONING_NET_PATH:-Qwen/Qwen3-Embedding-0.6B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/ALR-Stage1-DSR1-Qwen-1.5B}"
LATENT_TRAJECTORY_LENGTHS="${LATENT_TRAJECTORY_LENGTHS:-}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-}"
QUESTION="${QUESTION:-}"
FIXED_LATENT_TRAJECTORY_LENGTH="${FIXED_LATENT_TRAJECTORY_LENGTH:-}"
ALR_USE_SYSTEM_PROMPT="${ALR_USE_SYSTEM_PROMPT:-0}"
ALR_SYSTEM_PROMPT="${ALR_SYSTEM_PROMPT:-}"

EXTRA_ARGS=()
if [ -n "$LATENT_TRAJECTORY_LENGTHS" ]; then
    EXTRA_ARGS+=(--latent_trajectory_lengths "$LATENT_TRAJECTORY_LENGTHS")
fi
if [ -n "$SEED" ]; then
    EXTRA_ARGS+=(--seed "$SEED")
fi
if [ -n "$QUESTION" ]; then
    EXTRA_ARGS+=(--question "$QUESTION")
fi
if [ -n "$FIXED_LATENT_TRAJECTORY_LENGTH" ]; then
    EXTRA_ARGS+=(--fixed_latent_trajectory_length "$FIXED_LATENT_TRAJECTORY_LENGTH")
fi

echo "================================================"
echo "  ALR Stage1 Random-Length Inference"
echo "  Model:        $SLOW_THINKING_MODEL_PATH"
echo "  ReasoningNet: $REASONING_NET_PATH"
echo "  Checkpoint:   $CHECKPOINT_PATH"
if [ -n "$LATENT_TRAJECTORY_LENGTHS" ]; then
    echo "  Lengths:      $LATENT_TRAJECTORY_LENGTHS"
else
    echo "  Lengths:      from checkpoint config or default"
fi
if [ -n "$FIXED_LATENT_TRAJECTORY_LENGTH" ]; then
    echo "  Fixed length: $FIXED_LATENT_TRAJECTORY_LENGTH"
fi
if [[ "$ALR_USE_SYSTEM_PROMPT" =~ ^(1|true|TRUE|yes|YES|y|Y|on|ON)$ ]]; then
    echo "  SystemPrompt: enabled"
else
    echo "  SystemPrompt: disabled"
fi
echo "================================================"

python inference/run_alr_stage1_inference.py \
    --model_path "$SLOW_THINKING_MODEL_PATH" \
    --reasoning_net_path "$REASONING_NET_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --prompt_max_length "$PROMPT_MAX_LENGTH" \
    --temperature "$TEMPERATURE" \
    --device "$DEVICE" \
    "${EXTRA_ARGS[@]}"
