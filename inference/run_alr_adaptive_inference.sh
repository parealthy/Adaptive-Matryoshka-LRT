#!/bin/bash
set -euo pipefail

SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
REASONING_NET_PATH="${REASONING_NET_PATH:-Qwen/Qwen3-Embedding-0.6B}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-checkpoints/ALR-Stage1-DSR1-Qwen-1.5B}"
DIFFICULTY_CHECKPOINT_PATH="${DIFFICULTY_CHECKPOINT_PATH:-checkpoints/ALR-Stage2-Difficulty-DSR1-Qwen-1.5B}"
LATENT_TRAJECTORY_LENGTHS="${LATENT_TRAJECTORY_LENGTHS:-}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DEVICE="${DEVICE:-cuda:0}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
QUESTION="${QUESTION:-}"

EXTRA_ARGS=()
if [ -n "$LATENT_TRAJECTORY_LENGTHS" ]; then
    EXTRA_ARGS+=(--latent_trajectory_lengths "$LATENT_TRAJECTORY_LENGTHS")
fi
if [ -n "$QUESTION" ]; then
    EXTRA_ARGS+=(--question "$QUESTION")
fi

echo "================================================"
echo "  ALR Adaptive Inference"
echo "  Model:        $SLOW_THINKING_MODEL_PATH"
echo "  ReasoningNet: $REASONING_NET_PATH"
echo "  Stage1:       $STAGE1_CHECKPOINT_PATH"
echo "  Stage2:       $DIFFICULTY_CHECKPOINT_PATH"
echo "================================================"

python inference/run_alr_adaptive_inference.py \
    --model_path "$SLOW_THINKING_MODEL_PATH" \
    --reasoning_net_path "$REASONING_NET_PATH" \
    --stage1_checkpoint_path "$STAGE1_CHECKPOINT_PATH" \
    --difficulty_checkpoint_path "$DIFFICULTY_CHECKPOINT_PATH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --prompt_max_length "$PROMPT_MAX_LENGTH" \
    --temperature "$TEMPERATURE" \
    --device "$DEVICE" \
    --torch_dtype "$TORCH_DTYPE" \
    "${EXTRA_ARGS[@]}"
