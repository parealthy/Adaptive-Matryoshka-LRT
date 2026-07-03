#!/bin/bash
set -euo pipefail

SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
REASONING_NET_PATH="${REASONING_NET_PATH:-Qwen/Qwen3-Embedding-0.6B}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-checkpoints/ALR-Stage1-DSR1-Qwen-1.5B}"
DIFFICULTY_CHECKPOINT_PATH="${DIFFICULTY_CHECKPOINT_PATH:-checkpoints/ALR-Stage2-Difficulty-DSR1-Qwen-1.5B}"
HARNESS_PATH="${HARNESS_PATH:-/Users/focus/Desktop/paper/s1/eval/lm-evaluation-harness}"
LATENT_TRAJECTORY_LENGTHS="${LATENT_TRAJECTORY_LENGTHS:-64,128,192,256}"

OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/alr_lm_eval_1.5B}"
TASKS="${TASKS:-gsm8k math500}"
METHODS="${METHODS:-fixed-64 fixed-128 fixed-192 fixed-256 random adaptive}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
DEVICE="${DEVICE:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
SEED="${SEED:-42}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29513}"
OVERWRITE="${OVERWRITE:-true}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-false}"

export ALR_LM_EVAL_DIST_BACKEND="${ALR_LM_EVAL_DIST_BACKEND:-gloo}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

read -r -a TASK_ARGS <<< "$TASKS"
read -r -a METHOD_ARGS <<< "$METHODS"

EXTRA_ARGS=()
if [ -n "$LIMIT" ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi
if [ "$OVERWRITE" = "true" ]; then
    EXTRA_ARGS+=(--overwrite)
fi
if [ "$APPLY_CHAT_TEMPLATE" = "true" ]; then
    EXTRA_ARGS+=(--apply_chat_template)
fi

echo "================================================"
echo "  ALR lm-evaluation-harness Evaluation"
echo "  Harness: $HARNESS_PATH"
echo "  Stage1:  $STAGE1_CHECKPOINT_PATH"
echo "  Stage2:  $DIFFICULTY_CHECKPOINT_PATH"
echo "  Tasks:   $TASKS"
echo "  Methods: $METHODS"
echo "  Output:  $OUTPUT_DIR"
echo "  GPUs:    $NUM_GPUS"
echo "================================================"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MASTER_PORT" \
    evaluation/run_lm_eval_alr.py \
    --harness_path "$HARNESS_PATH" \
    --model_path "$SLOW_THINKING_MODEL_PATH" \
    --reasoning_net_path "$REASONING_NET_PATH" \
    --stage1_checkpoint_path "$STAGE1_CHECKPOINT_PATH" \
    --difficulty_checkpoint_path "$DIFFICULTY_CHECKPOINT_PATH" \
    --latent_trajectory_lengths "$LATENT_TRAJECTORY_LENGTHS" \
    --output_dir "$OUTPUT_DIR" \
    --tasks "${TASK_ARGS[@]}" \
    --methods "${METHOD_ARGS[@]}" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --prompt_max_length "$PROMPT_MAX_LENGTH" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --device "$DEVICE" \
    --torch_dtype "$TORCH_DTYPE" \
    --seed "$SEED" \
    "${EXTRA_ARGS[@]}"
