#!/bin/bash
set -euo pipefail

SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-checkpoints/ALR-Stage1-DSR1-Qwen-1.5B}"
LATENT_TRAJECTORY_LENGTHS="${LATENT_TRAJECTORY_LENGTHS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/ALR-Stage2-Difficulty-DSR1-Qwen-1.5B}"
CACHE_DIR="${CACHE_DIR:-cache/alr_stage2_pooled_1.5B}"

SAMPLE_PER_CLASS="${SAMPLE_PER_CLASS:-7473}"
LIMIT="${LIMIT:-}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.05}"
SEED="${SEED:-42}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-1024}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-20}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MLP_HIDDEN_SIZE="${MLP_HIDDEN_SIZE:-1024}"
DROPOUT="${DROPOUT:-0.1}"
DEVICE="${DEVICE:-cuda:0}"
HEAD_DEVICE="${HEAD_DEVICE:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
OVERWRITE_CACHE="${OVERWRITE_CACHE:-false}"

EXTRA_ARGS=()
if [ -n "$LATENT_TRAJECTORY_LENGTHS" ]; then
    EXTRA_ARGS+=(--latent_trajectory_lengths "$LATENT_TRAJECTORY_LENGTHS")
fi
if [ -n "$LIMIT" ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi
if [ "$OVERWRITE_CACHE" = "true" ]; then
    EXTRA_ARGS+=(--overwrite_cache)
fi

echo "================================================"
echo "  ALR Stage2 Difficulty Estimator Training"
echo "  Model:  $SLOW_THINKING_MODEL_PATH"
echo "  Stage1: $STAGE1_CHECKPOINT_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Cache:  $CACHE_DIR"
echo "================================================"

python stage2_train.py \
    --model_path "$SLOW_THINKING_MODEL_PATH" \
    --stage1_checkpoint_path "$STAGE1_CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --sample_per_class "$SAMPLE_PER_CLASS" \
    --validation_ratio "$VALIDATION_RATIO" \
    --seed "$SEED" \
    --prompt_max_length "$PROMPT_MAX_LENGTH" \
    --cache_batch_size "$CACHE_BATCH_SIZE" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --mlp_hidden_size "$MLP_HIDDEN_SIZE" \
    --dropout "$DROPOUT" \
    --device "$DEVICE" \
    --head_device "$HEAD_DEVICE" \
    --torch_dtype "$TORCH_DTYPE" \
    "${EXTRA_ARGS[@]}"
