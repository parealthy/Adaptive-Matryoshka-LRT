#!/bin/bash
set -euo pipefail

# Model
SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
REASONING_NET_PATH="${REASONING_NET_PATH:-Qwen/Qwen3-Embedding-0.6B}"
LATENT_TRAJECTORY_LENGTHS="${LATENT_TRAJECTORY_LENGTHS:-64,128,192,256}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/ALR-Stage1-DSR1-Qwen-7B}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RUN_NAME="${RUN_NAME:-alr-stage1-length-elastic-7B}"

# Data
DATASET_NAME="${DATASET_NAME:-open-r1/OpenR1-Math-220k}"

# Training
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2.yaml}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-1024}"
COMPLETION_MAX_LENGTH="${COMPLETION_MAX_LENGTH:-2048}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
BF16="${BF16:-true}"
TF32="${TF32:-false}"
LOG_ACTIVE_LENGTH="${LOG_ACTIVE_LENGTH:-true}"
ACTIVE_LENGTH_LOG_INTERVAL="${ACTIVE_LENGTH_LOG_INTERVAL:-100}"

# Distributed
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
NUM_NODES="${NUM_NODES:-4}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23467}"
TOTAL_PROCESSES=$((NUM_GPUS_PER_NODE * NUM_NODES))

EXTRA_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

if [ "$NODE_RANK" -eq 0 ]; then
    mkdir -p "$OUTPUT_DIR"
fi

echo "------------------------------------------------"
echo "ALR Stage1 Length-Elastic Training Configuration:"
echo "  Model: $SLOW_THINKING_MODEL_PATH"
echo "  ReasoningNet: $REASONING_NET_PATH"
echo "  Latent lengths: $LATENT_TRAJECTORY_LENGTHS"
echo "  Dataset: $DATASET_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  NUM_GPUS_PER_NODE: $NUM_GPUS_PER_NODE"
echo "  NUM_NODES: $NUM_NODES"
echo "  NODE_RANK: $NODE_RANK"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "------------------------------------------------"

accelerate launch \
    --config_file "$DEEPSPEED_CONFIG" \
    --num_processes "$TOTAL_PROCESSES" \
    --num_machines "$NUM_NODES" \
    --machine_rank "$NODE_RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    stage1_sft.py \
    --slow_thinking_model_path "$SLOW_THINKING_MODEL_PATH" \
    --reasoning_net_path "$REASONING_NET_PATH" \
    --latent_trajectory_lengths "$LATENT_TRAJECTORY_LENGTHS" \
    --dataset_name "$DATASET_NAME" \
    --prompt_max_length "$PROMPT_MAX_LENGTH" \
    --completion_max_length "$COMPLETION_MAX_LENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --num_train_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --bf16 "$BF16" \
    --tf32 "$TF32" \
    --log_active_length "$LOG_ACTIVE_LENGTH" \
    --active_length_log_interval "$ACTIVE_LENGTH_LOG_INTERVAL" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$OUTPUT_DIR/train-stage1-node${NODE_RANK}-$(date +%Y%m%d-%H%M%S).log"
