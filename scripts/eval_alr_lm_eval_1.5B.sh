#!/bin/bash
set -euo pipefail

SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
REASONING_NET_PATH="${REASONING_NET_PATH:-Qwen/Qwen3-Embedding-0.6B}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-checkpoints/ALR-Stage1-DSR1-Qwen-1.5B}"
DIFFICULTY_CHECKPOINT_PATH="${DIFFICULTY_CHECKPOINT_PATH:-checkpoints/ALR-Stage2-Difficulty-DSR1-Qwen-1.5B}"
HARNESS_PATH="${HARNESS_PATH:-/mnt/pami23/dzhu/s1/eval/lm-evaluation-harness}"
LATENT_TRAJECTORY_LENGTHS="${LATENT_TRAJECTORY_LENGTHS:-64,128,192,256}"

if [ -z "${METHODS+x}" ] && [ -n "${METHODs:-}" ]; then
    echo "Warning: detected METHODs; using it as METHODS."
    METHODS="$METHODs"
fi

OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/alr_lm_eval_1.5B}"
TASKS="${TASKS:-gsm8k math500}"
METHODS="${METHODS:-fixed-64 fixed-128 fixed-192 fixed-256 random adaptive}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
DEVICE="${DEVICE:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
SEED="${SEED:-42}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29513}"
OVERWRITE="${OVERWRITE:-true}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-false}"
USE_TRAINING_PROMPT_TEMPLATE="${USE_TRAINING_PROMPT_TEMPLATE:-true}"

export ALR_LM_EVAL_DIST_BACKEND="${ALR_LM_EVAL_DIST_BACKEND:-gloo}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

TASKS="${TASKS//math-500/math500}"

python - "$HARNESS_PATH" <<'PY'
import importlib.util
import sys

harness_path = sys.argv[1]
missing = [
    package
    for package in ("sacrebleu", "numpy", "datasets", "transformers", "accelerate")
    if importlib.util.find_spec(package) is None
]
if missing:
    print("Missing Python packages required by lm-evaluation-harness: " + ", ".join(missing))
    print("Install them in the active environment, for example:")
    print(f"  pip install -e {harness_path}")
    print("or at minimum:")
    print("  pip install " + " ".join(missing))
    sys.exit(1)

import transformers
if not hasattr(transformers, "AutoModelForVision2Seq"):
    print(
        "Warning: transformers has no AutoModelForVision2Seq; "
        "run_lm_eval_alr.py will apply a text-only compatibility shim."
    )
    print("If this error still appears, sync evaluation/lm_eval_alr_model.py too.")
PY

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
if [ "$USE_TRAINING_PROMPT_TEMPLATE" != "true" ]; then
    EXTRA_ARGS+=(--no_training_prompt_template)
fi
EXTRA_ARGS+=(--num_fewshot "$NUM_FEWSHOT")

echo "================================================"
echo "  ALR lm-evaluation-harness Evaluation"
echo "  Harness: $HARNESS_PATH"
echo "  Stage1:  $STAGE1_CHECKPOINT_PATH"
echo "  Stage2:  $DIFFICULTY_CHECKPOINT_PATH"
echo "  Tasks:   $TASKS"
echo "  Methods: $METHODS"
echo "  Fewshot: $NUM_FEWSHOT"
echo "  Training prompt template: $USE_TRAINING_PROMPT_TEMPLATE"
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
