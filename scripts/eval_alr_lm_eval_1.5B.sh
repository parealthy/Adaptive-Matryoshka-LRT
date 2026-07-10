#!/bin/bash
set -euo pipefail

SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
REASONING_NET_PATH="${REASONING_NET_PATH:-Qwen/Qwen3-Embedding-0.6B}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-checkpoints/ALR-Stage1-DSR1-Qwen-1.5B}"
DIFFICULTY_CHECKPOINT_PATH="${DIFFICULTY_CHECKPOINT_PATH:-checkpoints/ALR-Stage2-Difficulty-DSR1-Qwen-1.5B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
# Leave empty to use the lm-eval package installed from requirements.txt.
# Set this only when deliberately testing a local lm-evaluation-harness checkout.
HARNESS_PATH="${HARNESS_PATH:-}"
LATENT_TRAJECTORY_LENGTHS="${LATENT_TRAJECTORY_LENGTHS:-64,128,192,256}"

if [ -z "${METHODS+x}" ] && [ -n "${METHODs:-}" ]; then
    echo "Warning: detected METHODs; using it as METHODS."
    METHODS="$METHODs"
fi

OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/alr_lm_eval_1.5B}"
TASKS="${TASKS:-gsm8k math500}"
METHODS="${METHODS:-plain-ar zero-256 random-256 fixed-64 fixed-128 fixed-192 fixed-256 random adaptive}"
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
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-false}"
USE_FAST_TOKENIZER="${USE_FAST_TOKENIZER:-true}"

export ALR_LM_EVAL_DIST_BACKEND="${ALR_LM_EVAL_DIST_BACKEND:-gloo}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

TASKS="${TASKS//math-500/math500}"

python - <<'PY'
import importlib.util

missing = [
    package
    for package in ("lm_eval", "numpy", "datasets", "transformers", "accelerate")
    if importlib.util.find_spec(package) is None
]
if missing:
    raise SystemExit(
        "Missing evaluation dependencies: " + ", ".join(missing)
        + ". Run: pip install -r requirements.txt"
    )

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
if [ -n "$TOKENIZER_PATH" ]; then
    EXTRA_ARGS+=(--tokenizer_path "$TOKENIZER_PATH")
fi
if [ "$LOCAL_FILES_ONLY" = "true" ]; then
    EXTRA_ARGS+=(--local_files_only)
fi
if [ "$USE_FAST_TOKENIZER" != "true" ]; then
    EXTRA_ARGS+=(--use_slow_tokenizer)
fi
if [ -n "$HARNESS_PATH" ]; then
    EXTRA_ARGS+=(--harness_path "$HARNESS_PATH")
fi
EXTRA_ARGS+=(--num_fewshot "$NUM_FEWSHOT")

echo "================================================"
echo "  ALR lm-evaluation-harness Evaluation"
echo "  Harness: ${HARNESS_PATH:-installed lm-eval package}"
echo "  Stage1:  $STAGE1_CHECKPOINT_PATH"
echo "  Stage2:  $DIFFICULTY_CHECKPOINT_PATH"
echo "  Tokenizer: ${TOKENIZER_PATH:-auto-stage1-or-model}"
echo "  Tasks:   $TASKS"
echo "  Methods: $METHODS"
echo "  Fewshot: $NUM_FEWSHOT"
echo "  Training prompt template: $USE_TRAINING_PROMPT_TEMPLATE"
echo "  Local files only: $LOCAL_FILES_ONLY"
echo "  Fast tokenizer: $USE_FAST_TOKENIZER"
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
