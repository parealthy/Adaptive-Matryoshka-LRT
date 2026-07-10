# Adaptive Length-Elastic Latent Reasoning (ALR)

Experimental code for *Adaptive Length-Elastic Latent Reasoning for Efficient Mathematical Inference*.

ALR keeps the base language model frozen and learns a small reasoning network that produces a latent prefix. Stage 1 trains one model to work at several prefix lengths; Stage 2 learns to select the shortest suitable length for each problem. The paper compares fixed lengths (64/128/192/256), random selection, and adaptive selection on GSM8K and MATH-500.

## Repository layout

```
├── stage1_sft.py                    # Stage 1: length-elastic reasoning-network training
├── stage2_train.py                  # Stage 2: difficulty-head training
├── modeling/
│   ├── alr_stage1.py                # Length-elastic latent reasoning model
│   └── alr_difficulty.py            # Difficulty classifier and checkpoint I/O
├── inference/                       # Fixed-length and adaptive interactive inference
├── evaluation/                      # Direct evaluator and lm-evaluation-harness adapter
├── scripts/                         # Reproducible launch scripts
├── configs/                         # Accelerate / DeepSpeed configurations in active use
├── utils/                           # Dataset loading and mathematical answer verification
├── tests/                           # Lightweight unit tests
└── requirements.txt
```

Only ALR code is kept here. Legacy fixed-length LRT/RFT implementations, vendored third-party sources, paper drafting files, IDE settings, and generated cache files have been removed.

## Environment

Training is intended for Linux with NVIDIA GPUs. The supplied paper-scale scripts default to eight GPUs; `NUM_GPUS_PER_NODE`, `NUM_NODES`, batch size, and accumulation steps are all overridable environment variables. A CUDA-capable PyTorch build is required for training and for practical evaluation. Python 3.10 or 3.11 is recommended.

```bash
cd LRT
conda create -n alr python=3.10 -y
conda activate alr

# Select the wheel index matching the installed CUDA driver; this example is CUDA 12.1.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Optional but recommended: verify the launcher configuration before a long run.
accelerate env
```

`requirements.txt` defines compatible version ranges for the training stack, answer verifier, and `lm-eval`. Do not install the historical `trl/` source tree: Stage 1 uses the standard Hugging Face `Trainer` directly.

Log in to Hugging Face before downloading gated models or datasets:

```bash
huggingface-cli login
```

Datasets load from the Hugging Face Hub by default. To use an existing `datasets.save_to_disk` cache instead, set `LRT_DATA_ROOT`; for example:

```bash
export LRT_DATA_ROOT=/datasets/alr
```

The Stage 1 loader checks `$LRT_DATA_ROOT/open-r1/OpenR1-Math-220k` before downloading. Stage 2 uses the Hub datasets listed below and relies on the normal Hugging Face cache (`HF_HOME`) unless they have already been cached.

## Data and model settings

The paper configuration uses:

| Component | Default |
| --- | --- |
| Frozen base model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Reasoning backbone | `Qwen/Qwen3-Embedding-0.6B` |
| Latent lengths | `64,128,192,256` |
| Stage 1 data | `open-r1/OpenR1-Math-220k` |
| Stage 2 labels | GSM8K → 64, MATH minus MATH-500 → 128, DeepScaleR Preview → 192, Olympiads → 256 |
| Evaluation | GSM8K and MATH-500, greedy decoding (`temperature=0`), seed 42 |

Stage 2 samples 7,473 examples per source by default (29,892 total before the train/validation split). It saves pooled-feature caches under `cache/`; retain these caches to avoid rerunning frozen-model feature extraction.

## Reproduce the paper pipeline

All commands below are run from the `LRT` directory. The shell scripts expose their settings as environment variables, so no source editing is required.

### 1. Train the length-elastic reasoning network

The default script trains the 1.5B setup for three epochs with ZeRO-2. It writes model weights, tokenizer files, Trainer checkpoints, and `alr_stage1_config.json` to `OUTPUT_DIR`.

```bash
NUM_GPUS_PER_NODE=8 NUM_NODES=1 \
  bash scripts/train_alr_stage1_1.5B.sh
```

For a small smoke test, use a short subset and a local launch configuration. This only checks the pipeline; it does not reproduce paper quality.

```bash
NUM_GPUS_PER_NODE=1 NUM_NODES=1 PER_DEVICE_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 NUM_EPOCHS=1 \
OUTPUT_DIR=checkpoints/alr-stage1-smoke \
  bash scripts/train_alr_stage1_1.5B.sh
```

Useful overrides include `SLOW_THINKING_MODEL_PATH`, `REASONING_NET_PATH`, `LATENT_TRAJECTORY_LENGTHS`, `DATASET_NAME`, `PROMPT_MAX_LENGTH`, `COMPLETION_MAX_LENGTH`, and `RESUME_FROM_CHECKPOINT`. `RESUME_FROM_CHECKPOINT` reloads model weights before training; optimizer and scheduler state are intentionally not restored.

### 2. Train the difficulty estimator

This stage freezes the base model, pools prompt hidden states, and fits the MLP classifier. It reads the Stage 1 length set from the checkpoint configuration unless it is explicitly overridden.

```bash
STAGE1_CHECKPOINT_PATH=checkpoints/ALR-Stage1-DSR1-Qwen-1.5B \
NUM_GPUS=8 \
  bash scripts/train_alr_stage2_1.5B.sh
```

The result directory contains `alr_difficulty_config.json`, `alr_difficulty_head.pt`, `alr_stage2_config.json`, and `stage2_metrics.json`. To check data and code paths without creating the full cache, add `LIMIT=16`; that limit is per source dataset.

### 3. Evaluate all methods

The primary evaluator evaluates fixed, random, and adaptive ALR with identical prompts, writes per-example records, and produces `summary.json` containing accuracy, mean latent length, distribution, and saving relative to fixed-256.

```bash
STAGE1_CHECKPOINT_PATH=checkpoints/ALR-Stage1-DSR1-Qwen-1.5B \
DIFFICULTY_CHECKPOINT_PATH=checkpoints/ALR-Stage2-Difficulty-DSR1-Qwen-1.5B \
NUM_GPUS=8 \
  bash scripts/eval_alr_full_1.5B.sh
```

For the paper comparison, leave the default methods unchanged:

```text
fixed-64 fixed-128 fixed-192 fixed-256 random adaptive
```

Use `LIMIT=32` for a fast integration test, or set `TASKS="gsm8k"` / `METHODS="fixed-256 adaptive"` to restrict the run. Full results require `LIMIT` to be empty.

The optional lm-evaluation-harness route uses the `lm-eval` package installed by `requirements.txt` and the repository's MATH-500 task definition:

```bash
METHODS="fixed-64 fixed-128 fixed-192 fixed-256 random adaptive" \
NUM_GPUS=8 \
  bash scripts/eval_alr_lm_eval_1.5B.sh
```

If a locally modified harness is required, pass its path explicitly rather than relying on a machine-specific path:

```bash
HARNESS_PATH=/path/to/lm-evaluation-harness \
  bash scripts/eval_alr_lm_eval_1.5B.sh
```

## Inference

Run a trained Stage 1 model at a fixed length:

```bash
CHECKPOINT_PATH=checkpoints/ALR-Stage1-DSR1-Qwen-1.5B \
FIXED_LATENT_TRAJECTORY_LENGTH=128 \
QUESTION="What is the sum of the first 100 positive integers?" \
  bash inference/run_alr_stage1_inference.sh
```

Run adaptive inference after both stages are available:

```bash
STAGE1_CHECKPOINT_PATH=checkpoints/ALR-Stage1-DSR1-Qwen-1.5B \
DIFFICULTY_CHECKPOINT_PATH=checkpoints/ALR-Stage2-Difficulty-DSR1-Qwen-1.5B \
QUESTION="Solve x^2 - 5x + 6 = 0." \
  bash inference/run_alr_adaptive_inference.sh
```

Omit `QUESTION` in either command to enter interactive mode.

## Validation and reproducibility notes

Run the lightweight checks after installing the environment:

```bash
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py'
python -m py_compile stage1_sft.py stage2_train.py modeling/*.py inference/*.py evaluation/*.py
bash -n scripts/*.sh inference/*.sh
```

For comparable results, keep the model, dataset revisions, length set, prompt lengths, evaluation prompt template, `temperature=0`, and seed 42 unchanged. Multi-GPU generation and CUDA kernels can still introduce small run-to-run differences. Checkpoints and datasets are intentionally not bundled with the repository; they must be downloaded or trained using the commands above.

## License

This project is released under the MIT License.
