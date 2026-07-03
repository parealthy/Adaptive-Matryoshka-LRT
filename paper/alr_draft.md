# Adaptive Length-Elastic Latent Reasoning for Efficient Mathematical Inference

## Abstract

Large reasoning models often allocate the same amount of test-time computation to every query, even though many math problems can be solved with substantially less reasoning capacity. We study **Adaptive Length-Elastic Latent Reasoning (ALR)**, a two-stage latent reasoning framework that trains a reasoning network to operate under multiple latent trajectory lengths and then predicts the required latent length for each prompt. The method preserves the frozen base language model and uses a lightweight difficulty head to select among 64, 128, 192, and 256 latent tokens at inference time. Our experiments evaluate whether adaptive latent allocation reduces average latent reasoning cost while retaining the accuracy of a fixed long latent trajectory.

## 1. Introduction

Latent reasoning replaces long explicit chain-of-thought traces with continuous latent trajectories. This reduces visible output length, but a fixed latent trajectory still spends the same compute budget on easy and hard questions. The central observation behind ALR is simple: not all problems require the same reasoning depth. A grade-school arithmetic problem should not need the same latent capacity as an olympiad-style problem.

This draft investigates a practical question: can a small prompt-level classifier choose a latent trajectory length that preserves accuracy while lowering average latent cost?

## 2. Method

ALR has two stages.

**Stage 1: length-elastic latent reasoning.** We train the latent reasoning network with random latent trajectory lengths sampled from {64, 128, 192, 256}. The base model remains frozen. During training, each batch uses one active prefix length, so every configured prefix becomes a valid reasoning model after training.

**Stage 2: difficulty estimator.** We freeze the Stage 1 model and train a tiny MLP classifier over pooled prompt hidden states. The classifier predicts one of four latent lengths. Labels are assigned by dataset source: GSM8K maps to 64, MATH-minus-MATH-500 maps to 128, DeepScaleR Preview maps to 192, and Olympiads maps to 256.

**Adaptive inference.** At test time, the model performs one prompt prefill, pools the prompt hidden states, predicts a length, generates the corresponding latent prefix, and decodes the final answer with the frozen base model.

## 3. Experimental Setup

- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Stage 1 checkpoint: `checkpoints/ALR-Stage1-DSR1-Qwen-1.5B`
- Stage 2 checkpoint: `checkpoints/ALR-Stage2-Difficulty-DSR1-Qwen-1.5B`
- Latent lengths: 64, 128, 192, 256
- Evaluation tasks: GSM8K and MATH-500
- Main baselines: fixed 64/128/192/256 latent lengths, random uniform length, and adaptive ALR
- Metrics: answer accuracy, average latent length, and latent cost saving relative to fixed-256

## 4. Results

| Method | GSM8K Acc. (%) | MATH500 Acc. (%) | Avg. Latent | Saving vs 256 (%) |
| --- | --- | --- | --- | --- |
| Fixed 64 | TBD | TBD | TBD | TBD |
| Fixed 128 | TBD | TBD | TBD | TBD |
| Fixed 192 | TBD | TBD | TBD | TBD |
| Fixed 256 | TBD | TBD | TBD | TBD |
| Random uniform | TBD | TBD | TBD | TBD |
| Adaptive ALR | TBD | TBD | TBD | TBD |

The key comparison is adaptive ALR versus fixed-256. A successful run should show that adaptive ALR uses a lower average latent length while staying close to the fixed-256 accuracy, especially on the easier GSM8K split.

### Length Allocation

Length distributions will be filled after running evaluation.

### Stage 2 Diagnostics

Stage 2 validation accuracy and confusion matrix will be filled after running `scripts/train_alr_stage2_1.5B.sh`.

## 5. Discussion

The fixed-length sweep measures how much capability each latent budget provides. Random uniform allocation tests whether cost reduction alone is sufficient. Adaptive ALR tests whether prompt-conditioned allocation can improve the accuracy-cost frontier. If adaptive ALR outperforms random uniform at a similar average latent length, that supports the usefulness of the learned difficulty estimator.

The current label policy is intentionally simple and source-based. It follows the ALR design sketch, but it is not an oracle. Future versions can replace these labels with the shortest latent length that solves each training sample, which may produce a stronger difficulty estimator.

## 6. Limitations

- Source-based labels are coarse and may overestimate or underestimate individual sample difficulty.
- The current experiments focus on mathematical reasoning tasks only.
- Adaptive generation still requires the prompt prefill before length selection, so savings are measured on latent trajectory length rather than total end-to-end FLOPs.
- Accuracy depends on final-answer extraction and mathematical verification quality.

## 7. Future Work

- Train Stage 2 with oracle shortest-correct-length labels.
- Add AIME/AMC/OlympiadBench held-out evaluations.
- Compare adaptive latent allocation with explicit chain-of-thought budget forcing.
- Measure wall-clock latency and memory in addition to latent token count.
