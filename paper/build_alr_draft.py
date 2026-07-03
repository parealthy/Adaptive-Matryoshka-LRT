#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


DEFAULT_METHOD_ORDER = [
    "fixed-64",
    "fixed-128",
    "fixed-192",
    "fixed-256",
    "random",
    "adaptive",
]
DEFAULT_TASK_ORDER = ["gsm8k", "math500"]


def load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_percent(value: Optional[float]) -> str:
    if value is None:
        return "TBD"
    return f"{100.0 * value:.2f}"


def format_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "TBD"
    return f"{value:.{digits}f}"


def method_label(method: str) -> str:
    labels = {
        "fixed-64": "Fixed 64",
        "fixed-128": "Fixed 128",
        "fixed-192": "Fixed 192",
        "fixed-256": "Fixed 256",
        "random": "Random uniform",
        "adaptive": "Adaptive ALR",
    }
    return labels.get(method, method)


def collect_methods(summary: Optional[dict[str, Any]]) -> list[str]:
    if not summary:
        return DEFAULT_METHOD_ORDER
    methods = list(summary.get("results", {}).keys())
    ordered = [method for method in DEFAULT_METHOD_ORDER if method in methods]
    ordered.extend(method for method in methods if method not in ordered)
    return ordered or DEFAULT_METHOD_ORDER


def collect_tasks(summary: Optional[dict[str, Any]]) -> list[str]:
    if not summary:
        return DEFAULT_TASK_ORDER
    tasks = set()
    for method_results in summary.get("results", {}).values():
        tasks.update(method_results.keys())
    ordered = [task for task in DEFAULT_TASK_ORDER if task in tasks]
    ordered.extend(task for task in sorted(tasks) if task not in ordered)
    return ordered or DEFAULT_TASK_ORDER


def get_metric(summary: Optional[dict[str, Any]], method: str, task: str, key: str):
    if not summary:
        return None
    return summary.get("results", {}).get(method, {}).get(task, {}).get(key)


def build_results_table(summary: Optional[dict[str, Any]]) -> str:
    methods = collect_methods(summary)
    tasks = collect_tasks(summary)
    task_headers = [f"{task.upper()} Acc. (%)" for task in tasks]
    headers = ["Method", *task_headers, "Avg. Latent", "Saving vs 256 (%)"]
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for method in methods:
        accuracies = [
            format_percent(get_metric(summary, method, task, "accuracy"))
            for task in tasks
        ]
        latent_values = [
            get_metric(summary, method, task, "avg_latent_length")
            for task in tasks
        ]
        saving_values = [
            get_metric(summary, method, task, "latent_cost_saving_vs_fixed_max")
            for task in tasks
        ]
        latent_values = [value for value in latent_values if value is not None]
        saving_values = [value for value in saving_values if value is not None]
        avg_latent = (
            sum(latent_values) / len(latent_values)
            if latent_values
            else None
        )
        avg_saving = (
            sum(saving_values) / len(saving_values)
            if saving_values
            else None
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    method_label(method),
                    *accuracies,
                    format_float(avg_latent),
                    format_percent(avg_saving),
                ]
            )
            + " |"
        )

    return "\n".join(rows)


def build_length_distribution(summary: Optional[dict[str, Any]]) -> str:
    if not summary:
        return "Length distributions will be filled after running evaluation."

    rows = ["| Method | Task | 64 | 128 | 192 | 256 |", "| --- | --- | --- | --- | --- | --- |"]
    for method in collect_methods(summary):
        for task in collect_tasks(summary):
            metrics = summary.get("results", {}).get(method, {}).get(task)
            if not metrics:
                continue
            distribution = metrics.get("latent_length_distribution", {})
            total = max(int(metrics.get("total", 0)), 1)
            rows.append(
                "| "
                + " | ".join(
                    [
                        method_label(method),
                        task.upper(),
                        *[
                            f"{100.0 * int(distribution.get(str(length), 0)) / total:.1f}"
                            for length in (64, 128, 192, 256)
                        ],
                    ]
                )
                + " |"
            )
    return "\n".join(rows)


def build_stage2_diagnostics(stage2_metrics: Optional[dict[str, Any]]) -> str:
    if not stage2_metrics:
        return (
            "Stage 2 validation accuracy and confusion matrix will be filled after "
            "running `scripts/train_alr_stage2_1.5B.sh`."
        )

    best = stage2_metrics.get("best_validation") or {}
    confusion = best.get("confusion_matrix")
    lines = [
        f"- Best validation accuracy: {format_percent(best.get('accuracy'))}%",
        f"- Best validation loss: {format_float(best.get('loss'), digits=4)}",
        f"- Validation examples: {best.get('total', 'TBD')}",
    ]
    if confusion:
        lines.append("")
        lines.append("| Gold \\ Pred | 64 | 128 | 192 | 256 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for length, row in zip((64, 128, 192, 256), confusion):
            lines.append(
                "| "
                + " | ".join([str(length), *[str(value) for value in row]])
                + " |"
            )
    return "\n".join(lines)


def build_draft(
    summary: Optional[dict[str, Any]],
    stage2_metrics: Optional[dict[str, Any]],
) -> str:
    config = (summary or {}).get("config", {})
    model_path = config.get("model_path", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    stage1_path = config.get("stage1_checkpoint_path", "checkpoints/ALR-Stage1-DSR1-Qwen-1.5B")
    stage2_path = config.get("difficulty_checkpoint_path", "checkpoints/ALR-Stage2-Difficulty-DSR1-Qwen-1.5B")

    return f"""# Adaptive Length-Elastic Latent Reasoning for Efficient Mathematical Inference

## Abstract

Large reasoning models often allocate the same amount of test-time computation to every query, even though many math problems can be solved with substantially less reasoning capacity. We study **Adaptive Length-Elastic Latent Reasoning (ALR)**, a two-stage latent reasoning framework that trains a reasoning network to operate under multiple latent trajectory lengths and then predicts the required latent length for each prompt. The method preserves the frozen base language model and uses a lightweight difficulty head to select among 64, 128, 192, and 256 latent tokens at inference time. Our experiments evaluate whether adaptive latent allocation reduces average latent reasoning cost while retaining the accuracy of a fixed long latent trajectory.

## 1. Introduction

Latent reasoning replaces long explicit chain-of-thought traces with continuous latent trajectories. This reduces visible output length, but a fixed latent trajectory still spends the same compute budget on easy and hard questions. The central observation behind ALR is simple: not all problems require the same reasoning depth. A grade-school arithmetic problem should not need the same latent capacity as an olympiad-style problem.

This draft investigates a practical question: can a small prompt-level classifier choose a latent trajectory length that preserves accuracy while lowering average latent cost?

## 2. Method

ALR has two stages.

**Stage 1: length-elastic latent reasoning.** We train the latent reasoning network with random latent trajectory lengths sampled from {{64, 128, 192, 256}}. The base model remains frozen. During training, each batch uses one active prefix length, so every configured prefix becomes a valid reasoning model after training.

**Stage 2: difficulty estimator.** We freeze the Stage 1 model and train a tiny MLP classifier over pooled prompt hidden states. The classifier predicts one of four latent lengths. Labels are assigned by dataset source: GSM8K maps to 64, MATH-minus-MATH-500 maps to 128, DeepScaleR Preview maps to 192, and Olympiads maps to 256.

**Adaptive inference.** At test time, the model performs one prompt prefill, pools the prompt hidden states, predicts a length, generates the corresponding latent prefix, and decodes the final answer with the frozen base model.

## 3. Experimental Setup

- Base model: `{model_path}`
- Stage 1 checkpoint: `{stage1_path}`
- Stage 2 checkpoint: `{stage2_path}`
- Latent lengths: 64, 128, 192, 256
- Evaluation tasks: GSM8K and MATH-500
- Main baselines: fixed 64/128/192/256 latent lengths, random uniform length, and adaptive ALR
- Metrics: answer accuracy, average latent length, and latent cost saving relative to fixed-256

## 4. Results

{build_results_table(summary)}

The key comparison is adaptive ALR versus fixed-256. A successful run should show that adaptive ALR uses a lower average latent length while staying close to the fixed-256 accuracy, especially on the easier GSM8K split.

### Length Allocation

{build_length_distribution(summary)}

### Stage 2 Diagnostics

{build_stage2_diagnostics(stage2_metrics)}

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
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the English Markdown draft for the ALR experiment.",
    )
    parser.add_argument("--eval_dir", default="eval_outputs/alr_full_1.5B")
    parser.add_argument("--stage2_metrics", default=None)
    parser.add_argument("--output", default="paper/alr_draft.md")
    return parser.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    summary = load_json(eval_dir / "summary.json")

    stage2_metrics_path = (
        Path(args.stage2_metrics)
        if args.stage2_metrics
        else Path("checkpoints/ALR-Stage2-Difficulty-DSR1-Qwen-1.5B/stage2_metrics.json")
    )
    stage2_metrics = load_json(stage2_metrics_path)

    draft = build_draft(summary, stage2_metrics)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(draft, encoding="utf-8")
    print(f"Wrote ALR draft to {output_path}")


if __name__ == "__main__":
    main()
