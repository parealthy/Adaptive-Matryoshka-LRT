#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARNESS_PATH = Path("/Users/focus/Desktop/paper/s1/eval/lm-evaluation-harness")


def parse_list(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def normalize_task_name(task: str) -> str:
    aliases = {
        "math-500": "math500",
        "MATH-500": "math500",
    }
    return aliases.get(task, task)


def parse_limit(value: str):
    parsed = float(value)
    if parsed.is_integer():
        return int(parsed)
    return parsed


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")


def patch_transformers_harness_compat() -> None:
    import transformers

    try:
        transformers.AutoModelForVision2Seq
        return
    except AttributeError:
        pass

    fallback = None
    for fallback_name in ("AutoModelForSeq2SeqLM", "AutoModelForCausalLM"):
        try:
            fallback = getattr(transformers, fallback_name)
            break
        except Exception:
            continue
    if fallback is None:
        raise RuntimeError(
            "transformers is missing AutoModelForVision2Seq and no text fallback "
            "AutoModel class is available. Upgrade transformers."
        )

    transformers.__dict__["AutoModelForVision2Seq"] = fallback
    try:
        object.__setattr__(transformers, "AutoModelForVision2Seq", fallback)
    except Exception:
        pass
    if hasattr(transformers, "_objects"):
        transformers._objects["AutoModelForVision2Seq"] = fallback
    try:
        from transformers.utils.import_utils import _LazyModule
    except Exception:
        return
    if getattr(_LazyModule, "_alr_vision2seq_patch", False):
        return
    original_getattr = _LazyModule.__getattr__

    def patched_getattr(self, name):
        if name == "AutoModelForVision2Seq":
            return fallback
        return original_getattr(self, name)

    _LazyModule.__getattr__ = patched_getattr
    _LazyModule._alr_vision2seq_patch = True


def import_harness(harness_path: Path):
    harness_path = harness_path.expanduser().resolve()
    if not harness_path.exists():
        raise FileNotFoundError(f"lm-evaluation-harness path does not exist: {harness_path}")
    if str(harness_path) not in sys.path:
        sys.path.insert(0, str(harness_path))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    patch_transformers_harness_compat()
    import evaluation.lm_eval_alr_model  # noqa: F401
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager

    return lm_eval, evaluator, TaskManager


def select_accuracy_metric(task_result: dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    candidates = [
        key
        for key in task_result
        if not key.endswith("_stderr") and "exact_match" in key
    ]
    if not candidates:
        candidates = [
            key
            for key in task_result
            if not key.endswith("_stderr") and key in {"acc", "accuracy"}
        ]
    if not candidates:
        return None, None

    def priority(key: str) -> tuple[int, str]:
        if "flexible" in key:
            return (0, key)
        if key == "exact_match":
            return (1, key)
        if "strict" in key:
            return (3, key)
        return (2, key)

    selected = sorted(candidates, key=priority)[0]
    return selected, float(task_result[selected])


def read_traces(trace_glob: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for filename in sorted(glob.glob(trace_glob)):
        with Path(filename).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def summarize_cost(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "avg_latent_length": None,
            "latent_cost_saving_vs_fixed_256": None,
            "latent_length_distribution": {},
            "avg_output_tokens": None,
        }

    lengths = [int(record["latent_length"]) for record in records]
    output_tokens = [int(record.get("output_tokens", 0)) for record in records]
    avg_latent_length = sum(lengths) / len(lengths)
    avg_output_tokens = sum(output_tokens) / len(output_tokens)
    distribution = Counter(str(length) for length in lengths)
    return {
        "avg_latent_length": avg_latent_length,
        "latent_cost_saving_vs_fixed_256": 1.0 - avg_latent_length / 256.0,
        "latent_length_distribution": dict(sorted(distribution.items(), key=lambda item: int(item[0]))),
        "avg_output_tokens": avg_output_tokens,
    }


def effective_total(results: Optional[dict[str, Any]], task: str, fallback: int) -> int:
    if not results:
        return fallback
    samples = results.get("n-samples", {}).get(task, {})
    if "effective" in samples:
        return int(samples["effective"])
    return fallback


def run_one(evaluator, TaskManager, args, method: str, task: str) -> Optional[dict[str, Any]]:
    harness_output_dir = Path(args.output_dir) / "harness" / method / task
    trace_dir = Path(args.output_dir) / "traces" / method
    trace_path = trace_dir / f"{task}_rank{{rank:05d}}.jsonl"
    result_path = harness_output_dir / "results.json"

    if not args.overwrite and result_path.exists():
        if is_main_process():
            print(
                f"Skipping existing lm-eval result: task={task} method={method} "
                f"({result_path})",
                flush=True,
            )
        with result_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    if is_main_process() and args.overwrite:
        if harness_output_dir.exists():
            shutil.rmtree(harness_output_dir)

    task_manager = TaskManager(args.verbosity, include_path=args.task_include_path)
    model_args = {
        "lrt_root": str(PROJECT_ROOT),
        "model_path": args.model_path,
        "reasoning_net_path": args.reasoning_net_path,
        "stage1_checkpoint_path": args.stage1_checkpoint_path,
        "difficulty_checkpoint_path": args.difficulty_checkpoint_path,
        "tokenizer_path": args.tokenizer_path,
        "latent_trajectory_lengths": args.latent_trajectory_lengths,
        "method": method,
        "prompt_max_length": args.prompt_max_length,
        "max_gen_toks": args.max_new_tokens,
        "torch_dtype": args.torch_dtype,
        "trust_remote_code": not args.no_trust_remote_code,
        "top_p": args.top_p,
        "seed": args.seed,
        "trace_path": str(trace_path),
        "trace_task": task,
        "use_training_prompt_template": not args.no_training_prompt_template,
        "local_files_only": args.local_files_only,
        "use_fast_tokenizer": not args.use_slow_tokenizer,
    }
    gen_kwargs_items = [
        f"do_sample={'true' if args.temperature > 0 else 'false'}",
        f"max_gen_toks={args.max_new_tokens}",
    ]
    if args.temperature > 0:
        gen_kwargs_items.extend(
            [
                f"temperature={args.temperature}",
                f"top_p={args.top_p}",
            ]
        )
    gen_kwargs = ",".join(gen_kwargs_items)

    results = evaluator.simple_evaluate(
        model="alr",
        model_args=model_args,
        tasks=[task],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=None,
        device=args.device,
        limit=args.limit,
        log_samples=True,
        task_manager=task_manager,
        verbosity=args.verbosity,
        gen_kwargs=gen_kwargs,
        apply_chat_template=args.apply_chat_template,
        bootstrap_iters=0,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if results is None:
        return None

    samples = results.pop("samples", {})
    dump_json(harness_output_dir / "results.json", results)
    dump_json(harness_output_dir / "samples.json", samples)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ALR through lm-evaluation-harness and summarize latent cost traces.",
    )
    parser.add_argument("--harness_path", default=str(DEFAULT_HARNESS_PATH))
    parser.add_argument("--model_path", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--reasoning_net_path", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument(
        "--stage1_checkpoint_path",
        default="checkpoints/ALR-Stage1-DSR1-Qwen-1.5B",
    )
    parser.add_argument(
        "--difficulty_checkpoint_path",
        default="checkpoints/ALR-Stage2-Difficulty-DSR1-Qwen-1.5B",
    )
    parser.add_argument("--latent_trajectory_lengths", default="64,128,192,256")
    parser.add_argument("--output_dir", default="eval_outputs/alr_lm_eval_1.5B")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "math500"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["fixed-64", "fixed-128", "fixed-192", "fixed-256", "random", "adaptive"],
    )
    parser.add_argument("--task_include_path", default=str(PROJECT_ROOT / "evaluation" / "lm_eval_tasks"))
    parser.add_argument("--batch_size", default="1")
    parser.add_argument("--limit", type=parse_limit, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--prompt_max_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch_dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--verbosity", default="INFO")
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--no_training_prompt_template", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--no_trust_remote_code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.tasks = [normalize_task_name(task) for task in args.tasks]
    harness_path = Path(args.harness_path)
    _, evaluator, TaskManager = import_harness(harness_path)

    summary: dict[str, Any] = {
        "config": {
            "harness_path": str(harness_path.expanduser().resolve()),
            "model_path": args.model_path,
            "reasoning_net_path": args.reasoning_net_path,
            "tokenizer_path": args.tokenizer_path,
            "stage1_checkpoint_path": args.stage1_checkpoint_path,
            "difficulty_checkpoint_path": args.difficulty_checkpoint_path,
            "latent_trajectory_lengths": parse_list(args.latent_trajectory_lengths),
            "tasks": args.tasks,
            "methods": args.methods,
            "num_fewshot": args.num_fewshot,
            "use_training_prompt_template": not args.no_training_prompt_template,
            "local_files_only": args.local_files_only,
            "use_fast_tokenizer": not args.use_slow_tokenizer,
            "accuracy_source": "lm-evaluation-harness",
        },
        "results": {},
    }

    for method in args.methods:
        summary["results"].setdefault(method, {})
        for task in args.tasks:
            if is_main_process():
                print(f"\n=== lm-eval ALR: task={task} method={method} ===", flush=True)
            results = run_one(evaluator, TaskManager, args, method, task)
            if not is_main_process():
                continue

            trace_pattern = str(Path(args.output_dir) / "traces" / method / f"{task}_rank*.jsonl")
            trace_records = read_traces(trace_pattern)
            cost = summarize_cost(trace_records)
            task_result = (results or {}).get("results", {}).get(task, {})
            metric_name, accuracy = select_accuracy_metric(task_result)
            row = {
                "accuracy": accuracy,
                "accuracy_metric": metric_name,
                "total": effective_total(results, task, len(trace_records)),
                **cost,
            }
            summary["results"][method][task] = row
            dump_json(Path(args.output_dir) / "summary.json", summary)

    if is_main_process():
        dump_json(Path(args.output_dir) / "summary.json", summary)
        print(f"\nSaved summary to {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
