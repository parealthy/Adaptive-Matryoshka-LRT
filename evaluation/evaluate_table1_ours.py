"""
Evaluate the paper's "Ours" latent-reasoning method on Table 1 benchmarks.

This script only covers the requested datasets:
  - MATH-500
  - GSM8K

It loads the frozen base LLM, the trained reasoning-network checkpoint, runs
greedy generation, and scores answers with the same math-verification reward
used by RFT training.

Example:
    python evaluation/evaluate_table1_ours.py \
        --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --reasoning_net_path Qwen/Qwen3-Embedding-0.6B \
        --checkpoint_path checkpoints/DSR1-Qwen-1.5B-LRT-Math \
        --datasets math500 gsm8k
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.reason import TransformerReasoningNet
from utils.load_data import MATH_SUFFIX
from utils.reward_func import accuracy_reward


@dataclass(frozen=True)
class BenchmarkSpec:
    display_name: str
    dataset_id: str
    config_name: str | None
    split: str


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "math500": BenchmarkSpec(
        display_name="MATH-500",
        dataset_id="HuggingFaceH4/MATH-500",
        config_name=None,
        split="test",
    ),
    "gsm8k": BenchmarkSpec(
        display_name="GSM8K",
        dataset_id="openai/gsm8k",
        config_name="main",
        split="test",
    ),
}


DTYPE_MAP = {
    "auto": "auto",
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Table 1 Ours on MATH-500 and GSM8K.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Frozen base LLM path or Hugging Face id.",
    )
    parser.add_argument(
        "--reasoning_net_path",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Reasoning-network backbone path or Hugging Face id.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Checkpoint directory containing trained reasoning_network weights.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["math500", "gsm8k"],
        choices=sorted(BENCHMARKS),
        help="Benchmarks to evaluate.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("LRT_DATA_ROOT", str(PROJECT_ROOT / "data" / "datasets")),
        help="Optional local dataset cache root. Falls back to Hugging Face Hub.",
    )
    parser.add_argument("--output_dir", type=str, default="eval_outputs/table1_ours")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--prompt_max_length", type=int, default=1024)
    parser.add_argument("--latent_trajectory_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for single-device evaluation, or 'auto' for HF device_map auto.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bf16",
        choices=sorted(DTYPE_MAP),
        help="Base-model load dtype.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-dataset sample limit for smoke tests.",
    )
    parser.add_argument(
        "--no_math_suffix",
        action="store_true",
        help="Do not append the repository's boxed-answer math suffix to prompts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prediction files.",
    )
    return parser.parse_args()


def _resolve_hidden_size(config) -> int:
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    raise ValueError("Failed to resolve hidden_size from the base model config.")


def _build_device_map(device: str):
    if device == "auto":
        return "auto"
    if device == "cuda":
        return {"": 0}
    if device.startswith("cuda:"):
        return {"": int(device.split(":", 1)[1])}
    return {"": device}


def _model_input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def _last_layer_hidden_states(model, inputs_embeds, attention_mask):
    """
    Extract last decoder layer hidden states without materializing every layer.
    Falls back to output_hidden_states for model families without a standard
    decoder-layer attribute.
    """
    captured: dict[str, torch.Tensor] = {}

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        last_layer = model.model.layers[-1]
        base_model = model.model
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        last_layer = model.transformer.h[-1]
        base_model = model.transformer
    else:
        output = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        return output.hidden_states[-1]

    def hook(_module, _input, output):
        captured["hidden_states"] = output[0] if isinstance(output, tuple) else output

    handle = last_layer.register_forward_hook(hook)
    try:
        base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    finally:
        handle.remove()

    if "hidden_states" not in captured:
        raise RuntimeError("Failed to capture last-layer hidden states.")
    return captured["hidden_states"]


def _load_reasoning_weights(reasoning_network: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    safetensor_files = sorted(glob(str(checkpoint_dir / "*.safetensors")))
    state_dict: dict[str, torch.Tensor] = {}

    for filename in safetensor_files:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("reasoning_network."):
                    state_dict[key.removeprefix("reasoning_network.")] = handle.get_tensor(key)
                elif key in reasoning_network.state_dict():
                    state_dict[key] = handle.get_tensor(key)

    if not state_dict:
        pytorch_model = checkpoint_dir / "pytorch_model.bin"
        if pytorch_model.exists():
            raw_state = torch.load(str(pytorch_model), map_location="cpu")
            for key, value in raw_state.items():
                if key.startswith("reasoning_network."):
                    state_dict[key.removeprefix("reasoning_network.")] = value
                elif key in reasoning_network.state_dict():
                    state_dict[key] = value

    if not state_dict:
        raise ValueError(
            f"No reasoning-network weights found in checkpoint directory: {checkpoint_path}"
        )

    reasoning_network.load_state_dict(state_dict, strict=True)
    print(f"Loaded {len(state_dict)} reasoning-network tensors from {checkpoint_path}")


class LatentReasoningEvaluator:
    def __init__(
        self,
        *,
        model_path: str,
        reasoning_net_path: str,
        checkpoint_path: str,
        latent_trajectory_length: int,
        prompt_max_length: int,
        max_new_tokens: int,
        device: str,
        torch_dtype: str,
        top_p: float,
    ):
        self.prompt_max_length = prompt_max_length
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print(f"Loading base model: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=DTYPE_MAP[torch_dtype],
            device_map=_build_device_map(device),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.input_device = _model_input_device(self.model)

        hidden_size = _resolve_hidden_size(self.model.config)
        print(f"Loading reasoning network: {reasoning_net_path}")
        self.reasoning_network = TransformerReasoningNet(
            reasoning_net_path,
            latent_trajectory_length=latent_trajectory_length,
            hidden_size=hidden_size,
        )
        self.reasoning_network.to(self.input_device)
        self.reasoning_network.eval()
        _load_reasoning_weights(self.reasoning_network, checkpoint_path)

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str], *, temperature: float) -> list[str]:
        prompt_texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]
        inputs = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.prompt_max_length,
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(self.input_device)
        attention_mask = inputs["attention_mask"].to(self.input_device)

        prompt_embeddings = self.model.get_input_embeddings()(input_ids)
        prompt_embeddings = prompt_embeddings.to(self.model.dtype)

        hidden_states = _last_layer_hidden_states(
            self.model,
            prompt_embeddings,
            attention_mask,
        )

        reasoning_device = next(self.reasoning_network.parameters()).device
        latent_trajectory = self.reasoning_network(
            hidden_states.to(reasoning_device),
            attention_mask=attention_mask.to(reasoning_device),
        ).to(prompt_embeddings.device, dtype=prompt_embeddings.dtype)

        latent_mask = torch.ones(
            latent_trajectory.size(0),
            latent_trajectory.size(1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_embeds = torch.cat([prompt_embeddings, latent_trajectory], dim=1)
        combined_mask = torch.cat([attention_mask, latent_mask], dim=1)

        generate_kwargs: dict[str, Any] = {
            "inputs_embeds": combined_embeds,
            "attention_mask": combined_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": temperature > 0,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = self.top_p

        output_ids = self.model.generate(**generate_kwargs)
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def _resolve_split(dataset_obj, split: str):
    if isinstance(dataset_obj, DatasetDict):
        if split not in dataset_obj:
            raise KeyError(f"Split '{split}' not found. Available: {list(dataset_obj.keys())}")
        return dataset_obj[split]
    return dataset_obj


def _load_from_local_or_hub(spec: BenchmarkSpec, data_root: Path):
    local_path = data_root / spec.dataset_id
    if local_path.exists():
        try:
            return _resolve_split(load_from_disk(str(local_path)), spec.split)
        except Exception:
            if spec.config_name is None:
                return load_dataset(str(local_path), split=spec.split)
            return load_dataset(str(local_path), spec.config_name, split=spec.split)

    if spec.config_name is None:
        return load_dataset(spec.dataset_id, split=spec.split)
    return load_dataset(spec.dataset_id, spec.config_name, split=spec.split)


def _extract_boxed_answer(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    cursor = start + len(marker)
    depth = 1
    chars: list[str] = []
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        cursor += 1
    return None


def _gsm8k_final_answer(raw_answer: str) -> str:
    answer = raw_answer.split("####")[-1].strip()
    answer = re.sub(r"^[=$\s]+", "", answer)
    return answer.replace(",", "").strip()


def _math500_final_answer(example: dict[str, Any]) -> str:
    for key in ("answer", "final_answer", "target"):
        value = example.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    solution = str(example.get("solution", "")).strip()
    boxed = _extract_boxed_answer(solution)
    if boxed:
        return boxed
    return solution


def _project_example(dataset_name: str, example: dict[str, Any], *, add_math_suffix: bool) -> dict[str, str]:
    if dataset_name == "math500":
        problem = str(example.get("problem") or example.get("question") or "").strip()
        solution = _math500_final_answer(example)
    elif dataset_name == "gsm8k":
        problem = str(example.get("question") or example.get("problem") or "").strip()
        solution = _gsm8k_final_answer(str(example["answer"]))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if add_math_suffix:
        problem = f"{problem}{MATH_SUFFIX}"
    return {"problem": problem, "solution": solution}


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Prediction file already exists: {path}. Use --overwrite.")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_dataset(
    *,
    evaluator: LatentReasoningEvaluator,
    dataset_name: str,
    data_root: Path,
    batch_size: int,
    temperature: float,
    limit: int | None,
    add_math_suffix: bool,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    spec = BENCHMARKS[dataset_name]
    raw_dataset = _load_from_local_or_hub(spec, data_root)
    total_available = len(raw_dataset)
    total = min(total_available, limit) if limit is not None else total_available

    records: list[dict[str, Any]] = []
    correct = 0
    started = perf_counter()
    print(f"\nEvaluating {spec.display_name}: {total} examples")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        examples = [dict(raw_dataset[i]) for i in range(start, end)]
        projected = [
            _project_example(dataset_name, example, add_math_suffix=add_math_suffix)
            for example in examples
        ]

        predictions = evaluator.generate_batch(
            [item["problem"] for item in projected],
            temperature=temperature,
        )
        rewards = accuracy_reward(
            completions=[
                [{"role": "assistant", "content": prediction}]
                for prediction in predictions
            ],
            solution=[item["solution"] for item in projected],
        )

        for offset, (example, item, prediction, reward) in enumerate(
            zip(examples, projected, predictions, rewards)
        ):
            is_correct = bool(reward is not None and float(reward) > 0.0)
            correct += int(is_correct)
            records.append(
                {
                    "dataset": spec.display_name,
                    "index": start + offset,
                    "id": example.get("unique_id") or example.get("id") or start + offset,
                    "problem": item["problem"],
                    "solution": item["solution"],
                    "prediction": prediction,
                    "reward": None if reward is None else float(reward),
                    "correct": is_correct,
                }
            )

        accuracy = 100.0 * correct / end
        print(f"  {end:>5}/{total:<5} accuracy={accuracy:6.2f}%", flush=True)

    elapsed = perf_counter() - started
    output_path = output_dir / f"{dataset_name}_predictions.jsonl"
    _write_jsonl(output_path, records, overwrite=overwrite)

    metrics = {
        "dataset": spec.display_name,
        "correct": correct,
        "total": total,
        "accuracy": 100.0 * correct / total if total else 0.0,
        "elapsed_seconds": elapsed,
        "predictions_path": str(output_path),
    }
    return metrics


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).expanduser()

    evaluator = LatentReasoningEvaluator(
        model_path=args.model_path,
        reasoning_net_path=args.reasoning_net_path,
        checkpoint_path=args.checkpoint_path,
        latent_trajectory_length=args.latent_trajectory_length,
        prompt_max_length=args.prompt_max_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        torch_dtype=args.torch_dtype,
        top_p=args.top_p,
    )

    metrics: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        metrics.append(
            evaluate_dataset(
                evaluator=evaluator,
                dataset_name=dataset_name,
                data_root=data_root,
                batch_size=args.batch_size,
                temperature=args.temperature,
                limit=args.limit,
                add_math_suffix=not args.no_math_suffix,
                output_dir=output_dir,
                overwrite=args.overwrite,
            )
        )

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Metrics file already exists: {metrics_path}. Use --overwrite.")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print("\nTable 1 Ours evaluation")
    print("| Dataset | Correct | Total | Accuracy (%) |")
    print("|---|---:|---:|---:|")
    for item in metrics:
        print(
            f"| {item['dataset']} | {item['correct']} | {item['total']} | "
            f"{item['accuracy']:.2f} |"
        )
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
