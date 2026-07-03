#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from glob import glob
from pathlib import Path
from time import perf_counter
from typing import Any, Optional
from dataclasses import dataclass

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.alr_difficulty import DifficultyEstimator, normalize_latent_lengths
from modeling.alr_stage1 import LengthElasticTransformerReasoningNet
from utils.reward_func import accuracy_reward


DEFAULT_LATENT_TRAJECTORY_LENGTHS = [64, 128, 192, 256]
DEFAULT_PROMPT_TEMPLATE = (
    "{problem} Let's think step by step and output the final answer within \\boxed{}."
)
BUILTIN_TASKS = {"math500", "gsm8k"}


@dataclass(frozen=True)
class DistributedEnv:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def get_distributed_env() -> DistributedEnv:
    return DistributedEnv(
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )


def setup_distributed(env: DistributedEnv) -> None:
    if not env.is_distributed:
        return
    import torch.distributed as dist

    if torch.cuda.is_available():
        torch.cuda.set_device(env.local_rank)
    if dist.is_available() and not dist.is_initialized():
        backend = os.environ.get("ALR_EVAL_DIST_BACKEND", "gloo")
        dist.init_process_group(backend=backend, init_method="env://")


def distributed_barrier(env: DistributedEnv) -> None:
    if not env.is_distributed:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(env: DistributedEnv) -> None:
    if not env.is_distributed:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def progress(iterable, desc=None):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc)


def parse_latent_trajectory_lengths(value: Optional[str]) -> list[int]:
    if not value:
        return list(DEFAULT_LATENT_TRAJECTORY_LENGTHS)
    tokens = [token for token in re.split(r"[,\s]+", value.strip()) if token]
    return normalize_latent_lengths([int(token) for token in tokens])


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_latent_trajectory_lengths(args) -> list[int]:
    if args.latent_trajectory_lengths:
        return parse_latent_trajectory_lengths(args.latent_trajectory_lengths)

    difficulty_config = load_json_if_exists(
        Path(args.difficulty_checkpoint_path) / "alr_difficulty_config.json",
    )
    if difficulty_config.get("latent_trajectory_lengths"):
        return normalize_latent_lengths(difficulty_config["latent_trajectory_lengths"])

    stage1_config = load_json_if_exists(
        Path(args.stage1_checkpoint_path) / "alr_stage1_config.json",
    )
    if stage1_config.get("latent_trajectory_lengths"):
        return normalize_latent_lengths(stage1_config["latent_trajectory_lengths"])

    return list(DEFAULT_LATENT_TRAJECTORY_LENGTHS)


def resolve_hidden_size(config) -> int:
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    raise ValueError("Failed to resolve hidden_size from the base model config.")


def resolve_device_map(device: str, env: Optional[DistributedEnv] = None):
    if device == "auto":
        if not torch.cuda.is_available():
            return {"": "cpu"}
        if env is not None and env.is_distributed:
            return {"": env.local_rank}
        return {"": 0}
    if device == "cuda":
        if env is not None and env.is_distributed:
            return {"": env.local_rank}
        return {"": 0}
    if device.startswith("cuda:"):
        return {"": int(device.split(":", 1)[1])}
    return {"": device}


def dtype_from_arg(value: str):
    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if value not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return dtype_map[value]


def model_input_device(model):
    return model.get_input_embeddings().weight.device


def load_reasoning_weights(reasoning_network, checkpoint_path: str) -> None:
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Stage1 checkpoint path does not exist: {checkpoint_path}")

    target_keys = set(reasoning_network.state_dict().keys())
    state_dict = {}
    for filename in sorted(glob(str(checkpoint_dir / "*.safetensors"))):
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("reasoning_network."):
                    state_dict[key.removeprefix("reasoning_network.")] = handle.get_tensor(key)
                elif key in target_keys:
                    state_dict[key] = handle.get_tensor(key)

    if not state_dict:
        pytorch_model = checkpoint_dir / "pytorch_model.bin"
        if pytorch_model.exists():
            raw_state = torch.load(str(pytorch_model), map_location="cpu")
            for key, value in raw_state.items():
                if key.startswith("reasoning_network."):
                    state_dict[key.removeprefix("reasoning_network.")] = value
                elif key in target_keys:
                    state_dict[key] = value

    if not state_dict:
        raise ValueError(f"No reasoning-network weights found in {checkpoint_path}")

    reasoning_network.load_state_dict(state_dict, strict=True)
    print(f"Loaded {len(state_dict)} reasoning-network tensors from {checkpoint_path}")


def last_layer_hidden_states(model, inputs_embeds, attention_mask):
    captured = {}

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

    def hook(_module, _inputs, output):
        captured["hidden_states"] = output[0] if isinstance(output, tuple) else output

    handle = last_layer.register_forward_hook(hook)
    try:
        base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    finally:
        handle.remove()

    if "hidden_states" not in captured:
        raise RuntimeError("Failed to capture last-layer hidden states.")
    return captured["hidden_states"]


def parse_method(method: str) -> tuple[str, Optional[int]]:
    normalized = method.strip().lower().replace("_", "-")
    if normalized in {"adaptive", "random"}:
        return normalized, None
    if normalized.startswith("fixed-"):
        return "fixed", int(normalized.split("-", 1)[1])
    if normalized.startswith("fixed:"):
        return "fixed", int(normalized.split(":", 1)[1])
    raise ValueError(
        f"Unknown method '{method}'. Use fixed-64, fixed-128, fixed-192, "
        "fixed-256, random, or adaptive."
    )


def sanitize_name(value: str) -> str:
    value = value.strip().strip("/")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value or "unnamed"


class ALRBatchGenerator:
    def __init__(
        self,
        args,
        latent_trajectory_lengths: list[int],
        needs_adaptive: bool,
        env: Optional[DistributedEnv] = None,
    ):
        self.args = args
        self.latent_trajectory_lengths = latent_trajectory_lengths
        self.rng = random.Random(args.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=not args.no_trust_remote_code,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print(f"Loading base LLM: {args.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype_from_arg(args.torch_dtype),
            device_map=resolve_device_map(args.device, env),
            trust_remote_code=not args.no_trust_remote_code,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.input_device = model_input_device(self.model)

        print(f"Loading ALR Stage1 reasoning network: {args.reasoning_net_path}")
        self.reasoning_network = LengthElasticTransformerReasoningNet(
            args.reasoning_net_path,
            latent_trajectory_length=max(self.latent_trajectory_lengths),
            hidden_size=resolve_hidden_size(self.model.config),
        )
        self.reasoning_network.to(self.input_device)
        self.reasoning_network.eval()
        load_reasoning_weights(self.reasoning_network, args.stage1_checkpoint_path)

        self.difficulty_estimator = None
        if needs_adaptive:
            print(f"Loading ALR Stage2 difficulty estimator: {args.difficulty_checkpoint_path}")
            self.difficulty_estimator = DifficultyEstimator.from_pretrained(
                args.difficulty_checkpoint_path,
                map_location="cpu",
            )
            if self.difficulty_estimator.latent_trajectory_lengths != self.latent_trajectory_lengths:
                raise ValueError(
                    "Difficulty estimator length set does not match evaluation length set: "
                    f"{self.difficulty_estimator.latent_trajectory_lengths} vs "
                    f"{self.latent_trajectory_lengths}."
                )
            self.difficulty_estimator.to(self.input_device)
            self.difficulty_estimator.eval()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def render_prompts(self, prompts: list[str]) -> list[str]:
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]

    @torch.no_grad()
    def generate_batch(self, prompts: list[str], method: str) -> dict[str, Any]:
        method_kind, fixed_length = parse_method(method)
        prompt_texts = self.render_prompts(prompts)
        inputs = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.prompt_max_length,
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(self.input_device)
        attention_mask = inputs["attention_mask"].to(self.input_device)
        prompt_embeddings = self.model.get_input_embeddings()(input_ids).to(self.model.dtype)
        hidden_states = last_layer_hidden_states(
            self.model,
            prompt_embeddings,
            attention_mask,
        )

        probabilities: list[dict[str, float] | None] = [None for _ in prompts]
        predicted_labels: list[int | None] = [None for _ in prompts]
        if method_kind == "fixed":
            if fixed_length not in self.latent_trajectory_lengths:
                raise ValueError(
                    f"Fixed length {fixed_length} is not one of {self.latent_trajectory_lengths}."
                )
            latent_lengths = [int(fixed_length) for _ in prompts]
        elif method_kind == "random":
            latent_lengths = [
                self.rng.choice(self.latent_trajectory_lengths)
                for _ in prompts
            ]
        else:
            if self.difficulty_estimator is None:
                raise ValueError("Adaptive evaluation requires a difficulty checkpoint.")
            pooled = DifficultyEstimator.mean_pool_hidden_states(
                hidden_states,
                attention_mask,
            )
            logits = self.difficulty_estimator(pooled_hidden_states=pooled.float())
            probs = torch.softmax(logits, dim=-1)
            labels = logits.argmax(dim=-1)
            latent_lengths = [
                self.difficulty_estimator.label_to_length(int(label))
                for label in labels.cpu().tolist()
            ]
            predicted_labels = [int(label) for label in labels.cpu().tolist()]
            probabilities = [
                {
                    str(length): float(prob)
                    for length, prob in zip(
                        self.latent_trajectory_lengths,
                        row.detach().cpu().tolist(),
                    )
                }
                for row in probs
            ]

        completions = ["" for _ in prompts]
        output_token_counts = [0 for _ in prompts]
        for latent_length in sorted(set(latent_lengths)):
            indices = [
                index
                for index, length in enumerate(latent_lengths)
                if length == latent_length
            ]
            index_tensor = torch.tensor(indices, dtype=torch.long, device=self.input_device)
            batch_completions, batch_token_counts = self._generate_fixed_prefilled(
                prompt_embeddings.index_select(0, index_tensor),
                hidden_states.index_select(0, index_tensor),
                attention_mask.index_select(0, index_tensor),
                latent_length,
            )
            for local_idx, global_idx in enumerate(indices):
                completions[global_idx] = batch_completions[local_idx]
                output_token_counts[global_idx] = batch_token_counts[local_idx]

        return {
            "completions": completions,
            "latent_lengths": latent_lengths,
            "predicted_labels": predicted_labels,
            "length_probabilities": probabilities,
            "output_token_counts": output_token_counts,
            "prompt_texts": prompt_texts,
        }

    def _generate_fixed_prefilled(
        self,
        prompt_embeddings,
        hidden_states,
        attention_mask,
        latent_length: int,
    ) -> tuple[list[str], list[int]]:
        latent_trajectory = self.reasoning_network(
            hidden_states,
            attention_mask=attention_mask,
            active_latent_length=latent_length,
        ).to(prompt_embeddings.dtype)
        latent_mask = torch.ones(
            latent_trajectory.size(0),
            latent_trajectory.size(1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_embeds = torch.cat([prompt_embeddings, latent_trajectory], dim=1)
        combined_mask = torch.cat([attention_mask, latent_mask], dim=1)

        generate_kwargs = {
            "inputs_embeds": combined_embeds,
            "attention_mask": combined_mask,
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": self.args.temperature > 0,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.args.temperature > 0:
            generate_kwargs["temperature"] = self.args.temperature
            generate_kwargs["top_p"] = self.args.top_p

        output_ids = self.model.generate(**generate_kwargs)
        completions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if self.tokenizer.pad_token_id is None:
            output_token_counts = [len(row) for row in output_ids]
        else:
            output_token_counts = [
                int(row.ne(self.tokenizer.pad_token_id).sum().item())
                for row in output_ids
            ]
        return completions, output_token_counts


def first_nonempty(example: dict[str, Any], keys: tuple[str, ...]) -> Optional[Any]:
    for key in keys:
        value = example.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def project_math500(example: dict[str, Any]) -> dict[str, Any]:
    target = first_nonempty(example, ("answer", "final_answer", "target"))
    if target is None:
        raise ValueError("MATH-500 example is missing an answer field.")
    return {
        "problem": str(example.get("problem") or example.get("question") or "").strip(),
        "target_answer": str(target).strip(),
        "metadata": {
            "subject": example.get("subject"),
            "level": example.get("level"),
            "unique_id": example.get("unique_id"),
        },
    }


def project_gsm8k(example: dict[str, Any]) -> dict[str, Any]:
    raw_answer = str(example.get("answer", "")).strip()
    target = raw_answer.split("####")[-1].strip()
    target = target.lstrip("=$ ").replace(",", "").strip()
    return {
        "problem": str(example.get("question", "")).strip(),
        "target_answer": target,
        "metadata": {},
    }


def load_dataset_with_split_fallback(dataset_id: str, config_name: Optional[str], splits: list[str]):
    from datasets import load_dataset

    last_error = None
    for split in splits:
        try:
            if config_name is None:
                return load_dataset(dataset_id, split=split)
            return load_dataset(dataset_id, config_name, split=split)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Failed to load {dataset_id} with splits {splits}: {last_error}"
    )


def load_builtin_task(task_name: str) -> list[dict[str, Any]]:
    if task_name == "math500":
        dataset = load_dataset_with_split_fallback(
            "HuggingFaceH4/MATH-500",
            None,
            ["test", "train"],
        )
        return [project_math500(dict(dataset[i])) for i in range(len(dataset))]

    if task_name == "gsm8k":
        dataset = load_dataset_with_split_fallback(
            "openai/gsm8k",
            "main",
            ["test", "validation"],
        )
        return [project_gsm8k(dict(dataset[i])) for i in range(len(dataset))]

    raise ValueError(f"Unknown built-in task: {task_name}")


def load_local_task(data_dir: Path, task_name: str) -> list[dict[str, Any]]:
    path = Path(task_name)
    if not path.exists():
        path = data_dir / (task_name if task_name.endswith(".json") else f"{task_name}.json")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ("data", "examples", "records", "items", "results"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list or a dict with a data list.")

    records = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Local task {path} contains a non-object record.")
        problem = first_nonempty(item, ("problem", "question", "prompt", "input"))
        target = first_nonempty(item, ("target_answer", "answer", "solution", "target"))
        if problem is None or target is None:
            raise ValueError(
                f"Local task {path} requires problem/question and answer fields."
            )
        records.append(
            {
                "problem": str(problem).strip(),
                "target_answer": str(target).strip(),
                "metadata": {
                    key: value
                    for key, value in item.items()
                    if key not in {"problem", "question", "prompt", "input", "target_answer", "answer", "solution", "target"}
                },
            }
        )
    return records


def load_task(args, task_name: str) -> list[dict[str, Any]]:
    data_dir = Path(args.data_dir)
    local_path = data_dir / f"{task_name}.json"
    if local_path.exists():
        return load_local_task(data_dir, task_name)
    if task_name in BUILTIN_TASKS:
        return load_builtin_task(task_name)
    return load_local_task(data_dir, task_name)


def extract_boxed(text: str) -> Optional[str]:
    simple_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if simple_matches:
        return simple_matches[-1].strip()

    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    cursor = start + len("\\boxed{")
    depth = 1
    chars = []
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


def normalize_text_answer(text: str) -> str:
    text = text.strip().lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text


def score_completion(completion: str, target_answer: str) -> tuple[Optional[float], bool, Optional[str]]:
    extracted = extract_boxed(completion)
    try:
        reward = accuracy_reward(
            [[{"role": "assistant", "content": completion}]],
            [target_answer],
        )[0]
    except Exception:
        reward = None

    if reward is None:
        candidate = extracted if extracted is not None else completion
        correct = normalize_text_answer(candidate) == normalize_text_answer(target_answer)
        reward = 1.0 if correct else 0.0
    else:
        correct = bool(reward == 1.0)

    return reward, correct, extracted


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def evaluate_task(
    generator: ALRBatchGenerator,
    records: list[dict[str, Any]],
    task_name: str,
    method: str,
    args,
    record_indices: Optional[list[int]] = None,
):
    if args.limit is not None:
        records = records[: args.limit]
        if record_indices is not None:
            record_indices = record_indices[: args.limit]
    if not records:
        return [], summarize_rows([], task_name, method, generator.latent_trajectory_lengths)
    if record_indices is None:
        record_indices = list(range(len(records)))
    if len(record_indices) != len(records):
        raise ValueError("record_indices and records must have the same length.")

    result_rows = []
    started = perf_counter()
    for start in progress(range(0, len(records), args.batch_size), desc=f"{method}/{task_name}"):
        batch = records[start : start + args.batch_size]
        batch_indices = record_indices[start : start + args.batch_size]
        prompts = [
            args.prompt_template.format(problem=record["problem"])
            for record in batch
        ]
        generation = generator.generate_batch(prompts, method)
        for offset, record in enumerate(batch):
            completion = generation["completions"][offset]
            score, correct, extracted_answer = score_completion(
                completion,
                str(record["target_answer"]),
            )
            latent_length = int(generation["latent_lengths"][offset])
            row = {
                "index": int(batch_indices[offset]),
                "task": task_name,
                "method": method,
                "problem": record["problem"],
                "target_answer": str(record["target_answer"]),
                "completion": completion,
                "score": score,
                "correct": correct,
                "extracted_answer": extracted_answer,
                "latent_length": latent_length,
                "predicted_length": latent_length,
                "predicted_label": generation["predicted_labels"][offset],
                "length_probabilities": generation["length_probabilities"][offset],
                "output_tokens": int(generation["output_token_counts"][offset]),
                "metadata": record.get("metadata", {}),
            }
            if args.save_prompts:
                row["rendered_prompt"] = generation["prompt_texts"][offset]
            result_rows.append(row)

    elapsed = perf_counter() - started
    summary = summarize_rows(
        result_rows,
        task_name,
        method,
        generator.latent_trajectory_lengths,
        elapsed_seconds=elapsed,
    )
    return result_rows, summary


def summarize_rows(
    result_rows: list[dict[str, Any]],
    task_name: str,
    method: str,
    latent_trajectory_lengths: list[int],
    elapsed_seconds: Optional[float] = None,
):
    total = len(result_rows)
    if total == 0:
        return {
            "task": task_name,
            "method": method,
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "avg_latent_length": 0.0,
            "latent_cost_saving_vs_fixed_max": 0.0,
            "avg_output_tokens": 0.0,
            "latent_length_distribution": {},
            "elapsed_seconds": elapsed_seconds,
        }
    correct_count = sum(1 for row in result_rows if row["correct"])
    avg_latent_length = sum(row["latent_length"] for row in result_rows) / total
    avg_output_tokens = sum(row["output_tokens"] for row in result_rows) / total
    max_length = max(latent_trajectory_lengths)
    latent_distribution = Counter(str(row["latent_length"]) for row in result_rows)
    return {
        "task": task_name,
        "method": method,
        "total": total,
        "correct": correct_count,
        "accuracy": correct_count / total,
        "avg_latent_length": avg_latent_length,
        "latent_cost_saving_vs_fixed_max": 1.0 - (avg_latent_length / max_length),
        "avg_output_tokens": avg_output_tokens,
        "latent_length_distribution": dict(sorted(latent_distribution.items())),
        "elapsed_seconds": elapsed_seconds,
    }


def select_rank_records(records: list[dict[str, Any]], env: DistributedEnv):
    indexed = list(enumerate(records))
    selected = indexed[env.rank :: env.world_size]
    return [index for index, _record in selected], [record for _index, record in selected]


def shard_dir_for(output_dir: Path, method: str, task_name: str, world_size: int) -> Path:
    return output_dir / f"{sanitize_name(method)}_{sanitize_name(task_name)}_shards_world{world_size}"


def merge_result_shards(shard_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in shard_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing result shard: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Result shard must contain a JSON list: {path}")
        rows.extend(payload)
    rows.sort(key=lambda row: int(row["index"]))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ALR fixed, random, and adaptive latent lengths.",
    )
    parser.add_argument(
        "--model_path",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    parser.add_argument("--reasoning_net_path", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument(
        "--stage1_checkpoint_path",
        default="checkpoints/ALR-Stage1-DSR1-Qwen-1.5B",
    )
    parser.add_argument(
        "--difficulty_checkpoint_path",
        default="checkpoints/ALR-Stage2-Difficulty-DSR1-Qwen-1.5B",
    )
    parser.add_argument("--latent_trajectory_lengths", default=None)
    parser.add_argument("--data_dir", default="data/eval/math")
    parser.add_argument("--output_dir", default="eval_outputs/alr_full_1.5B")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "math500"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["fixed-64", "fixed-128", "fixed-192", "fixed-256", "random", "adaptive"],
    )
    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--prompt_max_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Device for generation. Use 'auto' with accelerate single-node "
            "multi-GPU launch so each rank uses its LOCAL_RANK GPU."
        ),
    )
    parser.add_argument("--torch_dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--no_trust_remote_code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    env = get_distributed_env()
    setup_distributed(env)
    if env.is_distributed:
        print(
            f"distributed_eval: rank={env.rank} local_rank={env.local_rank} "
            f"world_size={env.world_size}",
            flush=True,
        )
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    methods = [method.strip() for method in args.methods if method.strip()]
    parsed_methods = [parse_method(method) for method in methods]
    needs_adaptive = any(kind == "adaptive" for kind, _ in parsed_methods)
    latent_trajectory_lengths = resolve_latent_trajectory_lengths(args)
    if env.is_main_process:
        print(f"latent_trajectory_lengths={latent_trajectory_lengths}")

    output_dir = Path(args.output_dir)
    if env.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(env)
    generator = ALRBatchGenerator(
        args,
        latent_trajectory_lengths=latent_trajectory_lengths,
        needs_adaptive=needs_adaptive,
        env=env,
    )

    summary = {
        "config": {
            "model_path": args.model_path,
            "reasoning_net_path": args.reasoning_net_path,
            "stage1_checkpoint_path": args.stage1_checkpoint_path,
            "difficulty_checkpoint_path": args.difficulty_checkpoint_path,
            "latent_trajectory_lengths": latent_trajectory_lengths,
            "tasks": args.tasks,
            "methods": methods,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "prompt_max_length": args.prompt_max_length,
        },
        "results": {},
    }

    for task_name in args.tasks:
        if env.is_main_process:
            print(f"\nLoading task: {task_name}")
        records = load_task(args, task_name)
        if args.limit is not None:
            records = records[: args.limit]
            if env.is_main_process:
                print(f"Using first {len(records)} examples for {task_name}")

        for method in methods:
            result_path = output_dir / f"{sanitize_name(method)}_{sanitize_name(task_name)}_results.json"
            if result_path.exists() and not args.overwrite:
                if env.is_main_process:
                    print(f"Skipping existing result file: {result_path}")
                    with result_path.open("r", encoding="utf-8") as handle:
                        rows = json.load(handle)
                    method_summary = summarize_rows(
                        rows,
                        task_name,
                        method,
                        latent_trajectory_lengths,
                        elapsed_seconds=None,
                    )
                distributed_barrier(env)
            else:
                rank_indices, rank_records = select_rank_records(records, env)
                rows, _rank_summary = evaluate_task(
                    generator,
                    rank_records,
                    task_name,
                    method,
                    args,
                    record_indices=rank_indices,
                )
                if env.is_distributed:
                    shard_dir = shard_dir_for(output_dir, method, task_name, env.world_size)
                    shard_dir.mkdir(parents=True, exist_ok=True)
                    shard_path = shard_dir / f"rank{env.rank:05d}.json"
                    save_json(shard_path, rows)
                    print(f"rank={env.rank} saved {len(rows)} rows to {shard_path}", flush=True)
                    distributed_barrier(env)
                    if env.is_main_process:
                        shard_paths = [
                            shard_dir / f"rank{rank:05d}.json"
                            for rank in range(env.world_size)
                        ]
                        merged_rows = merge_result_shards(shard_paths)
                        method_summary = summarize_rows(
                            merged_rows,
                            task_name,
                            method,
                            latent_trajectory_lengths,
                        )
                        save_json(result_path, merged_rows)
                        print(f"Saved {len(merged_rows)} merged rows to {result_path}")
                    distributed_barrier(env)
                else:
                    method_summary = summarize_rows(
                        rows,
                        task_name,
                        method,
                        latent_trajectory_lengths,
                    )
                    save_json(result_path, rows)
                    print(f"Saved {len(rows)} rows to {result_path}")

            if env.is_main_process:
                method_summary["output_path"] = str(result_path)
                summary["results"].setdefault(method, {})[task_name] = method_summary
                print(
                    "{method}/{task}: acc={acc:.4f} avg_latent={latent:.2f} "
                    "saving={saving:.2%}".format(
                        method=method,
                        task=task_name,
                        acc=method_summary["accuracy"],
                        latent=method_summary["avg_latent_length"],
                        saving=method_summary["latent_cost_saving_vs_fixed_max"],
                    )
                )

    if env.is_main_process:
        summary_path = output_dir / "summary.json"
        save_json(summary_path, summary)
        print(f"\nSaved summary to {summary_path}")
    cleanup_distributed(env)


if __name__ == "__main__":
    main()
