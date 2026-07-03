#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Any, Callable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.alr_difficulty import DifficultyEstimator, normalize_latent_lengths


DEFAULT_LATENT_TRAJECTORY_LENGTHS = [64, 128, 192, 256]
MATH_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."


@dataclass(frozen=True)
class Stage2DatasetSpec:
    name: str
    dataset_id: str
    config_name: Optional[str]
    split: str
    target_length: int
    problem_field: str


STAGE2_DATASETS = [
    Stage2DatasetSpec(
        name="gsm8k",
        dataset_id="openai/gsm8k",
        config_name="main",
        split="train",
        target_length=64,
        problem_field="question",
    ),
    Stage2DatasetSpec(
        name="math_minus_math500",
        dataset_id="rasbt/math_full_minus_math500",
        config_name=None,
        split="train",
        target_length=128,
        problem_field="problem",
    ),
    Stage2DatasetSpec(
        name="deepscaler_preview",
        dataset_id="agentica-org/DeepScaleR-Preview-Dataset",
        config_name=None,
        split="train",
        target_length=192,
        problem_field="problem",
    ),
    Stage2DatasetSpec(
        name="olympiads",
        dataset_id="Metaskepsis/Olympiads",
        config_name=None,
        split="train",
        target_length=256,
        problem_field="problem",
    ),
]


def parse_latent_trajectory_lengths(value: Optional[str]) -> list[int]:
    if not value:
        return list(DEFAULT_LATENT_TRAJECTORY_LENGTHS)
    tokens = [token for token in re.split(r"[,\s]+", value.strip()) if token]
    return normalize_latent_lengths([int(token) for token in tokens])


def load_stage1_config(checkpoint_path: Optional[str]) -> dict[str, Any]:
    if not checkpoint_path:
        return {}
    config_path = Path(checkpoint_path) / "alr_stage1_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_latent_trajectory_lengths(args) -> list[int]:
    if args.latent_trajectory_lengths:
        return parse_latent_trajectory_lengths(args.latent_trajectory_lengths)
    config = load_stage1_config(args.stage1_checkpoint_path)
    if config.get("latent_trajectory_lengths"):
        return normalize_latent_lengths(config["latent_trajectory_lengths"])
    return list(DEFAULT_LATENT_TRAJECTORY_LENGTHS)


def resolve_hidden_size(config) -> int:
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    raise ValueError("Failed to resolve hidden_size from the base model config.")


def resolve_device_map(device: str):
    if device == "auto":
        return "auto"
    if device == "cuda":
        return {"": 0}
    if device.startswith("cuda:"):
        return {"": int(device.split(":", 1)[1])}
    return {"": device}


def model_input_device(model):
    return model.get_input_embeddings().weight.device


def dtype_from_arg(value: str):
    import torch

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


def load_dataset_split(spec: Stage2DatasetSpec):
    from datasets import load_dataset

    if spec.config_name is None:
        return load_dataset(spec.dataset_id, split=spec.split)
    return load_dataset(spec.dataset_id, spec.config_name, split=spec.split)


def project_problem(example: dict[str, Any], field_name: str) -> str:
    problem = str(example.get(field_name, "")).strip()
    if not problem:
        raise ValueError(f"Missing problem text in field '{field_name}'.")
    return problem + MATH_SUFFIX


def build_records(
    latent_trajectory_lengths: list[int],
    sample_per_class: int,
    seed: int,
    limit_per_class: Optional[int],
) -> list[dict[str, Any]]:
    if set(DEFAULT_LATENT_TRAJECTORY_LENGTHS) - set(latent_trajectory_lengths):
        raise ValueError(
            "Default Stage2 dataset labels require latent lengths "
            f"{DEFAULT_LATENT_TRAJECTORY_LENGTHS}; got {latent_trajectory_lengths}."
        )

    records: list[dict[str, Any]] = []
    for spec in STAGE2_DATASETS:
        dataset = load_dataset_split(spec)
        desired = limit_per_class or sample_per_class
        take = min(int(desired), len(dataset))
        if take < desired:
            print(
                f"Warning: requested {desired} examples for {spec.name}, "
                f"but only {take} are available.",
                flush=True,
            )

        shuffled = dataset.shuffle(seed=seed)
        for example in shuffled.select(range(take)):
            records.append(
                {
                    "problem": project_problem(dict(example), spec.problem_field),
                    "target_length": spec.target_length,
                    "label": latent_trajectory_lengths.index(spec.target_length),
                    "source": spec.name,
                    "dataset_id": spec.dataset_id,
                }
            )

    rng = random.Random(seed)
    rng.shuffle(records)
    return records


def stratified_split(
    records: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_label.setdefault(int(record["label"]), []).append(record)

    rng = random.Random(seed)
    train_records = []
    validation_records = []
    for label, group in sorted(by_label.items()):
        del label
        rng.shuffle(group)
        validation_count = max(1, int(round(len(group) * validation_ratio)))
        validation_count = min(validation_count, max(len(group) - 1, 1))
        validation_records.extend(group[:validation_count])
        train_records.extend(group[validation_count:])

    rng.shuffle(train_records)
    rng.shuffle(validation_records)
    return train_records, validation_records


def render_prompts(tokenizer, records: list[dict[str, Any]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": record["problem"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for record in records
    ]


def cache_features_for_records(args, records: list[dict[str, Any]], output_path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=not args.no_trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base LLM for feature caching: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_arg(args.torch_dtype),
        device_map=resolve_device_map(args.device),
        trust_remote_code=not args.no_trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    input_device = model_input_device(model)
    all_features = []
    all_labels = []
    all_lengths = []
    all_sources = []

    prompts = render_prompts(tokenizer, records)
    with torch.inference_mode():
        for start in range(0, len(records), args.cache_batch_size):
            end = min(start + args.cache_batch_size, len(records))
            batch_prompts = prompts[start:end]
            batch_records = records[start:end]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.prompt_max_length,
                add_special_tokens=False,
            )
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            prompt_embeddings = model.get_input_embeddings()(input_ids).to(model.dtype)
            hidden_states = last_layer_hidden_states(
                model,
                prompt_embeddings,
                attention_mask,
            )
            pooled = DifficultyEstimator.mean_pool_hidden_states(
                hidden_states,
                attention_mask,
            )
            all_features.append(pooled.detach().float().cpu())
            all_labels.extend(int(record["label"]) for record in batch_records)
            all_lengths.extend(int(record["target_length"]) for record in batch_records)
            all_sources.extend(str(record["source"]) for record in batch_records)
            print(f"Cached {end}/{len(records)} examples", flush=True)

    payload = {
        "features": torch.cat(all_features, dim=0),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "target_lengths": torch.tensor(all_lengths, dtype=torch.long),
        "sources": all_sources,
    }
    torch.save(payload, output_path)
    print(f"Saved cached features to {output_path}")


def load_or_cache_features(args, split_name: str, records: list[dict[str, Any]]):
    import torch

    cache_path = Path(args.cache_dir) / f"{split_name}_features.pt"
    if cache_path.exists() and not args.overwrite_cache:
        print(f"Loading cached {split_name} features from {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        cached_count = int(payload["features"].size(0))
        if cached_count == len(records):
            return payload
        print(
            f"Cached {split_name} count ({cached_count}) does not match the "
            f"requested count ({len(records)}); recomputing.",
            flush=True,
        )

    cache_features_for_records(args, records, cache_path)
    return torch.load(cache_path, map_location="cpu")


def make_loader(features, labels, batch_size: int, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_head(model, features, labels, batch_size: int, device):
    import torch
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    num_classes = model.num_classes
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    loader = make_loader(features, labels, batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(pooled_hidden_states=batch_features)
            loss = F.cross_entropy(logits, batch_labels)
            predictions = logits.argmax(dim=-1)
            total_loss += float(loss.item()) * batch_labels.numel()
            total += batch_labels.numel()
            correct += int((predictions == batch_labels).sum().item())
            for gold, pred in zip(batch_labels.cpu(), predictions.cpu()):
                confusion[int(gold), int(pred)] += 1

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "total": total,
        "confusion_matrix": confusion.tolist(),
    }


def train_head(args, latent_trajectory_lengths: list[int], train_payload, validation_payload):
    import torch
    import torch.nn.functional as F

    train_features = train_payload["features"]
    train_labels = train_payload["labels"]
    validation_features = validation_payload["features"]
    validation_labels = validation_payload["labels"]

    device = torch.device("cuda" if torch.cuda.is_available() and args.head_device == "cuda" else "cpu")
    model = DifficultyEstimator(
        hidden_size=train_features.size(1),
        latent_trajectory_lengths=latent_trajectory_lengths,
        mlp_hidden_size=args.mlp_hidden_size,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(
        train_features,
        train_labels,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    best_state = None
    best_metrics = None
    best_accuracy = -math.inf
    history = []

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(pooled_hidden_states=batch_features)
            loss = F.cross_entropy(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            predictions = logits.argmax(dim=-1)
            total_loss += float(loss.item()) * batch_labels.numel()
            total += batch_labels.numel()
            correct += int((predictions == batch_labels).sum().item())

        train_metrics = {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "total": total,
        }
        validation_metrics = evaluate_head(
            model,
            validation_features,
            validation_labels,
            args.eval_batch_size,
            device,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        print(
            "epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            "val_loss={val_loss:.4f} val_acc={val_acc:.4f}".format(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_acc=train_metrics["accuracy"],
                val_loss=validation_metrics["loss"],
                val_acc=validation_metrics["accuracy"],
            ),
            flush=True,
        )

        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            best_metrics = validation_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    model.save_pretrained(args.output_dir)
    return model, {
        "history": history,
        "best_validation": best_metrics,
        "latent_trajectory_lengths": latent_trajectory_lengths,
    }


def count_by(items: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item)
        counts[key] = counts.get(key, 0) + 1
    return counts


def save_training_artifacts(args, latent_trajectory_lengths, records, train_records, validation_records, metrics):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "method": "ALR Stage2 Difficulty Estimator",
        "model_path": args.model_path,
        "stage1_checkpoint_path": args.stage1_checkpoint_path,
        "latent_trajectory_lengths": latent_trajectory_lengths,
        "label_policy": "dataset_source",
        "datasets": [asdict(spec) for spec in STAGE2_DATASETS],
        "sample_per_class": args.sample_per_class,
        "limit_per_class": args.limit_per_class,
        "validation_ratio": args.validation_ratio,
        "prompt_max_length": args.prompt_max_length,
        "mlp_hidden_size": args.mlp_hidden_size,
        "dropout": args.dropout,
    }
    with (output_dir / "alr_stage2_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    metrics = dict(metrics)
    metrics["data_counts"] = {
        "all": count_by([record["source"] for record in records]),
        "train": count_by([record["source"] for record in train_records]),
        "validation": count_by([record["source"] for record in validation_records]),
    }
    with (output_dir / "stage2_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the ALR Stage2 difficulty estimator MLP head.",
    )
    parser.add_argument(
        "--model_path",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    parser.add_argument("--stage1_checkpoint_path", default=None)
    parser.add_argument(
        "--latent_trajectory_lengths",
        default=None,
        help="Comma-separated list. Defaults to Stage1 config or 64,128,192,256.",
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints/ALR-Stage2-Difficulty-DSR1-Qwen-1.5B",
    )
    parser.add_argument(
        "--cache_dir",
        default="cache/alr_stage2_pooled_1.5B",
    )
    parser.add_argument("--sample_per_class", type=int, default=7473)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test alias for --limit_per_class.",
    )
    parser.add_argument("--limit_per_class", type=int, default=None)
    parser.add_argument("--validation_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_max_length", type=int, default=1024)
    parser.add_argument("--cache_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mlp_hidden_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--torch_dtype", default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--head_device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--no_trust_remote_code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit is not None and args.limit_per_class is None:
        args.limit_per_class = args.limit
    if args.validation_ratio <= 0 or args.validation_ratio >= 1:
        raise ValueError("--validation_ratio must be between 0 and 1.")

    latent_trajectory_lengths = resolve_latent_trajectory_lengths(args)
    print(f"latent_trajectory_lengths={latent_trajectory_lengths}")

    records = build_records(
        latent_trajectory_lengths=latent_trajectory_lengths,
        sample_per_class=args.sample_per_class,
        seed=args.seed,
        limit_per_class=args.limit_per_class,
    )
    train_records, validation_records = stratified_split(
        records,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print(f"records={len(records)} train={len(train_records)} validation={len(validation_records)}")

    train_payload = load_or_cache_features(args, "train", train_records)
    validation_payload = load_or_cache_features(args, "validation", validation_records)
    _, metrics = train_head(
        args,
        latent_trajectory_lengths,
        train_payload,
        validation_payload,
    )
    save_training_artifacts(
        args,
        latent_trajectory_lengths,
        records,
        train_records,
        validation_records,
        metrics,
    )
    print(f"Saved ALR Stage2 difficulty estimator to {args.output_dir}")


if __name__ == "__main__":
    main()
