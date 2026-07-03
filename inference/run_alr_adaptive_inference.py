#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from glob import glob
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.alr_difficulty import DifficultyEstimator, normalize_latent_lengths
from modeling.alr_stage1 import LengthElasticTransformerReasoningNet


DEFAULT_LATENT_TRAJECTORY_LENGTHS = [64, 128, 192, 256]


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


def resolve_latent_trajectory_lengths(
    stage1_checkpoint_path: str,
    difficulty_checkpoint_path: str,
    cli_value: Optional[str],
) -> list[int]:
    if cli_value:
        return parse_latent_trajectory_lengths(cli_value)

    difficulty_config = load_json_if_exists(
        Path(difficulty_checkpoint_path) / "alr_difficulty_config.json",
    )
    if difficulty_config.get("latent_trajectory_lengths"):
        return normalize_latent_lengths(difficulty_config["latent_trajectory_lengths"])

    stage1_config = load_json_if_exists(
        Path(stage1_checkpoint_path) / "alr_stage1_config.json",
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


def resolve_device_map(device: str):
    if device == "auto":
        return "auto"
    if device == "cuda":
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
    print(f"  Loaded {len(state_dict)} reasoning-network tensors from {checkpoint_path}")


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


class ALRAdaptiveInteractive:
    def __init__(
        self,
        model_path: str,
        reasoning_net_path: str,
        stage1_checkpoint_path: str,
        difficulty_checkpoint_path: str,
        latent_trajectory_lengths: list[int],
        prompt_max_length: int = 1024,
        max_new_tokens: int = 2048,
        device: str = "cuda:0",
        torch_dtype: str = "bf16",
        no_trust_remote_code: bool = False,
    ):
        self.prompt_max_length = prompt_max_length
        self.max_new_tokens = max_new_tokens
        self.latent_trajectory_lengths = latent_trajectory_lengths

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=not no_trust_remote_code,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print("  Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype_from_arg(torch_dtype),
            device_map=resolve_device_map(device),
            trust_remote_code=not no_trust_remote_code,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.input_device = model_input_device(self.model)

        hidden_size = resolve_hidden_size(self.model.config)
        print("  Loading ALR Stage1 reasoning network...")
        self.reasoning_network = LengthElasticTransformerReasoningNet(
            reasoning_net_path,
            latent_trajectory_length=max(self.latent_trajectory_lengths),
            hidden_size=hidden_size,
        )
        self.reasoning_network.to(self.input_device)
        self.reasoning_network.eval()
        load_reasoning_weights(self.reasoning_network, stage1_checkpoint_path)

        print("  Loading ALR Stage2 difficulty estimator...")
        self.difficulty_estimator = DifficultyEstimator.from_pretrained(
            difficulty_checkpoint_path,
            map_location="cpu",
        )
        if self.difficulty_estimator.latent_trajectory_lengths != self.latent_trajectory_lengths:
            raise ValueError(
                "Difficulty estimator length set does not match inference length set: "
                f"{self.difficulty_estimator.latent_trajectory_lengths} vs "
                f"{self.latent_trajectory_lengths}."
            )
        self.difficulty_estimator.to(self.input_device)
        self.difficulty_estimator.eval()

    def render_prompt(self, user_input: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_input}],
            add_generation_prompt=True,
            tokenize=False,
        )

    @torch.no_grad()
    def generate(self, user_input: str, temperature: float = 0.0) -> dict:
        prompt_text = self.render_prompt(user_input)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.prompt_max_length,
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

        pooled = DifficultyEstimator.mean_pool_hidden_states(
            hidden_states,
            attention_mask,
        )
        logits = self.difficulty_estimator(pooled_hidden_states=pooled.float())
        probabilities = torch.softmax(logits, dim=-1)
        predicted_label = int(logits.argmax(dim=-1).item())
        predicted_length = self.difficulty_estimator.label_to_length(predicted_label)

        latent_trajectory = self.reasoning_network(
            hidden_states,
            attention_mask=attention_mask,
            active_latent_length=predicted_length,
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
            "max_new_tokens": self.max_new_tokens,
            "do_sample": temperature > 0,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = 0.95

        output_ids = self.model.generate(**generate_kwargs)
        completion = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {
            "completion": completion,
            "predicted_length": predicted_length,
            "predicted_label": predicted_label,
            "length_probabilities": {
                str(length): float(prob)
                for length, prob in zip(
                    self.latent_trajectory_lengths,
                    probabilities[0].detach().cpu().tolist(),
                )
            },
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="ALR adaptive interactive inference with a Stage2 difficulty head.",
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
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--prompt_max_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bf16")
    parser.add_argument("--question", default=None)
    parser.add_argument("--no_trust_remote_code", action="store_true")
    return parser.parse_args()


def print_result(result: dict) -> None:
    print(f"\n[predicted_length={result['predicted_length']}]")
    print(f"[length_probabilities={result['length_probabilities']}]")
    print(result["completion"])
    print()


def main():
    args = parse_args()
    latent_trajectory_lengths = resolve_latent_trajectory_lengths(
        args.stage1_checkpoint_path,
        args.difficulty_checkpoint_path,
        args.latent_trajectory_lengths,
    )
    print("\nLoading ALR adaptive inference model...")
    print(f"  Latent length candidates: {latent_trajectory_lengths}")
    model = ALRAdaptiveInteractive(
        model_path=args.model_path,
        reasoning_net_path=args.reasoning_net_path,
        stage1_checkpoint_path=args.stage1_checkpoint_path,
        difficulty_checkpoint_path=args.difficulty_checkpoint_path,
        latent_trajectory_lengths=latent_trajectory_lengths,
        prompt_max_length=args.prompt_max_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        torch_dtype=args.torch_dtype,
        no_trust_remote_code=args.no_trust_remote_code,
    )

    if args.question:
        print_result(model.generate(args.question, temperature=args.temperature))
        return

    print("Model loaded. Type your question (or 'exit' to quit).\n")
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break
        try:
            print_result(model.generate(user_input, temperature=args.temperature))
        except KeyboardInterrupt:
            print("\n[Generation interrupted]\n")


if __name__ == "__main__":
    main()
