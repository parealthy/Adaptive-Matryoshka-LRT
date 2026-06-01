"""
ALR Stage1 Length-Elastic Inference

Loads an ALR Stage1 checkpoint and samples one latent prefix length k before
each generation. This script does not use a difficulty predictor.

Usage:
    python inference/run_alr_stage1_inference.py \
        --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --reasoning_net_path Qwen/Qwen3-Embedding-0.6B \
        --checkpoint_path checkpoints/ALR-Stage1-DSR1-Qwen-1.5B
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from glob import glob
from typing import Optional

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modeling.alr_stage1 import LengthElasticTransformerReasoningNet


DEFAULT_LATENT_TRAJECTORY_LENGTHS = [64, 128, 192, 256]


def parse_latent_trajectory_lengths(value: Optional[str]) -> list[int]:
    if value is None:
        return list(DEFAULT_LATENT_TRAJECTORY_LENGTHS)

    tokens = [token for token in re.split(r"[,\s]+", value.strip()) if token]
    if not tokens:
        return list(DEFAULT_LATENT_TRAJECTORY_LENGTHS)

    lengths = []
    for token in tokens:
        try:
            length = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid latent trajectory length '{token}'. "
                "Use a comma-separated list like '64,128,192,256'."
            ) from exc
        if length <= 0:
            raise ValueError("latent trajectory lengths must be positive integers.")
        lengths.append(length)
    return sorted(set(lengths))


def load_stage1_config(checkpoint_path: str) -> dict:
    config_path = os.path.join(checkpoint_path, "alr_stage1_config.json")
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_latent_trajectory_lengths(
    checkpoint_path: str,
    cli_value: Optional[str],
) -> list[int]:
    if cli_value:
        return parse_latent_trajectory_lengths(cli_value)

    stage1_config = load_stage1_config(checkpoint_path)
    config_lengths = stage1_config.get("latent_trajectory_lengths")
    if config_lengths:
        return sorted({int(length) for length in config_lengths})

    return list(DEFAULT_LATENT_TRAJECTORY_LENGTHS)


def _resolve_hidden_size(config) -> int:
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    raise ValueError("Failed to resolve hidden_size from the base model config.")


def _load_reasoning_weights(reasoning_network, checkpoint_path: str) -> None:
    safetensor_files = glob(os.path.join(checkpoint_path, "*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {checkpoint_path}")

    target_keys = set(reasoning_network.state_dict().keys())
    state_dict = {}
    for filename in safetensor_files:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("reasoning_network."):
                    new_key = key.removeprefix("reasoning_network.")
                    state_dict[new_key] = handle.get_tensor(key)
                elif key in target_keys:
                    state_dict[key] = handle.get_tensor(key)

    if not state_dict:
        raise ValueError(
            f"No reasoning network weights found in checkpoint at {checkpoint_path}"
        )

    reasoning_network.load_state_dict(state_dict, strict=True)
    print(
        f"  Loaded {len(state_dict)} reasoning network weight tensors from checkpoint."
    )


def _last_layer_hidden_states(model, inputs_embeds, attention_mask):
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

    def _hook(module, input, output):
        del module, input
        captured["hidden_states"] = output[0] if isinstance(output, tuple) else output

    handle = last_layer.register_forward_hook(_hook)
    try:
        base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
    finally:
        handle.remove()

    return captured["hidden_states"]


class ALRStage1RandomLengthInteractive:
    def __init__(
        self,
        model_path: str,
        reasoning_net_path: str,
        checkpoint_path: str,
        latent_trajectory_lengths: list[int],
        prompt_max_length: int = 1024,
        max_new_tokens: int = 2048,
        device: str = "cuda",
        seed: Optional[int] = None,
        fixed_latent_trajectory_length: Optional[int] = None,
    ):
        self.prompt_max_length = prompt_max_length
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.latent_trajectory_lengths = latent_trajectory_lengths
        self.rng = random.Random(seed)

        if not self.latent_trajectory_lengths:
            raise ValueError("latent_trajectory_lengths must not be empty.")
        if fixed_latent_trajectory_length is not None:
            fixed_latent_trajectory_length = int(fixed_latent_trajectory_length)
            if fixed_latent_trajectory_length not in self.latent_trajectory_lengths:
                raise ValueError(
                    "fixed_latent_trajectory_length must be one of "
                    f"{self.latent_trajectory_lengths}, got "
                    f"{fixed_latent_trajectory_length}."
                )
        self.fixed_latent_trajectory_length = fixed_latent_trajectory_length

        print(f"  Latent length candidates: {self.latent_trajectory_lengths}")
        if self.fixed_latent_trajectory_length is not None:
            print(f"  Fixed latent length: {self.fixed_latent_trajectory_length}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("  Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        hidden_size = _resolve_hidden_size(self.model.config)
        print("  Loading ALR Stage1 reasoning network...")
        self.reasoning_network = LengthElasticTransformerReasoningNet(
            reasoning_net_path,
            latent_trajectory_length=max(self.latent_trajectory_lengths),
            hidden_size=hidden_size,
        )
        self.reasoning_network.to(device)
        self.reasoning_network.eval()
        _load_reasoning_weights(self.reasoning_network, checkpoint_path)

    def sample_latent_length(self) -> int:
        if self.fixed_latent_trajectory_length is not None:
            return self.fixed_latent_trajectory_length
        return self.rng.choice(self.latent_trajectory_lengths)

    @torch.no_grad()
    def generate(self, user_input: str, temperature: float = 0.0) -> tuple[str, int]:
        active_latent_length = self.sample_latent_length()

        messages = [{"role": "user", "content": user_input}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.prompt_max_length,
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        prompt_embeddings = self.model.get_input_embeddings()(input_ids)
        prompt_embeddings = prompt_embeddings.to(self.model.dtype)

        hidden_states = _last_layer_hidden_states(
            self.model,
            prompt_embeddings,
            attention_mask,
        )

        latent_trajectory = self.reasoning_network(
            hidden_states,
            attention_mask=attention_mask,
            active_latent_length=active_latent_length,
        ).to(prompt_embeddings.dtype)

        combined_embeds = torch.cat([prompt_embeddings, latent_trajectory], dim=1)
        combined_mask = torch.ones(
            1,
            combined_embeds.size(1),
            dtype=torch.long,
            device=self.device,
        )

        generate_kwargs = dict(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=temperature > 0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = 0.95

        output_ids = self.model.generate(**generate_kwargs)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text, active_latent_length


def parse_args():
    parser = argparse.ArgumentParser(
        description="ALR Stage1 random-length interactive inference",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--reasoning_net_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--latent_trajectory_lengths",
        type=str,
        default=None,
        help=(
            "Comma-separated candidate lengths. If omitted, the script reads "
            "alr_stage1_config.json from the checkpoint, then falls back to "
            "64,128,192,256."
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--prompt_max_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument(
        "--fixed_latent_trajectory_length",
        type=int,
        default=None,
        help="Use one fixed latent length for debugging instead of random sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    latent_trajectory_lengths = resolve_latent_trajectory_lengths(
        args.checkpoint_path,
        args.latent_trajectory_lengths,
    )

    print("\nLoading ALR Stage1 random-length inference model...")
    model = ALRStage1RandomLengthInteractive(
        model_path=args.model_path,
        reasoning_net_path=args.reasoning_net_path,
        checkpoint_path=args.checkpoint_path,
        latent_trajectory_lengths=latent_trajectory_lengths,
        prompt_max_length=args.prompt_max_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        seed=args.seed,
        fixed_latent_trajectory_length=args.fixed_latent_trajectory_length,
    )

    if args.question:
        answer, active_latent_length = model.generate(
            args.question,
            temperature=args.temperature,
        )
        print(f"\n[active_latent_length={active_latent_length}]\n{answer}\n")
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
        if user_input == ">>>":
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if user_input.lower() == ":paste":
            print("Paste your multi-line question. End with a line containing ':end'.")
            lines = []
            while True:
                try:
                    line = input()
                except (EOFError, KeyboardInterrupt):
                    print("\n[Paste interrupted]\n")
                    lines = []
                    break
                if line.strip().lower() == ":end":
                    break
                lines.append(line)
            user_input = "\n".join(lines).strip()
            if not user_input:
                continue
        if not any(char.isalnum() for char in user_input):
            continue

        try:
            answer, active_latent_length = model.generate(
                user_input,
                temperature=args.temperature,
            )
            print(f"\n[active_latent_length={active_latent_length}]\n{answer}\n")
        except KeyboardInterrupt:
            print("\n[Generation interrupted]\n")
            continue


if __name__ == "__main__":
    main()
