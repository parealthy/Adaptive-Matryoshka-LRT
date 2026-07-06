#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_LATENT_TRAJECTORY_LENGTHS = [64, 128, 192, 256]
MATH_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."


torch = None
safe_open = None
AutoModelForCausalLM = None
AutoTokenizer = None
DifficultyEstimator = None
LengthElasticTransformerReasoningNet = None
normalize_latent_lengths = None


def load_runtime_dependencies() -> None:
    global torch
    global safe_open
    global AutoModelForCausalLM
    global AutoTokenizer
    global DifficultyEstimator
    global LengthElasticTransformerReasoningNet
    global normalize_latent_lengths

    if torch is not None:
        return

    import torch as torch_module
    from safetensors import safe_open as safe_open_fn
    from transformers import AutoModelForCausalLM as auto_model_for_causal_lm
    from transformers import AutoTokenizer as auto_tokenizer

    from modeling.alr_difficulty import DifficultyEstimator as difficulty_estimator
    from modeling.alr_difficulty import normalize_latent_lengths as normalize_lengths
    from modeling.alr_stage1 import (
        LengthElasticTransformerReasoningNet as length_elastic_transformer_reasoning_net,
    )

    torch = torch_module
    safe_open = safe_open_fn
    AutoModelForCausalLM = auto_model_for_causal_lm
    AutoTokenizer = auto_tokenizer
    DifficultyEstimator = difficulty_estimator
    LengthElasticTransformerReasoningNet = length_elastic_transformer_reasoning_net
    normalize_latent_lengths = normalize_lengths


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
        Path(difficulty_checkpoint_path).expanduser() / "alr_difficulty_config.json",
    )
    if difficulty_config.get("latent_trajectory_lengths"):
        return normalize_latent_lengths(difficulty_config["latent_trajectory_lengths"])

    stage1_config = load_json_if_exists(
        Path(stage1_checkpoint_path).expanduser() / "alr_stage1_config.json",
    )
    if stage1_config.get("latent_trajectory_lengths"):
        return normalize_latent_lengths(stage1_config["latent_trajectory_lengths"])

    return list(DEFAULT_LATENT_TRAJECTORY_LENGTHS)


def has_tokenizer_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    tokenizer_files = {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
    }
    return any((path / filename).exists() for filename in tokenizer_files)


def resolve_tokenizer_path(
    model_path: str,
    stage1_checkpoint_path: str,
    tokenizer_path: Optional[str],
) -> str:
    if tokenizer_path:
        return tokenizer_path

    stage1_path = Path(stage1_checkpoint_path).expanduser()
    if has_tokenizer_files(stage1_path):
        return str(stage1_path)
    return model_path


def resolve_hidden_size(config) -> int:
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    raise ValueError("Failed to resolve hidden_size from the base model config.")


def resolve_torch_dtype(value: str):
    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(value).lower()
    if key not in dtype_map:
        raise ValueError(
            "torch_dtype must be one of auto, bf16, bfloat16, fp16, float16, fp32, float32."
        )
    return dtype_map[key]


def resolve_device_map(device: str):
    if device == "auto":
        if not torch.cuda.is_available():
            return {"": "cpu"}
        return {"": 0}
    if device == "cuda":
        return {"": 0}
    if device.startswith("cuda:"):
        return {"": int(device.split(":", 1)[1])}
    return {"": device}


def model_input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def load_reasoning_weights(reasoning_network, checkpoint_path: str) -> None:
    checkpoint_dir = Path(checkpoint_path).expanduser()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Stage1 checkpoint path does not exist: {checkpoint_path}")

    target_keys = set(reasoning_network.state_dict().keys())
    state_dict = {}
    for filename in sorted(glob.glob(str(checkpoint_dir / "*.safetensors"))):
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


def append_math_suffix(problem: str) -> str:
    problem = str(problem).strip()
    if not problem.endswith(MATH_SUFFIX):
        problem = problem + MATH_SUFFIX
    return problem


def strip_prompt_if_present(text: str, prompt: str) -> str:
    if prompt and text.startswith(prompt):
        return text[len(prompt) :]
    return text


def tensor_slice_for_print(tensor: torch.Tensor):
    detached = tensor.detach().float().cpu()
    if detached.ndim == 0:
        return detached.item()
    if detached.ndim == 1:
        return detached[: min(8, detached.size(0))].tolist()
    if detached.ndim == 2:
        return detached[: min(2, detached.size(0)), : min(8, detached.size(1))].tolist()
    return detached[
        : min(1, detached.size(0)),
        : min(2, detached.size(1)),
        : min(8, detached.size(2)),
    ].tolist()


def print_tensor_summary(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}:")
    print(f"  shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")
    print(f"  slice={tensor_slice_for_print(tensor)}")


class ALREvalFlowDebugRunner:
    def __init__(
        self,
        model_path: str,
        reasoning_net_path: str,
        stage1_checkpoint_path: str,
        difficulty_checkpoint_path: str,
        tokenizer_path: Optional[str],
        latent_trajectory_lengths: list[int],
        prompt_max_length: int,
        max_new_tokens: int,
        device: str,
        torch_dtype: str,
        top_p: float,
        trust_remote_code: bool,
        local_files_only: bool,
        use_fast_tokenizer: bool,
    ) -> None:
        self.model_path = model_path
        self.reasoning_net_path = reasoning_net_path
        self.stage1_checkpoint_path = stage1_checkpoint_path
        self.difficulty_checkpoint_path = difficulty_checkpoint_path
        self.latent_trajectory_lengths = latent_trajectory_lengths
        self.prompt_max_length = int(prompt_max_length)
        self.max_new_tokens = int(max_new_tokens)
        self.top_p = float(top_p)
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.use_fast_tokenizer = use_fast_tokenizer
        self.tokenizer_path = resolve_tokenizer_path(
            model_path=model_path,
            stage1_checkpoint_path=stage1_checkpoint_path,
            tokenizer_path=tokenizer_path,
        )

        print(f"Loading tokenizer from {self.tokenizer_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=self.trust_remote_code,
                use_fast=self.use_fast_tokenizer,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            if not self.use_fast_tokenizer:
                raise
            print(f"Fast tokenizer loading failed; retrying with use_fast=False. Error: {exc}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=self.trust_remote_code,
                use_fast=False,
                local_files_only=self.local_files_only,
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        print(f"Loading base model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=resolve_torch_dtype(torch_dtype),
            device_map=resolve_device_map(device),
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True,
            local_files_only=self.local_files_only,
        )
        self.model.eval()
        self.input_device = model_input_device(self.model)

        print(f"Loading reasoning net from {reasoning_net_path}")
        self.reasoning_network = LengthElasticTransformerReasoningNet(
            reasoning_net_path,
            latent_trajectory_length=max(self.latent_trajectory_lengths),
            hidden_size=resolve_hidden_size(self.model.config),
            local_files_only=self.local_files_only,
        )
        self.reasoning_network.to(self.input_device)
        self.reasoning_network.eval()
        load_reasoning_weights(self.reasoning_network, stage1_checkpoint_path)

        print(f"Loading difficulty estimator from {difficulty_checkpoint_path}")
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

    def render_prompt(self, question: str) -> tuple[str, str]:
        suffixed_problem = append_math_suffix(question)
        rendered_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": suffixed_problem}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return suffixed_problem, rendered_prompt

    def build_generation_kwargs(self, temperature: float) -> dict:
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": temperature > 0.0,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = float(temperature)
            generate_kwargs["top_p"] = self.top_p
        return generate_kwargs

    def maybe_dump_tensors(
        self,
        dump_tensors: Optional[str],
        hidden_states: torch.Tensor,
        pooled_hidden: torch.Tensor,
        latent_trajectory: torch.Tensor,
    ) -> None:
        if not dump_tensors:
            return
        output_dir = Path(dump_tensors).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(hidden_states.detach().cpu(), output_dir / "hidden_states.pt")
        torch.save(pooled_hidden.detach().cpu(), output_dir / "pooled_hidden.pt")
        torch.save(latent_trajectory.detach().cpu(), output_dir / "latent_trajectory.pt")
        print(f"Saved full tensors to {output_dir}")

    def run(self, question: str, temperature: float, dump_tensors: Optional[str]) -> str:
        suffixed_problem, rendered_prompt = self.render_prompt(question)

        print("\n===== Raw Question =====")
        print(question)
        print("\n===== Problem With Evaluation Suffix =====")
        print(suffixed_problem)
        print("\n===== Rendered Prompt Sent To Base Model =====")
        print(rendered_prompt)

        inputs = self.tokenizer(
            rendered_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.prompt_max_length,
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(self.input_device)
        attention_mask = inputs["attention_mask"].to(self.input_device)
        print("\n===== Tokenized Input =====")
        print(f"input_ids shape={tuple(input_ids.shape)}")
        print(f"attention_mask shape={tuple(attention_mask.shape)}")

        with torch.inference_mode():
            prompt_embeddings = self.model.get_input_embeddings()(input_ids).to(self.model.dtype)
            print_tensor_summary("\nPrompt embeddings", prompt_embeddings)

            hidden_states = last_layer_hidden_states(
                self.model,
                prompt_embeddings,
                attention_mask,
            )
            print_tensor_summary("\nBase model last hidden_state", hidden_states)

            pooled_hidden = DifficultyEstimator.mean_pool_hidden_states(
                hidden_states,
                attention_mask,
            )
            print_tensor_summary("\nPooled hidden_state for length predictor", pooled_hidden)

            logits = self.difficulty_estimator(pooled_hidden_states=pooled_hidden.float())
            probabilities = torch.softmax(logits, dim=-1)
            predicted_label = int(logits.argmax(dim=-1).item())
            predicted_length = self.difficulty_estimator.label_to_length(predicted_label)
            length_probabilities = {
                str(length): float(prob)
                for length, prob in zip(
                    self.latent_trajectory_lengths,
                    probabilities[0].detach().cpu().tolist(),
                )
            }

            print("\n===== Length Predictor Output =====")
            print(f"logits={logits.detach().float().cpu().tolist()[0]}")
            print(f"probabilities={length_probabilities}")
            print(f"predicted_label={predicted_label}")
            print(f"predicted_length={predicted_length}")

            latent_trajectory = self.reasoning_network(
                hidden_states,
                attention_mask=attention_mask,
                active_latent_length=predicted_length,
            ).to(prompt_embeddings.dtype)
            print_tensor_summary("\nReasoning net latent trajectory", latent_trajectory)
            print(f"latent trajectory length={latent_trajectory.size(1)}")

            latent_mask = torch.ones(
                latent_trajectory.size(0),
                latent_trajectory.size(1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            combined_embeds = torch.cat([prompt_embeddings, latent_trajectory], dim=1)
            combined_mask = torch.cat([attention_mask, latent_mask], dim=1)
            print("\n===== Concatenated Generation Input =====")
            print(f"combined_embeds shape={tuple(combined_embeds.shape)}")
            print(f"combined_mask shape={tuple(combined_mask.shape)}")

            self.maybe_dump_tensors(
                dump_tensors=dump_tensors,
                hidden_states=hidden_states,
                pooled_hidden=pooled_hidden,
                latent_trajectory=latent_trajectory,
            )

            output_ids = self.model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                **self.build_generation_kwargs(temperature),
            )
            decoded_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            response = strip_prompt_if_present(decoded_output, rendered_prompt)

        print("\n===== Final Response =====")
        print(response)
        print()
        return response


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug one ALR adaptive inference pass using the lm-eval data flow.",
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
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--latent_trajectory_lengths", default=None)
    parser.add_argument("--prompt_max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch_dtype", default="bf16")
    parser.add_argument("--question", default=None)
    parser.add_argument("--dump_tensors", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--no_trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    latent_trajectory_lengths = resolve_latent_trajectory_lengths(
        args.stage1_checkpoint_path,
        args.difficulty_checkpoint_path,
        args.latent_trajectory_lengths,
    )

    print("Loading ALR eval-flow debug runner...")
    print(f"Latent length candidates: {latent_trajectory_lengths}")
    runner = ALREvalFlowDebugRunner(
        model_path=args.model_path,
        reasoning_net_path=args.reasoning_net_path,
        stage1_checkpoint_path=args.stage1_checkpoint_path,
        difficulty_checkpoint_path=args.difficulty_checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        latent_trajectory_lengths=latent_trajectory_lengths,
        prompt_max_length=args.prompt_max_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        torch_dtype=args.torch_dtype,
        top_p=args.top_p,
        trust_remote_code=not args.no_trust_remote_code,
        local_files_only=args.local_files_only,
        use_fast_tokenizer=not args.use_slow_tokenizer,
    )

    if args.question:
        runner.run(
            question=args.question,
            temperature=args.temperature,
            dump_tensors=args.dump_tensors,
        )
        return

    print("Model loaded. Type a question, or 'exit' to quit.\n")
    while True:
        try:
            question = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        try:
            runner.run(
                question=question,
                temperature=args.temperature,
                dump_tensors=args.dump_tensors,
            )
        except KeyboardInterrupt:
            print("\n[Generation interrupted]\n")


if __name__ == "__main__":
    main()
