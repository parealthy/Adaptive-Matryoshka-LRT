#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "eval" / "math"
DEFAULT_DATA_ROOT = Path(os.environ.get("LRT_DATA_ROOT", PROJECT_ROOT / "data" / "datasets"))
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval_outputs" / "math"
DEFAULT_MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_REASONING_NET_PATH = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "DSR1-Qwen-1.5B-LRT-Math"
REPO_MATH_PROMPT_TEMPLATE = (
    "{problem} Let's think step by step and output the final answer within \\boxed{}."
)

BUILTIN_TASKS = {"math500", "gsm8k"}


def progress(iterable, desc=None):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc)


def sanitize_name(value: str) -> str:
    import re

    value = value.strip().strip("/")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value or "model"


def resolve_hidden_size(config) -> int:
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
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


def load_reasoning_weights(reasoning_network, checkpoint_path: str) -> None:
    import torch
    from safetensors import safe_open

    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    state_dict = {}
    for filename in sorted(glob.glob(str(checkpoint_dir / "*.safetensors"))):
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


class LatentReasoningGenerator:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from modeling.reason import TransformerReasoningNet

        dtype_map = {
            "auto": "auto",
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }

        self.args = args
        self.prompt_max_length = args.prompt_max_length
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p

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
            torch_dtype=dtype_map[args.torch_dtype],
            device_map=resolve_device_map(args.device),
            trust_remote_code=not args.no_trust_remote_code,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.input_device = model_input_device(self.model)

        print(f"Loading reasoning network: {args.reasoning_net_path}")
        self.reasoning_network = TransformerReasoningNet(
            args.reasoning_net_path,
            latent_trajectory_length=args.latent_trajectory_length,
            hidden_size=resolve_hidden_size(self.model.config),
        )
        self.reasoning_network.to(self.input_device)
        self.reasoning_network.eval()
        load_reasoning_weights(self.reasoning_network, args.checkpoint_path)

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

    def generate_batch(self, prompts: list[str]) -> tuple[list[str], list[str]]:
        import torch

        prompt_texts = self.render_prompts(prompts)
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

        with torch.inference_mode():
            prompt_embeddings = self.model.get_input_embeddings()(input_ids).to(self.model.dtype)
            hidden_states = last_layer_hidden_states(self.model, prompt_embeddings, attention_mask)
            latent_trajectory = self.reasoning_network(
                hidden_states,
                attention_mask=attention_mask,
            ).to(prompt_embeddings.device, dtype=prompt_embeddings.dtype)

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
                "do_sample": self.temperature > 0,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.temperature > 0:
                generate_kwargs["temperature"] = self.temperature
                generate_kwargs["top_p"] = self.top_p

            output_ids = self.model.generate(**generate_kwargs)

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True), prompt_texts


class VLLMGenerator:
    def __init__(self, args):
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.args = args
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )

        tokenizer_path = args.tokenizer_path
        model_dir = Path(args.model_path)
        if tokenizer_path is None:
            if "Marco-o1" in args.model_path:
                tokenizer_path = "AIDC-AI/Marco-o1"
            elif model_dir.exists() and (model_dir / "huggingface").exists():
                tokenizer_path = str(model_dir / "huggingface")
            else:
                tokenizer_path = args.model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=not args.no_trust_remote_code,
            revision=args.revision,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        tensor_parallel_size = args.tensor_parallel_size or max(torch.cuda.device_count(), 1)
        llm_kwargs = {
            "dtype": args.vllm_dtype,
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": not args.no_trust_remote_code,
        }
        if args.revision:
            llm_kwargs["revision"] = args.revision

        llm_model_path = self.prepare_vllm_model_path(args.model_path, tokenizer_path)
        self.llm = LLM(model=llm_model_path, tokenizer=tokenizer_path, **llm_kwargs)

    def prepare_vllm_model_path(self, model_path: str, tokenizer_path: str) -> str:
        model_dir = Path(model_path)
        if not model_dir.exists():
            return model_path

        merged_model = model_dir / "merged_model.pt"
        if not merged_model.exists():
            hf_dir = model_dir / "huggingface"
            if hf_dir.exists() and any(item.suffix in {".bin", ".safetensors"} for item in hf_dir.iterdir()):
                return str(hf_dir)
            return str(model_dir)

        hf_dir = model_dir / "huggingface"
        if not hf_dir.exists():
            raise ValueError(f"Found {merged_model}, but missing HuggingFace files under {hf_dir}")

        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name)
        for item in hf_dir.iterdir():
            destination = temp_path / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)
        try:
            os.symlink(merged_model, temp_path / "pytorch_model.bin")
        except (OSError, NotImplementedError):
            shutil.copy2(merged_model, temp_path / "pytorch_model.bin")

        self._temp_dir = temp_dir
        return str(temp_path)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def render_prompts(self, prompts: list[str]) -> list[str]:
        if not self.args.use_chat_template:
            return prompts
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]

    def generate_batch(self, prompts: list[str]) -> tuple[list[str], list[str]]:
        prompt_texts = self.render_prompts(prompts)
        outputs = self.llm.generate(prompt_texts, sampling_params=self.sampling_params)
        completions = [output.outputs[0].text if output.outputs else "" for output in outputs]
        return completions, prompt_texts


def load_builtin_task(task_name: str, data_root: Path) -> tuple[list[dict[str, Any]], str]:
    from datasets import DatasetDict, load_dataset, load_from_disk

    def resolve_split(dataset_obj, split: str):
        if isinstance(dataset_obj, DatasetDict):
            return dataset_obj[split]
        return dataset_obj

    if task_name == "math500":
        dataset_id = "HuggingFaceH4/MATH-500"
        split = "test"
        local_path = data_root / dataset_id
        if local_path.exists():
            try:
                dataset = resolve_split(load_from_disk(str(local_path)), split)
            except Exception:
                dataset = load_dataset(str(local_path), split=split)
        else:
            dataset = load_dataset(dataset_id, split=split)
        return [project_math500(dict(dataset[i])) for i in range(len(dataset))], task_name

    if task_name == "gsm8k":
        dataset_id = "openai/gsm8k"
        split = "test"
        local_path = data_root / dataset_id
        if local_path.exists():
            try:
                dataset = resolve_split(load_from_disk(str(local_path)), split)
            except Exception:
                dataset = load_dataset(str(local_path), "main", split=split)
        else:
            dataset = load_dataset(dataset_id, "main", split=split)
        return [project_gsm8k(dict(dataset[i])) for i in range(len(dataset))], task_name

    raise ValueError(f"Unknown built-in task: {task_name}")


def project_math500(example: dict[str, Any]) -> dict[str, str]:
    problem = str(example.get("problem") or example.get("question") or "").strip()
    target = first_nonempty(example, ("answer", "final_answer", "target"))
    if not target:
        target = extract_boxed(str(example.get("solution", ""))) or str(example.get("solution", "")).strip()
    return {"problem": problem, "target_answer": str(target).strip()}


def project_gsm8k(example: dict[str, Any]) -> dict[str, str]:
    raw_answer = str(example.get("answer", "")).strip()
    target = raw_answer.split("####")[-1].strip()
    target = target.lstrip("=$ ").replace(",", "").strip()
    return {"problem": str(example.get("question", "")).strip(), "target_answer": target}


def load_local_task(data_dir: Path, task_name: str) -> tuple[list[dict[str, Any]], str]:
    dataset_path = Path(task_name)
    if not dataset_path.exists():
        if dataset_path.suffix == "":
            dataset_path = data_dir / f"{task_name}.json"
        else:
            dataset_path = data_dir / dataset_path

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        for key in ("data", "examples", "records", "items", "results"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break

    if not isinstance(payload, list):
        raise ValueError(f"Dataset must be a JSON list: {dataset_path}")

    records = [project_local_example(item) for item in payload]
    return records, dataset_path.stem


def project_local_example(example: Any) -> dict[str, str]:
    if not isinstance(example, dict):
        return {"problem": to_text(example), "target_answer": ""}

    problem = first_nonempty(example, ("problem", "question", "prompt", "input"))
    target = first_nonempty(
        example,
        ("target_answer", "answer", "solution", "final_answer", "target", "ground_truth"),
    )
    return {"problem": to_text(problem).strip(), "target_answer": to_text(target).strip()}


def first_nonempty(example: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = example.get(key)
        if value is not None and str(value).strip():
            return value
    return ""


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def extract_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    cursor = start + len(marker)
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


def resolve_tasks(args) -> list[str]:
    if "all" not in args.tasks:
        return args.tasks

    data_dir = Path(args.data_dir)
    local_tasks = []
    if data_dir.exists():
        local_tasks = [
            path.stem
            for path in sorted(data_dir.glob("*.json"))
            if not path.name.startswith(("summary", "verified_"))
        ]
    if local_tasks:
        return local_tasks
    return ["math500", "gsm8k"]


def load_task(args, task_name: str) -> tuple[list[dict[str, Any]], str]:
    if task_name in BUILTIN_TASKS:
        return load_builtin_task(task_name, Path(args.data_root))
    return load_local_task(Path(args.data_dir), task_name)


def build_generator(args):
    if args.backend == "lrt":
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required when --backend lrt.")
        return LatentReasoningGenerator(args)
    if args.backend == "vllm":
        return VLLMGenerator(args)
    raise ValueError(f"Unknown backend: {args.backend}")


def evaluate_task(generator, records: list[dict[str, Any]], task_name: str, args, save_dir: Path):
    if args.limit is not None:
        records = records[: args.limit]

    output_path = save_dir / f"{task_name}_results.json"
    results = []
    total_output_tokens = 0
    started = perf_counter()

    for start in progress(range(0, len(records), args.batch_size), desc=f"Evaluating {task_name}"):
        batch = records[start : start + args.batch_size]
        prompts = [
            args.prompt_template.replace("{problem}", record["problem"].strip())
            for record in batch
        ]
        completions, rendered_prompts = generator.generate_batch(prompts)

        for offset, (record, prompt, rendered_prompt, completion) in enumerate(
            zip(batch, prompts, rendered_prompts, completions)
        ):
            output_length = generator.count_tokens(completion)
            total_output_tokens += output_length
            results.append(
                {
                    "task": task_name,
                    "index": start + offset,
                    "problem": record["problem"],
                    "prompt": prompt,
                    "target_answer": record["target_answer"],
                    "completion": completion,
                    "input_length": generator.count_tokens(rendered_prompt),
                    "output_length": output_length,
                }
            )

    save_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    elapsed = perf_counter() - started
    avg_output_length = total_output_tokens / len(results) if results else 0.0
    print(f"{task_name}: generated {len(results)} examples")
    print(f"{task_name}: average output length = {avg_output_length:.2f} tokens")
    print(f"{task_name}: saved results to {output_path}")
    return {
        "total_problems": len(results),
        "avg_output_length": avg_output_length,
        "elapsed_seconds": elapsed,
        "results_path": str(output_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate math-evaluation responses for LRT checkpoints or vLLM models.",
    )
    parser.add_argument("--backend", choices=("lrt", "vllm"), default="lrt")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reasoning_net_path", default=DEFAULT_REASONING_NET_PATH)
    parser.add_argument("--checkpoint_path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tasks", nargs="+", default=["math500", "gsm8k"])
    parser.add_argument("--prompt_template", default=REPO_MATH_PROMPT_TEMPLATE)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--prompt_max_length", type=int, default=1024)
    parser.add_argument("--latent_trajectory_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--torch_dtype",
        default="bf16",
        choices=("auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"),
    )
    parser.add_argument("--vllm_dtype", default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=16000)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--no_trust_remote_code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")

    if args.model_name is None:
        args.model_name = sanitize_name(args.checkpoint_path if args.backend == "lrt" else args.model_path)

    save_dir = Path(args.output_dir) / args.model_name
    tasks = resolve_tasks(args)

    print(f"Backend: {args.backend}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Output directory: {save_dir}")

    generator = build_generator(args)

    summary = {}
    for task_name in tasks:
        print(f"\n--- Evaluating {task_name} ---")
        records, resolved_name = load_task(args, task_name)
        summary[resolved_name] = evaluate_task(generator, records, resolved_name, args, save_dir)

    summary_path = save_dir / "generation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\nGeneration complete.")
    print(f"Saved generation summary to {summary_path}")
    print("Run verifier.py/verifier.sh on this output directory to compute accuracy.")


if __name__ == "__main__":
    main()
