#!/usr/bin/env python3
import argparse
import glob
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DEFAULT_DATA_DIR = "/ShorterBetter/eval_data/math"
DEFAULT_OUTPUT_DIR = "/ShorterBetter/eval_data/outputs/math"
DEFAULT_PROMPT_TEMPLATE = "{problem}\n\nGive your answer in \\boxed{}."


def cuda_device_count():
    return max(torch.cuda.device_count(), 1)


def load_model_and_tokenizer(
    model_path,
    tokenizer_path=None,
    dtype="bfloat16",
    max_model_len=16000,
    tensor_parallel_size=None,
    trust_remote_code=True,
    revision=None,
):
    print(f"Loading model from: {model_path}")
    tensor_parallel_size = tensor_parallel_size or cuda_device_count()
    model_dir = Path(model_path)

    if tokenizer_path is None:
        if "Marco-o1" in model_path:
            tokenizer_path = "AIDC-AI/Marco-o1"
        elif model_dir.exists() and (model_dir / "huggingface").exists():
            tokenizer_path = str(model_dir / "huggingface")
        else:
            tokenizer_path = model_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = {
        "dtype": dtype,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
    }
    if revision:
        llm_kwargs["revision"] = revision

    if model_dir.exists():
        index_file = model_dir / "model.safetensors.index.json"
        if index_file.exists():
            print(f"Found sharded safetensors model at: {model_dir}")
            return LLM(model=str(model_dir), tokenizer=tokenizer_path, **llm_kwargs), tokenizer

        merged_model_path = model_dir / "merged_model.pt"
        if merged_model_path.exists():
            return load_merged_model(merged_model_path, model_dir / "huggingface", tokenizer, llm_kwargs)

        hf_dir = model_dir / "huggingface"
        if hf_dir.exists() and any(item.suffix in {".bin", ".safetensors"} for item in hf_dir.iterdir()):
            print(f"Using HuggingFace directory with weights: {hf_dir}")
            return LLM(model=str(hf_dir), tokenizer=str(hf_dir), **llm_kwargs), tokenizer

    return LLM(model=model_path, tokenizer=tokenizer_path, **llm_kwargs), tokenizer


def load_merged_model(merged_model_path, hf_dir, tokenizer, llm_kwargs):
    if not hf_dir.exists():
        raise ValueError(f"HuggingFace directory not found at {hf_dir}")

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    print(f"Preparing temporary HF directory at: {temp_path}")

    for item in hf_dir.iterdir():
        destination = temp_path / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    try:
        os.symlink(merged_model_path, temp_path / "pytorch_model.bin")
        print("Created symlink to merged_model.pt")
    except (OSError, NotImplementedError):
        print("Symlink failed; copying merged_model.pt instead.")
        shutil.copy2(merged_model_path, temp_path / "pytorch_model.bin")

    llm = LLM(model=str(temp_path), tokenizer=str(hf_dir), **llm_kwargs)
    llm._math_eval_temp_dir = temp_dir
    return llm, tokenizer


def load_dataset(data_dir, task_name, max_datapoints=None):
    dataset_path = Path(task_name)
    if not dataset_path.exists():
        if not task_name.endswith(".json"):
            task_name = f"{task_name}.json"
        dataset_path = Path(data_dir) / task_name

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    if isinstance(dataset, dict):
        for key in ("data", "examples", "records", "items", "results"):
            if isinstance(dataset.get(key), list):
                dataset = dataset[key]
                break

    if not isinstance(dataset, list):
        raise ValueError(f"Dataset must be a JSON list: {dataset_path}")

    if max_datapoints is not None:
        dataset = dataset[:max_datapoints]

    print(f"Loaded {len(dataset)} problems from {dataset_path}")
    return dataset, dataset_path


def get_all_datasets(data_dir):
    dataset_files = glob.glob(os.path.join(data_dir, "*.json"))
    return [
        os.path.basename(path)
        for path in sorted(dataset_files)
        if not os.path.basename(path).startswith(("summary", "verified_"))
    ]


def get_value(item, field, fallback_fields):
    if field != "auto":
        return item.get(field, "")
    for key in fallback_fields:
        if key in item and item[key] is not None:
            return item[key]
    return ""


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def count_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def render_prompts(tokenizer, prompts, use_chat_template):
    if not use_chat_template:
        return prompts
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in prompts
    ]


def evaluate_dataset(llm, dataset, dataset_path, tokenizer, args, save_path):
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    results = []
    total_output_tokens = 0
    task_name = Path(dataset_path).stem

    for start in tqdm(range(0, len(dataset), args.batch_size), desc=f"Evaluating {task_name}"):
        batch = dataset[start : start + args.batch_size]
        raw_prompts = []
        problems = []

        for item in batch:
            problem = to_text(
                get_value(item, args.problem_field, ("problem", "question", "prompt", "input"))
            ).strip()
            problems.append(problem)
            raw_prompts.append(args.prompt_template.replace("{problem}", problem))

        prompts = render_prompts(tokenizer, raw_prompts, args.use_chat_template)
        outputs = llm.generate(prompts, sampling_params=sampling_params)

        for offset, item in enumerate(batch):
            completion = outputs[offset].outputs[0].text if outputs[offset].outputs else ""
            target_answer = to_text(
                get_value(
                    item,
                    args.answer_field,
                    ("answer", "target_answer", "solution", "final_answer", "target", "ground_truth"),
                )
            ).strip()
            input_tokens = count_tokens(tokenizer, prompts[offset])
            output_tokens = count_tokens(tokenizer, completion)
            total_output_tokens += output_tokens

            results.append(
                {
                    "problem": problems[offset],
                    "target_answer": target_answer,
                    "completion": completion,
                    "input_length": input_tokens,
                    "output_length": output_tokens,
                }
            )

    avg_output_length = total_output_tokens / len(results) if results else 0.0
    print("Dataset evaluation completed:")
    print(f"Average output length: {avg_output_length:.2f} tokens")

    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f"{task_name}_results.json")
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(f"Results saved to {filename}")
    return {
        "avg_output_length": avg_output_length,
        "total_problems": len(results),
        "results_path": filename,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a vLLM language model on math tasks.")
    parser.add_argument("--model_path", type=str, required=True, help="Path or HF id of the model to evaluate.")
    parser.add_argument("--model_name", type=str, required=True, help="Name used for saving results.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional tokenizer path or HF id.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing math JSON files.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Root directory for results.")
    parser.add_argument("--tasks", type=str, nargs="+", default=["all"], help="Task names/files or 'all'.")
    parser.add_argument("--problem_field", type=str, default="auto", help="Problem field name, or 'auto'.")
    parser.add_argument("--answer_field", type=str, default="auto", help="Answer field name, or 'auto'.")
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_datapoints", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--max_model_len", type=int, default=16000)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--no_trust_remote_code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    save_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    if "all" in args.tasks:
        task_files = get_all_datasets(args.data_dir)
    else:
        task_files = [task if task.endswith(".json") else f"{task}.json" for task in args.tasks]

    llm, tokenizer = load_model_and_tokenizer(
        args.model_path,
        tokenizer_path=args.tokenizer_path,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=not args.no_trust_remote_code,
        revision=args.revision,
    )

    summary = {}
    for task_file in task_files:
        print(f"\n--- Evaluating {task_file} ---")
        dataset, dataset_path = load_dataset(args.data_dir, task_file, args.max_datapoints)
        summary[Path(dataset_path).stem] = evaluate_dataset(
            llm=llm,
            dataset=dataset,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            args=args,
            save_path=save_dir,
        )

    summary_path = os.path.join(save_dir, "generation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("Evaluation complete.")
    print("Please use verifier.sh to verify the final answers.")


if __name__ == "__main__":
    main()
