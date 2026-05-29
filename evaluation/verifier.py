#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "eval_outputs" / "math" / "lrt-math"


def progress(iterable, desc=None):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc)


VERIFICATION_PROMPT_TEMPLATE = """You are a mathematical answer validation system. Your task is to determine if two mathematical expressions are equivalent.

Are these answers mathematically equivalent? Consider these factors:
1. Different forms of the same number, including fractions, decimals, and scientific notation.
2. Algebraic equivalence, including factorization and simplification.
3. Trigonometric equivalence.
4. Logical equivalence for boolean expressions.
5. Equivalence of sets, vectors, or matrices.
6. Tolerance for small rounding errors in numerical answers.

Answer with only "True" if the expressions are equivalent or "False" if they are not.

Examples:
Target Answer: 2.0
Extracted Answer: 2
Your response: True

Target Answer: 0.5
Extracted Answer: 1/2
Your response: True

Target Answer: k = 2
Extracted Answer: 2
Your response: True

Target Answer: x^2 - 4
Extracted Answer: (x-2)(x+2)
Your response: True

Target Answer: 2y + 2z
Extracted Answer: 2(y+z)
Your response: True

Input:
Target Answer: {target}
Extracted Answer: {extracted}
Your response:
"""


GPQA_VERIFICATION_PROMPT = """You are a mathematical and scientific answer validation system. Your task is to determine if two answers are equivalent.

Are these answers semantically equivalent? Consider these factors:
1. Different forms of the same number, including fractions, decimals, and scientific notation.
2. Algebraic equivalence, including factorization and simplification.
3. Scientific answers that convey the same meaning even with different wording.
4. Partial answers that correctly identify the main entity or concept asked.
5. Tolerance for small rounding errors in numerical answers.

Answer with only "True" if the answers are equivalent or "False" if they are not.

Examples:
Target Answer: Hybridization of carbon in methane is sp3
Extracted Answer: sp3
Your response: True

Target Answer: The electric field decreases as 1/r^2 from a point charge.
Extracted Answer: E proportional to 1/r^2
Your response: True

Input:
Target Answer: {target}
Extracted Answer: {extracted}
Your response:
"""


def extract_answer(response):
    simple_matches = re.findall(r"\\boxed\{([^{}]+)\}", response)
    if simple_matches:
        return simple_matches[-1].strip()

    start = response.rfind("\\boxed{")
    if start < 0:
        return None

    cursor = start + len("\\boxed{")
    depth = 1
    chars = []
    while cursor < len(response):
        char = response[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        cursor += 1

    return None


def parse_bool(text):
    match = re.search(r"\b(true|false)\b", text.strip().lower())
    if match is None:
        return False
    return match.group(1) == "true"


def load_json_list(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list.")
    return data


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def list_result_files(dataset_dir, tasks, gpqa=False):
    if gpqa:
        return ["gpqa_results.json"]

    if tasks and "all" not in tasks:
        return [task if task.endswith(".json") else f"{task}_results.json" for task in tasks]

    return [
        name
        for name in sorted(os.listdir(dataset_dir))
        if name.endswith(".json")
        and not name.startswith(("summary", "verified_"))
        and name != "generation_summary.json"
    ]


def verify_responses(args):
    import torch
    from vllm import LLM, SamplingParams

    print(f"Loading verification model: {args.model}")
    tensor_parallel_size = args.tensor_parallel_size or max(torch.cuda.device_count(), 1)
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=not args.no_trust_remote_code,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    output_dir = args.output_dir or args.dataset_dir
    os.makedirs(output_dir, exist_ok=True)

    prompt_template = GPQA_VERIFICATION_PROMPT if args.gpqa else VERIFICATION_PROMPT_TEMPLATE
    json_files = list_result_files(args.dataset_dir, args.tasks, gpqa=args.gpqa)
    summary = {}

    for file_name in json_files:
        file_path = os.path.join(args.dataset_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Result file not found: {file_path}")

        output_path = os.path.join(output_dir, f"verified_{file_name}")
        print(f"Processing {file_path}")
        problems = load_json_list(file_path)

        total_problems = len(problems)
        verified_count = 0
        correct_count = 0

        for start in progress(range(0, total_problems, args.batch_size), desc=f"Verifying {file_name}"):
            batch = problems[start : start + args.batch_size]
            batch_prompts = []
            valid_indices = []

            for offset, problem in enumerate(batch):
                target_answer = str(problem.get("target_answer", "")).strip()
                completion = str(problem.get("completion", ""))
                extracted_answer = extract_answer(completion)

                global_index = start + offset
                problems[global_index]["extracted_answer"] = extracted_answer

                if extracted_answer is None:
                    problems[global_index]["pass_or_fail"] = False
                    problems[global_index]["verification_response"] = ""
                    continue

                prompt = prompt_template.replace("{target}", target_answer).replace(
                    "{extracted}", extracted_answer
                )
                batch_prompts.append(prompt)
                valid_indices.append(global_index)

            if not batch_prompts:
                continue

            outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
            for index, output in zip(valid_indices, outputs):
                verification_result = output.outputs[0].text.strip() if output.outputs else "False"
                is_correct = parse_bool(verification_result)
                problems[index]["verification_response"] = verification_result
                problems[index]["pass_or_fail"] = is_correct

                verified_count += 1
                correct_count += int(is_correct)

            if (start + args.batch_size) % 100 == 0 or (start + args.batch_size) >= total_problems:
                print(f"Progress: {verified_count}/{total_problems} problems verified")
                save_json(output_path, problems)

        save_json(output_path, problems)

        dataset_name = file_name.replace("_results.json", "").replace(".json", "")
        accuracy = correct_count / total_problems if total_problems else 0.0
        summary[dataset_name] = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_problems,
            "verified_count": verified_count,
        }

        print(f"Verification completed for {file_name}")
        print(f"Total problems: {total_problems}")
        print(f"Verified problems: {verified_count}")
        print(f"Correct problems: {correct_count}")
        print(f"Accuracy: {accuracy:.4f}")

    summary_path = os.path.join(output_dir, "summary.json")
    save_json(summary_path, summary)
    print(f"Summary saved to {summary_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Verify math responses using LLM equivalence judging.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B", help="Verification model name.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Directory containing result JSON files.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save verified results.")
    parser.add_argument("--tasks", type=str, nargs="+", default=["all"], help="Tasks/files to verify or 'all'.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpqa", action="store_true", help="Use the science/GPQA verifier prompt.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--max_model_len", type=int, default=8000)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--no_trust_remote_code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    verify_responses(args)


if __name__ == "__main__":
    main()
