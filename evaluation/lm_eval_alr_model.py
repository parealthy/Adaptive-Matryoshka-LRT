from __future__ import annotations

import atexit
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open
from tqdm import tqdm


def _patch_transformers_harness_compat() -> None:
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


_patch_transformers_harness_compat()

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


DEFAULT_LATENT_TRAJECTORY_LENGTHS = [64, 128, 192, 256]
MATH_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _parse_latent_trajectory_lengths(value: Optional[str]) -> list[int]:
    if not value:
        return list(DEFAULT_LATENT_TRAJECTORY_LENGTHS)
    tokens = [token for token in re.split(r"[,\s]+", value.strip()) if token]
    return sorted({int(token) for token in tokens})


def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_latent_trajectory_lengths(
    stage1_checkpoint_path: str,
    difficulty_checkpoint_path: Optional[str],
    cli_value: Optional[str],
) -> list[int]:
    if cli_value:
        return _parse_latent_trajectory_lengths(cli_value)

    if difficulty_checkpoint_path:
        difficulty_config = _load_json_if_exists(
            Path(difficulty_checkpoint_path) / "alr_difficulty_config.json",
        )
        if difficulty_config.get("latent_trajectory_lengths"):
            return sorted({int(x) for x in difficulty_config["latent_trajectory_lengths"]})

    stage1_config = _load_json_if_exists(
        Path(stage1_checkpoint_path) / "alr_stage1_config.json",
    )
    if stage1_config.get("latent_trajectory_lengths"):
        return sorted({int(x) for x in stage1_config["latent_trajectory_lengths"]})

    return list(DEFAULT_LATENT_TRAJECTORY_LENGTHS)


def _resolve_hidden_size(config) -> int:
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    raise ValueError("Failed to resolve hidden_size from the base model config.")


def _resolve_torch_dtype(torch_dtype):
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return dtype_map[str(torch_dtype).lower()]
    except KeyError as exc:
        raise ValueError(
            "torch_dtype must be one of auto, bf16, bfloat16, fp16, float16, fp32, float32."
        ) from exc


def _distributed_env() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def _setup_distributed() -> tuple[int, int, int]:
    rank, local_rank, world_size = _distributed_env()
    if world_size <= 1:
        return rank, local_rank, world_size

    import torch.distributed as dist

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if dist.is_available() and not dist.is_initialized():
        backend = os.environ.get("ALR_LM_EVAL_DIST_BACKEND", "gloo")
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(days=7),
        )
    return rank, local_rank, world_size


class _TorchDistributedAccelerator:
    def __init__(self, rank: int, local_rank: int, world_size: int):
        self.local_process_index = rank
        self.process_index = rank
        self.num_processes = world_size
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

    @property
    def is_local_main_process(self) -> bool:
        return self.local_process_index == 0

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.num_processes <= 1:
            return tensor
        import torch.distributed as dist

        gather_tensor = tensor
        if dist.get_backend() == "gloo" and tensor.is_cuda:
            gather_tensor = tensor.detach().cpu()

        gathered = [torch.empty_like(gather_tensor) for _ in range(self.num_processes)]
        dist.all_gather(gathered, gather_tensor)
        if tensor.dim() == 0:
            return torch.stack(gathered)
        return torch.cat(gathered, dim=0)

    def wait_for_everyone(self) -> None:
        if self.num_processes <= 1:
            return
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()


def _resolve_device_map(device: str, local_rank: int, world_size: int):
    if device == "auto":
        if not torch.cuda.is_available():
            return {"": "cpu"}
        if world_size > 1:
            return {"": local_rank}
        return {"": 0}
    if device == "cuda":
        if world_size > 1:
            return {"": local_rank}
        return {"": 0}
    if device.startswith("cuda:"):
        return {"": int(device.split(":", 1)[1])}
    return {"": device}


def _model_input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def _load_reasoning_weights(reasoning_network, checkpoint_path: str) -> None:
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


def _chunks(items: List, size: int) -> Iterable[List]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _normalize_until(until) -> List[str]:
    if until is None:
        return []
    if isinstance(until, str):
        return [until]
    return [str(item) for item in until if str(item)]


def _truncate_at_stop(text: str, stop_sequences: List[str]) -> str:
    stop_positions = [text.find(stop) for stop in stop_sequences if stop]
    stop_positions = [pos for pos in stop_positions if pos >= 0]
    if not stop_positions:
        return text
    return text[: min(stop_positions)]


def _strip_prompt_if_present(text: str, prompt: str) -> str:
    if prompt and text.startswith(prompt):
        return text[len(prompt) :]
    return text


def _parse_method(method: str) -> tuple[str, Optional[int]]:
    normalized = method.strip().lower()
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


def _sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _append_math_suffix(problem: str) -> str:
    problem = str(problem).strip()
    if not problem.endswith(MATH_SUFFIX):
        problem = problem + MATH_SUFFIX
    return problem


def _has_tokenizer_files(path: Path) -> bool:
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


def _resolve_tokenizer_path(
    model_path: str,
    stage1_checkpoint_path: str,
    tokenizer_path: Optional[str],
) -> str:
    if tokenizer_path:
        return tokenizer_path

    stage1_path = Path(stage1_checkpoint_path).expanduser()
    if _has_tokenizer_files(stage1_path):
        return str(stage1_path)
    return model_path


@register_model("alr")
class ALRLM(LM):
    """ALR length-elastic generation backend for lm-evaluation-harness."""

    def __init__(
        self,
        lrt_root: str,
        model_path: str,
        reasoning_net_path: str,
        stage1_checkpoint_path: str,
        difficulty_checkpoint_path: Optional[str] = None,
        latent_trajectory_lengths: Optional[str] = None,
        method: str = "adaptive",
        prompt_max_length: int = 1024,
        max_gen_toks: int = 2048,
        batch_size=1,
        device: str = "auto",
        torch_dtype="bf16",
        trust_remote_code=True,
        top_p: float = 0.95,
        seed: int = 42,
        trace_path: Optional[str] = None,
        trace_task: Optional[str] = None,
        use_training_prompt_template=True,
        tokenizer_path: Optional[str] = None,
        local_files_only=False,
        use_fast_tokenizer=True,
        **_kwargs,
    ) -> None:
        super().__init__()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        rank, local_rank, world_size = _setup_distributed()
        self._rank = rank
        self._world_size = world_size
        self.local_rank = local_rank
        self.accelerator = _TorchDistributedAccelerator(rank, local_rank, world_size)

        lrt_root_path = Path(lrt_root).expanduser().resolve()
        if not lrt_root_path.exists():
            raise FileNotFoundError(f"lrt_root does not exist: {lrt_root}")
        if str(lrt_root_path) not in sys.path:
            sys.path.insert(0, str(lrt_root_path))

        from modeling.alr_difficulty import DifficultyEstimator
        from modeling.alr_stage1 import LengthElasticTransformerReasoningNet

        self.DifficultyEstimator = DifficultyEstimator
        self.lrt_root = str(lrt_root_path)
        self.model_path = model_path
        self.reasoning_net_path = reasoning_net_path
        self.stage1_checkpoint_path = stage1_checkpoint_path
        self.difficulty_checkpoint_path = difficulty_checkpoint_path
        self.latent_trajectory_lengths = _resolve_latent_trajectory_lengths(
            stage1_checkpoint_path,
            difficulty_checkpoint_path,
            latent_trajectory_lengths,
        )
        self.method = method
        self.method_kind, self.fixed_length = _parse_method(method)
        self.prompt_max_length = int(prompt_max_length)
        self._max_gen_toks = int(max_gen_toks)
        self._batch_size = 1 if str(batch_size) == "auto" else int(batch_size)
        self.top_p = float(top_p)
        self.seed = int(seed)
        self.rng = random.Random(self.seed + self.rank)
        self.trust_remote_code = _as_bool(trust_remote_code)
        self.trace_task = trace_task
        self.use_training_prompt_template = _as_bool(use_training_prompt_template)
        self.local_files_only = _as_bool(local_files_only)
        self.use_fast_tokenizer = _as_bool(use_fast_tokenizer)
        self.tokenizer_path = _resolve_tokenizer_path(
            model_path=model_path,
            stage1_checkpoint_path=stage1_checkpoint_path,
            tokenizer_path=tokenizer_path,
        )
        self._prompt_fallback_warning_printed = False
        self._trace_counter = 0
        self._trace_handle = self._open_trace(trace_path)

        if self.method_kind == "fixed" and self.fixed_length not in self.latent_trajectory_lengths:
            raise ValueError(
                f"Fixed length {self.fixed_length} is not one of "
                f"{self.latent_trajectory_lengths}."
            )
        if self.method_kind == "adaptive" and not difficulty_checkpoint_path:
            raise ValueError("method=adaptive requires difficulty_checkpoint_path.")

        if self.rank == 0:
            print(
                f"Loading tokenizer from {self.tokenizer_path} "
                f"(local_files_only={self.local_files_only}, use_fast={self.use_fast_tokenizer})",
                flush=True,
            )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=self.trust_remote_code,
                use_fast=self.use_fast_tokenizer,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            if not self.use_fast_tokenizer:
                raise RuntimeError(
                    f"Failed to load tokenizer from {self.tokenizer_path}. "
                    "Set TOKENIZER_PATH to a local tokenizer directory, usually the "
                    "Stage1 checkpoint directory."
                ) from exc
            if self.rank == 0:
                print(
                    "Fast tokenizer loading failed; retrying with use_fast=False. "
                    f"Original error: {exc}",
                    flush=True,
                )
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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=_resolve_torch_dtype(torch_dtype),
            device_map=_resolve_device_map(device, self.local_rank, self.world_size),
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True,
            local_files_only=self.local_files_only,
        )
        self.model.eval()
        self.input_device = _model_input_device(self.model)
        self._device = self.input_device

        self.reasoning_network = LengthElasticTransformerReasoningNet(
            reasoning_net_path,
            latent_trajectory_length=max(self.latent_trajectory_lengths),
            hidden_size=_resolve_hidden_size(self.model.config),
            local_files_only=self.local_files_only,
        )
        self.reasoning_network.to(self.input_device)
        self.reasoning_network.eval()
        _load_reasoning_weights(self.reasoning_network, stage1_checkpoint_path)

        self.difficulty_estimator = None
        if self.method_kind == "adaptive":
            self.difficulty_estimator = DifficultyEstimator.from_pretrained(
                difficulty_checkpoint_path,
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

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def tokenizer_name(self) -> str:
        return self.model_path

    @property
    def device(self) -> torch.device:
        return getattr(self, "_device", self.accelerator.device)

    def chat_template(self, chat_template=False) -> str:
        if not chat_template:
            return ""
        template = getattr(self.tokenizer, "chat_template", None)
        return template or ""

    def apply_chat_template(
        self,
        chat_history: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def loglikelihood(self, requests, disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
        raise NotImplementedError("ALRLM currently supports generate_until tasks only.")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        raise NotImplementedError("ALRLM does not implement loglikelihood_rolling.")

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        indexed_requests = [
            (index, request, request.args[0], dict(request.args[1]))
            for index, request in enumerate(requests)
        ]
        results: List[Optional[str]] = [None] * len(indexed_requests)

        grouped: Dict[str, List[Tuple[int, object, str, dict]]] = {}
        for item in indexed_requests:
            gen_kwargs = item[3]
            key = json.dumps(gen_kwargs, sort_keys=True, default=str)
            grouped.setdefault(key, []).append(item)

        pbar = tqdm(
            total=len(indexed_requests),
            disable=disable_tqdm or self.rank != 0,
            desc=f"Running ALR {self.method} generate_until requests",
        )
        for group in grouped.values():
            for batch in _chunks(group, self.batch_size):
                contexts = [item[2] for item in batch]
                prompt_infos = [
                    self._render_prompt_for_request(request=item[1], context=item[2])
                    for item in batch
                ]
                rendered_prompts = [prompt for prompt, _metadata in prompt_infos]
                prompt_metadata = [metadata for _prompt, metadata in prompt_infos]
                gen_kwargs = dict(batch[0][3])
                batch_result = self._generate_batch(rendered_prompts, gen_kwargs)
                outputs = batch_result["completions"]
                for local_index, ((original_index, request, context, original_kwargs), output) in enumerate(
                    zip(batch, outputs)
                ):
                    results[original_index] = output
                    self.cache_hook.add_partial(
                        "generate_until",
                        (context, original_kwargs),
                        output,
                    )
                    self._write_trace(
                        request=request,
                        context=context,
                        rendered_prompt=rendered_prompts[local_index],
                        prompt_metadata=prompt_metadata[local_index],
                        completion=output,
                        latent_length=batch_result["latent_lengths"][local_index],
                        predicted_label=batch_result["predicted_labels"][local_index],
                        length_probabilities=batch_result["length_probabilities"][local_index],
                        output_tokens=batch_result["output_token_counts"][local_index],
                    )
                    pbar.update(1)

        pbar.close()
        if self._trace_handle is not None:
            self._trace_handle.flush()
        return [result if result is not None else "" for result in results]

    def _extract_problem_from_request(self, request, context: str) -> Optional[str]:
        doc = getattr(request, "doc", None)
        if isinstance(doc, dict):
            for field_name in ("problem", "question"):
                value = doc.get(field_name)
                if value:
                    return str(value).strip()

        context = str(context)
        problem_matches = re.findall(
            r"Problem:\s*(.*?)(?:\n\s*\n\s*Solution:|\n\s*Answer:)",
            context,
            re.S,
        )
        if problem_matches:
            return problem_matches[-1].strip()

        question_matches = re.findall(r"Question:\s*(.*?)(?:\n\s*Answer:)", context, re.S)
        if question_matches:
            return question_matches[-1].strip()
        return None

    def _render_training_prompt(self, problem: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": _append_math_suffix(problem)}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _render_prompt_for_request(self, request, context: str) -> tuple[str, dict[str, str]]:
        if not self.use_training_prompt_template:
            return context, {"prompt_template": "harness_context"}

        problem = self._extract_problem_from_request(request, context)
        if problem is None:
            if self.rank == 0 and not self._prompt_fallback_warning_printed:
                print(
                    "Warning: failed to recover the raw problem from lm-eval request; "
                    "falling back to the harness context for this and similar samples.",
                    flush=True,
                )
                self._prompt_fallback_warning_printed = True
            return context, {"prompt_template": "harness_context_fallback"}

        rendered_prompt = self._render_training_prompt(problem)
        return rendered_prompt, {
            "prompt_template": "training_chat_math_suffix",
            "problem_sha1": _sha1_text(_append_math_suffix(problem)),
        }

    def _open_trace(self, trace_path: Optional[str]):
        if not trace_path:
            return None
        path_text = str(trace_path)
        if "{rank" in path_text:
            path = Path(path_text.format(rank=self.rank, local_rank=self.local_rank))
        else:
            path = Path(path_text)
            if path.suffix != ".jsonl":
                path = path / f"rank{self.rank:05d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8", buffering=1)
        atexit.register(handle.close)
        return handle

    def _write_trace(
        self,
        request,
        context: str,
        rendered_prompt: str,
        prompt_metadata: dict[str, str],
        completion: str,
        latent_length: int,
        predicted_label: Optional[int],
        length_probabilities: Optional[dict[str, float]],
        output_tokens: int,
    ) -> None:
        if self._trace_handle is None:
            return
        task_name = getattr(request, "task_name", None) or self.trace_task
        doc_id = getattr(request, "doc_id", None)
        payload = {
            "task": task_name,
            "method": self.method,
            "rank": self.rank,
            "request_index": self._trace_counter,
            "index": doc_id if doc_id is not None else self._trace_counter,
            "doc_id": doc_id,
            "instance_idx": getattr(request, "idx", None),
            "predicted_label": predicted_label,
            "predicted_length": int(latent_length),
            "latent_length": int(latent_length),
            "output_tokens": int(output_tokens),
            "context_sha1": _sha1_text(context),
            "rendered_prompt_sha1": _sha1_text(rendered_prompt),
            "completion_sha1": _sha1_text(completion),
        }
        payload.update(prompt_metadata)
        if length_probabilities is not None:
            payload["length_probabilities"] = length_probabilities
        self._trace_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._trace_counter += 1

    def _normalize_generation_kwargs(self, gen_kwargs: dict) -> tuple[int, List[str], dict]:
        kwargs = dict(gen_kwargs)
        max_new_tokens = int(
            kwargs.pop(
                "max_gen_toks",
                kwargs.pop("max_new_tokens", kwargs.pop("max_tokens", self.max_gen_toks)),
            )
        )
        until = _normalize_until(kwargs.pop("until", None))

        temperature = float(kwargs.pop("temperature", 0.0))
        do_sample = _as_bool(kwargs.pop("do_sample", temperature > 0.0))
        top_p = float(kwargs.pop("top_p", self.top_p))

        hf_kwargs = {
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            hf_kwargs["temperature"] = temperature
        if do_sample:
            hf_kwargs["top_p"] = top_p

        for unsupported_key in (
            "max_tokens_thinking",
            "thinking_n_ignore",
            "thinking_n_ignore_str",
            "thinking_start",
            "thinking_end",
            "until_thinking",
            "rejection_sample",
        ):
            kwargs.pop(unsupported_key, None)

        hf_kwargs.update(kwargs)
        return max_new_tokens, until, hf_kwargs

    @torch.no_grad()
    def _generate_batch(self, prompts: List[str], gen_kwargs: dict) -> dict:
        max_new_tokens, until, hf_kwargs = self._normalize_generation_kwargs(gen_kwargs)

        inputs = self.tokenizer(
            prompts,
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
            hidden_states = _last_layer_hidden_states(
                self.model,
                prompt_embeddings,
                attention_mask,
            )
            length_info = self._select_latent_lengths(hidden_states, attention_mask)

            completions = ["" for _ in prompts]
            output_token_counts = [0 for _ in prompts]
            latent_to_indices: dict[int, list[int]] = defaultdict(list)
            for index, latent_length in enumerate(length_info["latent_lengths"]):
                latent_to_indices[int(latent_length)].append(index)

            for latent_length, indices in sorted(latent_to_indices.items()):
                index_tensor = torch.tensor(indices, dtype=torch.long, device=self.input_device)
                batch_outputs = self._generate_prefilled_fixed_length(
                    prompt_embeddings.index_select(0, index_tensor),
                    hidden_states.index_select(0, index_tensor),
                    attention_mask.index_select(0, index_tensor),
                    [prompts[index] for index in indices],
                    latent_length,
                    max_new_tokens,
                    until,
                    hf_kwargs,
                )
                for local_index, global_index in enumerate(indices):
                    completions[global_index] = batch_outputs["completions"][local_index]
                    output_token_counts[global_index] = batch_outputs["output_token_counts"][
                        local_index
                    ]

        return {
            "completions": completions,
            "latent_lengths": length_info["latent_lengths"],
            "predicted_labels": length_info["predicted_labels"],
            "length_probabilities": length_info["length_probabilities"],
            "output_token_counts": output_token_counts,
        }

    def _select_latent_lengths(self, hidden_states, attention_mask) -> dict:
        batch_size = hidden_states.size(0)
        probabilities: list[dict[str, float] | None] = [None for _ in range(batch_size)]
        predicted_labels: list[int | None] = [None for _ in range(batch_size)]

        if self.method_kind == "fixed":
            latent_lengths = [int(self.fixed_length) for _ in range(batch_size)]
        elif self.method_kind == "random":
            latent_lengths = [self.rng.choice(self.latent_trajectory_lengths) for _ in range(batch_size)]
        else:
            pooled = self.DifficultyEstimator.mean_pool_hidden_states(
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

        return {
            "latent_lengths": latent_lengths,
            "predicted_labels": predicted_labels,
            "length_probabilities": probabilities,
        }

    def _generate_prefilled_fixed_length(
        self,
        prompt_embeddings,
        hidden_states,
        attention_mask,
        prompts: list[str],
        latent_length: int,
        max_new_tokens: int,
        until: list[str],
        hf_kwargs: dict,
    ) -> dict:
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

        output_ids = self.model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            **hf_kwargs,
        )
        decoded_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        completions = []
        output_token_counts = []
        for prompt, output in zip(prompts, decoded_outputs):
            continuation = _strip_prompt_if_present(output, prompt)
            continuation = _truncate_at_stop(continuation, until)
            completions.append(continuation)
            output_token_counts.append(
                len(self.tokenizer(continuation, add_special_tokens=False).input_ids)
            )
        return {
            "completions": completions,
            "output_token_counts": output_token_counts,
        }
