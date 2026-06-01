from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel


class LengthElasticReasoningNetBase(torch.nn.Module):
    def __init__(self, latent_trajectory_length=256, hidden_size=1024):
        super(LengthElasticReasoningNetBase, self).__init__()

        self.latent_trajectory = nn.Parameter(
            torch.randn(latent_trajectory_length, hidden_size),
            requires_grad=True,
        )

        self.latent_trajectory_length = latent_trajectory_length
        self.hidden_size = hidden_size

    def _validate_active_latent_length(self, active_latent_length: Optional[int]) -> int:
        if active_latent_length is None:
            return self.latent_trajectory_length

        active_latent_length = int(active_latent_length)
        if active_latent_length <= 0:
            raise ValueError("active_latent_length must be a positive integer.")
        if active_latent_length > self.latent_trajectory_length:
            raise ValueError(
                "active_latent_length cannot exceed the initialized "
                f"latent_trajectory_length ({self.latent_trajectory_length})."
            )
        return active_latent_length


class LengthElasticTransformerReasoningNet(LengthElasticReasoningNetBase):
    def __init__(self, model_name_or_path, latent_trajectory_length=256, hidden_size=1024):
        super(LengthElasticTransformerReasoningNet, self).__init__(
            latent_trajectory_length,
            hidden_size,
        )

        self.reasoning_network = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        self.reasoning_network.embed_tokens = None

        if self.reasoning_network.config.hidden_size != self.hidden_size:
            self.transform_layer = torch.nn.Linear(
                self.hidden_size,
                self.reasoning_network.config.hidden_size,
            )
            self.reverse_transform_layer = torch.nn.Linear(
                self.reasoning_network.config.hidden_size,
                self.hidden_size,
            )
            self.transform_layer.to(self.reasoning_network.dtype)
            self.reverse_transform_layer.to(self.reasoning_network.dtype)
        else:
            self.transform_layer = torch.nn.Identity()
            self.reverse_transform_layer = torch.nn.Identity()

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        active_latent_length: Optional[int] = None,
    ):
        active_latent_length = self._validate_active_latent_length(active_latent_length)
        hidden_state_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.reasoning_network.dtype)

        latent_trajectory_base = self.latent_trajectory[:active_latent_length]
        latent_trajectory_base = latent_trajectory_base.unsqueeze(0).expand(
            hidden_states.size(0),
            -1,
            -1,
        )
        latent_trajectory_base = latent_trajectory_base.to(self.reasoning_network.dtype)

        last_hidden_state = hidden_states[:, -1:, :]
        latent_trajectory_input = last_hidden_state * latent_trajectory_base

        transformed_hidden_states = self.transform_layer(hidden_states)
        transformed_latent_trajectory = self.transform_layer(latent_trajectory_input)
        reasoning_inputs = torch.cat(
            [transformed_hidden_states, transformed_latent_trajectory],
            dim=1,
        )

        reasoning_mask = None
        if attention_mask is not None:
            latent_mask = torch.ones(
                hidden_states.size(0),
                active_latent_length,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            reasoning_mask = torch.cat([attention_mask, latent_mask], dim=1)

        outputs = self.reasoning_network(
            inputs_embeds=reasoning_inputs,
            attention_mask=reasoning_mask,
            return_dict=True,
        )
        latent_trajectory_output = outputs.last_hidden_state[:, -active_latent_length:, :]
        latent_trajectory = self.reverse_transform_layer(latent_trajectory_output)
        return latent_trajectory.to(hidden_state_dtype)


class LengthElasticLatentTransformerReasoningModel(torch.nn.Module):
    def __init__(
        self,
        slow_reasoning_model,
        processor,
        reasoning_network,
        latent_trajectory_lengths: Sequence[int],
        active_latent_length: Optional[int] = None,
        log_active_length: bool = True,
        active_length_log_interval: int = 100,
        **kwargs,
    ):
        super(LengthElasticLatentTransformerReasoningModel, self).__init__(**kwargs)
        self.slow_reasoning_model = slow_reasoning_model
        self.processor = processor
        self.reasoning_network = reasoning_network
        self.latent_trajectory_length = reasoning_network.latent_trajectory_length
        self.latent_trajectory_lengths = self._validate_latent_trajectory_lengths(
            latent_trajectory_lengths,
        )
        self.active_latent_length = (
            None
            if active_latent_length is None
            else self._validate_active_latent_length(active_latent_length)
        )
        self.log_active_length = log_active_length
        self.active_length_log_interval = max(int(active_length_log_interval), 1)
        self._active_length_calls = 0

        self.slow_reasoning_model.requires_grad_(False)
        self.reasoning_network.requires_grad_(True)

        self.config = self.slow_reasoning_model.config

    def _validate_latent_trajectory_lengths(self, lengths: Sequence[int]) -> list[int]:
        if not lengths:
            raise ValueError("latent_trajectory_lengths must contain at least one length.")

        normalized = sorted({int(length) for length in lengths})
        if any(length <= 0 for length in normalized):
            raise ValueError("latent_trajectory_lengths must contain only positive integers.")
        if normalized[-1] > self.latent_trajectory_length:
            raise ValueError(
                "The largest configured latent trajectory length cannot exceed "
                f"the initialized latent_trajectory_length ({self.latent_trajectory_length})."
            )
        return normalized

    def _validate_active_latent_length(self, active_latent_length: int) -> int:
        active_latent_length = int(active_latent_length)
        if active_latent_length not in self.latent_trajectory_lengths:
            raise ValueError(
                "active_latent_length must be one of "
                f"{self.latent_trajectory_lengths}, got {active_latent_length}."
            )
        return active_latent_length

    @property
    def get_input_embeddings(self):
        return self.slow_reasoning_model.get_input_embeddings()

    @property
    def tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    @property
    def pad_token_id(self):
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("The tokenizer must define either pad_token_id or eos_token_id.")
        return pad_token_id

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.slow_reasoning_model, "gradient_checkpointing_enable"):
            try:
                if gradient_checkpointing_kwargs:
                    self.slow_reasoning_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
                    )
                else:
                    self.slow_reasoning_model.gradient_checkpointing_enable()
            except TypeError:
                self.slow_reasoning_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        if hasattr(self.slow_reasoning_model, "gradient_checkpointing_disable"):
            self.slow_reasoning_model.gradient_checkpointing_disable()

    def _select_active_latent_length(
        self,
        explicit_active_latent_length: Optional[int],
        device: torch.device,
    ) -> int:
        if explicit_active_latent_length is not None:
            return self._validate_active_latent_length(explicit_active_latent_length)

        if self.active_latent_length is not None:
            return self.active_latent_length

        if self.training:
            sampled_idx = torch.randint(
                len(self.latent_trajectory_lengths),
                (1,),
                device=device,
            ).item()
            return self.latent_trajectory_lengths[sampled_idx]

        return self.latent_trajectory_lengths[-1]

    def _maybe_log_active_length(self, active_latent_length: int) -> None:
        if not self.training or not self.log_active_length:
            return

        self._active_length_calls += 1
        if (
            self._active_length_calls <= 5
            or self._active_length_calls % self.active_length_log_interval == 0
        ):
            print(
                "[ALR Stage1] "
                f"active_latent_length={active_latent_length} "
                f"candidates={self.latent_trajectory_lengths}",
                flush=True,
            )

    def _prefill_prompt(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        prompt_embeddings = self.get_input_embeddings(input_ids).to(
            self.slow_reasoning_model.dtype,
        )

        output = self.slow_reasoning_model(
            inputs_embeds=prompt_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
            **kwargs,
        )

        prompt_hidden_states = output.hidden_states[-1].detach()
        return prompt_embeddings, prompt_hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        active_latent_length: Optional[int] = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided.")
        if labels is None:
            raise ValueError("labels must be provided for ALR Stage1 SFT training.")

        if attention_mask is None:
            attention_mask = torch.ones(
                (input_ids.size(0), input_ids.size(1) + labels.size(1)),
                device=input_ids.device,
                dtype=torch.long,
            )

        active_latent_length = self._select_active_latent_length(
            active_latent_length,
            input_ids.device,
        )
        self._maybe_log_active_length(active_latent_length)

        with torch.no_grad():
            prompt_mask = attention_mask[:, :input_ids.size(1)]
            prompt_embeddings, prompt_hidden_states = self._prefill_prompt(
                input_ids=input_ids,
                attention_mask=prompt_mask,
                position_ids=position_ids,
                **kwargs,
            )

        latent_trajectory = self.reasoning_network(
            prompt_hidden_states,
            attention_mask=prompt_mask,
            active_latent_length=active_latent_length,
        )
        latent_trajectory_mask = torch.ones(
            latent_trajectory.size(0),
            latent_trajectory.size(1),
            device=input_ids.device,
            dtype=prompt_mask.dtype,
        )

        label_embeddings = self.get_input_embeddings(labels).to(
            self.slow_reasoning_model.dtype,
        )
        labels_mask = attention_mask[:, input_ids.size(1):]

        input_embeddings = torch.cat(
            [prompt_embeddings, latent_trajectory, label_embeddings],
            dim=1,
        )
        input_mask = torch.cat(
            [prompt_mask, latent_trajectory_mask, labels_mask],
            dim=1,
        ).long()

        # Mask padding by attention mask rather than token id. Some chat
        # tokenizers reuse eos as pad, and masking by pad_token_id would remove
        # the supervised EOS target.
        labels = labels.masked_fill(labels_mask == 0, -100).long()
        ignore_prompt = torch.full(
            (labels.size(0), prompt_embeddings.size(1)),
            -100,
            device=labels.device,
            dtype=torch.long,
        )
        ignore_latent = torch.full(
            (labels.size(0), latent_trajectory.size(1)),
            -100,
            device=labels.device,
            dtype=torch.long,
        )
        labels = torch.cat((ignore_prompt, ignore_latent, labels), dim=1).long()

        outputs = self.slow_reasoning_model(
            inputs_embeds=input_embeddings,
            attention_mask=input_mask,
            labels=labels,
            return_dict=True,
            **kwargs,
        )

        return outputs
