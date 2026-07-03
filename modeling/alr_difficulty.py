from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import torch
import torch.nn as nn


DIFFICULTY_CONFIG_NAME = "alr_difficulty_config.json"
DIFFICULTY_WEIGHTS_NAME = "alr_difficulty_head.pt"


def normalize_latent_lengths(latent_trajectory_lengths: Sequence[int]) -> list[int]:
    lengths = sorted({int(length) for length in latent_trajectory_lengths})
    if not lengths:
        raise ValueError("latent_trajectory_lengths must contain at least one length.")
    if any(length <= 0 for length in lengths):
        raise ValueError("latent_trajectory_lengths must contain only positive integers.")
    return lengths


class DifficultyEstimator(nn.Module):
    """Tiny MLP head that predicts an ALR latent prefix length.

    The head consumes frozen LLM prompt hidden states, pools them with the
    attention mask, and classifies one of the configured latent lengths.
    """

    def __init__(
        self,
        hidden_size: int,
        latent_trajectory_lengths: Sequence[int],
        mlp_hidden_size: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.latent_trajectory_lengths = normalize_latent_lengths(
            latent_trajectory_lengths,
        )
        self.mlp_hidden_size = int(mlp_hidden_size)
        self.dropout = float(dropout)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_hidden_size, len(self.latent_trajectory_lengths)),
        )

    @property
    def num_classes(self) -> int:
        return len(self.latent_trajectory_lengths)

    def length_to_label(self, length: int) -> int:
        length = int(length)
        if length not in self.latent_trajectory_lengths:
            raise ValueError(
                f"Unknown latent length {length}; expected one of "
                f"{self.latent_trajectory_lengths}."
            )
        return self.latent_trajectory_lengths.index(length)

    def label_to_length(self, label: int) -> int:
        label = int(label)
        if label < 0 or label >= len(self.latent_trajectory_lengths):
            raise ValueError(
                f"Unknown difficulty label {label}; expected an integer in "
                f"[0, {len(self.latent_trajectory_lengths) - 1}]."
            )
        return self.latent_trajectory_lengths[label]

    @staticmethod
    def mean_pool_hidden_states(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch_size, seq_length, hidden_size]."
            )

        if attention_mask is None:
            return hidden_states.mean(dim=1)

        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch_size, seq_length].")
        if attention_mask.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                "attention_mask shape must match the first two hidden_states dimensions."
            )

        mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        mask = mask.unsqueeze(-1)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pooled_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pooled_hidden_states is None:
            if hidden_states is None:
                raise ValueError(
                    "Either hidden_states or pooled_hidden_states must be provided."
                )
            pooled_hidden_states = self.mean_pool_hidden_states(
                hidden_states,
                attention_mask,
            )

        if pooled_hidden_states.ndim != 2:
            raise ValueError(
                "pooled_hidden_states must have shape [batch_size, hidden_size]."
            )
        return self.classifier(pooled_hidden_states)

    @torch.no_grad()
    def predict_labels(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pooled_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            pooled_hidden_states=pooled_hidden_states,
        )
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def predict_lengths(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pooled_hidden_states: Optional[torch.Tensor] = None,
    ) -> list[int]:
        labels = self.predict_labels(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            pooled_hidden_states=pooled_hidden_states,
        )
        return [self.label_to_length(int(label)) for label in labels.cpu().tolist()]

    def to_config(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "latent_trajectory_lengths": self.latent_trajectory_lengths,
            "mlp_hidden_size": self.mlp_hidden_size,
            "dropout": self.dropout,
        }

    def save_pretrained(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, DIFFICULTY_CONFIG_NAME)
        weights_path = os.path.join(output_dir, DIFFICULTY_WEIGHTS_NAME)
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_config(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        torch.save(self.state_dict(), weights_path)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        map_location: str | torch.device = "cpu",
    ) -> "DifficultyEstimator":
        config_path = os.path.join(checkpoint_dir, DIFFICULTY_CONFIG_NAME)
        weights_path = os.path.join(checkpoint_dir, DIFFICULTY_WEIGHTS_NAME)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing difficulty config: {config_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing difficulty weights: {weights_path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        model = cls(**config)
        state_dict = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state_dict, strict=True)
        return model
