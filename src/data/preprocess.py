"""Tokenization and class-imbalance utilities for SCOTUS classification.

SCOTUS opinions frequently exceed the model's context window; we truncate to the first
`max_length` tokens rather than chunking or summarizing (documented limitation, not solved
in the initial version of this project).
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase


def tokenize_batch(examples: dict, tokenizer: PreTrainedTokenizerBase, max_length: int = 2048) -> dict:
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def compute_class_weights(labels: list[int], num_labels: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1.0, for weighted cross-entropy."""
    counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero for any unseen class
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(labels: list[int], num_labels: int) -> torch.utils.data.WeightedRandomSampler:
    """Per-example sampling weights (alternative comparison to loss weighting)."""
    class_weights = compute_class_weights(labels, num_labels)
    sample_weights = class_weights[torch.as_tensor(labels)]
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
