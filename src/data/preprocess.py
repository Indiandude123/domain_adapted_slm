"""Text cleaning, tokenization, and class-imbalance utilities for SCOTUS classification.

SCOTUS opinions frequently exceed the model's context window; we truncate to the first
`max_length` tokens rather than chunking or summarizing (documented limitation, not solved
in the initial version of this project).
"""
from __future__ import annotations

import re

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

# ~10% of SCOTUS train docs contain this OCR/scan placeholder where the original page of
# oral-argument transcript wasn't digitized. It carries no classification signal and just
# eats into the truncation budget, so it's stripped rather than left in.
_OMITTED_ARGUMENT_RE = re.compile(r"\[.*?intentionally omitted\]", re.IGNORECASE | re.DOTALL)
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Strip scanning artifacts and normalize whitespace in raw SCOTUS opinion text.

    Deliberately conservative: the citation/attorney front matter is left in place since its
    extent isn't reliably detectable across the dataset's document eras/formats, and cutting
    it with a fragile heuristic risks losing real content in edge cases.
    """
    text = _OMITTED_ARGUMENT_RE.sub(" ", text)
    text = _INLINE_WHITESPACE_RE.sub(" ", text)
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


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
