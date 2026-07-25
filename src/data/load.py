"""Load the LexGLUE SCOTUS dataset: 14-class, single-label, imbalanced legal case classification."""
from __future__ import annotations

from datasets import DatasetDict, load_dataset

from src.data.preprocess import clean_text

DATASET_NAME = "coastalcph/lex_glue"
DATASET_CONFIG = "scotus"

# Supreme Court Database "issueArea" categories used as SCOTUS labels in LexGLUE.
SCOTUS_LABELS = [
    "Criminal Procedure",
    "Civil Rights",
    "First Amendment",
    "Due Process",
    "Privacy",
    "Attorneys",
    "Unions",
    "Economic Activity",
    "Judicial Power",
    "Federalism",
    "Interstate Relations",
    "Federal Taxation",
    "Miscellaneous",
    "Private Action",
]


def load_scotus() -> DatasetDict:
    """Load train/validation/test splits for the SCOTUS subtask of LexGLUE, with basic
    text cleaning applied (see `src.data.preprocess.clean_text`)."""
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    return dataset.map(lambda ex: {"text": clean_text(ex["text"])})


def label_names() -> list[str]:
    return SCOTUS_LABELS
