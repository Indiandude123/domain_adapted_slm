"""QLoRA model construction: 4-bit quantized base + LoRA adapters for sequence classification."""
from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_quantized_classifier(
    model_name: str, num_labels: int, compute_dtype: torch.dtype = torch.float16
) -> AutoModelForSequenceClassification:
    """Load a 4-bit quantized sequence classifier.

    `compute_dtype` defaults to fp16, not bf16: on Turing GPUs (T4 — Kaggle's free-tier
    accelerator) bf16 isn't tensor-core accelerated and fell back to a path ~30x slower in
    practice (450s/step instead of the expected low single digits). bf16 only pays off on
    Ampere+ (A100, RTX 30xx and later).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="auto",
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    return model


def apply_lora(
    model,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
):
    """Wrap a quantized base model with trainable LoRA adapters.

    `modules_to_save=["score"]` keeps the classification head trainable in full precision
    even though the backbone stays 4-bit quantized.
    """
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or DEFAULT_TARGET_MODULES,
        modules_to_save=["score"],
        bias="none",
    )
    return get_peft_model(model, lora_config)
