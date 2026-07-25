"""CLI entrypoint for QLoRA training, invoked from the Kaggle notebook.

Usage:
    python -m src.train.run_train --config configs/scotus_phi3.yaml
    python -m src.train.run_train --config configs/scotus_phi3.yaml --weighted
    python -m src.train.run_train --config configs/scotus_phi3.yaml --weighted --push-to-hub <user>/<repo>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments

from src.data.load import label_names, load_scotus
from src.data.preprocess import compute_class_weights, tokenize_batch
from src.model.qlora import apply_lora, load_quantized_classifier
from src.train.trainer import WeightedLossTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weighted", action="store_true", help="Use class-weighted loss")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF Hub repo id to push the adapter to")
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    labels = label_names()
    num_labels = len(labels)

    dataset = load_scotus()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = dataset.map(
        lambda ex: tokenize_batch(ex, tokenizer, max_length=cfg["max_length"]),
        batched=True,
        remove_columns=["text"],
    )

    model = load_quantized_classifier(cfg["model_name"], num_labels)
    model = apply_lora(
        model,
        r=cfg.get("lora_r", 16),
        alpha=cfg.get("lora_alpha", 32),
        dropout=cfg.get("lora_dropout", 0.05),
    )
    model.print_trainable_parameters()

    class_weights = None
    if args.weighted:
        class_weights = compute_class_weights(tokenized["train"]["label"], num_labels)

    run_name = "weighted" if args.weighted else "unweighted"
    training_args = TrainingArguments(
        output_dir=f"{cfg['output_dir']}/{run_name}",
        per_device_train_batch_size=cfg.get("batch_size", 4),
        per_device_eval_batch_size=cfg.get("eval_batch_size", 8),
        gradient_accumulation_steps=cfg.get("grad_accum_steps", 4),
        num_train_epochs=cfg.get("epochs", 3),
        learning_rate=cfg.get("learning_rate", 2e-4),
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to=cfg.get("report_to", "none"),
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
        run_name=run_name,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )
    trainer.train()
    trainer.save_model(f"{cfg['output_dir']}/{run_name}/final")

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
