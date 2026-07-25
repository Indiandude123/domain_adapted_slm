"""Classical TF-IDF baselines (Logistic Regression, Random Forest, XGBoost) for SCOTUS
classification — gives the QLoRA fine-tuned SLM a real comparison point in the results table.

Usage:
    python -m src.eval.baseline
    python -m src.eval.baseline --max-features 100000 --output outputs/baseline_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data.load import label_names, load_scotus
from src.data.preprocess import compute_class_weights
from src.eval.metrics import full_report

MODEL_BUILDERS = {
    "logreg": lambda: LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0
    ),
    # XGBoost has no native class_weight param; imbalance is handled via per-sample weights
    # at fit time instead (same inverse-frequency weights used for the QLoRA weighted-loss run).
    "xgboost": lambda: XGBClassifier(
        n_estimators=300,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=0,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline_results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = label_names()
    num_labels = len(labels)

    dataset = load_scotus()
    y_train = np.array(dataset["train"]["label"])
    y_test = np.array(dataset["test"]["label"])

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(dataset["train"]["text"])
    X_test = vectorizer.transform(dataset["test"]["text"])

    class_weights = compute_class_weights(y_train.tolist(), num_labels).numpy()
    sample_weight = class_weights[y_train]

    results = {}
    for name, build_model in MODEL_BUILDERS.items():
        model = build_model()
        if name == "xgboost":
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)  # class_weight="balanced" handles LR/RF directly

        preds = model.predict(X_test)
        report = full_report(y_test, preds, labels)
        accuracy = float((preds == y_test).mean())
        results[name] = {"accuracy": accuracy, **report}

        print(
            f"{name:>15}  acc={accuracy:.3f}  "
            f"macro_f1={report['macro_f1']:.3f}  weighted_f1={report['weighted_f1']:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
