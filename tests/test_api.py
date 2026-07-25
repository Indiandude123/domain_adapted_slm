"""Smoke test for the FastAPI /predict endpoint using a stubbed model (no GPU/download needed)."""
from __future__ import annotations

import torch
from fastapi.testclient import TestClient
from transformers import BatchEncoding

from src.api import main


class _StubOutputs:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _StubModel:
    def __call__(self, **kwargs) -> _StubOutputs:
        logits = torch.tensor([[5.0] + [0.0] * (len(main.LABELS) - 1)])
        return _StubOutputs(logits)

    def parameters(self):
        yield torch.zeros(1)


class _StubTokenizer:
    def __call__(self, text, **kwargs) -> BatchEncoding:
        return BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})


def test_predict_returns_top_label():
    main.state["model"] = _StubModel()
    main.state["tokenizer"] = _StubTokenizer()

    # Plain (non-context-manager) TestClient use skips lifespan, so no real model download happens.
    client = TestClient(main.app)
    response = client.post(
        "/predict",
        json={"text": "The petitioner challenges the search under the Fourth Amendment."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == main.LABELS[0]
    assert len(body["probabilities"]) == len(main.LABELS)


def test_health():
    client = TestClient(main.app)
    assert client.get("/health").json() == {"status": "ok"}
