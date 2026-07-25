"""FastAPI inference service for the QLoRA-fine-tuned SCOTUS classifier."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from peft import PeftModel
from transformers import AutoTokenizer

from src.api.schemas import ClassProbability, PredictRequest, PredictResponse
from src.data.load import label_names
from src.model.qlora import load_quantized_classifier

MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/weighted/final")
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "2048"))

LABELS = label_names()

# Populated by the lifespan handler at startup; tests inject stubs here directly instead of
# running lifespan, since plain (non-context-manager) TestClient use skips ASGI lifespan events.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = load_quantized_classifier(BASE_MODEL_NAME, num_labels=len(LABELS))
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()

    state["tokenizer"] = tokenizer
    state["model"] = model
    yield
    state.clear()


app = FastAPI(title="SCOTUS Legal Case Classifier", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    tokenizer = state["tokenizer"]
    model = state["model"]
    device = next(model.parameters()).device

    inputs = tokenizer(request.text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = F.softmax(logits, dim=-1).tolist()

    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return PredictResponse(
        predicted_label=LABELS[best_idx],
        confidence=probs[best_idx],
        probabilities=[ClassProbability(label=label, probability=p) for label, p in zip(LABELS, probs)],
    )
