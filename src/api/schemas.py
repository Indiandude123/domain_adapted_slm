"""Pydantic request/response schemas for the classification API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Legal case / opinion text to classify")


class ClassProbability(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: list[ClassProbability]
