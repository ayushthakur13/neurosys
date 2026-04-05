from __future__ import annotations

from pydantic import BaseModel


class SequenceInput(BaseModel):
    block_id: str
    events: list[str]


class DetectResponse(BaseModel):
    block_id: str
    score: float
    is_anomaly: bool


class LatentResponse(BaseModel):
    block_id: str
    latent: list[float]


class RootCauseResponse(BaseModel):
    block_id: str
    explanation: dict
