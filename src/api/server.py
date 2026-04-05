from __future__ import annotations

from fastapi import FastAPI

from .schemas import DetectResponse, LatentResponse, RootCauseResponse, SequenceInput
from .service import NeuroSysService

app = FastAPI(title="NeuroSys API", version="0.1.0")
service = NeuroSysService("results/default_run/artifacts")


@app.on_event("startup")
def startup() -> None:
    try:
        service.load()
    except Exception:
        # The API can still start before models are trained.
        pass


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(payload: SequenceInput) -> DetectResponse:
    score, is_anomaly = service.detect(payload.events)
    return DetectResponse(block_id=payload.block_id, score=score, is_anomaly=is_anomaly)


@app.post("/latent", response_model=LatentResponse)
def latent(payload: SequenceInput) -> LatentResponse:
    z = service.latent(payload.events)
    return LatentResponse(block_id=payload.block_id, latent=z)


@app.post("/root-cause", response_model=RootCauseResponse)
def root_cause(payload: SequenceInput) -> RootCauseResponse:
    explanation = service.root_cause(payload.events)
    return RootCauseResponse(block_id=payload.block_id, explanation=explanation)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
