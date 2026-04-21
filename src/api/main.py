"""fastapi app for fraud detection inference."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.api.model_loader import FraudModel, load_from_env
from src.api.schemas import (
    HealthResponse,
    InfoResponse,
    PredictionResponse,
    TransactionRequest,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# module-level state. uvicorn creates one instance per worker, so this is safe.
STATE: dict[str, FraudModel | None] = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """load model on startup, clean up on shutdown."""
    log.info("startup: loading model")
    STATE["model"] = load_from_env()
    log.info("startup: model ready")
    yield
    log.info("shutdown")
    STATE["model"] = None


app = FastAPI(
    title="fraud detection api",
    version="0.1.0",
    lifespan=lifespan,
)


def _get_model() -> FraudModel:
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return model


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=STATE["model"] is not None,
    )


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    model = _get_model()
    return InfoResponse(
        model_run_id=model.run_id,
        n_features=len(model.feature_names),
        threshold=model.threshold,
        expected_features_sample=model.feature_names[:20],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest) -> PredictionResponse:
    model = _get_model()
    # request.model_dump() gives us every field including extras
    payload: dict[str, Any] = request.model_dump(exclude_none=False)

    try:
        proba, is_fraud = model.predict(payload)
    except Exception as e:
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail=f"prediction failed: {e}")

    return PredictionResponse(
        fraud_probability=proba,
        is_fraud=is_fraud,
        threshold=model.threshold,
        model_run_id=model.run_id,
    )


@app.exception_handler(Exception)
async def unhandled(request, exc) -> JSONResponse:
    log.exception("unhandled error on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error"},
    )