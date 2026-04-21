"""fastapi app for fraud detection inference."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.api.model_loader import FraudModel, load_from_env
from src.api.schemas import (
    HealthResponse,
    InfoResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.monitoring.db import init_db, write_prediction_log

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


STATE: dict[str, FraudModel | None] = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """load model + init db on startup."""
    log.info("startup: loading model")
    STATE["model"] = load_from_env()
    log.info("startup: model ready")

    try:
        init_db()
        log.info("startup: db ready")
    except Exception:
        # db is optional for serving; log but don't block startup
        log.exception("startup: db init failed, continuing without logging")

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
def predict(
    request: TransactionRequest,
    background: BackgroundTasks,
) -> PredictionResponse:
    model = _get_model()
    payload: dict[str, Any] = request.model_dump(exclude_none=False)

    start = time.perf_counter()
    try:
        proba, is_fraud = model.predict(payload)
    except Exception as e:
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail=f"prediction failed: {e}")
    latency_ms = (time.perf_counter() - start) * 1000

    # log asynchronously so db writes don't slow down responses
    background.add_task(
        write_prediction_log,
        model_run_id=model.run_id,
        features=payload,
        fraud_probability=proba,
        is_fraud=is_fraud,
        threshold=model.threshold,
        latency_ms=latency_ms,
    )

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
