"""pydantic schemas for the fraud detection api."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    """a transaction to score.

    all fields are optional. the api accepts a subset and fills the rest
    with nan, which the model handles natively. extra fields are ignored.
    """

    model_config = ConfigDict(extra="allow")

    TransactionAmt: float | None = Field(
        default=None, description="transaction amount in usd", ge=0,
    )
    ProductCD: str | None = Field(default=None, description="product code")
    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None


class PredictionResponse(BaseModel):
    """response from /predict."""

    fraud_probability: float = Field(
        description="probability of fraud, in [0, 1]", ge=0, le=1,
    )
    is_fraud: bool = Field(
        description="model decision at the configured threshold",
    )
    threshold: float = Field(description="threshold used for the decision")
    model_run_id: str = Field(description="mlflow run id of the loaded model")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    model_run_id: str
    n_features: int
    threshold: float
    expected_features_sample: list[str]