"""loads a trained model and its category mappings from mlflow."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.features import apply_category_mappings

log = logging.getLogger(__name__)


class FraudModel:
    """wraps an mlflow-logged xgboost model + category mappings.

    loads once on init, then serves predictions in-memory.
    """

    def __init__(
        self,
        run_id: str,
        tracking_uri: str,
        threshold: float = 0.5,
    ) -> None:
        self.run_id = run_id
        self.threshold = threshold

        mlflow.set_tracking_uri(tracking_uri)
        log.info("loading model from mlflow run %s", run_id)

        # xgboost model
        model_uri = f"runs:/{run_id}/model"
        self.model = mlflow.xgboost.load_model(model_uri)
        log.info("model loaded, type=%s", type(self.model).__name__)

        # category mappings (logged as artifact in train.py)
        with tempfile.TemporaryDirectory() as tmp:
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="category_mappings.json",
                dst_path=tmp,
            )
            self.mappings: dict[str, dict[str, int]] = json.loads(Path(path).read_text())
        log.info("loaded %d category mappings", len(self.mappings))

        # the model knows what features it was trained on and their order
        booster = self.model.get_booster()
        self.feature_names: list[str] = list(booster.feature_names or [])
        log.info("model expects %d features", len(self.feature_names))

    def _prepare_row(self, payload: dict[str, Any]) -> pd.DataFrame:
        """turn an incoming dict into a 1-row dataframe with the right columns.

        steps:
        1. start with a frame containing all expected features as nan.
        2. overlay whatever the caller provided.
        3. apply category mappings (unseen/nan -> -1) to categorical cols.
        """
        row = {name: np.nan for name in self.feature_names}
        for k, v in payload.items():
            if k in row:
                row[k] = v
        df = pd.DataFrame([row], columns=self.feature_names)

        df = apply_category_mappings(df, self.mappings)

        # numeric columns that weren't mapped may still be "object" dtype if
        # the caller passed strings like "123.4". coerce to float.
        for col in df.columns:
            if col not in self.mappings:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def predict(self, payload: dict[str, Any]) -> tuple[float, bool]:
        """return (fraud_probability, is_fraud_at_threshold)."""
        x = self._prepare_row(payload)
        proba = float(self.model.predict_proba(x)[0, 1])
        return proba, proba >= self.threshold


def load_from_env() -> FraudModel:
    """read run id and tracking uri from env vars, with sensible defaults."""
    run_id = os.environ.get("MODEL_RUN_ID")
    if not run_id:
        raise RuntimeError("MODEL_RUN_ID env var not set. set it to a valid mlflow run id.")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    threshold = float(os.environ.get("DECISION_THRESHOLD", "0.5"))
    return FraudModel(run_id=run_id, tracking_uri=tracking_uri, threshold=threshold)
