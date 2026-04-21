"""loads a trained model + category mappings from mlflow OR a local folder.

in dev we load from mlflow (lets us swap runs with one env var).
in production we load from a local folder baked into the docker image,
so the deployed service has no runtime dependency on mlflow.
"""
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
import xgboost as xgb

from src.features import apply_category_mappings

log = logging.getLogger(__name__)


class FraudModel:
    """holds an xgboost model + category mappings, exposes predict()."""

    def __init__(
        self,
        model: xgb.XGBClassifier,
        mappings: dict[str, dict[str, int]],
        run_id: str,
        threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.mappings = mappings
        self.run_id = run_id
        self.threshold = threshold

        booster = self.model.get_booster()
        self.feature_names: list[str] = list(booster.feature_names or [])
        log.info(
            "FraudModel ready: run_id=%s, features=%d, mappings=%d",
            run_id, len(self.feature_names), len(self.mappings),
        )

    def _prepare_row(self, payload: dict[str, Any]) -> pd.DataFrame:
        row = {name: np.nan for name in self.feature_names}
        for k, v in payload.items():
            if k in row:
                row[k] = v
        df = pd.DataFrame([row], columns=self.feature_names)

        df = apply_category_mappings(df, self.mappings)

        for col in df.columns:
            if col not in self.mappings:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def predict(self, payload: dict[str, Any]) -> tuple[float, bool]:
        x = self._prepare_row(payload)
        proba = float(self.model.predict_proba(x)[0, 1])
        return proba, proba >= self.threshold


def _load_from_mlflow(run_id: str, tracking_uri: str, threshold: float) -> FraudModel:
    mlflow.set_tracking_uri(tracking_uri)
    log.info("loading from mlflow run %s", run_id)

    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")

    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="category_mappings.json",
            dst_path=tmp,
        )
        mappings = json.loads(Path(path).read_text())

    return FraudModel(model=model, mappings=mappings, run_id=run_id, threshold=threshold)


def _load_from_local(local_dir: Path, threshold: float) -> FraudModel:
    log.info("loading from local dir %s", local_dir)

    # mlflow saves the model under <local_dir>/model/
    model_path = local_dir / "model"
    if not model_path.exists():
        raise FileNotFoundError(f"no model dir at {model_path}")
    model = mlflow.xgboost.load_model(str(model_path))

    mappings_path = local_dir / "category_mappings.json"
    mappings = json.loads(mappings_path.read_text())

    run_id_file = local_dir / "run_id.txt"
    run_id = run_id_file.read_text().strip() if run_id_file.exists() else "local"

    return FraudModel(model=model, mappings=mappings, run_id=run_id, threshold=threshold)


def load_from_env() -> FraudModel:
    """load model using env vars.

    precedence: MODEL_LOCAL_DIR > MODEL_RUN_ID.

    MODEL_LOCAL_DIR: path to a folder with `model/` and `category_mappings.json`.
                     used in production (docker image).
    MODEL_RUN_ID: mlflow run id. used in dev.
    """
    threshold = float(os.environ.get("DECISION_THRESHOLD", "0.5"))

    local_dir = os.environ.get("MODEL_LOCAL_DIR")
    if local_dir:
        return _load_from_local(Path(local_dir), threshold)

    run_id = os.environ.get("MODEL_RUN_ID")
    if run_id:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        return _load_from_mlflow(run_id, tracking_uri, threshold)

    raise RuntimeError(
        "set either MODEL_LOCAL_DIR (prod) or MODEL_RUN_ID (dev) env var"
    )
