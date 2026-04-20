"""train a baseline xgboost model for fraud detection."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import xgboost as xgb

from src.config import MODELS_DIR, RANDOM_SEED
from src.data import load_raw, time_split
from src.evaluate import compute_metrics
from src.features import build_features

log = logging.getLogger(__name__)


def train_xgb(
    x_train, y_train, x_val, y_val,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
) -> xgb.XGBClassifier:
    """train xgboost with early stopping on validation set."""
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)
    log.info("scale_pos_weight=%.2f (pos=%d neg=%d)", scale_pos_weight, pos, neg)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        verbose=50,
    )
    log.info("best iteration: %d", model.best_iteration)
    return model


def save_artifacts(model, metrics: dict, feature_names: list[str]) -> Path:
    """save model, metrics, and feature list to a timestamped folder."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MODELS_DIR / f"baseline_{timestamp}"
    run_dir.mkdir()

    joblib.dump(model, run_dir / "model.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "features.json").write_text(json.dumps(feature_names, indent=2))

    log.info("saved artifacts to %s", run_dir)
    return run_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_raw()
    train, val, test = time_split(df)
    x_train, y_train, x_val, y_val, x_test, y_test = build_features(train, val, test)

    model = train_xgb(x_train, y_train, x_val, y_val)

    val_pred = model.predict_proba(x_val)[:, 1]
    test_pred = model.predict_proba(x_test)[:, 1]

    val_metrics = compute_metrics(y_val.to_numpy(), val_pred)
    test_metrics = compute_metrics(y_test.to_numpy(), test_pred)

    log.info("val metrics: %s", val_metrics.to_dict())
    log.info("test metrics: %s", test_metrics.to_dict())

    save_artifacts(
        model,
        {"val": val_metrics.to_dict(), "test": test_metrics.to_dict()},
        feature_names=x_train.columns.tolist(),
    )


if __name__ == "__main__":
    main()