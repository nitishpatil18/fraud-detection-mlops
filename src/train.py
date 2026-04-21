"""train xgboost on pre-built features, log to mlflow."""

from __future__ import annotations

import logging

import hydra
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf

from src.config import PROCESSED_DIR, RANDOM_SEED, TARGET_COL
from src.evaluate import compute_metrics

log = logging.getLogger(__name__)


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """load a parquet split and separate features from target."""
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    y = df[TARGET_COL]
    x = df.drop(columns=[TARGET_COL])
    log.info("loaded %s: x=%s y_mean=%.4f", name, x.shape, y.mean())
    return x, y


def train_xgb(x_train, y_train, x_val, y_val, params: dict) -> xgb.XGBClassifier:
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)
    log.info("scale_pos_weight=%.2f (pos=%d neg=%d)", scale_pos_weight, pos, neg)

    model = xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        early_stopping_rounds=params["early_stopping_rounds"],
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=50)
    log.info("best iteration: %d", model.best_iteration)
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=cfg.run_name) as run:
        log.info("mlflow run_id: %s", run.info.run_id)

        mlflow.log_params(dict(cfg.model))

        x_train, y_train = load_split("train")
        x_val, y_val = load_split("val")
        x_test, y_test = load_split("test")

        mlflow.log_params(
            {
                "n_train": len(x_train),
                "n_val": len(x_val),
                "n_test": len(x_test),
                "n_features": x_train.shape[1],
            }
        )

        model = train_xgb(x_train, y_train, x_val, y_val, params=dict(cfg.model))
        mlflow.log_metric("best_iteration", model.best_iteration)

        val_pred = model.predict_proba(x_val)[:, 1]
        test_pred = model.predict_proba(x_test)[:, 1]
        val_metrics = compute_metrics(y_val.to_numpy(), val_pred)
        test_metrics = compute_metrics(y_test.to_numpy(), test_pred)

        for k, v in val_metrics.to_dict().items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.to_dict().items():
            mlflow.log_metric(f"test_{k}", v)

        log.info("val metrics: %s", val_metrics.to_dict())
        log.info("test metrics: %s", test_metrics.to_dict())

        # also log the category mappings file as an artifact, so the model
        # run is self-contained (model + mappings always go together).
        mlflow.log_artifact(str(PROCESSED_DIR / "category_mappings.json"))

        mlflow.xgboost.log_model(model, artifact_path="model")
        log.info("model logged to mlflow")


if __name__ == "__main__":
    main()
