"""train a baseline xgboost model with mlflow tracking and hydra config."""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import mlflow
import mlflow.xgboost
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf

from src.config import RANDOM_SEED
from src.data import load_raw, time_split
from src.evaluate import compute_metrics
from src.features import build_features

log = logging.getLogger(__name__)


def train_xgb(
    x_train, y_train, x_val, y_val, params: dict,
) -> xgb.XGBClassifier:
    """train xgboost with early stopping on validation set."""
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

        # log hyperparameters
        mlflow.log_params(dict(cfg.model))
        mlflow.log_params({
            "val_frac": cfg.split.val_frac,
            "test_frac": cfg.split.test_frac,
        })

        # data prep
        df = load_raw()
        train, val, test = time_split(
            df, val_frac=cfg.split.val_frac, test_frac=cfg.split.test_frac,
        )
        x_train, y_train, x_val, y_val, x_test, y_test = build_features(
            train, val, test,
        )

        mlflow.log_params({
            "n_train": len(x_train),
            "n_val": len(x_val),
            "n_test": len(x_test),
            "n_features": x_train.shape[1],
        })

        # train
        model = train_xgb(x_train, y_train, x_val, y_val, params=dict(cfg.model))
        mlflow.log_metric("best_iteration", model.best_iteration)

        # evaluate
        val_pred = model.predict_proba(x_val)[:, 1]
        test_pred = model.predict_proba(x_test)[:, 1]
        val_metrics = compute_metrics(y_val.to_numpy(), val_pred)
        test_metrics = compute_metrics(y_test.to_numpy(), test_pred)

        # log metrics with val_/test_ prefix so they're comparable in the ui
        for k, v in val_metrics.to_dict().items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.to_dict().items():
            mlflow.log_metric(f"test_{k}", v)

        log.info("val metrics: %s", val_metrics.to_dict())
        log.info("test metrics: %s", test_metrics.to_dict())

        # log model as artifact
        mlflow.xgboost.log_model(model, artifact_path="model")
        log.info("model logged to mlflow")


if __name__ == "__main__":
    main()