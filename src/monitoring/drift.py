"""drift detection: compare recent predictions vs training baseline.

usage:
  python -m src.monitoring.drift
  python -m src.monitoring.drift --hours 24
  python -m src.monitoring.drift --sample 5000
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
from sqlalchemy import select

from src.config import PROCESSED_DIR, TARGET_COL
from src.monitoring.db import init_db, session_scope
from src.monitoring.schema import PredictionLog

log = logging.getLogger(__name__)

REPORT_DIR = Path("reports/drift")


def load_reference() -> pd.DataFrame:
    """load training features as the reference distribution."""
    df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    return df.drop(columns=[TARGET_COL])


def load_current(
    hours: int,
    sample: int | None,
    mappings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """load recent predictions from postgres as the current distribution.

    applies the same category mappings the model used so the current frame
    has the same encoded-int representation as the reference parquet.
    """
    init_db()
    cutoff = datetime.now(UTC) - timedelta(hours=hours)

    with session_scope() as s:
        q = (
            select(PredictionLog)
            .where(PredictionLog.ts >= cutoff)
            .order_by(PredictionLog.ts.desc())
        )
        rows = s.execute(q).scalars().all()

    if not rows:
        raise RuntimeError(
            f"no predictions in the last {hours} hours. " "run some /predict calls first."
        )

    records = [r.features for r in rows]
    df = pd.DataFrame(records)

    # encode categoricals using the model's mappings (strings -> ints)
    from src.features import apply_category_mappings

    df = apply_category_mappings(df, mappings)

    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=42)
    log.info("loaded %d recent predictions", len(df))
    return df


def align_columns(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """make sure both frames have the same columns in the same order.

    columns that are entirely null in current are dropped from both frames,
    since drift tests can't run on empty data. this is expected: the api
    only receives a subset of features per request, the rest arrive as nan.
    """
    for col in reference.columns:
        if col not in current.columns:
            current[col] = pd.NA
    current = current[reference.columns]

    # drop columns that are 100% null in current
    empty_in_current = [c for c in current.columns if current[c].isna().all()]
    if empty_in_current:
        log.info(
            "dropping %d columns that are fully null in current (e.g. %s)",
            len(empty_in_current),
            empty_in_current[:5],
        )
        reference = reference.drop(columns=empty_in_current)
        current = current.drop(columns=empty_in_current)

    # cast dtypes to match reference
    for col in reference.columns:
        ref_dtype = reference[col].dtype
        try:
            current[col] = current[col].astype(ref_dtype)
        except (ValueError, TypeError):
            current[col] = pd.to_numeric(current[col], errors="coerce")

    return reference, current


def build_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    """run evidently's data drift + data quality presets."""
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24, help="lookback window")
    parser.add_argument("--sample", type=int, default=5000, help="rows to sample")
    args = parser.parse_args()

    log.info("loading category mappings")
    from src.features import load_mappings

    mappings = load_mappings(PROCESSED_DIR / "category_mappings.json")

    log.info("loading reference (training) data")
    reference = load_reference()
    log.info("reference shape: %s", reference.shape)

    if len(reference) > args.sample:
        reference = reference.sample(args.sample, random_state=42)

    log.info("loading current data (last %d hours)", args.hours)
    current = load_current(hours=args.hours, sample=args.sample, mappings=mappings)
    log.info("current shape: %s", current.shape)

    reference, current = align_columns(reference, current)

    log.info("computing drift report (may take ~30s)")
    report = build_report(reference, current)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = REPORT_DIR / f"drift_{timestamp}.html"
    report.save_html(str(html_path))

    result = report.as_dict()
    drift_metric = next(m for m in result["metrics"] if m["metric"] == "DatasetDriftMetric")
    drifted = drift_metric["result"]["number_of_drifted_columns"]
    total = drift_metric["result"]["number_of_columns"]
    drift_share = drift_metric["result"]["share_of_drifted_columns"]

    log.info(
        "drift summary: %d/%d columns drifted (%.1f%%)",
        drifted,
        total,
        drift_share * 100,
    )
    log.info("report saved: %s", html_path)


if __name__ == "__main__":
    main()
