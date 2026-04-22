"""run drift detection against production predictions, exit non-zero on drift.

designed to be called by a cron / scheduled job. in real mlops this would
trigger a retraining workflow or page a human. here we just print and exit.

usage:
  DATABASE_URL=... python -m scripts.check_drift
  DATABASE_URL=... python -m scripts.check_drift --hours 24 --threshold 0.3
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.config import PROCESSED_DIR
from src.features import load_mappings
from src.monitoring.drift import (
    align_columns,
    build_report,
    load_current,
    load_reference,
)

log = logging.getLogger(__name__)


def check_drift(hours: int, sample: int, threshold: float) -> int:
    mappings = load_mappings(PROCESSED_DIR / "category_mappings.json")

    reference = load_reference()
    if len(reference) > sample:
        reference = reference.sample(sample, random_state=42)

    current = load_current(hours=hours, sample=sample, mappings=mappings)
    reference, current = align_columns(reference, current)

    report = build_report(reference, current)
    result = report.as_dict()
    drift_metric = next(
        m for m in result["metrics"] if m["metric"] == "DatasetDriftMetric"
    )
    drift_share = drift_metric["result"]["share_of_drifted_columns"]
    drifted = drift_metric["result"]["number_of_drifted_columns"]
    total = drift_metric["result"]["number_of_columns"]

    log.info("drift share: %.3f (%d/%d columns)", drift_share, drifted, total)
    log.info("alert threshold: %.3f", threshold)

    if drift_share > threshold:
        log.warning("DRIFT ALERT: %.1f%% > %.1f%%", drift_share * 100, threshold * 100)
        return 1

    log.info("ok, no alert")
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--sample", type=int, default=5000)
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="drift share above which to alert (0.3 = 30%)",
    )
    args = parser.parse_args()

    exit_code = check_drift(args.hours, args.sample, args.threshold)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()