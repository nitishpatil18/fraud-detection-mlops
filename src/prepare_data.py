"""build features once and save to data/processed as parquet.

run this whenever the raw data changes. training reads the parquet files,
not the raw csvs, so it's fast and reproducible.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, TARGET_COL
from src.data import load_raw, time_split
from src.features import (
    apply_category_mappings,
    fit_category_mappings,
    identify_column_types,
    save_mappings,
    split_x_y,
)

log = logging.getLogger(__name__)

MAPPINGS_PATH = PROCESSED_DIR / "category_mappings.json"


def save_split(x: pd.DataFrame, y: pd.Series, path: Path) -> None:
    """save features + target as a single parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = x.copy()
    df[TARGET_COL] = y.values
    df.to_parquet(path, compression="snappy", index=False)
    log.info("saved %s (shape=%s, size=%.1fMB)", path.name, df.shape, path.stat().st_size / 1e6)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    log.info("step 1: load raw data")
    df = load_raw()

    log.info("step 2: time-based split")
    train, val, test = time_split(df)

    log.info("step 3: split x/y")
    x_train, y_train = split_x_y(train)
    x_val, y_val = split_x_y(val)
    x_test, y_test = split_x_y(test)

    log.info("step 4: identify column types")
    num_cols, cat_cols = identify_column_types(x_train)
    log.info("numeric=%d categorical=%d", len(num_cols), len(cat_cols))

    log.info("step 5: fit category mappings on train")
    mappings = fit_category_mappings(x_train, cat_cols)
    save_mappings(mappings, MAPPINGS_PATH)

    log.info("step 6: apply mappings to all splits")
    x_train = apply_category_mappings(x_train, mappings)
    x_val = apply_category_mappings(x_val, mappings)
    x_test = apply_category_mappings(x_test, mappings)

    log.info("step 7: save splits as parquet")
    save_split(x_train, y_train, PROCESSED_DIR / "train.parquet")
    save_split(x_val, y_val, PROCESSED_DIR / "val.parquet")
    save_split(x_test, y_test, PROCESSED_DIR / "test.parquet")

    log.info("done. outputs in %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
