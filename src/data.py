"""data loading and time-based splitting."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    ID_COL,
    RAW_ID_PATH,
    RAW_TXN_PATH,
    TARGET_COL,
    TIME_COL,
)

log = logging.getLogger(__name__)


def load_raw() -> pd.DataFrame:
    """load transaction and identity tables, left-join on TransactionID."""
    log.info("loading raw csvs")
    txn = pd.read_csv(RAW_TXN_PATH)
    ident = pd.read_csv(RAW_ID_PATH)
    df = txn.merge(ident, on=ID_COL, how="left")
    log.info("loaded shape=%s", df.shape)
    return df


def time_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """split by TransactionDT so val/test are strictly after train in time.

    we sort by time, then take the last test_frac as test, the chunk before
    that as val, and the rest as train. no shuffling, no random state.
    """
    if not 0 < val_frac < 1 or not 0 < test_frac < 1:
        raise ValueError("fractions must be in (0, 1)")
    if val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must be < 1")

    df_sorted = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df_sorted)
    test_start = int(n * (1 - test_frac))
    val_start = int(n * (1 - test_frac - val_frac))

    train = df_sorted.iloc[:val_start].copy()
    val = df_sorted.iloc[val_start:test_start].copy()
    test = df_sorted.iloc[test_start:].copy()

    log.info(
        "split sizes: train=%d val=%d test=%d", len(train), len(val), len(test)
    )
    log.info(
        "fraud rates: train=%.4f val=%.4f test=%.4f",
        train[TARGET_COL].mean(),
        val[TARGET_COL].mean(),
        test[TARGET_COL].mean(),
    )
    return train, val, test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_raw()
    train, val, test = time_split(df)