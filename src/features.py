"""feature engineering for fraud detection baseline."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import ID_COL, TARGET_COL, TIME_COL

log = logging.getLogger(__name__)

COLS_TO_DROP = [ID_COL, TIME_COL, TARGET_COL]


def split_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """separate features (x) from target (y)."""
    y = df[TARGET_COL].astype(np.int8)
    x = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    return x, y


def identify_column_types(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    """return (numeric_cols, categorical_cols) based on dtype."""
    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = x.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_cols, categorical_cols


def encode_categoricals(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """label-encode categorical columns using train values only.

    fit on train: each unique value gets an int code. unseen values in val/test
    get -1. nans get -1 too. this matches what a real pipeline does (you only
    see training-time categories at fit time).
    """
    x_train = x_train.copy()
    x_val = x_val.copy()
    x_test = x_test.copy()

    for col in cat_cols:
        train_vals = x_train[col].astype("string")
        unique_vals = train_vals.dropna().unique()
        mapping = {v: i for i, v in enumerate(unique_vals)}

        def encode(s: pd.Series) -> pd.Series:
            return (
                s.astype("string")
                .map(mapping)
                .fillna(-1)
                .astype(np.int32)
            )

        x_train[col] = encode(x_train[col])
        x_val[col] = encode(x_val[col])
        x_test[col] = encode(x_test[col])

    return x_train, x_val, x_test


def build_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
]:
    """full feature pipeline: split x/y, identify types, encode categoricals."""
    x_train, y_train = split_x_y(train)
    x_val, y_val = split_x_y(val)
    x_test, y_test = split_x_y(test)

    num_cols, cat_cols = identify_column_types(x_train)
    log.info(
        "feature counts: numeric=%d categorical=%d", len(num_cols), len(cat_cols)
    )

    x_train, x_val, x_test = encode_categoricals(x_train, x_val, x_test, cat_cols)

    log.info(
        "final shapes: train=%s val=%s test=%s",
        x_train.shape, x_val.shape, x_test.shape,
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from src.data import load_raw, time_split

    df = load_raw()
    train, val, test = time_split(df)
    x_train, y_train, x_val, y_val, x_test, y_test = build_features(train, val, test)
    print("\nsanity check:")
    print("x_train dtypes sample:", x_train.dtypes.value_counts().to_dict())
    print("y_train positive rate:", y_train.mean())