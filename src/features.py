"""feature engineering for fraud detection."""
from __future__ import annotations

import json
import logging
from pathlib import Path

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


def fit_category_mappings(
    x_train: pd.DataFrame, cat_cols: list[str]
) -> dict[str, dict[str, int]]:
    """learn a value->int mapping for each categorical column from train only."""
    mappings: dict[str, dict[str, int]] = {}
    for col in cat_cols:
        unique_vals = x_train[col].astype("string").dropna().unique()
        mappings[col] = {str(v): i for i, v in enumerate(unique_vals)}
    return mappings


def apply_category_mappings(
    x: pd.DataFrame, mappings: dict[str, dict[str, int]]
) -> pd.DataFrame:
    """encode categorical cols using pre-fit mappings. unseen/nan -> -1."""
    x = x.copy()
    for col, mapping in mappings.items():
        if col not in x.columns:
            x[col] = -1
            continue
        x[col] = (
            x[col].astype("string").map(mapping).fillna(-1).astype(np.int32)
        )
    return x


def save_mappings(mappings: dict[str, dict[str, int]], path: Path) -> None:
    """write mappings to json for reuse at inference time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mappings, indent=2))
    log.info("saved mappings to %s", path)


def load_mappings(path: Path) -> dict[str, dict[str, int]]:
    return json.loads(path.read_text())