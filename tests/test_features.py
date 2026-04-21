"""tests for feature engineering logic."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (
    apply_category_mappings,
    fit_category_mappings,
)


def test_mappings_are_deterministic() -> None:
    """fitting twice on the same data produces the same mappings."""
    df = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
    m1 = fit_category_mappings(df, ["color"])
    m2 = fit_category_mappings(df, ["color"])
    assert m1 == m2


def test_unseen_values_map_to_minus_one() -> None:
    """values not seen in train become -1 at apply time."""
    train = pd.DataFrame({"color": ["red", "blue"]})
    test = pd.DataFrame({"color": ["red", "blue", "yellow"]})
    mappings = fit_category_mappings(train, ["color"])
    encoded = apply_category_mappings(test, mappings)
    assert encoded["color"].iloc[2] == -1


def test_nan_maps_to_minus_one() -> None:
    """nan becomes -1 at apply time."""
    train = pd.DataFrame({"color": ["red", "blue"]})
    test = pd.DataFrame({"color": ["red", None]})
    mappings = fit_category_mappings(train, ["color"])
    encoded = apply_category_mappings(test, mappings)
    assert encoded["color"].iloc[1] == -1


def test_encoded_dtype_is_int32() -> None:
    """encoded columns should be int32 for memory efficiency."""
    train = pd.DataFrame({"color": ["red", "blue"]})
    mappings = fit_category_mappings(train, ["color"])
    encoded = apply_category_mappings(train, mappings)
    assert encoded["color"].dtype == np.int32