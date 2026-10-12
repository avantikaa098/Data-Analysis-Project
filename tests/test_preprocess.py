"""Unit tests for cleaning and preprocessing."""

from __future__ import annotations

import pandas as pd

from src.preprocess import clean_dataframe, split_and_save


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [45, 60, 52, 48, 70, 55],
            "sex": [1, 0, 1, 0, 1, 1],
            "cp": [0, 2, 1, 3, 1, 0],
            "trestbps": [120, 140, 130, 128, 150, 135],
            "chol": [200, 240, 220, 210, 260, 230],
            "fbs": [0, 1, 0, 0, 1, 0],
            "restecg": [1, 0, 1, 0, 1, 1],
            "thalach": [160, 140, 150, 155, 130, 145],
            "exang": [0, 1, 0, 0, 1, 0],
            "oldpeak": [0.5, 1.8, 1.0, 0.2, 2.2, 1.1],
            "slope": [1, 2, 1, 1, 2, 1],
            "ca": [0, 1, 0, 0, 2, 1],
            "thal": [2, 3, 2, 2, 3, 2],
            "target": [0, 1, 0, 0, 1, 1],
        }
    )


def test_clean_dataframe_binary_target():
    cleaned = clean_dataframe(_sample_df())
    assert set(cleaned["target"].unique()).issubset({0, 1})
    assert cleaned.isna().sum().sum() == 0


def test_split_shapes():
    X_train, X_test, y_train, y_test, features = split_and_save(
        _sample_df(), test_size=0.33, random_state=42
    )
    assert len(features) == 13
    assert len(X_train) + len(X_test) == 6
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
