"""Data cleaning and preprocessing for the heart disease dataset."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED, TARGET_COL


EXPECTED_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns, coerce types, and drop/repair invalid rows."""
    data = df.copy()
    data.columns = [c.strip().lower() for c in data.columns]

    rename_map = {
        "chest pain type": "cp",
        "resting bp s": "trestbps",
        "cholesterol": "chol",
        "fasting blood sugar": "fbs",
        "resting ecg": "restecg",
        "max heart rate": "thalach",
        "exercise angina": "exang",
        "st slope": "slope",
    }
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

    if TARGET_COL not in data.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataset.")

    # Keep known features when present; otherwise use all non-target columns
    feature_cols = [c for c in EXPECTED_FEATURES if c in data.columns]
    if not feature_cols:
        feature_cols = [c for c in data.columns if c != TARGET_COL]

    data = data[feature_cols + [TARGET_COL]]

    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Binary target: presence of disease (some sources use 0–4 severity)
    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data = data.dropna(subset=[TARGET_COL])
    data[TARGET_COL] = (data[TARGET_COL] > 0).astype(int)

    # Drop rows that are almost entirely missing
    data = data.dropna(thresh=max(3, len(feature_cols) // 2))
    data = data.drop_duplicates().reset_index(drop=True)
    return data


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipe, feature_names)],
        remainder="drop",
    )


def split_and_save(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cleaned = clean_dataframe(df)
    cleaned.to_csv(DATA_PROCESSED / "heart_cleaned.csv", index=False)

    feature_names = [c for c in cleaned.columns if c != TARGET_COL]
    X = cleaned[feature_names]
    y = cleaned[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, feature_names
