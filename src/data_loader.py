"""Download and load the public heart disease dataset."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_RAW, HEART_DATASET_URL


def ensure_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)


def download_dataset(force: bool = False) -> Path:
    """Download the educational heart disease CSV into data/raw/."""
    ensure_dirs()
    out_path = DATA_RAW / "heart.csv"
    if out_path.exists() and not force:
        return out_path

    response = requests.get(HEART_DATASET_URL, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    # Normalize target column name if needed
    if "target" not in df.columns and "Target" in df.columns:
        df = df.rename(columns={"Target": "target"})
    if "target" not in df.columns and df.columns[-1].lower() in {"target", "num", "output"}:
        df = df.rename(columns={df.columns[-1]: "target"})
    df.to_csv(out_path, index=False)
    return out_path


def load_raw(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or download_dataset()
    return pd.read_csv(csv_path)
