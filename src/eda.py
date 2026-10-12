"""Exploratory data analysis plots and summary statistics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES, REPORTS, TARGET_COL


def _save_fig(name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def run_eda(df: pd.DataFrame) -> dict:
    """Create EDA figures and a JSON summary report."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_rows": int(len(df)),
        "n_features": int(df.shape[1] - 1),
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "class_balance": {str(k): int(v) for k, v in df[TARGET_COL].value_counts().to_dict().items()},
        "describe": df.describe().round(3).to_dict(),
    }

    sns.set_theme(style="whitegrid", context="notebook")

    # Target distribution
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=TARGET_COL, hue=TARGET_COL, palette="Set2", legend=False)
    ax.set_title("Target distribution (0 = lower risk pattern, 1 = higher risk pattern)")
    ax.set_xlabel("Target class")
    ax.set_ylabel("Count")
    _save_fig("target_distribution.png")

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
    plt.title("Feature correlation heatmap")
    _save_fig("correlation_heatmap.png")

    # Age vs target
    if "age" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df, x=TARGET_COL, y="age", hue=TARGET_COL, palette="pastel", legend=False)
        plt.title("Age distribution by target class")
        plt.xlabel("Target class")
        _save_fig("age_by_target.png")

    # Cholesterol vs max heart rate
    if {"chol", "thalach"}.issubset(df.columns):
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=df,
            x="chol",
            y="thalach",
            hue=TARGET_COL,
            palette="Set1",
            alpha=0.75,
        )
        plt.title("Cholesterol vs max heart rate by class")
        _save_fig("chol_vs_thalach.png")

    # Numeric feature distributions
    numeric_cols = [c for c in df.columns if c != TARGET_COL]
    n = len(numeric_cols)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#4C78A8")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Feature distributions", y=1.01)
    _save_fig("feature_distributions.png")

    with open(REPORTS / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
