"""Explain which features influence model predictions."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.config import FEATURE_NAMES, FIGURES, REPORTS


def extract_native_importance(pipeline, feature_names: list[str]) -> pd.DataFrame | None:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        kind = "tree_feature_importances"
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
        kind = "absolute_coefficients"
    else:
        return None

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
            "label": [FEATURE_NAMES.get(f, f) for f in feature_names],
            "method": kind,
        }
    ).sort_values("importance", ascending=False)
    return df


def compute_permutation_importance(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    random_state: int = 42,
) -> pd.DataFrame:
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=15,
        random_state=random_state,
        scoring="roc_auc",
    )
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": result.importances_mean,
            "std": result.importances_std,
            "label": [FEATURE_NAMES.get(f, f) for f in feature_names],
            "method": "permutation_importance",
        }
    ).sort_values("importance", ascending=False)
    return df


def plot_importance(df: pd.DataFrame, title: str, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plot_df = df.head(12).iloc[::-1]
    sns.barplot(data=plot_df, x="importance", y="label", color="#4C78A8")
    if "std" in plot_df.columns:
        plt.errorbar(
            plot_df["importance"],
            np.arange(len(plot_df)),
            xerr=plot_df["std"],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.close()


def explain_model(pipeline, X_test, y_test, feature_names: list[str]) -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)

    native = extract_native_importance(pipeline, feature_names)
    perm = compute_permutation_importance(pipeline, X_test, y_test, feature_names)

    if native is not None:
        plot_importance(native, "Model-native feature importance", "feature_importance_native.png")
        native.to_csv(REPORTS / "feature_importance_native.csv", index=False)

    plot_importance(
        perm,
        "Permutation importance (ROC-AUC drop)",
        "feature_importance_permutation.png",
    )
    perm.to_csv(REPORTS / "feature_importance_permutation.csv", index=False)

    payload = {
        "native": None if native is None else native.to_dict(orient="records"),
        "permutation": perm.to_dict(orient="records"),
    }
    with open(REPORTS / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload
