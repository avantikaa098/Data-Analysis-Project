"""Train and compare multiple classification models."""

from __future__ import annotations

import json
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import FIGURES, MODELS, REPORTS
from src.preprocess import build_preprocessor


def get_candidate_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=random_state),
    }


def evaluate_model(model, X_test, y_test) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_true": y_test.tolist() if hasattr(y_test, "tolist") else list(y_test),
        "y_proba": y_proba.tolist(),
    }


def train_and_compare(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
) -> tuple[Pipeline, pd.DataFrame, dict]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    results = {}
    pipelines: dict[str, Pipeline] = {}

    for name, clf in get_candidate_models().items():
        pipe = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(feature_names)),
                ("model", clf),
            ]
        )
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(pipe, X_test, y_test)
        results[name] = metrics
        pipelines[name] = pipe

    comparison = pd.DataFrame(
        {
            name: {
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
            }
            for name, m in results.items()
        }
    ).T.sort_values("roc_auc", ascending=False)
    comparison.to_csv(REPORTS / "model_comparison.csv")

    best_name = comparison.index[0]
    best_model = pipelines[best_name]
    joblib.dump(
        {
            "pipeline": best_model,
            "model_name": best_name,
            "feature_names": feature_names,
            "metrics": results[best_name],
            "comparison": comparison.to_dict(orient="index"),
        },
        MODELS / "best_model.joblib",
    )

    # Comparison bar chart
    plt.figure(figsize=(8, 5))
    melted = comparison.reset_index().melt(id_vars="index", var_name="metric", value_name="score")
    melted = melted.rename(columns={"index": "model"})
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Model performance comparison")
    plt.tight_layout()
    plt.savefig(FIGURES / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ROC curves
    plt.figure(figsize=(7, 6))
    for name, metrics in results.items():
        fpr, tpr, _ = roc_curve(metrics["y_true"], metrics["y_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Confusion matrix for best model
    cm = np.array(results[best_name]["confusion_matrix"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion matrix — {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix_best.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Strip large arrays before saving JSON
    compact = {}
    for name, m in results.items():
        compact[name] = {
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "classification_report": m["classification_report"],
            "confusion_matrix": m["confusion_matrix"],
        }
    with open(REPORTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "results": compact}, f, indent=2)

    return best_model, comparison, results
