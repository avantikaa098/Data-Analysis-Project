"""Train, cross-validate, tune, and compare multiple classification models."""

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
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import CV_FOLDS, FIGURES, MODELS, RANDOM_STATE, REPORTS
from src.logging_utils import get_logger
from src.preprocess import build_preprocessor

logger = get_logger(__name__)


def get_candidate_models(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "SVM (RBF)": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def get_param_grids() -> dict[str, dict]:
    """Compact grids for interview-ready tuning without excessive runtime."""
    return {
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 3.0],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"],
        },
        "Random Forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [6, 10, None],
            "model__min_samples_leaf": [1, 2],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
        "SVM (RBF)": {
            "model__C": [0.5, 1.0, 2.0],
            "model__gamma": ["scale", 0.01],
        },
    }


def evaluate_model(model, X_test, y_test) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_true": y_test.tolist() if hasattr(y_test, "tolist") else list(y_test),
        "y_proba": y_proba.tolist(),
    }


def cross_validate_model(pipe: Pipeline, X, y) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
    }
    cv_results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        metric: float(np.mean(cv_results[f"test_{metric}"]))
        for metric in scoring
    } | {
        f"{metric}_std": float(np.std(cv_results[f"test_{metric}"]))
        for metric in scoring
    }


def tune_model(pipe: Pipeline, param_grid: dict, X, y) -> tuple[Pipeline, dict]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
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

    results: dict[str, Any] = {}
    pipelines: dict[str, Pipeline] = {}
    tuning_notes: dict[str, Any] = {}
    param_grids = get_param_grids()

    for name, clf in get_candidate_models().items():
        logger.info("Training %s ...", name)
        base_pipe = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(feature_names)),
                ("model", clf),
            ]
        )

        # Baseline CV on untuned pipeline for transparency
        cv_baseline = cross_validate_model(base_pipe, X_train, y_train)

        tuned_pipe, tune_info = tune_model(
            base_pipe, param_grids[name], X_train, y_train
        )
        metrics = evaluate_model(tuned_pipe, X_test, y_test)
        metrics["cv_mean_roc_auc"] = cv_baseline["roc_auc"]
        metrics["cv_std_roc_auc"] = cv_baseline["roc_auc_std"]
        metrics["tuned_cv_roc_auc"] = tune_info["best_cv_roc_auc"]
        metrics["best_params"] = tune_info["best_params"]

        results[name] = metrics
        pipelines[name] = tuned_pipe
        tuning_notes[name] = tune_info
        logger.info(
            "%s | test ROC-AUC=%.3f | tuned CV ROC-AUC=%.3f",
            name,
            metrics["roc_auc"],
            metrics["tuned_cv_roc_auc"],
        )

    comparison = pd.DataFrame(
        {
            name: {
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "cv_roc_auc": m["tuned_cv_roc_auc"],
            }
            for name, m in results.items()
        }
    ).T.sort_values("roc_auc", ascending=False)
    comparison.to_csv(REPORTS / "model_comparison.csv")

    best_name = str(comparison.index[0])
    best_model = pipelines[best_name]

    selection_rationale = (
        f"{best_name} selected using held-out ROC-AUC as the primary ranking metric, "
        "with precision/recall/F1 and cross-validated ROC-AUC reviewed for robustness."
    )

    joblib.dump(
        {
            "pipeline": best_model,
            "model_name": best_name,
            "feature_names": feature_names,
            "metrics": {
                k: v
                for k, v in results[best_name].items()
                if k not in {"y_true", "y_proba"}
            },
            "comparison": comparison.to_dict(orient="index"),
            "tuning": tuning_notes,
            "selection_rationale": selection_rationale,
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
        },
        MODELS / "best_model.joblib",
    )

    # Comparison chart
    plt.figure(figsize=(9, 5))
    plot_df = comparison[["accuracy", "f1", "roc_auc", "pr_auc"]].copy()
    melted = plot_df.reset_index().melt(id_vars="index", var_name="metric", value_name="score")
    melted = melted.rename(columns={"index": "model"})
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Model performance comparison (hold-out test set)")
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

    # Confusion matrix
    cm = np.array(results[best_name]["confusion_matrix"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion matrix — {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix_best.png", dpi=150, bbox_inches="tight")
    plt.close()

    compact = {}
    for name, m in results.items():
        compact[name] = {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "cv_mean_roc_auc": m["cv_mean_roc_auc"],
            "tuned_cv_roc_auc": m["tuned_cv_roc_auc"],
            "best_params": m["best_params"],
            "classification_report": m["classification_report"],
            "confusion_matrix": m["confusion_matrix"],
        }

    with open(REPORTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_name,
                "selection_rationale": selection_rationale,
                "results": compact,
                "tuning": tuning_notes,
            },
            f,
            indent=2,
        )

    with open(REPORTS / "model_selection.md", "w", encoding="utf-8") as f:
        f.write("# Model Selection Summary\n\n")
        f.write(f"**Selected model:** {best_name}\n\n")
        f.write(f"**Rationale:** {selection_rationale}\n\n")
        f.write("```\n")
        f.write(comparison.round(4).to_string())
        f.write("\n```\n")

    return best_model, comparison, results
