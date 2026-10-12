"""
Streamlit dashboard for the educational heart disease risk estimation project.

This application is for research / education only and is not a medical device.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import DISCLAIMER, FEATURE_NAMES, FIGURES, MODELS, REPORTS

st.set_page_config(
    page_title="Heart Risk Estimation (Educational)",
    page_icon="🫀",
    layout="wide",
)


@st.cache_resource
def load_bundle():
    path = MODELS / "best_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def show_image(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing figure: {path.name}. Run `python train.py` first.")


def sidebar_inputs(feature_names: list[str]) -> dict:
    st.sidebar.header("Sample inputs")
    st.sidebar.caption("Adjust values to estimate a statistical risk score.")

    defaults = {
        "age": 55,
        "sex": 1,
        "cp": 1,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 2,
    }

    values = {}
    for feat in feature_names:
        label = FEATURE_NAMES.get(feat, feat)
        default = defaults.get(feat, 0)
        if feat in {"sex", "fbs", "exang"}:
            values[feat] = st.sidebar.selectbox(label, options=[0, 1], index=int(default))
        elif feat == "cp":
            values[feat] = st.sidebar.selectbox(label, options=[0, 1, 2, 3], index=int(default))
        elif feat in {"restecg", "slope"}:
            values[feat] = st.sidebar.selectbox(label, options=[0, 1, 2], index=min(int(default), 2))
        elif feat == "ca":
            values[feat] = st.sidebar.selectbox(label, options=[0, 1, 2, 3], index=int(default))
        elif feat == "thal":
            values[feat] = st.sidebar.selectbox(label, options=[0, 1, 2, 3], index=min(int(default), 3))
        elif feat == "oldpeak":
            values[feat] = st.sidebar.slider(label, 0.0, 6.0, float(default), 0.1)
        elif feat == "age":
            values[feat] = st.sidebar.slider(label, 20, 100, int(default))
        elif feat == "trestbps":
            values[feat] = st.sidebar.slider(label, 80, 200, int(default))
        elif feat == "chol":
            values[feat] = st.sidebar.slider(label, 100, 600, int(default))
        elif feat == "thalach":
            values[feat] = st.sidebar.slider(label, 70, 220, int(default))
        else:
            values[feat] = st.sidebar.number_input(label, value=float(default))
    return values


def main():
    st.title("Heart Disease Risk Estimation")
    st.subheader("Educational machine learning demonstration")
    st.warning(DISCLAIMER)

    bundle = load_bundle()
    metrics = load_json(REPORTS / "metrics.json")
    comparison = load_csv(REPORTS / "model_comparison.csv")
    eda_summary = load_json(REPORTS / "eda_summary.json")
    importance = load_json(REPORTS / "feature_importance.json")

    if bundle is None:
        st.error("No trained model found. Run `python train.py` from the project root, then refresh.")
        st.code("python train.py\nstreamlit run app.py")
        return

    tabs = st.tabs(
        [
            "Overview",
            "Exploratory Analysis",
            "Model Comparison",
            "Feature Influence",
            "Try a Prediction",
        ]
    )

    with tabs[0]:
        st.markdown(
            """
            This project trains multiple classifiers on a **public heart disease dataset**
            to estimate the likelihood of a higher-risk pattern in the data.

            **Pipeline**
            1. Data cleaning & preprocessing  
            2. Exploratory data analysis  
            3. Train Logistic Regression, Random Forest, Gradient Boosting, and SVM  
            4. Compare accuracy, F1, and ROC-AUC  
            5. Explain influential features via model importances and permutation importance  
            """
        )
        c1, c2, c3 = st.columns(3)
        best_name = bundle.get("model_name", "N/A")
        best_metrics = bundle.get("metrics", {})
        c1.metric("Selected model", best_name)
        c2.metric("ROC-AUC", f"{best_metrics.get('roc_auc', 0):.3f}")
        c3.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.3f}")

        if eda_summary:
            st.write(
                f"Dataset size after cleaning: **{eda_summary['n_rows']}** rows, "
                f"**{eda_summary['n_features']}** features. "
                f"Class balance: `{eda_summary['class_balance']}`."
            )

    with tabs[1]:
        left, right = st.columns(2)
        with left:
            show_image(FIGURES / "target_distribution.png", "Target class counts")
            show_image(FIGURES / "age_by_target.png", "Age by class")
        with right:
            show_image(FIGURES / "correlation_heatmap.png", "Correlation heatmap")
            show_image(FIGURES / "chol_vs_thalach.png", "Cholesterol vs max heart rate")
        show_image(FIGURES / "feature_distributions.png", "Feature distributions")

    with tabs[2]:
        if comparison is not None:
            st.dataframe(comparison, use_container_width=True)
        show_image(FIGURES / "model_comparison.png", "Metric comparison across models")
        left, right = st.columns(2)
        with left:
            show_image(FIGURES / "roc_curves.png", "ROC curves")
        with right:
            show_image(FIGURES / "confusion_matrix_best.png", "Best-model confusion matrix")
        if metrics:
            st.json(metrics)

    with tabs[3]:
        st.markdown(
            "Feature influence helps interpret **which inputs most affect predictions** "
            "in this educational model. Higher bars mean stronger influence on the score."
        )
        left, right = st.columns(2)
        with left:
            show_image(FIGURES / "feature_importance_native.png", "Native model importance")
        with right:
            show_image(
                FIGURES / "feature_importance_permutation.png",
                "Permutation importance (ROC-AUC)",
            )
        if importance and importance.get("permutation"):
            st.subheader("Top features (permutation importance)")
            st.dataframe(pd.DataFrame(importance["permutation"]), use_container_width=True)

    with tabs[4]:
        st.info(
            "The slider panel estimates a **probability-like risk score** from the trained "
            "model. It does not diagnose disease and must not guide medical decisions."
        )
        feature_names = bundle["feature_names"]
        values = sidebar_inputs(feature_names)
        row = pd.DataFrame([values])[feature_names]
        pipe = bundle["pipeline"]
        proba = float(pipe.predict_proba(row)[0, 1])
        pred = int(pipe.predict(row)[0])

        st.subheader("Estimated risk pattern")
        st.progress(min(max(proba, 0.0), 1.0))
        m1, m2 = st.columns(2)
        m1.metric("Model probability (class 1)", f"{proba:.1%}")
        m2.metric(
            "Predicted class",
            "Higher-risk pattern" if pred == 1 else "Lower-risk pattern",
        )
        st.caption(DISCLAIMER)
        with st.expander("Input values used"):
            st.dataframe(row.T.rename(columns={0: "value"}), use_container_width=True)


if __name__ == "__main__":
    main()
