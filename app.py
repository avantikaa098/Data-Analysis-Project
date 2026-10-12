"""
CardioRisk Insight — Streamlit stakeholder dashboard.

Educational / research prototype only. Not a medical device or diagnosis tool.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    BUSINESS_CONTEXT,
    DISCLAIMER,
    FEATURE_NAMES,
    FIGURES,
    MODELS,
    PROJECT_NAME,
    PROJECT_TAGLINE,
    REPORTS,
    RESPONSIBLE_AI,
)

st.set_page_config(
    page_title=f"{PROJECT_NAME} | Educational ML Prototype",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@600&display=swap');

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', sans-serif;
}
.block-container { padding-top: 1.2rem; max-width: 1200px; }
.hero {
  background: linear-gradient(135deg, #0B1F33 0%, #123A56 55%, #1F6F8B 100%);
  color: #F4F7FA;
  padding: 1.4rem 1.6rem;
  border-radius: 12px;
  margin-bottom: 1rem;
  border: 1px solid rgba(255,255,255,0.08);
}
.hero h1 {
  font-family: 'IBM Plex Serif', serif;
  font-size: 1.85rem;
  margin: 0 0 0.35rem 0;
  letter-spacing: -0.02em;
}
.hero p { margin: 0; opacity: 0.92; font-size: 0.98rem; }
.kpi {
  background: #F7FAFC;
  border: 1px solid #D9E2EC;
  border-radius: 10px;
  padding: 0.85rem 1rem;
}
.section-label {
  color: #486581;
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 600;
  margin-bottom: 0.25rem;
}
.stAlert { border-radius: 10px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
        st.info(f"Missing figure: `{path.name}`. Run `python train.py` first.")


def sidebar_inputs(feature_names: list[str]) -> dict:
    st.sidebar.markdown("### Scenario inputs")
    st.sidebar.caption("Simulate a structured clinical profile for educational scoring.")

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
    st.markdown(
        f"""
        <div class="hero">
          <div class="section-label">Analytics Prototype</div>
          <h1>{PROJECT_NAME}</h1>
          <p>{PROJECT_TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(DISCLAIMER)

    bundle = load_bundle()
    metrics = load_json(REPORTS / "metrics.json")
    comparison = load_csv(REPORTS / "model_comparison.csv")
    eda_summary = load_json(REPORTS / "eda_summary.json")
    importance = load_json(REPORTS / "feature_importance.json")

    if bundle is None:
        st.error("No trained model found. Run `python train.py`, then refresh this page.")
        st.code("python train.py\nstreamlit run app.py", language="bash")
        return

    tabs = st.tabs(
        [
            "Executive Brief",
            "Exploratory Analysis",
            "Model Lab",
            "Explainability",
            "Scenario Simulator",
            "Responsible AI",
        ]
    )

    best_name = bundle.get("model_name", "N/A")
    best_metrics = bundle.get("metrics", {})

    with tabs[0]:
        st.markdown("#### Business problem")
        st.write(BUSINESS_CONTEXT["problem"])
        st.markdown("#### Solution objective")
        st.write(BUSINESS_CONTEXT["objective"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected model", best_name)
        c2.metric("ROC-AUC", f"{best_metrics.get('roc_auc', 0):.3f}")
        c3.metric("Recall", f"{best_metrics.get('recall', 0):.3f}")
        c4.metric("Precision", f"{best_metrics.get('precision', 0):.3f}")

        st.markdown("#### Why this matters")
        for item in BUSINESS_CONTEXT["value"]:
            st.markdown(f"- {item}")

        if eda_summary:
            st.markdown(
                f"Cleaned dataset: **{eda_summary['n_rows']}** records, "
                f"**{eda_summary['n_features']}** features, "
                f"class balance `{eda_summary['class_balance']}`."
            )

        rationale = bundle.get("selection_rationale") or (
            metrics.get("selection_rationale") if metrics else None
        )
        if rationale:
            st.info(rationale)

        st.markdown("#### Delivery architecture")
        st.code(
            "Raw data → Cleaning/Preprocessing → EDA → CV + Tuning → "
            "Model selection → Explainability → Streamlit dashboard",
            language="text",
        )

    with tabs[1]:
        left, right = st.columns(2)
        with left:
            show_image(FIGURES / "target_distribution.png", "Target class distribution")
            show_image(FIGURES / "age_by_target.png", "Age by class")
        with right:
            show_image(FIGURES / "correlation_heatmap.png", "Correlation heatmap")
            show_image(FIGURES / "chol_vs_thalach.png", "Cholesterol vs max heart rate")
        show_image(FIGURES / "feature_distributions.png", "Feature distributions")

    with tabs[2]:
        st.markdown(
            "Models were trained with stratified splits, **5-fold cross-validation**, "
            "and lightweight **grid search** on ROC-AUC."
        )
        if comparison is not None:
            display = comparison.copy()
            if display.columns[0] in {"Unnamed: 0", "index"}:
                display = display.rename(columns={display.columns[0]: "model"})
            st.dataframe(display, use_container_width=True)
        show_image(FIGURES / "model_comparison.png", "Hold-out metric comparison")
        left, right = st.columns(2)
        with left:
            show_image(FIGURES / "roc_curves.png", "ROC curves")
        with right:
            show_image(FIGURES / "confusion_matrix_best.png", "Best-model confusion matrix")

    with tabs[3]:
        st.markdown(
            "Explainability supports auditability: stakeholders can see which inputs "
            "most influence the educational risk score."
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
            st.dataframe(pd.DataFrame(importance["permutation"]), use_container_width=True)

    with tabs[4]:
        st.info(
            "This simulator produces a statistical score for demonstration. "
            "It must not be used for diagnosis, triage, or treatment decisions."
        )
        feature_names = bundle["feature_names"]
        values = sidebar_inputs(feature_names)
        row = pd.DataFrame([values])[feature_names]
        pipe = bundle["pipeline"]
        proba = float(pipe.predict_proba(row)[0, 1])
        pred = int(pipe.predict(row)[0])

        st.markdown("#### Estimated risk pattern")
        st.progress(min(max(proba, 0.0), 1.0))
        m1, m2, m3 = st.columns(3)
        m1.metric("Model probability (class 1)", f"{proba:.1%}")
        m2.metric(
            "Predicted class",
            "Higher-risk pattern" if pred == 1 else "Lower-risk pattern",
        )
        m3.metric("Model used", best_name)
        st.caption(DISCLAIMER)
        with st.expander("Input vector"):
            pretty = row.T.rename(columns={0: "value"})
            pretty.index = [FEATURE_NAMES.get(i, i) for i in pretty.index]
            st.dataframe(pretty, use_container_width=True)

    with tabs[5]:
        st.markdown("#### Guardrails for responsible use")
        for item in RESPONSIBLE_AI:
            st.markdown(f"- {item}")
        st.markdown(
            """
            #### Interview-ready talking points
            - Framed as an analytics prototype with explicit non-clinical scope
            - Compared multiple algorithms with cross-validation and tuning
            - Selected model using a clear primary metric (ROC-AUC) plus supporting metrics
            - Added explainability for trust, audit, and stakeholder communication
            - Packaged as a reproducible pipeline + dashboard for demo readiness
            """
        )


if __name__ == "__main__":
    main()
