# CardioRisk Insight

**Educational machine learning prototype** that estimates cardiovascular risk patterns from a public dataset and presents results through a stakeholder-ready dashboard.

> **Important:** Research / education only. Not a medical diagnosis tool, clinical product, or substitute for professional healthcare advice.

![Python](https://img.shields.io/badge/Python-3.10+-0B1F33)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-1F6F8B)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-123A56)
![License](https://img.shields.io/badge/License-MIT-486581)

## Why this project

Built as an end-to-end analytics delivery sample for consulting-style interviews (e.g., Accenture):

- Clear business framing and measurable outcomes
- Reproducible ML pipeline (not just a notebook)
- Multi-model comparison with cross-validation and tuning
- Explainability for trust and audit conversations
- Responsible AI boundaries stated up front

## Business problem

Providers and analytics teams need transparent ways to explore which structured indicators associate with higher-risk patterns. This prototype demonstrates how to turn raw tabular clinical-style data into an explainable scoring workflow for **education and research demos**.

## Solution highlights

| Capability | Implementation |
|---|---|
| Data preparation | Cleaning, type coercion, stratified split, leakage-safe pipeline |
| EDA | Distribution, correlation, and class-balance visuals |
| Modeling | Logistic Regression, Random Forest, Gradient Boosting, SVM |
| Validation | Stratified 5-fold CV + grid search on ROC-AUC |
| Selection | Hold-out ROC-AUC primary; Precision/Recall/F1/PR-AUC supporting |
| Explainability | Native importances + permutation importance |
| Demo layer | Streamlit executive dashboard |

## Architecture

```text
Raw data → Cleaning/Preprocessing → EDA → CV + Tuning → Model selection
        → Explainability → Streamlit dashboard
```

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Project structure

```text
.
├── app.py                 # Streamlit stakeholder dashboard
├── train.py               # End-to-end training entrypoint
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── figures/
│   ├── models/
│   └── reports/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── METHODOLOGY.md
│   └── INTERVIEW_GUIDE.md
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── train_models.py
│   ├── feature_importance.py
│   └── logging_utils.py
└── tests/
    └── test_preprocess.py
```

## Quick start

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train + generate artifacts
python train.py

# Launch dashboard
streamlit run app.py

# Run unit tests
python -m pytest -q
```

## Key results

After training, open:

- `artifacts/reports/model_comparison.csv`
- `artifacts/reports/metrics.json`
- `artifacts/reports/model_selection.md`
- Dashboard tabs: **Executive Brief**, **Model Lab**, **Explainability**

## Responsible AI

- Explicit non-clinical scope and disclaimer on every major surface
- Bias and dataset limitation callouts
- Human-in-the-loop expectation for any real healthcare workflow
- Recommendation path for monitoring, fairness audits, and validation

## Interview materials

- Methodology: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
- Accenture interview guide: [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md)

## Disclaimer

Outputs are statistical estimates from historical public data. They can be wrong, biased, or incomplete. Do **not** use them for diagnosis, treatment, triage, or any real-world medical decision.
