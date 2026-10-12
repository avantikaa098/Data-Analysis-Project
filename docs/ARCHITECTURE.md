# Solution Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                 CardioRisk Insight Prototype               │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────┐   ┌────────────────┐   ┌──────────────────┐
│ Public CSV   │──▶│ Cleaning &     │──▶│ EDA + Reports    │
│ (raw data)   │   │ Preprocessing  │   │ (figures/JSON)   │
└──────────────┘   └────────────────┘   └──────────────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ Model Workbench        │
               │ - Logistic Regression  │
               │ - Random Forest        │
               │ - Gradient Boosting    │
               │ - SVM (RBF)            │
               │ + CV & Grid Search     │
               └────────────────────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ Model Selection        │
               │ Primary: ROC-AUC       │
               │ Support: P/R/F1/PR-AUC │
               └────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌──────────────────────┐     ┌──────────────────────────┐
│ Explainability       │     │ Streamlit Dashboard      │
│ Native + Permutation │     │ Exec / EDA / Lab / Sim   │
└──────────────────────┘     └──────────────────────────┘
```

## Design principles

- **Modularity:** each stage is an importable module under `src/`
- **Reproducibility:** fixed random seed, saved metrics, serialized pipeline
- **Transparency:** model comparison + feature influence for stakeholder trust
- **Governance-first messaging:** educational scope and responsible-AI guardrails
