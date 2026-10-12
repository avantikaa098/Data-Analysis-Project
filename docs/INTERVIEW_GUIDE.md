# Accenture Interview Guide — CardioRisk Insight

Use this project to demonstrate **consulting-ready delivery**: business framing, rigorous analytics, explainability, and responsible AI.

## 60-second pitch

> I built an end-to-end educational ML prototype that estimates cardiovascular risk patterns from a public dataset. It includes data cleaning, EDA, multi-model training with cross-validation and tuning, transparent model selection, feature explainability, and a Streamlit dashboard for stakeholder demos. It is explicitly scoped as research/education—not clinical diagnosis—so it also shows governance awareness.

## STAR story (recommended)

**Situation:** Healthcare analytics demos often stop at notebooks and lack stakeholder packaging.  
**Task:** Deliver a reproducible prototype that can be explained to both technical and business audiences.  
**Action:** Implemented a modular Python pipeline (preprocess → EDA → CV/tuning → explainability → dashboard), compared 4 models, and documented methodology + responsible-AI constraints.  
**Result:** Achieved strong hold-out discrimination (see `artifacts/reports/model_comparison.csv`) and a demo-ready UI that communicates trade-offs and limitations clearly.

## Likely interview questions & answers

### Why this problem?
It mirrors Accenture-style work: ambiguous business need → structured analytics approach → explainable prototype → clear caveats for production readiness.

### Why ROC-AUC as primary metric?
It evaluates ranking quality across thresholds and is robust for comparing classifiers before choosing an operating point. I still report Precision/Recall/F1 for decision-threshold conversations.

### How did you prevent data leakage?
Imputation and scaling run inside sklearn `Pipeline` fitted on training folds only; test data is transformed with learned parameters.

### How would you productionize this at Accenture?
1. Secure approved clinical data + legal/ethics review  
2. Fairness & subgroup performance audits  
3. Model monitoring (drift, calibration, alert thresholds)  
4. Human-in-the-loop workflow integration  
5. MLOps: CI tests, model registry, rollback, audit logs  
6. Clinical validation before any care-path impact

### What are the limitations?
Small public dataset, potential bias, no prospective validation, and no clinical claim. The value is the **delivery pattern** and transparent analysis, not medical authority.

## Demo script (5 minutes)

1. Executive Brief: problem, value, KPIs  
2. Model Lab: comparison table + ROC  
3. Explainability: top drivers  
4. Scenario Simulator: change inputs live  
5. Responsible AI: constraints and next steps

## Keywords to use naturally

CRISP-DM · stratified validation · hyperparameter tuning · model governance · explainability · stakeholder communication · reusable accelerator pattern
