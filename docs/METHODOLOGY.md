# Methodology (CRISP-DM Aligned)

## 1. Business understanding

Healthcare analytics teams often need transparent prototypes that show how structured clinical indicators relate to higher-risk patterns. This project delivers an educational decision-support **simulation** (not a clinical product) that:

- Estimates probability-like scores from public data
- Compares multiple algorithms with reproducible validation
- Explains which features drive predictions

## 2. Data understanding

- Source: public UCI-derived heart disease dataset (educational mirror)
- Target: binary label (`0` lower-risk pattern, `1` higher-risk pattern)
- Features: age, sex, chest pain type, blood pressure, cholesterol, ECG indicators, exercise metrics, vessel counts, and related attributes

## 3. Data preparation

- Column standardization and numeric coercion
- Duplicate removal and missing-value handling
- Stratified train/test split for class balance preservation
- Median imputation + standard scaling inside sklearn pipelines (no leakage)

## 4. Modeling

Candidate algorithms:

1. Logistic Regression (interpretable baseline)
2. Random Forest (non-linear ensemble)
3. Gradient Boosting (boosted trees)
4. SVM with RBF kernel (margin-based classifier)

Each model is evaluated with:

- Stratified 5-fold cross-validation
- Grid search on ROC-AUC
- Hold-out test metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC

## 5. Evaluation & selection

Primary ranking metric: **ROC-AUC** (ranking quality under class imbalance sensitivity).  
Supporting metrics: Precision/Recall/F1 for operating-point behavior and stakeholder trade-off discussion.

## 6. Deployment (demo)

- Serialized best pipeline (`artifacts/models/best_model.joblib`)
- Streamlit dashboard for executive walkthrough, EDA, model lab, explainability, and scenario simulation
- Explicit responsible-AI and non-clinical-use messaging
