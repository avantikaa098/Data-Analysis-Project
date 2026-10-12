# Heart Disease Risk Estimation (Educational ML Project)

An educational machine learning system that estimates the **statistical likelihood** of a higher-risk heart disease pattern using a public dataset.

> **Important:** This is a **research / educational prediction system**, not a medical diagnosis tool, clinical product, or substitute for professional healthcare advice.

## What this project includes

- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Training multiple ML models (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- Model performance comparison (Accuracy, F1, ROC-AUC)
- Feature influence explanations (native importances + permutation importance)
- Interactive Streamlit dashboard

## Tech stack

Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit

## Project structure

```text
.
├── app.py                 # Streamlit dashboard
├── train.py               # End-to-end training pipeline
├── requirements.txt
├── data/
│   ├── raw/               # Downloaded public dataset
│   └── processed/         # Cleaned data
├── artifacts/
│   ├── figures/           # EDA + evaluation plots
│   ├── models/            # Saved best model
│   └── reports/           # Metrics & importance tables
└── src/
    ├── config.py
    ├── data_loader.py
    ├── preprocess.py
    ├── eda.py
    ├── train_models.py
    └── feature_importance.py
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train models

```bash
python train.py
```

This will:

1. Download the public heart disease dataset
2. Clean / preprocess the data
3. Generate EDA figures
4. Train and compare models
5. Save the best model and feature-importance reports

## Launch dashboard

```bash
streamlit run app.py
```

## Dataset

Uses a publicly available UCI-derived heart disease dataset (educational mirror).  
Target is encoded as a binary label for research-style classification:

- `0` = lower-risk pattern in the dataset
- `1` = higher-risk pattern in the dataset

## Disclaimer

Outputs are statistical estimates learned from historical public data. They can be wrong, biased, or incomplete. Do **not** use them for diagnosis, treatment, triage, or any real-world medical decision.
