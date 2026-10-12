from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
MODELS = ARTIFACTS / "models"
REPORTS = ARTIFACTS / "reports"
DOCS = PROJECT_ROOT / "docs"

# Classic UCI-derived heart disease CSV (educational public mirror)
HEART_DATASET_URL = (
    "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
)

TARGET_COL = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

PROJECT_NAME = "CardioRisk Insight"
PROJECT_TAGLINE = "Educational ML decision-support prototype for cardiovascular risk patterning"

FEATURE_NAMES = {
    "age": "Age (years)",
    "sex": "Sex (1=male, 0=female)",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG results",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Number of major vessels (0-3)",
    "thal": "Thalassemia status",
}

BUSINESS_CONTEXT = {
    "problem": (
        "Healthcare providers and payers need scalable ways to prioritize preventive "
        "follow-up using structured clinical indicators—without replacing clinician judgment."
    ),
    "objective": (
        "Build a reproducible ML prototype that estimates higher-risk patterns from a "
        "public dataset, compares models transparently, and explains drivers of predictions."
    ),
    "value": [
        "Faster triage of high-signal cases for educational simulation / analytics demos",
        "Transparent model comparison supporting evidence-based method selection",
        "Feature influence views that improve stakeholder trust and auditability",
        "Reusable pipeline pattern applicable to other structured clinical datasets",
    ],
}

DISCLAIMER = (
    "Educational / research demonstration only. This system estimates statistical "
    "risk patterns from a public dataset and is NOT a medical diagnosis, clinical "
    "decision tool, medical device, or substitute for professional healthcare advice."
)

RESPONSIBLE_AI = [
    "Scope limited to research/education; no clinical deployment claim.",
    "Dataset is historical and may contain demographic or collection bias.",
    "Metrics alone do not imply fairness across subgroups without dedicated audits.",
    "Human-in-the-loop review is required for any real-world healthcare workflow.",
    "Model outputs must be logged, monitored, and periodically revalidated before use.",
]
