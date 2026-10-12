from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
MODELS = ARTIFACTS / "models"
REPORTS = ARTIFACTS / "reports"

# Classic UCI-derived heart disease CSV (educational public mirror)
HEART_DATASET_URL = (
    "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
)

TARGET_COL = "target"

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

DISCLAIMER = (
    "Educational / research demonstration only. This system estimates statistical "
    "risk patterns from a public dataset and is NOT a medical diagnosis, clinical "
    "decision tool, or substitute for professional healthcare advice."
)
