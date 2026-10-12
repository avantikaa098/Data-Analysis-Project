"""End-to-end training pipeline for the educational heart-risk project."""

from __future__ import annotations

import json

from src.config import DISCLAIMER, REPORTS
from src.data_loader import load_raw
from src.eda import run_eda
from src.feature_importance import explain_model
from src.preprocess import clean_dataframe, split_and_save
from src.train_models import train_and_compare


def main() -> None:
    print(DISCLAIMER)
    print("\n[1/5] Loading public heart disease dataset...")
    raw = load_raw()

    print("[2/5] Cleaning and splitting data...")
    X_train, X_test, y_train, y_test, feature_names = split_and_save(raw)
    cleaned = clean_dataframe(raw)

    print("[3/5] Running exploratory data analysis...")
    eda_summary = run_eda(cleaned)
    print(
        f"    rows={eda_summary['n_rows']}, features={eda_summary['n_features']}, "
        f"class_balance={eda_summary['class_balance']}"
    )

    print("[4/5] Training and comparing models...")
    best_model, comparison, _ = train_and_compare(
        X_train, X_test, y_train, y_test, feature_names
    )
    print(comparison.round(4).to_string())
    print(f"\n    Best model: {comparison.index[0]}")

    print("[5/5] Explaining feature influence...")
    importance = explain_model(best_model, X_test, y_test, feature_names)
    top = importance["permutation"][:5]
    print("    Top permutation-importance features:")
    for row in top:
        print(f"      - {row['label']}: {row['importance']:.4f}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "run_complete.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "disclaimer": DISCLAIMER,
                "best_model": comparison.index[0],
                "comparison": comparison.to_dict(orient="index"),
            },
            f,
            indent=2,
        )

    print("\nDone. Artifacts saved under artifacts/. Launch the dashboard with:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
