"""End-to-end training pipeline for CardioRisk Insight."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.config import DISCLAIMER, PROJECT_NAME, RANDOM_STATE, REPORTS, RESPONSIBLE_AI
from src.data_loader import load_raw
from src.eda import run_eda
from src.feature_importance import explain_model
from src.logging_utils import get_logger
from src.preprocess import clean_dataframe, split_and_save
from src.train_models import train_and_compare

logger = get_logger("train")


def main() -> None:
    logger.info("%s | starting pipeline", PROJECT_NAME)
    logger.info(DISCLAIMER)

    logger.info("[1/5] Loading public heart disease dataset")
    raw = load_raw()

    logger.info("[2/5] Cleaning and splitting data")
    X_train, X_test, y_train, y_test, feature_names = split_and_save(raw)
    cleaned = clean_dataframe(raw)

    logger.info("[3/5] Running exploratory data analysis")
    eda_summary = run_eda(cleaned)
    logger.info(
        "rows=%s features=%s class_balance=%s",
        eda_summary["n_rows"],
        eda_summary["n_features"],
        eda_summary["class_balance"],
    )

    logger.info("[4/5] Training, cross-validating, and tuning models")
    best_model, comparison, _ = train_and_compare(
        X_train, X_test, y_train, y_test, feature_names
    )
    print(comparison.round(4).to_string())
    logger.info("Best model: %s", comparison.index[0])

    logger.info("[5/5] Explaining feature influence")
    importance = explain_model(best_model, X_test, y_test, feature_names)
    top = importance["permutation"][:5]
    logger.info("Top permutation-importance features:")
    for row in top:
        logger.info("  - %s: %.4f", row["label"], row["importance"])

    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "run_complete.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "project": PROJECT_NAME,
                "disclaimer": DISCLAIMER,
                "responsible_ai": RESPONSIBLE_AI,
                "best_model": comparison.index[0],
                "comparison": comparison.to_dict(orient="index"),
                "random_state": RANDOM_STATE,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )

    logger.info("Artifacts saved under artifacts/")
    logger.info("Launch dashboard: streamlit run app.py")


if __name__ == "__main__":
    main()
