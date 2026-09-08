"""End-to-end held-out evaluation of the IPL live chase win-probability models.

Temporal (season-based) split: train = seasons < max(season), test = season == max(season).
Models replicated with the exact recipes from src/ipl_analytics/models.py::train_live_chase_classifier,
fitted ONLY on the training seasons to avoid leakage, then scored on the held-out season.

Run: python scripts/run_full_evaluation.py
"""
from __future__ import annotations

import platform
import random
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from ipl_analytics.data import load_raw_datasets
from ipl_analytics.features import build_match_context_dataset
from preprocess_test import (
    LIVE_CATEGORICAL,
    LIVE_FEATURES,
    LIVE_NUMERIC,
    SEED,
    TARGET,
    build_classification_pipeline,
    build_live_model_specs,
    temporal_season_split,
)

MODEL_ORDER = ["live_random_forest", "live_xgboost", "live_logistic_regression"]
DISPLAY_NAMES = {
    "live_random_forest": "RandomForest",
    "live_xgboost": "XGBoost",
    "live_logistic_regression": "LogisticRegression",
}
THRESHOLD = 0.5

RESULTS_CSV = PROJECT_ROOT / "evaluation_results.csv"
STEPS_TXT = PROJECT_ROOT / "evaluation_steps.txt"
ENV_TXT = PROJECT_ROOT / "environment.txt"
TEST_SET_CSV = PROJECT_ROOT / "data" / "test_set.csv"


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    bundle = load_raw_datasets(strict_players=True)
    context = build_match_context_dataset(bundle.matches, bundle.deliveries)

    test_season = int(context["season"].max())
    train_frame, test_frame = temporal_season_split(context, test_season)

    TEST_SET_CSV.parent.mkdir(parents=True, exist_ok=True)
    test_frame.to_csv(TEST_SET_CSV, index=False)

    X_train, y_train = train_frame[LIVE_FEATURES], train_frame[TARGET].astype(int)
    X_test, y_test = test_frame[LIVE_FEATURES], test_frame[TARGET].astype(int)

    specs = build_live_model_specs()
    rows = []
    for name in MODEL_ORDER:
        estimator, scale_numeric = specs[name]
        pipeline = build_classification_pipeline(LIVE_CATEGORICAL, LIVE_NUMERIC, estimator, scale_numeric=scale_numeric)
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= THRESHOLD).astype(int)
        rows.append(
            {
                "model": DISPLAY_NAMES[name],
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
                "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_CSV, index=False, float_format="%.4f")

    pip_freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    ENV_TXT.write_text(
        f"python=={platform.python_version()}\n"
        f"platform=={platform.platform()}\n"
        f"scikit-learn=={sklearn.__version__}\n"
        f"xgboost=={xgboost.__version__}\n"
        f"pandas=={pd.__version__}\n"
        f"numpy=={np.__version__}\n\n"
        "--- pip freeze ---\n" + pip_freeze,
        encoding="utf-8",
    )

    train_pos_rate = float(y_train.mean())
    test_pos_rate = float(y_test.mean())
    steps = f"""HELD-OUT EVALUATION LOG - IPL LIVE CHASE WIN-PROBABILITY MODELS
Generated: {datetime.now(timezone.utc).isoformat()}
Working directory: {PROJECT_ROOT}

1. Seeds fixed: numpy.random.seed({SEED}); random.seed({SEED}); every estimator receives random_state={SEED}.
   XGBoost additionally: objective='binary:logistic', eval_metric='logloss', n_jobs=1.
2. Data loaded via ipl_analytics.data.load_raw_datasets(strict_players=True) from
   data/raw/matches.csv ({len(bundle.matches)} matches) and data/raw/deliveries.csv ({len(bundle.deliveries)} deliveries).
3. Match-state frame built via ipl_analytics.features.build_match_context_dataset(matches, deliveries)
   -> {len(context)} rows (one row per ball of each second innings), target column 'won' (int 0/1;
   won=1 when batting_team == winner; positive class = chasing team wins).
4. Temporal season-based split (scripts/preprocess_test.py::temporal_season_split):
   test_season = int(context['season'].max()) = {test_season}
   train_frame = context[context['season'] < {test_season}]  # {len(train_frame)} rows, positive rate {train_pos_rate:.4f}
   test_frame  = context[context['season'] == {test_season}] # {len(test_frame)} rows, positive rate {test_pos_rate:.4f}
   Test set persisted to data/test_set.csv.
5. Feature columns (exact, from src/ipl_analytics/models.py::train_live_chase_classifier):
   categorical = {LIVE_CATEGORICAL}
   numeric     = {LIVE_NUMERIC}
6. Preprocessing per model (scripts/preprocess_test.py::build_preprocessor, identical to
   src/ipl_analytics/models.py::_preprocessor): categorical -> SimpleImputer(strategy='most_frequent')
   -> OneHotEncoder(handle_unknown='ignore'); numeric -> SimpleImputer(strategy='median')
   -> StandardScaler applied ONLY for LogisticRegression (matching original pipeline; no scaling for
   RandomForest/XGBoost).
7. Models retrained on training seasons ONLY (saved *.joblib artifacts were fitted on all seasons incl.
   {test_season} and would leak; this mirrors the repo's own _chrono_evaluate_classifier refit pattern):
   - RandomForest: Pipeline(preprocessor, RandomForestClassifier(n_estimators=100, max_depth=9,
     min_samples_leaf=5, random_state={SEED}, n_jobs=1))
   - XGBoost: Pipeline(preprocessor, XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.05,
     subsample=0.85, colsample_bytree=0.85, objective='binary:logistic', eval_metric='logloss',
     random_state={SEED}, n_jobs=1)) [use_label_encoder omitted: removed in xgboost >= 2.0]
   - LogisticRegression: Pipeline(preprocessor+StandardScaler, LogisticRegression(max_iter=1500))
8. Prediction and metrics per model m in [RandomForest, XGBoost, LogisticRegression]:
   pipeline.fit(X_train, y_train)
   y_proba = pipeline.predict_proba(X_test)[:, 1]
   y_pred = (y_proba >= {THRESHOLD}).astype(int)
   accuracy_score(y_test, y_pred); roc_auc_score(y_test, y_proba);
   precision_score(y_test, y_pred, zero_division=0); recall_score(y_test, y_pred, zero_division=0)
   rounded to 4 decimals and written to evaluation_results.csv (columns: model,accuracy,roc_auc,precision,recall).
9. Versions recorded in environment.txt (python, scikit-learn, xgboost, pandas, numpy + pip freeze).

RESULT ROWS (model,accuracy,roc_auc,precision,recall):
""" + "\n".join(results.to_csv(index=False).strip().splitlines()) + "\n"

    STEPS_TXT.write_text(steps, encoding="utf-8")

    print(results.to_string(index=False))
    print(f"\nWrote: {RESULTS_CSV.name}, {STEPS_TXT.name}, {ENV_TXT.name}, data/test_set.csv")


if __name__ == "__main__":
    main()
