"""Preprocessing utilities replicating src/ipl_analytics/models.py exactly for the
held-out evaluation of the live chase win-probability models.

Recipe (identical to ``_preprocessor`` / ``_classification_pipeline``):
  categorical -> SimpleImputer(strategy="most_frequent") -> OneHotEncoder(handle_unknown="ignore")
  numeric     -> SimpleImputer(strategy="median")        -> StandardScaler (only when scale_numeric=True)
ColumnTransformer order: categorical block first, then numeric block.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42

LIVE_CATEGORICAL = ["batting_team", "bowling_team", "venue"]
LIVE_NUMERIC = [
    "target",
    "current_score",
    "runs_needed",
    "overs_remaining",
    "balls_remaining",
    "wickets_remaining",
    "current_run_rate",
    "required_run_rate",
    "run_rate_gap",
    "balls_per_wicket",
    "pace_ratio",
]
LIVE_FEATURES = LIVE_CATEGORICAL + LIVE_NUMERIC
TARGET = "won"


def _onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(categorical: list[str], numeric: list[str], scale_numeric: bool = True) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _onehot())]),
                categorical,
            ),
            ("numeric", Pipeline(numeric_steps), numeric),
        ]
    )


def build_classification_pipeline(categorical: list[str], numeric: list[str], estimator, scale_numeric: bool = True) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(categorical, numeric, scale_numeric)), ("model", estimator)])


def build_live_model_specs() -> dict[str, tuple[object, bool]]:
    specs: dict[str, tuple[object, bool]] = {
        "live_logistic_regression": (
            LogisticRegression(max_iter=1500),
            True,
        ),
        "live_random_forest": (
            RandomForestClassifier(n_estimators=100, max_depth=9, min_samples_leaf=5, random_state=SEED, n_jobs=1),
            False,
        ),
        "live_xgboost": (
            XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=1,
            ),
            False,
        ),
    }
    return specs


def temporal_season_split(frame, test_season: int):
    train_frame = frame[frame["season"] < test_season].copy()
    test_frame = frame[frame["season"] == test_season].copy()
    return train_frame, test_frame
