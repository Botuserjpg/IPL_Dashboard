from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .data import MODELS_DIR, PROCESSED_DIR
from .features import label_clusters

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None


@dataclass
class TrainedModels:
    classifiers: Dict[str, Pipeline]
    score_regressors: Dict[str, Pipeline]
    player_regressors: Dict[str, Pipeline]
    metrics: pd.DataFrame
    cluster_frame: pd.DataFrame
    kmeans_model: dict


def _onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _preprocessor(categorical: list[str], numeric: list[str], scale_numeric: bool = True) -> ColumnTransformer:
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


def _classification_pipeline(categorical: list[str], numeric: list[str], estimator, scale_numeric: bool = True) -> Pipeline:
    return Pipeline([("preprocessor", _preprocessor(categorical, numeric, scale_numeric)), ("model", estimator)])


def _regression_pipeline(categorical: list[str], numeric: list[str], estimator, scale_numeric: bool = True) -> Pipeline:
    return Pipeline([("preprocessor", _preprocessor(categorical, numeric, scale_numeric)), ("model", estimator)])


def _chrono_evaluate_classifier(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, dates: pd.Series, max_train_rows: int = 30000) -> tuple[float, float]:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "y": y})
    ordered = frame.sort_values("date")
    split_idx = int(len(frame) * 0.8)
    train_ix = ordered.iloc[:split_idx].index
    test_ix = ordered.iloc[split_idx:].index
    X_ct, X_ce = X.loc[train_ix], X.loc[test_ix]
    y_ct, y_ce = y.loc[train_ix], y.loc[test_ix]
    if len(X_ce) < 10 or len(np.unique(y_ct)) < 2 or len(np.unique(y_ce)) < 2:
        return np.nan, np.nan
    if len(X_ct) > max_train_rows:
        sample_ix = X_ct.sample(n=max_train_rows, random_state=42).index
        X_ct, y_ct = X_ct.loc[sample_ix], y_ct.loc[sample_ix]
    from sklearn.base import clone

    chrono_pipe = clone(pipeline)
    chrono_pipe.fit(X_ct, y_ct)
    proba_chrono = chrono_pipe.predict_proba(X_ce)[:, 1]
    preds_chrono = (proba_chrono >= 0.5).astype(int)
    return (
        round(float(accuracy_score(y_ce, preds_chrono)), 4),
        round(float(roc_auc_score(y_ce, proba_chrono)), 4),
    )


def _chrono_evaluate_regressor(model: Pipeline, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> tuple[float, float]:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "y": y})
    ordered = frame.sort_values("date")
    split_idx = int(len(frame) * 0.8)
    train_ix = ordered.iloc[:split_idx].index
    test_ix = ordered.iloc[split_idx:].index
    X_ct, X_ce = X.loc[train_ix], X.loc[test_ix]
    y_ct, y_ce = y.loc[train_ix], y.loc[test_ix]
    if len(X_ce) < 10:
        return np.nan, np.nan
    from sklearn.base import clone

    chrono_model = clone(model)
    chrono_model.fit(X_ct, y_ct)
    preds_chrono = chrono_model.predict(X_ce)
    return (
        round(float(mean_absolute_error(y_ce, preds_chrono)), 3),
        round(float(r2_score(y_ce, preds_chrono)), 4),
    )


def _classify_score(name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, dates: pd.Series | None = None) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    chrono_accuracy = np.nan
    chrono_roc_auc = np.nan
    if dates is not None and len(dates) == len(X):
        chrono_accuracy, chrono_roc_auc = _chrono_evaluate_classifier(pipeline, X, y, dates)
    return {
        "model": name,
        "purpose": "match_winner",
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        "chrono_accuracy": chrono_accuracy,
        "chrono_roc_auc": chrono_roc_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_classifiers(match_frame: pd.DataFrame) -> tuple[Dict[str, Pipeline], pd.DataFrame]:
    categorical = ["venue", "team1", "team2", "toss_decision"]
    numeric = [
        "win_pct_diff",
        "recent5_form_diff",
        "recent10_form_diff",
        "toss_rate_diff",
        "chase_diff",
        "defend_diff",
        "venue_win_pct_diff",
        "season_win_pct_diff",
        "avg_runs_diff",
        "avg_conceded_diff",
        "squad_strength_diff",
        "h2h_team1_win_rate",
        "toss_winner_is_team1",
        "season",
    ]
    missing = [col for col in categorical + numeric + ["team1_won"] if col not in match_frame.columns]
    if missing:
        raise ValueError(f"Match training frame missing columns: {missing}")
    X = match_frame[categorical + numeric]
    y = match_frame["team1_won"]
    dates = match_frame["date"]

    specs: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=1500),
        "random_forest": RandomForestClassifier(n_estimators=160, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=1),
        "xgboost": XGBClassifier(
            n_estimators=220,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        ),
    }
    if LGBMClassifier is not None:
        specs["lightgbm"] = LGBMClassifier(n_estimators=220, learning_rate=0.04, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    if CatBoostClassifier is not None:
        specs["catboost"] = CatBoostClassifier(iterations=160, learning_rate=0.04, depth=4, random_seed=42, verbose=False)

    fitted: Dict[str, Pipeline] = {}
    metrics: list[dict] = []
    for name, estimator in specs.items():
        pipeline = _classification_pipeline(categorical, numeric, estimator, scale_numeric=name == "logistic_regression")
        if name == "random_forest":
            search = GridSearchCV(
                pipeline,
                {
                    "model__max_depth": [4, 6],
                    "model__min_samples_leaf": [4, 6],
                },
                scoring="roc_auc",
                cv=3,
                n_jobs=1,
            )
            search.fit(X, y)
            pipeline = search.best_estimator_
        metric = _classify_score(name, pipeline, X, y, dates=dates)
        fitted[name] = pipeline
        metrics.append(metric)
    return fitted, pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)


def train_live_chase_classifier(context_frame: pd.DataFrame) -> tuple[Dict[str, Pipeline], pd.DataFrame]:
    categorical = ["batting_team", "bowling_team", "venue"]
    numeric = [
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
    missing = [col for col in categorical + numeric + ["won"] if col not in context_frame.columns]
    if missing:
        raise ValueError(f"Live chase frame missing columns: {missing}")
    X = context_frame[categorical + numeric]
    y = context_frame["won"]
    dates = context_frame["date"]
    specs = {
        "live_logistic_regression": LogisticRegression(max_iter=1500),
        "live_random_forest": RandomForestClassifier(n_estimators=100, max_depth=9, min_samples_leaf=5, random_state=42, n_jobs=1),
        "live_xgboost": XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        ),
    }
    fitted: Dict[str, Pipeline] = {}
    metrics = []
    for name, estimator in specs.items():
        pipeline = _classification_pipeline(categorical, numeric, estimator, scale_numeric=name == "live_logistic_regression")
        metric = _classify_score(name, pipeline, X, y, dates=dates)
        metric["purpose"] = "live_chase_win_probability"
        fitted[name] = pipeline
        metrics.append(metric)
    return fitted, pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)


def _regression_metrics(name: str, purpose: str, model: Pipeline, X: pd.DataFrame, y: pd.Series, do_cv: bool = True, dates: pd.Series | None = None) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    if do_cv:
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        cv_mae_mean = round(float(-cv_scores.mean()), 3)
        cv_mae_std = round(float(cv_scores.std()), 3)
    else:
        cv_mae_mean = np.nan
        cv_mae_std = np.nan
    chrono_mae = np.nan
    chrono_r2 = np.nan
    if dates is not None and len(dates) == len(X):
        chrono_mae, chrono_r2 = _chrono_evaluate_regressor(model, X, y, dates)
    return {
        "model": name,
        "purpose": purpose,
        "mae": round(mean_absolute_error(y_test, preds), 3),
        "rmse": round(root_mean_squared_error(y_test, preds), 3),
        "r2": round(r2_score(y_test, preds), 4),
        "cv_mae_mean": cv_mae_mean,
        "cv_mae_std": cv_mae_std,
        "chrono_mae": chrono_mae,
        "chrono_r2": chrono_r2,
    }


def train_score_regressors(score_frame: pd.DataFrame) -> tuple[Dict[str, Pipeline], pd.DataFrame]:
    categorical = ["venue", "first_batting_team", "first_bowling_team", "toss_decision"]
    numeric = [
        "season",
        "toss_winner_bats_first",
        "batting_team_avg_first_score",
        "bowling_team_avg_first_conceded",
        "venue_avg_first_score",
        "recent_batting_first_score",
        "recent10_batting_first_score",
        "recent10_bowling_conceded",
        "recent20_venue_score",
        "venue_team_avg_score",
        "season_avg_first_score",
        "era_recent_avg_score",
    ]
    missing = [col for col in categorical + numeric + ["first_innings_score"] if col not in score_frame.columns]
    if missing:
        raise ValueError(f"Score training frame missing columns: {missing}")
    X = score_frame[categorical + numeric]
    y = score_frame["first_innings_score"]
    dates = score_frame["date"]
    specs = {
        "linear_regression": LinearRegression(),
        "random_forest_regressor": RandomForestRegressor(n_estimators=120, max_depth=6, min_samples_leaf=4, random_state=42, n_jobs=1),
        "xgboost_regressor": XGBRegressor(n_estimators=160, max_depth=3, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=1),
    }
    fitted: Dict[str, Pipeline] = {}
    metrics = []
    for name, estimator in specs.items():
        model = _regression_pipeline(categorical, numeric, estimator, scale_numeric=name == "linear_regression")
        metrics.append(_regression_metrics(name, "first_innings_score", model, X, y, dates=dates))
        fitted[name] = model
    return fitted, pd.DataFrame(metrics).sort_values("mae")


def train_player_regressors(player_frame: pd.DataFrame) -> tuple[Dict[str, Pipeline], pd.DataFrame]:
    categorical = ["venue", "team", "opposition", "player"]
    numeric = [
        "season",
        "recent_runs",
        "recent_wickets",
        "recent_strike_rate",
        "recent_fantasy_points",
        "venue_runs_avg",
        "opposition_runs_avg",
    ]
    specs = {
        "player_runs_rf": ("runs", RandomForestRegressor(n_estimators=40, max_depth=6, random_state=42, n_jobs=1)),
        "player_wickets_rf": ("wickets", RandomForestRegressor(n_estimators=40, max_depth=6, random_state=42, n_jobs=1)),
        "player_strike_rate_rf": ("strike_rate", RandomForestRegressor(n_estimators=40, max_depth=6, random_state=42, n_jobs=1)),
        "player_fantasy_points_rf": ("fantasy_points", RandomForestRegressor(n_estimators=40, max_depth=6, random_state=42, n_jobs=1)),
    }
    X = player_frame[categorical + numeric]
    fitted: Dict[str, Pipeline] = {}
    metrics = []
    for name, (target, estimator) in specs.items():
        y = player_frame[target]
        model = _regression_pipeline(categorical, numeric, estimator, scale_numeric=False)
        metrics.append(_regression_metrics(name, f"player_{target}", model, X, y, do_cv=False))
        fitted[name] = model
    return fitted, pd.DataFrame(metrics).sort_values(["purpose", "mae"])


def train_player_clusters(cluster_input: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, dict]:
    features = [
        "seasons",
        "batting_runs",
        "batting_average",
        "strike_rate",
        "runs_per_match",
        "bowling_wickets",
        "economy",
        "wickets_per_match",
        "fantasy_points",
    ]
    X = cluster_input[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_scaled)
    labeled = cluster_input.copy()
    labeled["cluster_id"] = clusters
    labeled["cluster_name"] = label_clusters(cluster_input, clusters)
    return labeled, {"scaler": scaler, "model": kmeans}


def _feature_importance(frame: pd.DataFrame, models: Dict[str, Pipeline]) -> pd.DataFrame:
    rows = []
    for name, pipeline in models.items():
        estimator = pipeline.named_steps["model"]
        if not hasattr(estimator, "feature_importances_"):
            continue
        preprocessor = pipeline.named_steps["preprocessor"]
        try:
            features = preprocessor.get_feature_names_out()
        except Exception:
            features = [f"feature_{idx}" for idx in range(len(estimator.feature_importances_))]
        for feature, importance in zip(features, estimator.feature_importances_):
            rows.append({"model": name, "feature": feature, "importance": importance})
    if not rows:
        return pd.DataFrame(columns=["model", "feature", "importance"])
    return pd.DataFrame(rows).sort_values(["model", "importance"], ascending=[True, False])


def save_artifacts(
    classifiers: Dict[str, Pipeline],
    score_regressors: Dict[str, Pipeline],
    player_regressors: Dict[str, Pipeline],
    metrics: pd.DataFrame,
    cluster_frame: pd.DataFrame,
    kmeans_model: dict,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in classifiers.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    for name, model in score_regressors.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    for name, model in player_regressors.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    joblib.dump(kmeans_model, MODELS_DIR / "kmeans_player_clusters.joblib")
    metrics.to_csv(MODELS_DIR / "model_metrics.csv", index=False)
    cluster_frame.to_csv(MODELS_DIR / "player_clusters.csv", index=False)
    _feature_importance(pd.DataFrame(), classifiers | score_regressors | player_regressors).to_csv(
        PROCESSED_DIR / "feature_importance.csv",
        index=False,
    )


def load_model(name: str) -> Pipeline:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def load_classifier(name: str) -> Pipeline:
    return load_model(name)


def load_model_metrics() -> pd.DataFrame:
    path = MODELS_DIR / "model_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def predict_win_probability(model: Pipeline, payload: pd.DataFrame) -> np.ndarray:
    try:
        return model.predict_proba(payload)[:, 1]
    except AttributeError as exc:
        for stale in MODELS_DIR.glob("*.joblib"):
            stale.unlink(missing_ok=True)
        stamp = MODELS_DIR / "sklearn_version.txt"
        if stamp.exists():
            stamp.unlink(missing_ok=True)
        raise RuntimeError(
            "Stale model artifacts detected and removed. Refresh or rerun the app to retrain models."
        ) from exc


def predict_score_interval(model: Pipeline, payload: pd.DataFrame, residual_mae: float = 12.0) -> dict[str, float]:
    expected = float(model.predict(payload)[0])
    return {
        "expected_score": round(expected, 1),
        "low_score": int(round(expected - residual_mae)),
        "high_score": int(round(expected + residual_mae)),
        "confidence_interval_low": round(expected - 1.96 * residual_mae, 1),
        "confidence_interval_high": round(expected + 1.96 * residual_mae, 1),
    }
