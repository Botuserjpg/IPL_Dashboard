from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from ipl_analytics.analytics import (
    build_head_to_head_player_performance,
    build_head_to_head_summary,
    build_player_dashboard_table,
    build_venue_stats,
    leaderboard_tables,
)
from ipl_analytics.data import MODELS_DIR, PROCESSED_DIR, dataset_summary, load_raw_datasets, raw_data_exists, save_processed_csv
from ipl_analytics.features import (
    build_match_context_dataset,
    build_match_prediction_dataset,
    build_phase_stats,
    build_player_cluster_dataset,
    build_player_match_features,
    build_player_season_features,
    build_score_prediction_dataset,
    build_team_match_results,
)
from ipl_analytics.models import (
    save_artifacts,
    train_classifiers,
    train_live_chase_classifier,
    train_player_clusters,
    train_player_regressors,
    train_score_regressors,
)


POWERBI_DIR = PROJECT_ROOT / "powerbi"


def ensure_raw_data() -> None:
    if raw_data_exists():
        return
    raise FileNotFoundError(
        "Verified IPL raw data is required. Add data/raw/matches.csv and data/raw/deliveries.csv. "
        "Synthetic/demo data generation has been disabled."
    )


def save_powerbi_csv(frame, name: str) -> None:
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(POWERBI_DIR / name, index=False)


def build_match_probability_export(match_prediction, model):
    features = [
        "venue",
        "team1",
        "team2",
        "toss_decision",
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
    export = match_prediction.copy()
    export["team1_win_probability"] = model.predict_proba(export[features])[:, 1]
    export["team2_win_probability"] = 1 - export["team1_win_probability"]
    export["team1_win_probability_pct"] = (export["team1_win_probability"] * 100).round(2)
    export["team2_win_probability_pct"] = (export["team2_win_probability"] * 100).round(2)
    return export


def build_live_win_probability_export(context, model):
    features = [
        "batting_team",
        "bowling_team",
        "venue",
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
    export = context.copy()
    export["chase_win_probability"] = model.predict_proba(export[features])[:, 1]
    export["defend_win_probability"] = 1 - export["chase_win_probability"]
    export["chase_win_probability_pct"] = (export["chase_win_probability"] * 100).round(2)
    export["defend_win_probability_pct"] = (export["defend_win_probability"] * 100).round(2)
    export["overs_remaining_display"] = (export["balls_remaining"] // 6).astype(str) + "." + (export["balls_remaining"] % 6).astype(str)
    return export


def main() -> None:
    ensure_raw_data()
    bundle = load_raw_datasets(strict_players=True)

    player_matches = build_player_match_features(bundle.matches, bundle.deliveries)
    match_prediction = build_match_prediction_dataset(bundle.matches, bundle.deliveries, player_matches)
    context = build_match_context_dataset(bundle.matches, bundle.deliveries)
    score_prediction = build_score_prediction_dataset(bundle.matches, bundle.deliveries)
    player_season = build_player_season_features(bundle.matches, bundle.deliveries)
    cluster_input = build_player_cluster_dataset(player_season)

    classifiers, classifier_metrics = train_classifiers(match_prediction)
    live_classifiers, live_metrics = train_live_chase_classifier(context)
    score_regressors, score_metrics = train_score_regressors(score_prediction)
    player_regressors, player_metrics = train_player_regressors(player_matches)
    cluster_frame, kmeans_model = train_player_clusters(cluster_input)

    classifiers.update(live_classifiers)
    metrics = pd.concat([classifier_metrics, live_metrics, score_metrics, player_metrics], ignore_index=True, sort=False)

    save_artifacts(classifiers, score_regressors, player_regressors, metrics, cluster_frame, kmeans_model)

    player_dashboard = build_player_dashboard_table(player_season)
    venue_stats = build_venue_stats(bundle.matches, bundle.deliveries)
    team_results = build_team_match_results(bundle.matches, bundle.deliveries)
    h2h_summary = build_head_to_head_summary(team_results)
    h2h_players = build_head_to_head_player_performance(bundle.matches, bundle.deliveries)
    batting_phase, bowling_phase = build_phase_stats(bundle.deliveries)
    matchup_dimension = h2h_summary[["team", "opponent", "matchup_key"]].drop_duplicates()
    player_performance = player_dashboard.merge(
        cluster_frame[["player", "cluster_id", "cluster_name"]],
        on="player",
        how="left",
    )
    live_probability = build_match_probability_export(match_prediction, classifiers["xgboost"])
    live_chase_probability = build_live_win_probability_export(context, classifiers["live_xgboost"])

    outputs = {
        "match_prediction.csv": match_prediction,
        "match_context.csv": context,
        "score_prediction.csv": score_prediction,
        "player_match_features.csv": player_matches,
        "player_season.csv": player_season,
        "player_dashboard.csv": player_dashboard,
        "venue_stats.csv": venue_stats,
        "team_match_results.csv": team_results,
        "head_to_head_summary.csv": h2h_summary,
        "head_to_head_player_performance.csv": h2h_players,
        "batting_phase_stats.csv": batting_phase,
        "bowling_phase_stats.csv": bowling_phase,
        "dim_matchup.csv": matchup_dimension,
        "player_performance_powerbi.csv": player_performance,
        "live_win_probability.csv": live_probability,
        "live_chase_probability.csv": live_chase_probability,
        "model_metrics.csv": metrics,
    }
    batting_lb, bowling_lb = leaderboard_tables(player_season)
    outputs["batting_leaderboard.csv"] = batting_lb
    outputs["bowling_leaderboard.csv"] = bowling_lb

    for name, frame in outputs.items():
        save_processed_csv(frame, name)
    for name, frame in outputs.items():
        if name in {
            "model_metrics.csv",
            "player_performance_powerbi.csv",
            "venue_stats.csv",
            "team_match_results.csv",
            "head_to_head_summary.csv",
            "head_to_head_player_performance.csv",
            "dim_matchup.csv",
            "live_win_probability.csv",
        }:
            save_powerbi_csv(frame, name)

    seasons, matches_count, deliveries_count = dataset_summary(bundle)
    classifier_ranked = classifier_metrics.sort_values("chrono_roc_auc", ascending=False)
    score_ranked = score_metrics.sort_values("cv_mae_mean", ascending=True)
    summary = {
        "seasons": seasons,
        "matches": matches_count,
        "deliveries": deliveries_count,
        "teams": sorted(set(bundle.matches["team1"]).union(bundle.matches["team2"])),
        "best_match_model": classifier_ranked.iloc[0]["model"],
        "best_score_model": score_ranked.iloc[0]["model"],
        "data_source": "data/raw/matches.csv + data/raw/deliveries.csv",
        "synthetic_data_used": False,
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import sklearn

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "sklearn_version.txt").write_text(sklearn.__version__, encoding="utf-8")

    print("Training complete.")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
