from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from ipl_analytics.analytics import (
    build_head_to_head_player_performance,
    build_head_to_head_summary,
    build_team_match_results,
)
from ipl_analytics.data import load_processed_csv, load_raw_datasets, save_processed_csv
from ipl_analytics.models import load_classifier, load_model_metrics
from train_models import build_live_win_probability_export


POWERBI_DIRS = [PROJECT_ROOT / "powerbi", PROJECT_ROOT / "powerbi2"]


def save_powerbi_csv(frame: pd.DataFrame, name: str) -> None:
    for directory in POWERBI_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / name, index=False)


def main() -> None:
    bundle = load_raw_datasets()
    player_dashboard = load_processed_csv("player_dashboard.csv")
    context = load_processed_csv("match_context.csv")
    venue_stats = load_processed_csv("venue_stats.csv")
    model_metrics = load_model_metrics()
    player_clusters = pd.read_csv(PROJECT_ROOT / "models" / "player_clusters.csv")

    team_results = build_team_match_results(bundle.matches, bundle.deliveries)
    h2h_summary = build_head_to_head_summary(team_results)
    h2h_players = build_head_to_head_player_performance(bundle.matches, bundle.deliveries)
    matchup_dimension = h2h_summary[["team", "opponent", "matchup_key"]].drop_duplicates()
    player_performance = player_dashboard.merge(
        player_clusters[["player", "cluster_id", "cluster_name"]],
        on="player",
        how="left",
    )
    live_probability = build_live_win_probability_export(context, load_classifier("xgboost"))

    outputs = {
        "player_performance_powerbi.csv": player_performance,
        "venue_stats.csv": venue_stats,
        "team_match_results.csv": team_results,
        "head_to_head_summary.csv": h2h_summary,
        "head_to_head_player_performance.csv": h2h_players,
        "dim_matchup.csv": matchup_dimension,
        "live_win_probability.csv": live_probability,
        "model_metrics.csv": model_metrics,
    }

    for name, frame in outputs.items():
        save_processed_csv(frame, name)
        save_powerbi_csv(frame, name)

    print("Power BI export complete.")
    for name, frame in outputs.items():
        print(f"{name}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
