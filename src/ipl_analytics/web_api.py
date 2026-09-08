from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ipl_analytics.analytics import build_head_to_head
from ipl_analytics.data import CURRENT_IPL_TEAMS, MODELS_DIR, PROCESSED_DIR, load_processed_csv, load_raw_datasets
from ipl_analytics.models import load_classifier, load_model, load_model_metrics, predict_score_interval, predict_win_probability


def _sklearn_version() -> str:
    import sklearn

    return sklearn.__version__


def _needs_retrain() -> bool:
    required = [
        MODELS_DIR / "xgboost.joblib",
        MODELS_DIR / "xgboost_regressor.joblib",
        MODELS_DIR / "model_metrics.csv",
        PROCESSED_DIR / "player_dashboard.csv",
        PROCESSED_DIR / "venue_stats.csv",
        PROCESSED_DIR / "match_prediction.csv",
        PROCESSED_DIR / "score_prediction.csv",
        PROCESSED_DIR / "player_match_features.csv",
        MODELS_DIR / "live_xgboost.joblib",
    ]
    if not all(path.exists() for path in required):
        return True
    stamp = MODELS_DIR / "sklearn_version.txt"
    return (not stamp.exists()) or stamp.read_text().strip() != _sklearn_version()


def _run_training() -> None:
    import runpy

    runpy.run_path(str(PROJECT_ROOT / "scripts" / "train_models.py"), run_name="__main__")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "sklearn_version.txt").write_text(_sklearn_version())


def ensure_assets() -> None:
    if _needs_retrain():
        for stale in MODELS_DIR.glob("*.joblib"):
            stale.unlink(missing_ok=True)
        _run_training()


def load_data() -> dict:
    ensure_assets()
    bundle = load_raw_datasets(strict_players=True)
    return {
        "bundle": bundle,
        "match_prediction": load_processed_csv("match_prediction.csv"),
        "score_prediction": load_processed_csv("score_prediction.csv"),
        "player_matches": load_processed_csv("player_match_features.csv"),
        "player_season": load_processed_csv("player_season.csv"),
        "player_dashboard": load_processed_csv("player_dashboard.csv"),
        "venue_stats": load_processed_csv("venue_stats.csv"),
        "team_results": load_processed_csv("team_match_results.csv"),
        "h2h": load_processed_csv("head_to_head_summary.csv"),
        "batting_phase": load_processed_csv("batting_phase_stats.csv"),
        "bowling_phase": load_processed_csv("bowling_phase_stats.csv"),
        "batting_lb": load_processed_csv("batting_leaderboard.csv"),
        "bowling_lb": load_processed_csv("bowling_leaderboard.csv"),
        "metrics": load_model_metrics(),
        "feature_importance": load_processed_csv("feature_importance.csv"),
        "summary": json.loads((PROCESSED_DIR / "summary.json").read_text(encoding="utf-8")),
    }


def load_models() -> dict:
    ensure_assets()
    return {
        "winner": load_classifier("xgboost"),
        "live": load_classifier("live_xgboost"),
        "score": load_model("xgboost_regressor"),
    }


# --------------------------------------------------------------------------
# Payload builders
# --------------------------------------------------------------------------
def _team_squad_strength(player_matches: pd.DataFrame, team: str, fallback: float) -> float:
    pm = player_matches[player_matches["team"] == team].copy()
    if pm.empty:
        return fallback
    latest = pm.sort_values(["date", "match_id"]).groupby("player", as_index=False).last()
    if latest.empty:
        return fallback
    return float(latest["recent_fantasy_points"].nlargest(5).sum())


def _squad_fallback(player_matches: pd.DataFrame) -> float:
    pm = player_matches.sort_values(
        ["match_id", "team", "recent_fantasy_points"], ascending=[True, True, False]
    )
    top5 = pm.groupby(["match_id", "team"]).head(5).groupby("match_id")["recent_fantasy_points"].sum()
    return float(top5.mean())


def _team_profile(
    team_results: pd.DataFrame, player_matches: pd.DataFrame, team: str, venue: str, season: int, fallback: float
) -> dict[str, float]:
    team_rows = team_results[team_results["team"] == team].sort_values(["date", "match_id"])
    if team_rows.empty:
        return {
            "team_win_pct": 0.5,
            "team_recent5_form": 0.5,
            "team_recent10_form": 0.5,
            "team_toss_win_rate": 0.5,
            "team_chase_success_rate": 0.5,
            "team_defend_success_rate": 0.5,
            "team_venue_win_pct": 0.5,
            "team_season_win_pct": 0.5,
            "team_avg_runs": 165.0,
            "team_avg_conceded": 165.0,
            "squad_strength": fallback,
        }
    chase = team_rows[team_rows["chasing"] == 1]
    defend = team_rows[team_rows["batting_first"] == 1]
    venue_rows = team_rows[team_rows["venue"] == venue]
    season_rows = team_rows[team_rows["season"] == season]
    return {
        "team_win_pct": float(team_rows["won"].mean()),
        "team_recent5_form": float(team_rows.tail(5)["won"].mean()),
        "team_recent10_form": float(team_rows.tail(10)["won"].mean()),
        "team_toss_win_rate": float(team_rows["toss_won"].mean()),
        "team_chase_success_rate": float(chase["won"].mean()) if not chase.empty else 0.5,
        "team_defend_success_rate": float(defend["won"].mean()) if not defend.empty else 0.5,
        "team_venue_win_pct": float(venue_rows["won"].mean()) if not venue_rows.empty else 0.5,
        "team_season_win_pct": float(season_rows["won"].mean()) if not season_rows.empty else 0.5,
        "team_avg_runs": float(team_rows["runs"].mean()),
        "team_avg_conceded": float(team_rows["opponent_runs"].mean()),
        "squad_strength": _team_squad_strength(player_matches, team, fallback),
    }


def _winner_payload(
    team_results: pd.DataFrame,
    h2h: pd.DataFrame,
    player_matches: pd.DataFrame,
    venue: str,
    team1: str,
    team2: str,
    toss_winner: str,
    toss_decision: str,
    season: int,
) -> pd.DataFrame:
    fallback = _squad_fallback(player_matches)
    p1 = _team_profile(team_results, player_matches, team1, venue, season, fallback)
    p2 = _team_profile(team_results, player_matches, team2, venue, season, fallback)
    h2h_row = h2h[(h2h["team"] == team1) & (h2h["opponent"] == team2)]
    h2h_rate = float(h2h_row["win_rate"].iloc[0] / 100) if not h2h_row.empty else 0.5
    return pd.DataFrame(
        [
            {
                "venue": venue,
                "team1": team1,
                "team2": team2,
                "toss_decision": toss_decision,
                "win_pct_diff": p1["team_win_pct"] - p2["team_win_pct"],
                "recent5_form_diff": p1["team_recent5_form"] - p2["team_recent5_form"],
                "recent10_form_diff": p1["team_recent10_form"] - p2["team_recent10_form"],
                "toss_rate_diff": p1["team_toss_win_rate"] - p2["team_toss_win_rate"],
                "chase_diff": p1["team_chase_success_rate"] - p2["team_chase_success_rate"],
                "defend_diff": p1["team_defend_success_rate"] - p2["team_defend_success_rate"],
                "venue_win_pct_diff": p1["team_venue_win_pct"] - p2["team_venue_win_pct"],
                "season_win_pct_diff": p1["team_season_win_pct"] - p2["team_season_win_pct"],
                "avg_runs_diff": p1["team_avg_runs"] - p2["team_avg_runs"],
                "avg_conceded_diff": p1["team_avg_conceded"] - p2["team_avg_conceded"],
                "squad_strength_diff": p1["squad_strength"] - p2["squad_strength"],
                "h2h_team1_win_rate": h2h_rate,
                "toss_winner_is_team1": int(toss_winner == team1),
                "season": season,
            }
        ]
    )


def _score_payload(
    score_prediction: pd.DataFrame,
    venue: str,
    batting_team: str,
    bowling_team: str,
    toss_winner: str,
    toss_decision: str,
) -> pd.DataFrame:
    frame = score_prediction.sort_values(["date", "match_id"])
    history = frame[frame["first_batting_team"] == batting_team]
    if history.empty:
        history = frame
    bowling_history = frame[frame["first_bowling_team"] == bowling_team]
    venue_history = frame[frame["venue"] == venue]
    overall = frame["first_innings_score"]
    season = int(frame["season"].max())
    season_history = frame[frame["season"] == season]
    venue_team = frame[(frame["venue"] == venue) & (frame["first_batting_team"] == batting_team)]["first_innings_score"]

    def safe(series, fallback):
        return float(series.mean()) if not series.empty else float(fallback)

    return pd.DataFrame(
        [
            {
                "venue": venue,
                "first_batting_team": batting_team,
                "first_bowling_team": bowling_team,
                "toss_decision": toss_decision,
                "season": season,
                "toss_winner_bats_first": int(toss_winner == batting_team and toss_decision == "bat"),
                "batting_team_avg_first_score": safe(history["first_innings_score"], overall.mean()),
                "bowling_team_avg_first_conceded": safe(bowling_history["first_innings_score"], overall.mean()),
                "venue_avg_first_score": safe(venue_history["first_innings_score"], overall.mean()),
                "recent_batting_first_score": safe(history["first_innings_score"].tail(5), overall.mean()),
                "recent10_batting_first_score": safe(history["first_innings_score"].tail(10), overall.mean()),
                "recent10_bowling_conceded": safe(bowling_history["first_innings_score"].tail(10), overall.mean()),
                "recent20_venue_score": safe(venue_history["first_innings_score"].tail(20), overall.mean()),
                "venue_team_avg_score": safe(venue_team, overall.mean()),
                "season_avg_first_score": safe(season_history["first_innings_score"], overall.mean()),
                "era_recent_avg_score": float(overall.tail(50).mean()),
            }
        ]
    )


def _live_features(
    chasing_team: str,
    defending_team: str,
    venue: str,
    target: int,
    current_score: int,
    runs_needed: int,
    balls_remaining: int,
    wickets_remaining: int,
) -> dict:
    overs_remaining = balls_remaining / 6
    balls_bowled = max(120 - balls_remaining, 1)
    current_run_rate = current_score / (balls_bowled / 6)
    required_run_rate = runs_needed / overs_remaining if overs_remaining > 0 else 99.0
    return {
        "batting_team": chasing_team,
        "bowling_team": defending_team,
        "venue": venue,
        "target": target,
        "current_score": current_score,
        "runs_needed": runs_needed,
        "overs_remaining": overs_remaining,
        "balls_remaining": balls_remaining,
        "wickets_remaining": wickets_remaining,
        "current_run_rate": current_run_rate,
        "required_run_rate": required_run_rate,
        "run_rate_gap": required_run_rate - current_run_rate,
        "balls_per_wicket": balls_remaining / wickets_remaining if wickets_remaining > 0 else 0.0,
        "pace_ratio": current_run_rate / required_run_rate if required_run_rate > 0 else 1.0,
    }


def _top_performers(player_matches: pd.DataFrame, teams: list[str], venue: str) -> pd.DataFrame:
    candidates = player_matches[player_matches["team"].isin(teams)].copy()
    if candidates.empty:
        return pd.DataFrame()
    venue_bonus = candidates["venue"].eq(venue).astype(int)
    candidates["prediction_score"] = candidates["recent_fantasy_points"] + venue_bonus * 5
    cols = ["player", "team", "recent_runs", "recent_wickets", "recent_strike_rate", "recent_fantasy_points", "prediction_score"]
    return (
        candidates.sort_values(["prediction_score", "date"], ascending=[False, False])
        .drop_duplicates("player")
        .head(10)[cols]
        .round(2)
    )


# --------------------------------------------------------------------------
# API orchestration
# --------------------------------------------------------------------------
class WebApi:
    def __init__(self, data: dict | None = None, models: dict | None = None):
        self.data = data if data is not None else load_data()
        self.models = models if models is not None else load_models()
        bundle = self.data["bundle"]
        self.teams = sorted(
            [team for team in set(bundle.matches["team1"]).union(bundle.matches["team2"]) if team in CURRENT_IPL_TEAMS]
        )
        self.venues = sorted(bundle.matches["venue"].dropna().unique())
        self.seasons = sorted(self.data["player_dashboard"]["season"].astype(int).unique())
        self.current_season = int(self.seasons[-1])

    def meta(self) -> dict:
        summary = self.data["summary"]
        metrics = self.data["metrics"]
        xgb_row = metrics[metrics["model"] == "xgboost"]
        chrono_auc = float(xgb_row["chrono_roc_auc"].iloc[0]) if not xgb_row.empty and pd.notna(xgb_row["chrono_roc_auc"].iloc[0]) else None
        live_row = metrics[metrics["model"] == "live_xgboost"]
        live_auc = float(live_row["chrono_roc_auc"].iloc[0]) if not live_row.empty and pd.notna(live_row["chrono_roc_auc"].iloc[0]) else None
        return {
            "seasons": summary["seasons"],
            "matches": summary["matches"],
            "deliveries": summary["deliveries"],
            "teams": self.teams,
            "venues": self.venues,
            "seasons_list": self.seasons,
            "current_season": self.current_season,
            "best_match_model": summary.get("best_match_model", "xgboost"),
            "best_score_model": summary.get("best_score_model", "xgboost_regressor"),
            "winner_chrono_auc": chrono_auc,
            "live_chrono_auc": live_auc,
        }

    def predict_match(self, team1: str, team2: str, venue: str, toss_winner: str, toss_decision: str, season: int | None = None) -> dict:
        data = self.data
        season = int(season) if season else self.current_season
        payload = _winner_payload(
            data["team_results"], data["h2h"], data["player_matches"], venue, team1, team2, toss_winner, toss_decision, season
        )
        team1_prob = float(predict_win_probability(self.models["winner"], payload)[0]) * 100
        team2_prob = 100 - team1_prob
        winner = team1 if team1_prob >= team2_prob else team2
        return {
            "team1": team1,
            "team2": team2,
            "team1_prob": round(team1_prob, 2),
            "team2_prob": round(team2_prob, 2),
            "winner": winner,
            "confidence": round(max(team1_prob, team2_prob), 2),
            "venue": venue,
            "season": season,
        }

    def predict_score(self, batting_team: str, bowling_team: str, venue: str, toss_winner: str, toss_decision: str, season: int | None = None) -> dict:
        data = self.data
        season = int(season) if season else self.current_season
        frame = data["score_prediction"].sort_values(["date", "match_id"])
        frame = frame[frame["season"] == season] if season in frame["season"].astype(int).values else frame
        payload = _score_payload(
            frame, venue, batting_team, bowling_team, toss_winner, toss_decision
        )
        score_metrics_row = data["metrics"].query("model == 'xgboost_regressor' and purpose == 'first_innings_score'")
        residual = float(score_metrics_row["mae"].fillna(12).iloc[0]) if not score_metrics_row.empty else 12.0
        interval = predict_score_interval(self.models["score"], payload, residual_mae=residual)
        return {
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "expected_score": int(round(interval["expected_score"])),
            "low_score": int(round(interval["low_score"])),
            "high_score": int(round(interval["high_score"])),
            "season": season,
        }

    def predict_live(self, chasing_team: str, defending_team: str, venue: str, target: int, current_score: int,
                     runs_needed: int, balls_remaining: int, wickets_remaining: int) -> dict:
        feats = _live_features(chasing_team, defending_team, venue, target, current_score, runs_needed, balls_remaining, wickets_remaining)
        payload = pd.DataFrame([feats])
        chase_prob = float(predict_win_probability(self.models["live"], payload)[0]) * 100
        return {
            "chasing_team": chasing_team,
            "defending_team": defending_team,
            "chase_prob": round(chase_prob, 2),
            "defend_prob": round(100 - chase_prob, 2),
            "current_score": current_score,
            "wickets_lost": 10 - wickets_remaining,
            "required_run_rate": round(feats["required_run_rate"], 2),
        }

    def players_to_watch(self, team1: str, team2: str, venue: str) -> dict:
        top = _top_performers(self.data["player_matches"], [team1, team2], venue)
        return {"columns": list(top.columns), "rows": top.values.tolist()}

    def head_to_head(self, team_a: str, team_b: str) -> dict:
        summary, top = build_head_to_head(self.data["bundle"].matches, self.data["bundle"].deliveries, team_a, team_b)
        return {
            "team_a": team_a,
            "team_b": team_b,
            "summary": {"columns": list(summary.columns), "rows": summary.astype(object).where(pd.notnull(summary), None).values.tolist()} if not summary.empty else None,
            "top_performers": {"columns": list(top.columns), "rows": top.astype(object).where(pd.notnull(top), None).values.tolist()} if not top.empty else None,
        }

    def team_records(self) -> dict:
        trows = self.data["team_results"]
        overall = (
            trows.groupby("team")
            .agg(
                matches=("match_id", "nunique"),
                wins=("won", "sum"),
                losses=("lost", "sum"),
                avg_runs=("runs", "mean"),
                avg_conceded=("opponent_runs", "mean"),
            )
            .reset_index()
        )
        overall["win_rate"] = (overall["wins"] / overall["matches"] * 100).round(1)
        overall["net_run_margin"] = (overall["avg_runs"] - overall["avg_conceded"]).round(1)
        overall = overall.sort_values("win_rate", ascending=False).round(1)
        return {"columns": list(overall.columns), "rows": overall.values.tolist()}

    def player_season(self, season: int) -> dict:
        view = self.data["player_dashboard"][self.data["player_dashboard"]["season"].astype(int) == int(season)]
        top_fantasy = (
            view.sort_values("fantasy_points", ascending=False)
            .head(20)[["player", "batting_runs", "strike_rate", "batting_average", "bowling_wickets", "economy", "fantasy_points"]]
            .round(2)
        )
        batting_cols = [c for c in ["player", "batting_runs", "batting_average", "strike_rate"] if c in view.columns]
        best_batsmen = view.sort_values("batting_runs", ascending=False).head(15)[batting_cols].round(2)
        bowl_cols = [c for c in ["player", "bowling_wickets", "economy"] if c in view.columns]
        best_bowlers = view.sort_values("bowling_wickets", ascending=False).head(15)[bowl_cols].round(2)
        scatter_cols = ["player", "strike_rate", "batting_average", "batting_runs", "bowling_wickets"]
        scatter = None
        if all(col in view.columns for col in scatter_cols):
            scatter = view.dropna(subset=["strike_rate", "batting_average"])[scatter_cols].to_dict(orient="records")
        return {
            "season": int(season),
            "top_fantasy": {"columns": list(top_fantasy.columns), "rows": top_fantasy.values.tolist()},
            "best_batsmen": {"columns": list(best_batsmen.columns), "rows": best_batsmen.values.tolist()},
            "best_bowlers": {"columns": list(best_bowlers.columns), "rows": best_bowlers.values.tolist()},
            "scatter": scatter,
        }

    def venue_stats(self) -> dict:
        table_cols = [
            "venue",
            "matches",
            "avg_first_innings_score",
            "avg_second_innings_score",
            "chase_win_rate",
            "defend_win_rate",
            "toss_win_match_win_rate",
        ]
        stats = self.data["venue_stats"]
        table = stats[table_cols].sort_values("matches", ascending=False)
        return {
            "columns": list(table.columns),
            "rows": table.round(2).values.tolist(),
            "highest_scoring": stats.sort_values("avg_first_innings_score", ascending=False).head(12)[
                ["venue", "matches", "avg_first_innings_score"]
            ].round(1).to_dict(orient="records"),
            "chase_defend": stats.sort_values("matches", ascending=False).head(12)[
                ["venue", "matches", "chase_win_rate", "defend_win_rate"]
            ].round(1).to_dict(orient="records"),
        }

    def model_insights(self) -> dict:
        metrics_df = self.data["metrics"]
        purposes = {
            "match_winner": "Match Winner Classification",
            "live_chase_win_probability": "Live Chase Win Probability",
            "first_innings_score": "First-Innings Score Regression",
            "player_runs": "Player Performance Regression",
        }
        groups = {}
        for purpose, title in purposes.items():
            subset = metrics_df[metrics_df["purpose"] == purpose]
            if subset.empty:
                continue
            display_cols = [c for c in subset.columns if c not in {"purpose", "tn", "fp", "fn", "tp", "cv_roc_auc_std", "cv_mae_std"}]
            subset = subset[display_cols].round(3)
            groups[purpose] = {
                "title": title,
                "columns": list(subset.columns),
                "rows": subset.astype(object).where(pd.notnull(subset), None).values.tolist(),
            }
        fi = self.data["feature_importance"]
        xgb_fi = fi[fi["model"] == "xgboost"].sort_values("importance", ascending=False).head(15)
        feature_importance = (
            xgb_fi[["feature", "importance"]]
            .assign(feature=lambda df: df["feature"].str.replace(r"^(venue|team1|team2|toss_decision)_", "category·", regex=True))
            .round(3)
            .to_dict(orient="records")
        )
        return {"groups": groups, "feature_importance": feature_importance}

    def season_history(self) -> dict:
        matches = self.data["bundle"].matches
        finals = matches.sort_values(["date", "match_id"]).groupby("season", as_index=False).tail(1)
        champions = finals[finals["winner"].notna()][["season", "winner", "venue"]].rename(
            columns={"winner": "champion", "venue": "final_venue"}
        )
        players = self.data["player_season"]
        top_bat = (
            players.sort_values("batting_runs", ascending=False)
            .drop_duplicates("season")[["season", "player", "batting_runs"]]
            .rename(columns={"player": "top_batsman", "batting_runs": "top_batsman_runs"})
        )
        top_bowl = (
            players.sort_values("bowling_wickets", ascending=False)
            .drop_duplicates("season")[["season", "player", "bowling_wickets"]]
            .rename(columns={"player": "top_bowler", "bowling_wickets": "top_bowler_wickets"})
        )
        table = (
            champions.merge(top_bat, on="season", how="left")
            .merge(top_bowl, on="season", how="left")
            .sort_values("season")
        )
        return {
            "columns": list(table.columns),
            "rows": table.astype(object).where(pd.notnull(table), None).values.tolist(),
            "champions": table[["season", "champion"]].to_dict(orient="records"),
        }

    def season_review(self, season: int) -> dict:
        season = int(season)
        matches = self.data["bundle"].matches
        season_matches = matches[matches["season"] == season]
        finals = season_matches.sort_values(["date", "match_id"]).tail(1)
        champion = None
        runner_up = None
        final_venue = None
        if not finals.empty:
            final = finals.iloc[0]
            champion = final["winner"] if pd.notna(final["winner"]) else None
            final_venue = final["venue"]
            if champion is not None:
                runner_up = final["team1"] if champion == final["team2"] else final["team2"]

        players = self.data["player_season"]
        view = players[players["season"] == season]
        top_batsman = None
        top_bowler = None
        if not view.empty:
            bat = view.sort_values("batting_runs", ascending=False).head(1).iloc[0]
            top_batsman = {
                "player": bat["player"],
                "runs": int(bat["batting_runs"]),
                "average": round(float(bat["batting_average"]), 2),
                "strike_rate": round(float(bat["strike_rate"]), 2),
            }
            bowl = view.sort_values("bowling_wickets", ascending=False).head(1).iloc[0]
            top_bowler = {
                "player": bowl["player"],
                "wickets": int(bowl["bowling_wickets"]),
                "economy": round(float(bowl["economy"]), 2),
            }

        highest = None
        scores = self.data["score_prediction"]
        scores = scores[scores["season"] == season] if "season" in scores.columns else scores
        if not scores.empty:
            top = scores.sort_values("first_innings_score", ascending=False).head(1).iloc[0]
            highest = {
                "team": top["first_batting_team"],
                "score": int(top["first_innings_score"]),
                "venue": top["venue"],
                "date": top["date"],
            }

        return {
            "season": season,
            "matches": int(len(season_matches)),
            "champion": champion,
            "runner_up": runner_up,
            "final_venue": final_venue,
            "top_batsman": top_batsman,
            "top_bowler": top_bowler,
            "highest_total": highest,
        }

    def team_form(self, team: str, limit: int = 10) -> dict:
        rows = self.data["team_results"]
        team_rows = rows[rows["team"] == team].sort_values(["date", "match_id"], ascending=[False, False])
        recent = team_rows.head(int(limit)).copy()
        recent["result"] = recent["won"].map({1: "Won", 0: "Lost"})
        cols = ["date", "opponent", "venue", "result", "runs", "opponent_runs", "run_margin", "toss_decision"]
        cols = [c for c in cols if c in recent.columns]
        table = recent[cols].round(1)
        return {
            "team": team,
            "limit": int(limit),
            "columns": list(table.columns),
            "rows": table.astype(object).where(pd.notnull(table), None).values.tolist(),
            "recent_wins": int(recent["won"].sum()),
            "recent_played": int(len(recent)),
        }

    def player_profile(self, name: str) -> dict:
        players = self.data["player_season"]
        players = players.rename(columns={"player": "player"})
        exact = players[players["player"].str.lower() == name.strip().lower()]
        if exact.empty:
            candidates = (
                players[players["player"].str.lower().str.contains(name.strip().lower(), regex=False)]["player"]
                .unique()
                .tolist()
            )
            return {"matched": False, "query": name, "candidates": sorted(candidates)[:25]}
        career = exact.copy()
        seasons = (
            career.sort_values("season")
            .groupby("season", as_index=False)
            .agg(
                matches=("batting_matches", "sum"),
                runs=("batting_runs", "sum"),
                balls_faced=("batting_balls", "sum"),
                dismissals=("dismissals", "sum"),
                boundaries=("boundaries", "sum"),
                balls_bowled=("balls_bowled", "sum"),
                wickets=("bowling_wickets", "sum"),
                bowling_runs=("bowling_runs", "sum"),
                fantasy_points=("fantasy_points", "sum"),
            )
        )
        seasons["average"] = np.where(
            seasons["dismissals"] > 0, seasons["runs"] / seasons["dismissals"], seasons["runs"]
        )
        seasons["strike_rate"] = np.where(
            seasons["balls_faced"] > 0, seasons["runs"] * 100 / seasons["balls_faced"], 0.0
        )
        seasons["economy"] = np.where(
            seasons["balls_bowled"] > 0, seasons["bowling_runs"] / (seasons["balls_bowled"] / 6), 0.0
        )
        seasons["wicket_frequency"] = np.where(
            seasons["bowling_runs"] > 0, seasons["bowling_runs"] / seasons["wickets"].replace(0, np.nan), 0.0
        )
        seasons["wicket_frequency"] = seasons["wicket_frequency"].fillna(0.0).round(2)
        seasons = seasons.round(2)
        total_matches = int(seasons["matches"].sum())
        total_runs = int(seasons["runs"].sum())
        total_balls = int(seasons["balls_faced"].sum())
        total_dismissals = int(seasons["dismissals"].sum())
        total_boundaries = int(seasons["boundaries"].sum())
        total_wickets = int(seasons["wickets"].sum())
        total_bowling_runs = int(seasons["bowling_runs"].sum())
        total_balls_bowled = int(seasons["balls_bowled"].sum())
        total_fantasy = round(float(seasons["fantasy_points"].sum()), 1)
        best_season = seasons.loc[seasons["runs"].idxmax()]
        profile = {
            "player": exact["player"].iloc[0],
            "seasons_played": int(len(seasons)),
            "matches": total_matches,
            "runs": total_runs,
            "boundaries": total_boundaries,
            "average": round(total_runs / total_dismissals, 2) if total_dismissals else round(float(total_runs), 2),
            "strike_rate": round(total_runs * 100 / total_balls, 2) if total_balls else 0.0,
            "wickets": total_wickets,
            "economy": round(total_bowling_runs / (total_balls_bowled / 6), 2) if total_balls_bowled else 0.0,
            "fantasy_points": total_fantasy,
            "best_season": {
                "season": int(best_season["season"]),
                "runs": int(best_season["runs"]),
                "wickets": int(best_season["wickets"]),
                "average": round(float(best_season["average"]), 2),
                "strike_rate": round(float(best_season["strike_rate"]), 2),
                "economy": round(float(best_season["economy"]), 2),
            },
        }
        seasons_cols = [c for c in ["season", "matches", "runs", "average", "strike_rate", "wickets", "economy", "fantasy_points"] if c in seasons.columns]
        return {
            "matched": True,
            "profile": profile,
            "seasons": {"columns": seasons_cols, "rows": seasons[seasons_cols].values.tolist()},
        }

    def toss_impact(self) -> dict:
        rows = self.data["team_results"]
        grouped = (
            rows.groupby("team", as_index=False)
            .agg(
                matches=("match_id", "nunique"),
                toss_wins=("toss_won", "sum"),
                wins=("won", "sum"),
            )
        )
        toss_won_matches = rows[rows["toss_won"] == 1].groupby("team").agg(toss_win_match_wins=("won", "sum"))
        grouped = grouped.merge(toss_won_matches, on="team", how="left").fillna(0)
        grouped["toss_win_rate"] = (grouped["toss_wins"] / grouped["matches"] * 100).round(1)
        grouped["win_rate"] = (grouped["wins"] / grouped["matches"] * 100).round(1)
        grouped["toss_win_match_win_rate"] = (grouped["toss_win_match_wins"] / grouped["toss_wins"].replace(0, np.nan) * 100).round(1)
        grouped = grouped.sort_values("toss_win_match_win_rate", ascending=False)
        table = grouped[["team", "matches", "toss_win_rate", "win_rate", "toss_win_match_win_rate"]]
        overall_toss = rows["toss_won"].mean() * 100
        toss_won_rows = rows[rows["toss_won"] == 1]
        overall_toss_to_win = toss_won_rows["won"].mean() * 100
        return {
            "columns": list(table.columns),
            "rows": table.values.tolist(),
            "overall_toss_win_rate": round(overall_toss, 1),
            "overall_toss_to_match_win": round(overall_toss_to_win, 1),
        }

    def inplay_tracker(
        self,
        chasing_team: str,
        defending_team: str,
        venue: str,
        target: int,
        current_score: int,
        balls_remaining: int,
        wickets_remaining: int,
        series: list[dict] | None = None,
    ) -> dict:
        balls_bowled = 120 - balls_remaining
        over_done = int(balls_bowled / 6)
        wickets_lost_now = 10 - wickets_remaining
        if series:
            states = [
                {
                    "over": int(s["over"]),
                    "score": int(s["score"]),
                    "wickets": int(s.get("wickets", s.get("wickets_lost", 0))),
                }
                for s in series
            ]
        else:
            states = [{"over": 0, "score": 0, "wickets": 0}]
            for o in range(1, over_done + 1):
                states.append(
                    {
                        "over": o,
                        "score": int(round(current_score * o / over_done)) if over_done > 0 else 0,
                        "wickets": int(round(wickets_lost_now * o / over_done)) if over_done > 0 else 0,
                    }
                )
        points = []
        for st in states:
            balls_left = max(120 - 6 * st["over"], 0)
            runs_needed = max(target - st["score"], 0)
            feats = _live_features(
                chasing_team, defending_team, venue, target, st["score"], runs_needed, balls_left, st["wickets"]
            )
            prob = float(predict_win_probability(self.models["live"], pd.DataFrame([feats]))[0]) * 100
            points.append(
                {
                    "over": st["over"],
                    "score": st["score"],
                    "wickets_lost": st["wickets"],
                    "chase_prob": round(prob, 2),
                    "defend_prob": round(100 - prob, 2),
                    "required_run_rate": round(feats["required_run_rate"], 2),
                }
            )
        return {
            "chasing_team": chasing_team,
            "defending_team": defending_team,
            "venue": venue,
            "target": target,
            "source": "stub" if not series else "live",
            "note": (
                "Stub feed: over-by-over states linearly interpolated from the current score as a demo of the "
                "tracker. This is not real match data; pass a real feed via `series` to replace it."
                if not series
                else "Series supplied by the caller."
            ),
            "points": points,
            "latest": points[-1],
        }

    def simulate_match(
        self,
        team1: str,
        team2: str,
        venue: str,
        toss_winner: str,
        toss_decision: str,
        batting_first: str,
        season: int | None = None,
        n_sims: int = 2000,
    ) -> dict:
        data = self.data
        season = int(season) if season else self.current_season
        n_sims = max(100, min(int(n_sims), 5000))
        rng = np.random.default_rng(7)

        winner_payload = _winner_payload(
            data["team_results"], data["h2h"], data["player_matches"], venue, team1, team2, toss_winner, toss_decision, season
        )
        team1_prob = float(predict_win_probability(self.models["winner"], winner_payload)[0]) * 100
        team2_prob = 100 - team1_prob
        se = 100 * (team1_prob / 100 * team2_prob / 100 / n_sims) ** 0.5
        team1_ci_lo = max(0.0, team1_prob - 1.96 * se)
        team1_ci_hi = min(100.0, team1_prob + 1.96 * se)

        frame = data["score_prediction"].sort_values(["date", "match_id"])
        frame = frame[frame["season"] == season] if season in frame["season"].astype(int).values else frame
        bowling_first = team2 if batting_first == team1 else team1
        score_payload = _score_payload(frame, venue, batting_first, bowling_first, toss_winner, toss_decision)
        expected = float(self.models["score"].predict(score_payload)[0])
        score_row = data["metrics"].query("model == 'xgboost_regressor' and purpose == 'first_innings_score'")
        mae = float(score_row["mae"].fillna(12).iloc[0]) if not score_row.empty else 12.0

        samples = np.clip(np.round(rng.laplace(loc=expected, scale=mae, size=n_sims)), 60, 300).astype(int)
        p5, p25, p50, p75, p95 = [float(v) for v in np.percentile(samples, [5, 25, 50, 75, 95])]
        hist_counts, edges = np.histogram(samples, bins=range(60, 310, 10))
        histogram = [
            {"bucket": f"{int(edges[i])}-{int(edges[i + 1])}", "count": int(c), "pct": round(float(c) / n_sims * 100, 1)}
            for i, c in enumerate(hist_counts)
        ]

        chaser = bowling_first
        chase_rows = pd.DataFrame(
            [_live_features(chaser, batting_first, venue, int(t), 0, int(t), 120, 10) for t in samples]
        )
        chase_probs = predict_win_probability(self.models["live"], chase_rows) * 100
        chase_lo, chase_hi = [float(v) for v in np.percentile(chase_probs, [5, 95])]

        return {
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "season": season,
            "n_sims": n_sims,
            "winner": {
                "team1_prob": round(team1_prob, 2),
                "team2_prob": round(team2_prob, 2),
                "team1_ci_low": round(team1_ci_lo, 2),
                "team1_ci_high": round(team1_ci_hi, 2),
            },
            "score": {
                "expected": int(round(expected)),
                "mae_used": round(mae, 2),
                "p5": int(round(p5)),
                "p25": int(round(p25)),
                "p50": int(round(p50)),
                "p75": int(round(p75)),
                "p95": int(round(p95)),
                "histogram": histogram,
            },
            "chase": {
                "chaser": chaser,
                "defender": batting_first,
                "win_pct": round(float(chase_probs.mean()), 2),
                "win_p5": round(chase_lo, 2),
                "win_p95": round(chase_hi, 2),
            },
            "note": (
                "Scores are sampled from the model's predictive distribution (Laplace, scale = the model's real "
                "MAE on held-out matches), never from synthetic match data. Winner probabilities are the classifier "
                "point estimate with a binomial 95% CI from the simulation count."
            ),
        }

    def explain_prediction(
        self,
        team1: str,
        team2: str,
        venue: str,
        toss_winner: str,
        toss_decision: str,
        season: int | None = None,
    ) -> dict:
        data = self.data
        season = int(season) if season else self.current_season
        payload = _winner_payload(
            data["team_results"], data["h2h"], data["player_matches"], venue, team1, team2, toss_winner, toss_decision, season
        )
        base = float(predict_win_probability(self.models["winner"], payload)[0]) * 100
        ablations = [
            ("recent form", ["recent5_form_diff", "recent10_form_diff"], 0.0, "Form over the last 5 / 10 matches"),
            ("toss", ["toss_rate_diff", "toss_winner_is_team1"], 0.0, "Toss win-rate edge and who won the toss"),
            ("venue", ["venue_win_pct_diff"], 0.0, "Historical win percentage at this venue"),
            ("head-to-head", ["h2h_team1_win_rate"], 0.5, "Direct head-to-head record"),
            ("squad strength", ["squad_strength_diff"], 0.0, "Recent fantasy-point squad strength"),
            ("career & season", ["win_pct_diff", "season_win_pct_diff"], 0.0, "Career and season win rates"),
            ("scoring", ["avg_runs_diff", "avg_conceded_diff"], 0.0, "Average runs scored and conceded"),
        ]
        factors = []
        for name, keys, neutral, desc in ablations:
            mod = payload.copy()
            for key in keys:
                mod.loc[0, key] = neutral
            prob = float(predict_win_probability(self.models["winner"], mod)[0]) * 100
            contrib = round(base - prob, 2)
            factors.append(
                {
                    "factor": name,
                    "team1_contribution": contrib,
                    "favors": "team1" if contrib > 0.05 else ("team2" if contrib < -0.05 else "neutral"),
                    "description": desc,
                }
            )
        factors.sort(key=lambda f: abs(f["team1_contribution"]), reverse=True)
        return {
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "season": season,
            "base_prob": round(base, 2),
            "note": (
                "Factor contributions are one-at-a-time ablations: the change in Team1's win probability when that "
                "factor's edge is removed, holding everything else fixed. Positive favours Team1, negative favours Team2."
            ),
            "factors": factors,
        }
