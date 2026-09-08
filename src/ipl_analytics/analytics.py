from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_team_match_results


def build_player_dashboard_table(player_season: pd.DataFrame) -> pd.DataFrame:
    frame = player_season.copy()
    for column in ["batting_average", "strike_rate", "economy", "boundary_pct", "dot_ball_pct", "wicket_frequency"]:
        if column in frame.columns:
            frame[column] = frame[column].round(3)
    return frame.sort_values(["season", "batting_runs"], ascending=[True, False])


def build_venue_stats(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    scores = (
        deliveries.groupby(["match_id", "inning"], as_index=False)["total_runs"].sum()
        .pivot(index="match_id", columns="inning", values="total_runs")
        .reset_index()
        .rename(columns={1: "avg_first_innings_score_source", 2: "avg_second_innings_score_source"})
    )
    first_batting = deliveries[deliveries["inning"] == 1][["match_id", "batting_team"]].drop_duplicates()
    second_batting = deliveries[deliveries["inning"] == 2][["match_id", "batting_team"]].drop_duplicates()
    base = (
        matches[["match_id", "venue", "winner", "toss_winner", "toss_decision"]]
        .merge(scores, on="match_id", how="left")
        .merge(first_batting.rename(columns={"batting_team": "first_batting_team"}), on="match_id", how="left")
        .merge(second_batting.rename(columns={"batting_team": "second_batting_team"}), on="match_id", how="left")
    )
    base["chasing_won"] = (base["winner"] == base["second_batting_team"]).astype(int)
    base["defending_won"] = (base["winner"] == base["first_batting_team"]).astype(int)
    base["toss_winner_won"] = (base["winner"] == base["toss_winner"]).astype(int)
    base["field_first_won"] = ((base["toss_decision"] == "field") & (base["winner"] == base["toss_winner"])).astype(int)

    venue = (
        base.groupby("venue", as_index=False)
        .agg(
            matches=("match_id", "count"),
            avg_first_innings_score=("avg_first_innings_score_source", "mean"),
            avg_second_innings_score=("avg_second_innings_score_source", "mean"),
            chase_win_rate=("chasing_won", "mean"),
            defend_win_rate=("defending_won", "mean"),
            toss_win_match_win_rate=("toss_winner_won", "mean"),
            field_first_toss_success_rate=("field_first_won", "mean"),
        )
        .sort_values("matches", ascending=False)
    )
    for column in ["chase_win_rate", "defend_win_rate", "toss_win_match_win_rate", "field_first_toss_success_rate"]:
        venue[column] = (venue[column] * 100).round(2)
    for column in ["avg_first_innings_score", "avg_second_innings_score"]:
        venue[column] = venue[column].round(2)
    venue["pace_spin_note"] = "Requires verified bowling-style metadata in players.csv"
    return venue


def build_head_to_head_summary(team_results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        team_results.groupby(["team", "opponent"], as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            wins=("won", "sum"),
            losses=("lost", "sum"),
            avg_runs=("runs", "mean"),
            avg_opponent_runs=("opponent_runs", "mean"),
            avg_run_margin=("run_margin", "mean"),
        )
        .sort_values(["team", "wins"], ascending=[True, False])
    )
    summary["win_rate"] = (summary["wins"] / summary["matches"] * 100).round(2)
    summary["matchup_key"] = summary["team"] + " vs " + summary["opponent"]
    return summary.round(2)


def build_head_to_head_player_performance(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    match_teams = pd.concat(
        [
            matches[["match_id", "team1", "team2"]].rename(columns={"team1": "team", "team2": "opponent"}),
            matches[["match_id", "team1", "team2"]].rename(columns={"team2": "team", "team1": "opponent"}),
        ],
        ignore_index=True,
    )
    batting = (
        deliveries.merge(match_teams, left_on=["match_id", "batting_team"], right_on=["match_id", "team"], how="inner")
        .groupby(["team", "opponent", "batter"], as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            batting_runs=("batsman_runs", "sum"),
            balls_faced=("ball", "count"),
            dismissals=("is_wicket", "sum"),
        )
        .rename(columns={"batter": "player"})
    )
    batting["batting_average"] = batting["batting_runs"] / batting["dismissals"].replace(0, np.nan)
    batting["batting_average"] = batting["batting_average"].fillna(batting["batting_runs"])
    batting["strike_rate"] = np.where(batting["balls_faced"] > 0, batting["batting_runs"] * 100 / batting["balls_faced"], 0.0)

    bowling = (
        deliveries.merge(match_teams, left_on=["match_id", "bowling_team"], right_on=["match_id", "team"], how="inner")
        .groupby(["team", "opponent", "bowler"], as_index=False)
        .agg(
            bowling_matches=("match_id", "nunique"),
            bowling_wickets=("is_wicket", "sum"),
            bowling_runs=("total_runs", "sum"),
            balls_bowled=("ball", "count"),
        )
        .rename(columns={"bowler": "player"})
    )
    bowling["economy"] = np.where(bowling["balls_bowled"] > 0, bowling["bowling_runs"] / (bowling["balls_bowled"] / 6), 0.0)
    combined = batting.merge(bowling, on=["team", "opponent", "player"], how="outer").fillna(0)
    combined["matchup_key"] = combined["team"] + " vs " + combined["opponent"]
    combined["total_impact"] = combined["batting_runs"] + (combined["bowling_wickets"] * 25)
    for column in ["batting_average", "strike_rate", "economy", "total_impact"]:
        combined[column] = combined[column].round(2)
    return combined.sort_values(["team", "opponent", "total_impact"], ascending=[True, True, False])


def build_head_to_head(matches: pd.DataFrame, deliveries: pd.DataFrame, team_a: str, team_b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = matches[
        ((matches["team1"] == team_a) & (matches["team2"] == team_b))
        | ((matches["team1"] == team_b) & (matches["team2"] == team_a))
    ].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = subset.groupby("winner", as_index=False).agg(wins=("match_id", "count")).sort_values("wins", ascending=False)
    relevant = deliveries[deliveries["match_id"].isin(subset["match_id"])].copy()
    batting = (
        relevant.groupby("batter", as_index=False)
        .agg(runs=("batsman_runs", "sum"), balls=("ball", "count"))
        .sort_values("runs", ascending=False)
        .head(10)
        .rename(columns={"batter": "player"})
    )
    batting["strike_rate"] = np.where(batting["balls"] > 0, batting["runs"] * 100 / batting["balls"], 0.0).round(2)
    batting["role"] = "Batter"
    bowling = (
        relevant.groupby("bowler", as_index=False)
        .agg(wickets=("is_wicket", "sum"), runs_conceded=("total_runs", "sum"), balls=("ball", "count"))
        .sort_values(["wickets", "runs_conceded"], ascending=[False, True])
        .head(10)
        .rename(columns={"bowler": "player"})
    )
    bowling["economy"] = np.where(bowling["balls"] > 0, bowling["runs_conceded"] / (bowling["balls"] / 6), 0.0).round(2)
    bowling["role"] = "Bowler"
    return summary, pd.concat([batting, bowling], ignore_index=True, sort=False).fillna(0)


def leaderboard_tables(player_season: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    batting = (
        player_season.groupby("player", as_index=False)
        .agg(
            runs=("batting_runs", "sum"),
            average=("batting_average", "mean"),
            strike_rate=("strike_rate", "mean"),
            boundary_pct=("boundary_pct", "mean"),
            dot_ball_pct=("dot_ball_pct", "mean"),
        )
        .sort_values(["runs", "strike_rate"], ascending=[False, False])
        .head(25)
    )
    bowling = (
        player_season.groupby("player", as_index=False)
        .agg(
            wickets=("bowling_wickets", "sum"),
            economy=("economy", "mean"),
            wicket_frequency=("wicket_frequency", "mean"),
        )
        .query("wickets > 0")
        .sort_values(["wickets", "economy"], ascending=[False, True])
        .head(25)
    )
    return batting.round(3), bowling.round(3)
