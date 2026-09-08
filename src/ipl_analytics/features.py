from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _over_phase(over: pd.Series) -> pd.Series:
    cricket_over = over + 1 if over.min() == 0 else over
    return np.select(
        [cricket_over <= 6, cricket_over >= 17],
        ["powerplay", "death"],
        default="middle",
    )


def _prepare_match_innings(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    scores = (
        deliveries.groupby(["match_id", "inning", "batting_team", "bowling_team"], as_index=False)
        .agg(runs=("total_runs", "sum"), wickets=("is_wicket", "sum"))
        .sort_values(["match_id", "inning"])
    )
    first = scores[scores["inning"] == 1].rename(
        columns={
            "batting_team": "first_batting_team",
            "bowling_team": "first_bowling_team",
            "runs": "first_innings_score",
            "wickets": "first_innings_wickets",
        }
    )
    second = scores[scores["inning"] == 2].rename(
        columns={
            "batting_team": "second_batting_team",
            "bowling_team": "second_bowling_team",
            "runs": "second_innings_score",
            "wickets": "second_innings_wickets",
        }
    )
    out = matches.merge(
        first[
            [
                "match_id",
                "first_batting_team",
                "first_bowling_team",
                "first_innings_score",
                "first_innings_wickets",
            ]
        ],
        on="match_id",
        how="left",
    ).merge(
        second[
            [
                "match_id",
                "second_batting_team",
                "second_bowling_team",
                "second_innings_score",
                "second_innings_wickets",
            ]
        ],
        on="match_id",
        how="left",
    )
    out["chasing_team"] = out["second_batting_team"]
    out["defending_team"] = out["first_batting_team"]
    out["chase_success"] = (out["winner"] == out["chasing_team"]).astype(int)
    out["defend_success"] = (out["winner"] == out["defending_team"]).astype(int)
    return out


def build_team_match_results(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    base = _prepare_match_innings(matches, deliveries)
    rows = []
    for _, match in base.iterrows():
        for team, opponent in [(match["team1"], match["team2"]), (match["team2"], match["team1"])]:
            batting_first = int(match["first_batting_team"] == team)
            chasing = int(match["second_batting_team"] == team)
            team_runs = match["first_innings_score"] if batting_first else match["second_innings_score"]
            opp_runs = match["second_innings_score"] if batting_first else match["first_innings_score"]
            rows.append(
                {
                    "match_id": match["match_id"],
                    "season": match["season"],
                    "date": match["date"],
                    "venue": match["venue"],
                    "team": team,
                    "opponent": opponent,
                    "winner": match["winner"],
                    "won": int(match["winner"] == team),
                    "lost": int(pd.notna(match["winner"]) and match["winner"] == opponent),
                    "toss_won": int(match["toss_winner"] == team),
                    "toss_decision": match["toss_decision"],
                    "batting_first": batting_first,
                    "chasing": chasing,
                    "runs": team_runs,
                    "opponent_runs": opp_runs,
                    "run_margin": (team_runs or 0) - (opp_runs or 0),
                }
            )
    directed = pd.DataFrame(rows).sort_values(["date", "match_id", "team"])
    directed["matchup_key"] = directed["team"] + " vs " + directed["opponent"]
    return directed


def build_team_historical_features(team_results: pd.DataFrame) -> pd.DataFrame:
    frame = team_results.sort_values(["date", "match_id", "team"]).copy()
    global_avg_runs = frame["runs"].mean()
    frames = []
    for team, group in frame.groupby("team", sort=False):
        group = group.copy()
        prior_matches = group.groupby("team").cumcount()
        group["team_matches_before"] = prior_matches
        group["team_win_pct"] = group["won"].cumsum().shift(fill_value=0) / prior_matches.replace(0, np.nan)
        group["team_recent5_form"] = group["won"].shift().rolling(5, min_periods=1).mean()
        group["team_recent10_form"] = group["won"].shift().rolling(10, min_periods=1).mean()
        group["team_toss_win_rate"] = group["toss_won"].cumsum().shift(fill_value=0) / prior_matches.replace(0, np.nan)
        chase_mask = group["chasing"] == 1
        defend_mask = group["batting_first"] == 1
        group["team_chase_success_rate"] = (
            (group["won"] * chase_mask).cumsum().shift(fill_value=0)
            / chase_mask.cumsum().shift(fill_value=0).replace(0, np.nan)
        )
        group["team_defend_success_rate"] = (
            (group["won"] * defend_mask).cumsum().shift(fill_value=0)
            / defend_mask.cumsum().shift(fill_value=0).replace(0, np.nan)
        )
        group["team_venue_win_pct"] = group.groupby("venue")["won"].transform(
            lambda s: s.shift().expanding().mean()
        )
        group["team_season_win_pct"] = group.groupby("season")["won"].transform(
            lambda s: s.shift().expanding().mean()
        )
        group["team_avg_runs"] = group["runs"].shift().expanding().mean()
        group["team_avg_conceded"] = group["opponent_runs"].shift().expanding().mean()
        frames.append(group)
    out = pd.concat(frames, ignore_index=True)
    defaults = {
        "team_win_pct": 0.5,
        "team_recent5_form": 0.5,
        "team_recent10_form": 0.5,
        "team_toss_win_rate": 0.5,
        "team_chase_success_rate": 0.5,
        "team_defend_success_rate": 0.5,
        "team_venue_win_pct": 0.5,
        "team_season_win_pct": 0.5,
        "team_avg_runs": global_avg_runs,
        "team_avg_conceded": global_avg_runs,
    }
    return out.fillna(defaults)


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


def build_team_squad_strength(matches: pd.DataFrame, player_matches: pd.DataFrame) -> pd.DataFrame:
    """Sum of a team's top-5 players' recent fantasy points per match, then a
    temporal (prior-match) team average. `recent_fantasy_points` is already a
    rolling average over a player's prior matches, so no future leakage occurs.
    """
    if player_matches is None or player_matches.empty:
        return pd.DataFrame(columns=["match_id", "team", "squad_strength"])
    frame = player_matches[["match_id", "team", "date", "recent_fantasy_points"]].copy()
    frame = frame.sort_values(["match_id", "team", "recent_fantasy_points"], ascending=[True, True, False])
    top5 = (
        frame.groupby(["match_id", "team"])
        .head(5)
        .groupby(["match_id", "team"], as_index=False)["recent_fantasy_points"]
        .sum()
        .rename(columns={"recent_fantasy_points": "team_squad_points"})
    )
    meta = matches[["match_id", "date"]].drop_duplicates("match_id")
    top5 = top5.merge(meta, on="match_id", how="left").sort_values(["team", "date", "match_id"])
    top5["squad_strength"] = top5.groupby("team")["team_squad_points"].transform(
        lambda s: s.shift().expanding().mean()
    )
    return top5[["match_id", "team", "squad_strength"]]


def build_match_prediction_dataset(matches: pd.DataFrame, deliveries: pd.DataFrame, player_matches: pd.DataFrame | None = None) -> pd.DataFrame:
    team_results = build_team_historical_features(build_team_match_results(matches, deliveries))
    team1 = team_results.rename(columns={col: f"team1_{col}" for col in team_results.columns if col not in {"match_id"}})
    team2 = team_results.rename(columns={col: f"team2_{col}" for col in team_results.columns if col not in {"match_id"}})
    base = matches[
        ["match_id", "season", "date", "venue", "team1", "team2", "winner", "toss_winner", "toss_decision"]
    ].copy()
    dataset = base.merge(team1, left_on=["match_id", "team1"], right_on=["match_id", "team1_team"], how="left")
    dataset = dataset.merge(team2, left_on=["match_id", "team2"], right_on=["match_id", "team2_team"], how="left")
    dataset["team1_won"] = (dataset["winner"] == dataset["team1"]).astype(int)
    dataset["toss_winner_is_team1"] = (dataset["toss_winner"] == dataset["team1"]).astype(int)
    dataset["h2h_team1_win_rate"] = 0.5

    h2h_rates = []
    prior = []
    for _, row in dataset.sort_values(["date", "match_id"]).iterrows():
        key = tuple(sorted([row["team1"], row["team2"]]))
        played = [item for item in prior if item["key"] == key]
        if played:
            wins = sum(1 for item in played if item["winner"] == row["team1"])
            h2h_rates.append(wins / len(played))
        else:
            h2h_rates.append(0.5)
        prior.append({"key": key, "winner": row["winner"]})
    dataset.loc[dataset.sort_values(["date", "match_id"]).index, "h2h_team1_win_rate"] = h2h_rates

    squad = build_team_squad_strength(matches, player_matches)
    if not squad.empty:
        s1 = squad.rename(columns={"squad_strength": "team1_squad_strength", "team": "team1"})
        s2 = squad.rename(columns={"squad_strength": "team2_squad_strength", "team": "team2"})
        dataset = dataset.merge(s1[["match_id", "team1", "team1_squad_strength"]], on=["match_id", "team1"], how="left")
        dataset = dataset.merge(s2[["match_id", "team2", "team2_squad_strength"]], on=["match_id", "team2"], how="left")
    else:
        dataset["team1_squad_strength"] = 0.0
        dataset["team2_squad_strength"] = 0.0

    dataset["win_pct_diff"] = dataset["team1_team_win_pct"] - dataset["team2_team_win_pct"]
    dataset["recent5_form_diff"] = dataset["team1_team_recent5_form"] - dataset["team2_team_recent5_form"]
    dataset["recent10_form_diff"] = dataset["team1_team_recent10_form"] - dataset["team2_team_recent10_form"]
    dataset["toss_rate_diff"] = dataset["team1_team_toss_win_rate"] - dataset["team2_team_toss_win_rate"]
    dataset["chase_diff"] = dataset["team1_team_chase_success_rate"] - dataset["team2_team_chase_success_rate"]
    dataset["defend_diff"] = dataset["team1_team_defend_success_rate"] - dataset["team2_team_defend_success_rate"]
    dataset["venue_win_pct_diff"] = dataset["team1_team_venue_win_pct"] - dataset["team2_team_venue_win_pct"]
    dataset["season_win_pct_diff"] = dataset["team1_team_season_win_pct"] - dataset["team2_team_season_win_pct"]
    dataset["avg_runs_diff"] = dataset["team1_team_avg_runs"] - dataset["team2_team_avg_runs"]
    dataset["avg_conceded_diff"] = dataset["team1_team_avg_conceded"] - dataset["team2_team_avg_conceded"]
    dataset["squad_strength_diff"] = dataset["team1_squad_strength"] - dataset["team2_squad_strength"]

    numeric_cols = [
        "team1_team_win_pct",
        "team2_team_win_pct",
        "team1_team_recent5_form",
        "team2_team_recent5_form",
        "team1_team_toss_win_rate",
        "team2_team_toss_win_rate",
        "team1_team_chase_success_rate",
        "team2_team_chase_success_rate",
        "team1_team_defend_success_rate",
        "team2_team_defend_success_rate",
        "h2h_team1_win_rate",
        "toss_winner_is_team1",
    ]
    diff_cols = [
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
    ]
    for col in numeric_cols + diff_cols + ["team1_team_recent10_form", "team2_team_recent10_form", "team1_team_venue_win_pct", "team2_team_venue_win_pct", "team1_team_season_win_pct", "team2_team_season_win_pct", "team1_team_avg_runs", "team2_team_avg_runs", "team1_team_avg_conceded", "team2_team_avg_conceded", "team1_squad_strength", "team2_squad_strength"]:
        if col not in dataset.columns:
            dataset[col] = 0.5 if "win_pct" in col or "form" in col or "rate" in col else 0.0
    dataset[numeric_cols + diff_cols] = dataset[numeric_cols + diff_cols].fillna(0.5)
    return dataset[
        [
            "match_id",
            "season",
            "date",
            "venue",
            "team1",
            "team2",
            "toss_winner",
            "toss_decision",
            "team1_won",
        ]
        + numeric_cols
        + diff_cols
        + [
            "team1_team_recent10_form",
            "team2_team_recent10_form",
            "team1_team_venue_win_pct",
            "team2_team_venue_win_pct",
            "team1_team_season_win_pct",
            "team2_team_season_win_pct",
            "team1_team_avg_runs",
            "team2_team_avg_runs",
            "team1_team_avg_conceded",
            "team2_team_avg_conceded",
            "team1_squad_strength",
            "team2_squad_strength",
        ]
    ]


def build_match_context_dataset(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    match_meta = matches[
        ["match_id", "season", "date", "venue", "team1", "team2", "winner", "toss_winner", "toss_decision"]
    ].copy()
    inning1 = deliveries[deliveries["inning"] == 1].groupby("match_id", as_index=False)["total_runs"].sum()
    inning1 = inning1.rename(columns={"total_runs": "first_innings_score"})
    second = deliveries[deliveries["inning"] == 2].copy()
    if second.empty:
        raise ValueError("No second innings data available for win-probability training.")
    over_offset = 0 if deliveries["over"].min() == 0 else 1
    second["ball_number"] = (second["over"] - over_offset) * 6 + second["ball"].clip(upper=6)
    second = second.sort_values(["match_id", "ball_number"])
    second["cum_runs"] = second.groupby("match_id")["total_runs"].cumsum()
    second["cum_wickets"] = second.groupby("match_id")["is_wicket"].cumsum()
    second = second.merge(inning1, on="match_id", how="left").merge(match_meta, on="match_id", how="left")
    second["target"] = second["first_innings_score"] + 1
    second["balls_remaining"] = 120 - second["ball_number"]
    second["overs_remaining"] = second["balls_remaining"] / 6.0
    second["wickets_remaining"] = 10 - second["cum_wickets"]
    second["runs_needed"] = second["target"] - second["cum_runs"]
    second["current_run_rate"] = np.where(second["ball_number"] > 0, second["cum_runs"] / (second["ball_number"] / 6.0), 0.0)
    second["required_run_rate"] = np.where(second["balls_remaining"] > 0, second["runs_needed"] / (second["balls_remaining"] / 6.0), 0.0)
    second["won"] = (second["winner"] == second["batting_team"]).astype(int)
    second = second[(second["balls_remaining"] >= 0) & (second["runs_needed"] >= 0) & (second["wickets_remaining"] >= 0)].copy()
    second["run_rate_gap"] = second["required_run_rate"] - second["current_run_rate"]
    second["balls_per_wicket"] = np.where(second["wickets_remaining"] > 0, second["balls_remaining"] / second["wickets_remaining"], 0.0)
    second["pace_ratio"] = np.where(
        second["required_run_rate"] > 0,
        second["current_run_rate"] / second["required_run_rate"],
        1.0,
    )
    return second[
        [
            "match_id",
            "season",
            "date",
            "venue",
            "batting_team",
            "bowling_team",
            "target",
            "cum_runs",
            "runs_needed",
            "overs_remaining",
            "balls_remaining",
            "wickets_remaining",
            "current_run_rate",
            "required_run_rate",
            "run_rate_gap",
            "balls_per_wicket",
            "pace_ratio",
            "won",
        ]
    ].rename(columns={"cum_runs": "current_score"})


def build_score_prediction_dataset(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    base = _prepare_match_innings(matches, deliveries)
    phase = deliveries[deliveries["inning"] == 1].copy()
    phase["phase"] = _over_phase(phase["over"])
    phase_runs = phase.pivot_table(index="match_id", columns="phase", values="total_runs", aggfunc="sum", fill_value=0)
    phase_runs = phase_runs.rename(columns={col: f"{col}_runs" for col in phase_runs.columns}).reset_index()
    dataset = base.merge(phase_runs, on="match_id", how="left").fillna(0)
    dataset["toss_winner_bats_first"] = (dataset["toss_winner"] == dataset["first_batting_team"]).astype(int)
    dataset = dataset.sort_values(["date", "match_id"]).copy()
    dataset["batting_team_avg_first_score"] = dataset.groupby("first_batting_team")["first_innings_score"].transform(
        lambda s: s.shift().expanding().mean()
    )
    dataset["bowling_team_avg_first_conceded"] = dataset.groupby("first_bowling_team")["first_innings_score"].transform(
        lambda s: s.shift().expanding().mean()
    )
    dataset["venue_avg_first_score"] = dataset.groupby("venue")["first_innings_score"].transform(
        lambda s: s.shift().expanding().mean()
    )
    dataset["recent_batting_first_score"] = dataset.groupby("first_batting_team")["first_innings_score"].transform(
        lambda s: s.shift().rolling(5, min_periods=1).mean()
    )
    dataset["recent10_batting_first_score"] = dataset.groupby("first_batting_team")["first_innings_score"].transform(
        lambda s: s.shift().rolling(10, min_periods=1).mean()
    )
    dataset["recent10_bowling_conceded"] = dataset.groupby("first_bowling_team")["first_innings_score"].transform(
        lambda s: s.shift().rolling(10, min_periods=1).mean()
    )
    dataset["recent20_venue_score"] = dataset.groupby("venue")["first_innings_score"].transform(
        lambda s: s.shift().rolling(20, min_periods=1).mean()
    )
    dataset["venue_team_avg_score"] = dataset.groupby(["venue", "first_batting_team"])["first_innings_score"].transform(
        lambda s: s.shift().expanding().mean()
    )
    dataset["season_avg_first_score"] = dataset.groupby("season")["first_innings_score"].transform(
        lambda s: s.shift().expanding().mean()
    )
    dataset["era_recent_avg_score"] = dataset["first_innings_score"].shift().rolling(50, min_periods=1).mean()
    global_prior = dataset["first_innings_score"].expanding().mean().shift().fillna(dataset["first_innings_score"].mean())
    prior_columns = [
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
    for column in prior_columns:
        dataset[column] = dataset[column].fillna(global_prior)
    return dataset[
        [
            "match_id",
            "season",
            "date",
            "venue",
            "first_batting_team",
            "first_bowling_team",
            "toss_winner",
            "toss_decision",
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
            "powerplay_runs",
            "middle_runs",
            "death_runs",
            "first_innings_score",
        ]
    ]


def build_player_match_features(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    meta = matches[["match_id", "season", "date", "venue"]].copy()
    base = deliveries.merge(meta, on="match_id", how="left")
    base["phase"] = _over_phase(base["over"])
    legal_balls = ~base["extras_type"].isin(["wides"])

    batting = (
        base.groupby(["match_id", "season", "date", "venue", "batting_team", "bowling_team", "batter"], as_index=False)
        .agg(
            runs=("batsman_runs", "sum"),
            balls=("ball", "count"),
            dismissals=("is_wicket", "sum"),
            boundaries=("batsman_runs", lambda s: int((s.isin([4, 6])).sum())),
            dot_balls=("total_runs", lambda s: int((s == 0).sum())),
        )
        .rename(columns={"batter": "player", "batting_team": "team", "bowling_team": "opposition"})
    )
    batting["strike_rate"] = np.where(batting["balls"] > 0, batting["runs"] * 100 / batting["balls"], 0.0)
    batting["boundary_pct"] = np.where(batting["balls"] > 0, batting["boundaries"] / batting["balls"], 0.0)
    batting["dot_ball_pct"] = np.where(batting["balls"] > 0, batting["dot_balls"] / batting["balls"], 0.0)

    wickets = base[base["dismissal_kind"].fillna("").str.lower() != "run out"].copy()
    bowling = (
        wickets.groupby(["match_id", "season", "date", "venue", "bowling_team", "batting_team", "bowler"], as_index=False)
        .agg(
            balls_bowled=("ball", "count"),
            runs_conceded=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
        )
        .rename(columns={"bowler": "player", "bowling_team": "team", "batting_team": "opposition"})
    )
    bowling["economy"] = np.where(bowling["balls_bowled"] > 0, bowling["runs_conceded"] / (bowling["balls_bowled"] / 6), 0.0)
    combined = batting.merge(
        bowling,
        on=["match_id", "season", "date", "venue", "team", "opposition", "player"],
        how="outer",
    ).fillna(0)
    combined["fantasy_points"] = (
        combined["runs"]
        + combined["boundaries"]
        + combined["wickets"] * 25
        + (combined["dismissals"] == 0).astype(int) * 4
    )
    combined = combined.sort_values(["player", "date", "match_id"])
    for target in ["runs", "wickets", "strike_rate", "fantasy_points"]:
        combined[f"recent_{target}"] = combined.groupby("player")[target].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    combined["venue_runs_avg"] = combined.groupby(["player", "venue"])["runs"].transform(lambda s: s.shift().expanding().mean())
    combined["opposition_runs_avg"] = combined.groupby(["player", "opposition"])["runs"].transform(lambda s: s.shift().expanding().mean())
    fill_cols = [col for col in combined.columns if col.startswith("recent_")] + ["venue_runs_avg", "opposition_runs_avg"]
    combined[fill_cols] = combined[fill_cols].fillna(0.0)
    return combined


def build_player_season_features(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    player_matches = build_player_match_features(matches, deliveries)
    season = (
        player_matches.groupby(["season", "player"], as_index=False)
        .agg(
            batting_runs=("runs", "sum"),
            batting_balls=("balls", "sum"),
            dismissals=("dismissals", "sum"),
            batting_matches=("match_id", "nunique"),
            boundaries=("boundaries", "sum"),
            dot_balls=("dot_balls", "sum"),
            bowling_runs=("runs_conceded", "sum"),
            balls_bowled=("balls_bowled", "sum"),
            bowling_wickets=("wickets", "sum"),
            fantasy_points=("fantasy_points", "sum"),
        )
    )
    season["batting_average"] = season["batting_runs"] / season["dismissals"].replace(0, np.nan)
    season["batting_average"] = season["batting_average"].fillna(season["batting_runs"])
    season["strike_rate"] = np.where(season["batting_balls"] > 0, season["batting_runs"] * 100.0 / season["batting_balls"], 0.0)
    season["boundary_pct"] = np.where(season["batting_balls"] > 0, season["boundaries"] / season["batting_balls"], 0.0)
    season["dot_ball_pct"] = np.where(season["batting_balls"] > 0, season["dot_balls"] / season["batting_balls"], 0.0)
    season["overs_bowled"] = season["balls_bowled"] / 6.0
    season["economy"] = np.where(season["overs_bowled"] > 0, season["bowling_runs"] / season["overs_bowled"], 0.0)
    season["wicket_frequency"] = np.where(season["balls_bowled"] > 0, season["bowling_wickets"] / season["balls_bowled"], 0.0)
    return season.fillna(0)


def build_phase_stats(deliveries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = deliveries.copy()
    frame["phase"] = _over_phase(frame["over"])
    batting = (
        frame.groupby(["batting_team", "phase"], as_index=False)
        .agg(runs=("total_runs", "sum"), balls=("ball", "count"), boundaries=("batsman_runs", lambda s: int(s.isin([4, 6]).sum())), dots=("total_runs", lambda s: int((s == 0).sum())))
    )
    batting["run_rate"] = batting["runs"] / (batting["balls"] / 6).replace(0, np.nan)
    batting["boundary_pct"] = batting["boundaries"] / batting["balls"].replace(0, np.nan)
    batting["dot_ball_pct"] = batting["dots"] / batting["balls"].replace(0, np.nan)
    bowling = (
        frame.groupby(["bowling_team", "phase"], as_index=False)
        .agg(runs=("total_runs", "sum"), balls=("ball", "count"), wickets=("is_wicket", "sum"))
    )
    bowling["economy"] = bowling["runs"] / (bowling["balls"] / 6).replace(0, np.nan)
    bowling["wicket_frequency"] = bowling["wickets"] / bowling["balls"].replace(0, np.nan)
    return batting.round(4), bowling.round(4)


def build_player_cluster_dataset(player_season: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        player_season.groupby("player", as_index=False)
        .agg(
            seasons=("season", "nunique"),
            batting_runs=("batting_runs", "sum"),
            batting_balls=("batting_balls", "sum"),
            dismissals=("dismissals", "sum"),
            batting_matches=("batting_matches", "sum"),
            bowling_runs=("bowling_runs", "sum"),
            balls_bowled=("balls_bowled", "sum"),
            bowling_wickets=("bowling_wickets", "sum"),
            fantasy_points=("fantasy_points", "sum"),
        )
    )
    grouped["batting_average"] = grouped["batting_runs"] / grouped["dismissals"].replace(0, np.nan)
    grouped["batting_average"] = grouped["batting_average"].fillna(grouped["batting_runs"])
    grouped["strike_rate"] = np.where(grouped["batting_balls"] > 0, grouped["batting_runs"] * 100.0 / grouped["batting_balls"], 0.0)
    grouped["economy"] = np.where(grouped["balls_bowled"] > 0, grouped["bowling_runs"] / (grouped["balls_bowled"] / 6.0), 0.0)
    grouped["wickets_per_match"] = np.where(grouped["batting_matches"] > 0, grouped["bowling_wickets"] / grouped["batting_matches"], 0.0)
    grouped["runs_per_match"] = np.where(grouped["batting_matches"] > 0, grouped["batting_runs"] / grouped["batting_matches"], 0.0)
    return grouped.fillna(0)


def label_clusters(cluster_frame: pd.DataFrame, labels: Iterable[int]) -> pd.Series:
    cluster_frame = cluster_frame.copy()
    cluster_frame["cluster"] = list(labels)
    profile = cluster_frame.groupby("cluster").agg(
        strike_rate=("strike_rate", "mean"),
        batting_average=("batting_average", "mean"),
        economy=("economy", "mean"),
        wickets=("bowling_wickets", "mean"),
        runs=("batting_runs", "mean"),
    )
    ordered = {}
    for cluster_id, row in profile.iterrows():
        if row["wickets"] >= profile["wickets"].quantile(0.75) and row["economy"] <= profile["economy"].median():
            ordered[cluster_id] = "Economy Bowlers"
        elif row["strike_rate"] >= profile["strike_rate"].quantile(0.75):
            ordered[cluster_id] = "Power Hitters"
        elif row["batting_average"] >= profile["batting_average"].median() and row["strike_rate"] < profile["strike_rate"].median():
            ordered[cluster_id] = "Anchors"
        else:
            ordered[cluster_id] = "Finishers / All-Rounders"
    return cluster_frame["cluster"].map(ordered)
