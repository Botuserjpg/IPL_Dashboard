from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _copy_kaggle_csvs(source: Path) -> None:
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            csv_names = {Path(name).name.lower(): name for name in archive.namelist() if name.lower().endswith(".csv")}
            match_name, delivery_name = _find_kaggle_names(csv_names)
            with archive.open(match_name) as match_file, archive.open(delivery_name) as delivery_file:
                _write_raw(pd.read_csv(match_file), pd.read_csv(delivery_file))
        return

    csvs = {path.name.lower(): path for path in source.glob("*.csv")}
    match_name, delivery_name = _find_kaggle_names(csvs)
    _write_raw(_read_csv(csvs[Path(match_name).name.lower()]), _read_csv(csvs[Path(delivery_name).name.lower()]))


def _find_kaggle_names(csvs: dict[str, object]) -> tuple[str, str]:
    match_candidates = [
        "matches.csv",
        "ipl matches 2008-2020.csv",
        "ipl_matches_2008_2020.csv",
        "match_data.csv",
    ]
    delivery_candidates = [
        "deliveries.csv",
        "ipl ball-by-ball 2008-2020.csv",
        "ipl_ball_by_ball_2008_2020.csv",
        "ball_by_ball.csv",
    ]
    match_name = next((name for name in match_candidates if name in csvs), None)
    delivery_name = next((name for name in delivery_candidates if name in csvs), None)
    if match_name is None or delivery_name is None:
        names = ", ".join(sorted(csvs))
        raise FileNotFoundError(
            "Could not identify Kaggle matches/deliveries CSVs. "
            f"Found: {names}. Rename files to matches.csv and deliveries.csv or use --matches/--deliveries."
        )
    return str(csvs[match_name]), str(csvs[delivery_name])


def _normalize_outcome_winner(info: dict) -> str | None:
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    if winner:
        return winner
    return None


def _cricsheet_match_and_deliveries(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    info = data["info"]
    match_id = int(path.stem)
    dates = info.get("dates") or []
    date = dates[0] if dates else None
    teams = info.get("teams") or [None, None]
    toss = info.get("toss", {})
    match = {
        "match_id": match_id,
        "season": int(str(date)[:4]) if date else None,
        "date": date,
        "venue": info.get("venue"),
        "team1": teams[0],
        "team2": teams[1],
        "winner": _normalize_outcome_winner(info),
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "result": "normal" if _normalize_outcome_winner(info) else "no result",
    }
    rows: list[dict] = []
    for inning_no, innings in enumerate(data.get("innings", []), start=1):
        batting_team = innings.get("team")
        bowling_team = next((team for team in teams if team != batting_team), None)
        for over_block in innings.get("overs", []):
            over = int(over_block.get("over", 0)) + 1
            for ball_no, delivery in enumerate(over_block.get("deliveries", []), start=1):
                wickets = delivery.get("wickets") or []
                rows.append(
                    {
                        "match_id": match_id,
                        "inning": inning_no,
                        "over": over,
                        "ball": ball_no,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "batter": delivery.get("batter"),
                        "non_striker": delivery.get("non_striker"),
                        "bowler": delivery.get("bowler"),
                        "batsman_runs": delivery.get("runs", {}).get("batter", 0),
                        "extra_runs": delivery.get("runs", {}).get("extras", 0),
                        "total_runs": delivery.get("runs", {}).get("total", 0),
                        "is_wicket": int(bool(wickets)),
                        "player_dismissed": wickets[0].get("player_out") if wickets else None,
                        "dismissal_kind": wickets[0].get("kind") if wickets else None,
                        "extras_type": next(iter((delivery.get("extras") or {}).keys()), None),
                    }
                )
    return match, rows


def _convert_cricsheet_json(source: Path) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = Path(temp_dir)
        if source.is_file() and source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as archive:
                archive.extractall(source_dir)
        else:
            source_dir = source
        json_files = sorted(source_dir.rglob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No Cricsheet JSON files found in {source}")
        matches = []
        deliveries = []
        for json_file in json_files:
            match, rows = _cricsheet_match_and_deliveries(json_file)
            if match["team1"] and match["team2"]:
                matches.append(match)
                deliveries.extend(rows)
        _write_raw(pd.DataFrame(matches), pd.DataFrame(deliveries))


def _write_raw(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(RAW_DIR / "matches.csv", index=False)
    deliveries.to_csv(RAW_DIR / "deliveries.csv", index=False)
    print(f"Wrote {len(matches):,} matches to {RAW_DIR / 'matches.csv'}")
    print(f"Wrote {len(deliveries):,} deliveries to {RAW_DIR / 'deliveries.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare verified IPL data for the prediction system.")
    parser.add_argument("--source", type=Path, required=True, help="Folder or zip containing Kaggle CSVs or Cricsheet JSON.")
    parser.add_argument("--format", choices=["kaggle", "cricsheet-json"], required=True)
    parser.add_argument("--matches", type=Path, help="Explicit Kaggle matches CSV path.")
    parser.add_argument("--deliveries", type=Path, help="Explicit Kaggle deliveries CSV path.")
    args = parser.parse_args()

    if args.format == "kaggle":
        if args.matches and args.deliveries:
            _write_raw(_read_csv(args.matches), _read_csv(args.deliveries))
        else:
            _copy_kaggle_csvs(args.source)
    else:
        _convert_cricsheet_json(args.source)


if __name__ == "__main__":
    main()
