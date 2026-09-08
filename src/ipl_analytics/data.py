from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


CURRENT_IPL_TEAMS = {
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Punjab Kings",
    "Gujarat Titans",
    "Lucknow Super Giants",
}

TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

# Legacy official sides can appear in historical verified IPL datasets. The dashboard
# defaults to current teams, but the loader keeps these records authentic.
LEGACY_IPL_TEAMS = {
    "Rising Pune Supergiants",
    "Deccan Chargers",
    "Pune Warriors",
    "Gujarat Lions",
    "Kochi Tuskers Kerala",
}

ALLOWED_IPL_TEAMS = CURRENT_IPL_TEAMS | LEGACY_IPL_TEAMS

MATCH_COLUMN_ALIASES = {
    "id": "match_id",
    "match_id": "match_id",
    "season": "season",
    "date": "date",
    "venue": "venue",
    "team1": "team1",
    "team2": "team2",
    "winner": "winner",
    "toss_winner": "toss_winner",
    "toss_decision": "toss_decision",
    "result": "result",
    "player_of_match": "player_of_match",
}

DELIVERY_COLUMN_ALIASES = {
    "match_id": "match_id",
    "id": "match_id",
    "inning": "inning",
    "innings": "inning",
    "over": "over",
    "ball": "ball",
    "batting_team": "batting_team",
    "bowling_team": "bowling_team",
    "batter": "batter",
    "batsman": "batter",
    "striker": "batter",
    "non_striker": "non_striker",
    "non-striker": "non_striker",
    "bowler": "bowler",
    "batsman_runs": "batsman_runs",
    "batter_runs": "batsman_runs",
    "total_runs": "total_runs",
    "extra_runs": "extra_runs",
    "is_wicket": "is_wicket",
    "player_dismissed": "player_dismissed",
    "dismissal_kind": "dismissal_kind",
    "extras_type": "extras_type",
}

FAKE_PLAYER_PATTERNS = [
    re.compile(r"\b[A-Z]{2,4}\s+Player\s+\d+\b", re.IGNORECASE),
    re.compile(r"\b(Player|Dummy|Sample|Test)\s*\d*\b", re.IGNORECASE),
    re.compile(r"\bUnknown\s+Player\b", re.IGNORECASE),
]


@dataclass(frozen=True)
class DatasetBundle:
    matches: pd.DataFrame
    deliveries: pd.DataFrame
    players: pd.DataFrame | None = None


def _normalize_columns(frame: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        key = column.strip().lower()
        if key in alias_map:
            renamed[column] = alias_map[key]
    return frame.rename(columns=renamed)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _normalize_season(value: object) -> int:
    text = str(value).strip()
    if "/" not in text:
        return int(float(text))
    start_text, end_text = text.split("/", 1)
    start_year = int(start_text)
    if len(end_text) == 2:
        century = start_year // 100
        end_year = century * 100 + int(end_text)
        if end_year < start_year:
            end_year += 100
        return end_year
    return int(end_text)


def clean_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    return text or None


def normalize_team_name(value: object) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    return TEAM_NAME_MAP.get(text, text)


def clean_player_name(value: object) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    return text.replace("\u00a0", " ")


def _looks_fake_player(name: object) -> bool:
    text = clean_player_name(name)
    if not text:
        return False
    return any(pattern.search(text) for pattern in FAKE_PLAYER_PATTERNS)


def raw_data_exists() -> bool:
    return (RAW_DIR / "matches.csv").exists() and (RAW_DIR / "deliveries.csv").exists()


def validate_real_ipl_data(matches: pd.DataFrame, deliveries: pd.DataFrame, strict_players: bool = True) -> pd.DataFrame:
    issues: list[dict[str, object]] = []

    for column in ["team1", "team2", "winner", "toss_winner"]:
        unknown = sorted(set(matches[column].dropna()) - ALLOWED_IPL_TEAMS)
        if unknown:
            issues.append(
                {
                    "severity": "error",
                    "area": "teams",
                    "column": column,
                    "message": f"Unrecognized IPL team names after normalization: {unknown[:10]}",
                    "count": len(unknown),
                }
            )

    for column in ["batting_team", "bowling_team"]:
        unknown = sorted(set(deliveries[column].dropna()) - ALLOWED_IPL_TEAMS)
        if unknown:
            issues.append(
                {
                    "severity": "error",
                    "area": "teams",
                    "column": column,
                    "message": f"Unrecognized IPL team names after normalization: {unknown[:10]}",
                    "count": len(unknown),
                }
            )

    player_columns = [col for col in ["batter", "non_striker", "bowler", "player_dismissed"] if col in deliveries.columns]
    fake_names = sorted(
        {
            name
            for column in player_columns
            for name in deliveries[column].dropna().unique()
            if _looks_fake_player(name)
        }
    )
    if fake_names:
        issues.append(
            {
                "severity": "error" if strict_players else "warning",
                "area": "players",
                "column": ",".join(player_columns),
                "message": f"Placeholder or fake player names detected: {fake_names[:15]}",
                "count": len(fake_names),
            }
        )

    if matches["match_id"].duplicated().any():
        issues.append(
            {
                "severity": "warning",
                "area": "matches",
                "column": "match_id",
                "message": "Duplicate match_id rows were removed during preprocessing.",
                "count": int(matches["match_id"].duplicated().sum()),
            }
        )

    report = pd.DataFrame(issues, columns=["severity", "area", "column", "message", "count"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORTS_DIR / "data_quality_report.csv", index=False)

    if not report.empty and (report["severity"] == "error").any():
        detail = "; ".join(report.loc[report["severity"] == "error", "message"].head(3).tolist())
        raise ValueError(
            "Raw IPL data failed authenticity validation. Replace data/raw/matches.csv and "
            f"data/raw/deliveries.csv with verified IPL ball-by-ball data. Details: {detail}"
        )
    return report


def load_players_if_available() -> pd.DataFrame | None:
    path = RAW_DIR / "players.csv"
    if not path.exists():
        return None
    players = pd.read_csv(path)
    players.columns = [col.strip().lower().replace(" ", "_") for col in players.columns]
    for column in players.select_dtypes(include="object").columns:
        players[column] = players[column].map(clean_text)
    return players.drop_duplicates()


def load_raw_datasets(strict_players: bool = True) -> DatasetBundle:
    matches = _normalize_columns(_load_csv(RAW_DIR / "matches.csv"), MATCH_COLUMN_ALIASES)
    deliveries = _normalize_columns(_load_csv(RAW_DIR / "deliveries.csv"), DELIVERY_COLUMN_ALIASES)

    required_matches = {"match_id", "season", "date", "venue", "team1", "team2", "winner"}
    required_deliveries = {
        "match_id",
        "inning",
        "over",
        "ball",
        "batting_team",
        "bowling_team",
        "batter",
        "bowler",
        "batsman_runs",
        "total_runs",
        "is_wicket",
    }
    missing_matches = required_matches - set(matches.columns)
    missing_deliveries = required_deliveries - set(deliveries.columns)
    if missing_matches:
        raise ValueError(f"matches.csv missing columns: {sorted(missing_matches)}")
    if missing_deliveries:
        raise ValueError(f"deliveries.csv missing columns: {sorted(missing_deliveries)}")

    matches = matches.drop_duplicates(subset=["match_id"]).copy()
    deliveries = deliveries.drop_duplicates().copy()

    matches["match_id"] = pd.to_numeric(matches["match_id"], errors="coerce").astype("Int64")
    deliveries["match_id"] = pd.to_numeric(deliveries["match_id"], errors="coerce").astype("Int64")
    matches = matches.dropna(subset=["match_id"]).copy()
    deliveries = deliveries.dropna(subset=["match_id"]).copy()
    matches["match_id"] = matches["match_id"].astype(int)
    deliveries["match_id"] = deliveries["match_id"].astype(int)

    matches["_source_season"] = matches["season"]
    matches["season"] = matches["season"].apply(_normalize_season)
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.dropna(subset=["date", "season", "team1", "team2", "venue"]).copy()
    date_year = matches["date"].dt.year
    slash_season = matches["_source_season"].astype(str).str.contains("/", regex=False)
    matches.loc[slash_season & (matches["season"] != date_year), "season"] = date_year
    matches = matches.drop(columns=["_source_season"])
    matches["date"] = matches["date"].dt.strftime("%Y-%m-%d")
    matches["venue"] = matches["venue"].map(clean_text)

    for column in ["team1", "team2", "winner", "toss_winner"]:
        if column in matches.columns:
            matches[column] = matches[column].map(normalize_team_name)
    for column in ["batting_team", "bowling_team"]:
        deliveries[column] = deliveries[column].map(normalize_team_name)
    for column in ["batter", "non_striker", "bowler", "player_dismissed"]:
        if column in deliveries.columns:
            deliveries[column] = deliveries[column].map(clean_player_name)

    deliveries["inning"] = pd.to_numeric(deliveries["inning"], errors="coerce").fillna(0).astype(int)
    deliveries["over"] = pd.to_numeric(deliveries["over"], errors="coerce").fillna(0).astype(int)
    deliveries["ball"] = pd.to_numeric(deliveries["ball"], errors="coerce").fillna(0).astype(int)
    deliveries["batsman_runs"] = pd.to_numeric(deliveries["batsman_runs"], errors="coerce").fillna(0).astype(int)
    deliveries["total_runs"] = pd.to_numeric(deliveries["total_runs"], errors="coerce").fillna(0).astype(int)
    deliveries["is_wicket"] = pd.to_numeric(deliveries["is_wicket"], errors="coerce").fillna(0).astype(int)
    deliveries["extra_runs"] = pd.to_numeric(deliveries.get("extra_runs", 0), errors="coerce").fillna(0).astype(int)

    if "non_striker" not in deliveries.columns:
        deliveries["non_striker"] = None
    if "player_dismissed" not in deliveries.columns:
        deliveries["player_dismissed"] = None
    if "dismissal_kind" not in deliveries.columns:
        deliveries["dismissal_kind"] = None
    if "extras_type" not in deliveries.columns:
        deliveries["extras_type"] = None
    if "toss_winner" not in matches.columns:
        matches["toss_winner"] = matches["team1"]
    if "toss_decision" not in matches.columns:
        matches["toss_decision"] = "field"
    if "result" not in matches.columns:
        matches["result"] = "normal"

    validate_real_ipl_data(matches, deliveries, strict_players=strict_players)
    return DatasetBundle(matches=matches, deliveries=deliveries, players=load_players_if_available())


def load_processed_csv(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_processed_csv(frame: pd.DataFrame, name: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    frame.to_csv(path, index=False)
    return path


def dataset_summary(bundle: DatasetBundle) -> Tuple[int, int, int]:
    seasons = bundle.matches["season"].nunique()
    matches = len(bundle.matches)
    deliveries = len(bundle.deliveries)
    return seasons, matches, deliveries
