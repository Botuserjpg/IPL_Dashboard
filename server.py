from __future__ import annotations

import functools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import numpy as np

from ipl_analytics.web_api import WebApi

app = FastAPI(title="IPL Match Prediction System", version="1.0.0")


def _json_safe_value(value):
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_safe_value(value.tolist())
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return str(value)
    try:
        import pandas as pd

        if value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except Exception:
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def json_safe(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return _json_safe_value(func(*args, **kwargs))

    return wrapper


api = WebApi()

WEB_DIR = PROJECT_ROOT / "web"


class MatchRequest(BaseModel):
    team1: str
    team2: str
    venue: str
    toss_winner: str
    toss_decision: str = Field(pattern="^(bat|field)$")
    season: int | None = None


class ScoreRequest(BaseModel):
    batting_team: str
    bowling_team: str
    venue: str
    toss_winner: str
    toss_decision: str = Field(pattern="^(bat|field)$")
    season: int | None = None


class LiveRequest(BaseModel):
    chasing_team: str
    defending_team: str
    venue: str
    target: int = Field(ge=1, le=300)
    current_score: int = Field(ge=0, le=300)
    runs_needed: int = Field(ge=0, le=300)
    balls_remaining: int = Field(ge=0, le=120)
    wickets_remaining: int = Field(ge=0, le=10)


class TeamsRequest(BaseModel):
    team_a: str
    team_b: str


class WatchRequest(BaseModel):
    team1: str
    team2: str
    venue: str


class TrackerRequest(BaseModel):
    chasing_team: str
    defending_team: str
    venue: str
    target: int = Field(ge=1, le=300)
    current_score: int = Field(ge=0, le=300)
    balls_remaining: int = Field(ge=0, le=120)
    wickets_remaining: int = Field(ge=0, le=10)
    series: list[dict] | None = None


class SimulateRequest(BaseModel):
    team1: str
    team2: str
    venue: str
    toss_winner: str
    toss_decision: str = Field(pattern="^(bat|field)$")
    batting_first: str
    season: int | None = None
    n_sims: int = Field(2000, ge=100, le=5000)


@app.post("/api/predict/explain")
@json_safe
def explain_prediction(req: MatchRequest) -> dict:
    return api.explain_prediction(req.team1, req.team2, req.venue, req.toss_winner, req.toss_decision, req.season)


@app.post("/api/predict/tracker")
@json_safe
def inplay_tracker(req: TrackerRequest) -> dict:
    return api.inplay_tracker(
        req.chasing_team,
        req.defending_team,
        req.venue,
        req.target,
        req.current_score,
        req.balls_remaining,
        req.wickets_remaining,
        req.series,
    )


@app.post("/api/simulate/match")
@json_safe
def simulate_match(req: SimulateRequest) -> dict:
    return api.simulate_match(
        req.team1,
        req.team2,
        req.venue,
        req.toss_winner,
        req.toss_decision,
        req.batting_first,
        req.season,
        req.n_sims,
    )


@app.get("/api/meta")
@json_safe
def get_meta() -> dict:
    return api.meta()


@app.post("/api/predict/match")
@json_safe
def predict_match(req: MatchRequest) -> dict:
    return api.predict_match(req.team1, req.team2, req.venue, req.toss_winner, req.toss_decision, req.season)


@app.post("/api/predict/score")
@json_safe
def predict_score(req: ScoreRequest) -> dict:
    return api.predict_score(req.batting_team, req.bowling_team, req.venue, req.toss_winner, req.toss_decision, req.season)


@app.post("/api/predict/live")
@json_safe
def predict_live(req: LiveRequest) -> dict:
    return api.predict_live(
        req.chasing_team,
        req.defending_team,
        req.venue,
        req.target,
        req.current_score,
        req.runs_needed,
        req.balls_remaining,
        req.wickets_remaining,
    )


@app.post("/api/players/watch")
@json_safe
def players_to_watch(req: WatchRequest) -> dict:
    return api.players_to_watch(req.team1, req.team2, req.venue)


@app.post("/api/teams/h2h")
@json_safe
def teams_h2h(req: TeamsRequest) -> dict:
    return api.head_to_head(req.team_a, req.team_b)


@app.get("/api/teams/records")
@json_safe
def team_records() -> dict:
    return api.team_records()


@app.get("/api/players/season")
@json_safe
def player_season(season: int = Query(...)) -> dict:
    return api.player_season(season)


@app.get("/api/venues")
@json_safe
def venue_stats() -> dict:
    return api.venue_stats()


@app.get("/api/models")
@json_safe
def model_insights() -> dict:
    return api.model_insights()


@app.get("/api/seasons/history")
@json_safe
def season_history() -> dict:
    return api.season_history()


@app.get("/api/seasons/review")
@json_safe
def season_review(season: int = Query(...)) -> dict:
    return api.season_review(season)


@app.get("/api/teams/form")
@json_safe
def team_form(team: str = Query(...), limit: int = Query(10, ge=1, le=30)) -> dict:
    return api.team_form(team, limit)


@app.get("/api/players/profile")
@json_safe
def player_profile(name: str = Query(...)) -> dict:
    return api.player_profile(name)


@app.get("/api/toss")
@json_safe
def toss_impact() -> dict:
    return api.toss_impact()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
