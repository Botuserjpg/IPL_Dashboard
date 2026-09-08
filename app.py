from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ipl_analytics.analytics import build_head_to_head
from ipl_analytics.data import CURRENT_IPL_TEAMS, MODELS_DIR, PROCESSED_DIR, load_processed_csv, load_raw_datasets
from ipl_analytics.models import load_classifier, load_model, load_model_metrics, predict_score_interval, predict_win_probability

st.set_page_config(
    page_title="IPL Match Prediction System",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Theme (custom CSS)
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
  --ipl-purple: #6d28d9;
  --ipl-purple-light: #8b5cf6;
  --ipl-gold: #f5b004;
  --ipl-gold-light: #ffd24a;
  --ipl-bg: #0b0f24;
  --ipl-card: #131737;
  --ipl-card-border: rgba(255,255,255,0.08);
  --ipl-text: #e6e9ff;
  --ipl-muted: #8b90b5;
}

.stApp {
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(109,40,217,0.28), transparent 60%),
    radial-gradient(900px 400px at -10% 0%, rgba(245,176,4,0.12), transparent 55%),
    linear-gradient(160deg, #0b0f24 0%, #0d1130 55%, #101536 100%);
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  color: var(--ipl-text);
}

#MainMenu, footer { visibility: hidden; }

.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }

h1, h2, h3, h4 { color: var(--ipl-text); font-weight: 800; letter-spacing: -0.02em; }

[data-testid="stHeader"] { background: transparent; }

[data-testid="stSidebar"] {
  background: rgba(19,23,55,0.92);
  border-right: 1px solid var(--ipl-card-border);
}
[data-testid="stSidebar"] * { color: var(--ipl-text); }

[data-baseweb="tab-list"] {
  gap: 8px;
  border-bottom: 1px solid var(--ipl-card-border);
}
[data-baseweb="tab"] {
  background: rgba(255,255,255,0.04);
  border-radius: 12px 12px 0 0;
  padding: 8px 20px;
  color: var(--ipl-muted) !important;
  font-weight: 600;
}
[data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, rgba(109,40,217,0.9), rgba(139,92,246,0.7));
  color: #fff !important;
}

[data-testid="stMetric"] {
  background: linear-gradient(150deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
  border: 1px solid var(--ipl-card-border);
  border-radius: 16px;
  padding: 18px 20px;
}
[data-testid="stMetricLabel"] { color: var(--ipl-muted); font-weight: 600; }
[data-testid="stMetricValue"] { color: var(--ipl-text); font-weight: 800; }

.stSelectbox > div > div, .stNumberInput > div > div, .stSlider > div > div {
  border-radius: 10px;
}
.stSelectbox [data-baseweb="select"] > div {
  background: rgba(255,255,255,0.06);
  border: 1px solid var(--ipl-card-border);
}
.stSelectbox [data-baseweb="select"] span, .stNumberInput input, .stSlider span {
  color: var(--ipl-text);
}

.stButton > button {
  background: linear-gradient(135deg, var(--ipl-purple), var(--ipl-purple-light));
  color: #fff;
  border: none;
  border-radius: 12px;
  padding: 10px 24px;
  font-weight: 700;
  transition: transform 0.15s ease;
}
.stButton > button:hover { transform: translateY(-1px); }

[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }

div[data-testid="stExpander"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--ipl-card-border);
  border-radius: 14px;
}

/* ---- Custom component styles ---- */
.hero {
  background: linear-gradient(120deg, rgba(109,40,217,0.95), rgba(139,92,246,0.75) 45%, rgba(245,176,4,0.75));
  border-radius: 22px;
  padding: 30px 36px;
  margin-bottom: 22px;
  color: #fff;
  box-shadow: 0 18px 50px -18px rgba(109,40,217,0.65);
}
.hero h1 { color: #fff; font-size: 2.3rem; margin: 0 0 6px 0; }
.hero p { color: rgba(255,255,255,0.92); font-size: 1.02rem; margin: 0; }
.hero .chip {
  display: inline-block;
  background: rgba(0,0,0,0.22);
  border: 1px solid rgba(255,255,255,0.3);
  border-radius: 999px;
  padding: 4px 14px;
  margin: 12px 8px 0 0;
  font-size: 0.82rem;
  font-weight: 600;
}

.kpi {
  background: linear-gradient(150deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
  border: 1px solid var(--ipl-card-border);
  border-radius: 16px;
  padding: 18px 20px;
  height: 100%;
}
.kpi .label { color: var(--ipl-muted); font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
.kpi .value { color: var(--ipl-text); font-size: 1.9rem; font-weight: 800; margin-top: 4px; }
.kpi .sub { color: var(--ipl-muted); font-size: 0.82rem; margin-top: 2px; }

.winner-card {
  background: linear-gradient(135deg, rgba(109,40,217,0.85), rgba(139,92,246,0.55));
  border-radius: 18px;
  padding: 26px 28px;
  color: #fff;
  text-align: center;
  box-shadow: 0 14px 40px -14px rgba(109,40,217,0.8);
}
.winner-card .win-label { font-size: 0.82rem; font-weight: 700; letter-spacing: 0.1em; opacity: 0.85; }
.winner-card .win-team { font-size: 1.7rem; font-weight: 900; margin: 8px 0 2px; }
.winner-card .win-sub { font-size: 0.9rem; opacity: 0.9; }

.score-card {
  background: linear-gradient(135deg, rgba(245,176,4,0.85), rgba(255,210,74,0.55));
  border-radius: 18px;
  padding: 26px 28px;
  color: #3a2c00;
  text-align: center;
  box-shadow: 0 14px 40px -14px rgba(245,176,4,0.7);
}
.score-card .score-label { font-size: 0.82rem; font-weight: 800; letter-spacing: 0.1em; opacity: 0.85; }
.score-card .score-range { font-size: 2rem; font-weight: 900; margin: 6px 0; }
.score-card .score-avg { font-size: 0.9rem; opacity: 0.85; }

.prob-row { display: flex; align-items: center; margin: 10px 0; gap: 12px; }
.prob-team { min-width: 200px; font-weight: 700; color: var(--ipl-text); font-size: 0.95rem; text-align: right; }
.prob-track { flex: 1; height: 26px; background: rgba(255,255,255,0.06); border-radius: 999px; overflow: hidden; position: relative; }
.prob-fill { height: 100%; border-radius: 999px; display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; font-weight: 800; font-size: 0.85rem; color: #fff; min-width: 44px; transition: width 0.5s ease; }
.prob-fill.p1 { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
.prob-fill.p2 { background: linear-gradient(90deg, #d97706, #f5b004); }
.prob-fill.p3 { background: linear-gradient(90deg, #0ea5e9, #38bdf8); }

.stat-note {
  background: rgba(245,176,4,0.12);
  border: 1px solid rgba(245,176,4,0.35);
  border-radius: 12px;
  padding: 10px 16px;
  font-size: 0.85rem;
  color: var(--ipl-gold-light);
}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Helpers / data loading
# --------------------------------------------------------------------------
def _sklearn_version() -> str:
    import sklearn

    return sklearn.__version__


def _version_stamp_path() -> Path:
    return MODELS_DIR / "sklearn_version.txt"


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
    stamp = _version_stamp_path()
    return (not stamp.exists()) or stamp.read_text().strip() != _sklearn_version()


def _run_training() -> None:
    import runpy

    runpy.run_path(str(PROJECT_ROOT / "scripts" / "train_models.py"), run_name="__main__")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _version_stamp_path().write_text(_sklearn_version())


def ensure_assets() -> None:
    if _needs_retrain():
        for stale in MODELS_DIR.glob("*.joblib"):
            stale.unlink(missing_ok=True)
        _run_training()


@st.cache_data
def load_data():
    ensure_assets()
    bundle = load_raw_datasets(strict_players=True)
    return {
        "bundle": bundle,
        "match_prediction": load_processed_csv("match_prediction.csv"),
        "score_prediction": load_processed_csv("score_prediction.csv"),
        "player_matches": load_processed_csv("player_match_features.csv"),
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


@st.cache_resource
def load_models():
    ensure_assets()
    return {
        "winner": load_classifier("xgboost"),
        "live": load_classifier("live_xgboost"),
        "score": load_model("xgboost_regressor"),
    }


# --------------------------------------------------------------------------
# Prediction payload builders
# --------------------------------------------------------------------------
def _team_squad_strength(player_matches: pd.DataFrame, team: str, fallback: float) -> float:
    pm = player_matches[player_matches["team"] == team].copy()
    if pm.empty:
        return fallback
    latest = pm.sort_values(["date", "match_id"]).groupby("player", as_index=False).last()
    if latest.empty:
        return fallback
    return float(latest["recent_fantasy_points"].nlargest(5).sum())


@st.cache_data
def _squad_fallback(player_matches: pd.DataFrame) -> float:
    pm = player_matches.sort_values(
        ["match_id", "team", "recent_fantasy_points"], ascending=[True, True, False]
    )
    top5 = pm.groupby(["match_id", "team"]).head(5).groupby("match_id")["recent_fantasy_points"].sum()
    return float(top5.mean())


def _team_profile(team_results: pd.DataFrame, player_matches: pd.DataFrame, team: str, venue: str, season: int, fallback: float) -> dict[str, float]:
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


def _winner_payload(team_results: pd.DataFrame, h2h: pd.DataFrame, player_matches: pd.DataFrame, venue: str, team1: str, team2: str, toss_winner: str, toss_decision: str, season: int) -> pd.DataFrame:
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


def _score_payload(score_prediction: pd.DataFrame, venue: str, batting_team: str, bowling_team: str, toss_winner: str, toss_decision: str) -> pd.DataFrame:
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
# Visual helpers
# --------------------------------------------------------------------------
def _prob_bars(rows: list[tuple[str, float, str]]) -> str:
    html = ""
    for label, prob, cls in rows:
        prob = max(0.0, min(100.0, prob))
        html += (
            f'<div class="prob-row"><div class="prob-team">{label}</div>'
            f'<div class="prob-track"><div class="prob-fill {cls}" style="width:{prob:.1f}%">{prob:.1f}%</div></div></div>'
        )
    return html


def _chart_theme(fig, height=380, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6e9ff", family="Inter, sans-serif"),
        title=dict(text=title, font=dict(size=18, color="#ffffff")) if title else None,
        height=height,
        margin=dict(l=20, r=20, t=60 if title else 30, b=30),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


def _win_gauge(team: str, prob: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"suffix": "%", "font": {"size": 34, "color": "#ffffff"}},
            title={"text": f"{team} win probability", "font": {"size": 16, "color": "#c4b5fd"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b90b5", "tickwidth": 1},
                "bar": {"color": "#8b5cf6"},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(217,119,6,0.55)"},
                    {"range": [50, 100], "color": "rgba(109,40,217,0.75)"},
                ],
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=30, r=30, t=40, b=10),
    )
    return fig


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------
try:
    data = load_data()
    models = load_models()
except Exception as exc:
    st.title("IPL Match Prediction System")
    st.error(str(exc))
    report_path = PROJECT_ROOT / "reports" / "data_quality_report.csv"
    if report_path.exists():
        st.dataframe(pd.read_csv(report_path), width="stretch", hide_index=True)
    st.stop()


bundle = data["bundle"]
summary = data["summary"]
teams = sorted([team for team in set(bundle.matches["team1"]).union(bundle.matches["team2"]) if team in CURRENT_IPL_TEAMS])
venues = sorted(bundle.matches["venue"].dropna().unique())
seasons = sorted(data["player_dashboard"]["season"].astype(int).unique())
current_season = int(seasons[-1])

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 🏏 IPL Forecasting")
    st.markdown(
        "Historical ball-by-ball analytics and machine-learning match prediction, "
        "trained exclusively on verified IPL match data."
    )
    st.markdown("---")
    st.markdown("**Data snapshot**")
    st.markdown(
        f"- **Seasons:** {summary['seasons']}\n"
        f"- **Matches:** {summary['matches']:,}\n"
        f"- **Deliveries:** {summary['deliveries']:,}\n"
        f"- **Winner model:** `{summary.get('best_match_model','xgboost')}`\n"
        f"- **Score model:** `{summary.get('best_score_model','xgboost_regressor')}`"
    )
    st.markdown("---")
    st.caption("Source: `data/raw/matches.csv` + `data/raw/deliveries.csv`")

# ---------------- Hero ----------------
st.markdown(
    f"""
    <div class="hero">
      <h1>IPL Match Prediction System</h1>
      <p>Analytics-driven pre-match and live win probabilities, first-innings score forecasts and player insights.</p>
      <span class="chip">🏆 {summary['seasons']} seasons</span>
      <span class="chip">📊 {summary['matches']:,} matches</span>
      <span class="chip">🏏 {summary['deliveries']:,} deliveries</span>
      <span class="chip">🤖 XGBoost · LightGBM · CatBoost · Random Forest</span>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.markdown(
    f'<div class="kpi"><div class="label">Seasons Covered</div><div class="value">{summary["seasons"]}</div>'
    f'<div class="sub">2008 → {current_season}</div></div>',
    unsafe_allow_html=True,
)
k2.markdown(
    f'<div class="kpi"><div class="label">Matches Analysed</div><div class="value">{summary["matches"]:,}</div>'
    f'<div class="sub">Every completed IPL fixture</div></div>',
    unsafe_allow_html=True,
)
k3.markdown(
    f'<div class="kpi"><div class="label">Ball-by-Ball Records</div><div class="value">{summary["deliveries"]:,}</div>'
    f'<div class="sub">Full delivery-level detail</div></div>',
    unsafe_allow_html=True,
)
metrics_w = data["metrics"]
xgb_w_row = metrics_w[metrics_w["model"] == "xgboost"].iloc[0] if "xgboost" in metrics_w["model"].values else None
k4.markdown(
    f'<div class="kpi"><div class="label">Winner Model (Chrono AUC)</div><div class="value">'
    f'{xgb_w_row["chrono_roc_auc"]:.2f}</div><div class="sub">Train-on-past · test-on-future</div></div>'
    if xgb_w_row is not None and pd.notna(xgb_w_row["chrono_roc_auc"])
    else f'<div class="kpi"><div class="label">Models</div><div class="value">5</div><div class="sub">Ensemble ready</div></div>',
    unsafe_allow_html=True,
)

st.markdown("---")

tab_match, tab_team, tab_player, tab_venue, tab_models = st.tabs(
    ["🎯 Match Prediction", "⚔️ Team Comparison", "🏏 Player Analytics", "🏟️ Venue Analytics", "🧠 Model Insights"]
)

# ===================== MATCH PREDICTION =====================
with tab_match:
    st.subheader("Pre-Match Prediction")
    left, right = st.columns([1, 1])
    with left:
        team1 = st.selectbox("Team 1", teams, index=teams.index("Mumbai Indians") if "Mumbai Indians" in teams else 0)
        team2_options = [team for team in teams if team != team1]
        team2 = st.selectbox("Team 2", team2_options, index=team2_options.index("Chennai Super Kings") if "Chennai Super Kings" in team2_options else 0)
        venue = st.selectbox("Venue", venues, index=0)
    with right:
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
        toss_decision = st.selectbox("Toss Decision", ["field", "bat"])
        batting_first = st.selectbox("First Innings Batting Team", [team1, team2])
        bowling_first = team2 if batting_first == team1 else team1

    payload = _winner_payload(data["team_results"], data["h2h"], data["player_matches"], venue, team1, team2, toss_winner, toss_decision, current_season)
    team1_prob = float(predict_win_probability(models["winner"], payload)[0]) * 100
    team2_prob = 100 - team1_prob
    winner = team1 if team1_prob >= team2_prob else team2

    score_payload = _score_payload(data["score_prediction"], venue, batting_first, bowling_first, toss_winner, toss_decision)
    score_metrics_row = data["metrics"].query("model == 'xgboost_regressor' and purpose == 'first_innings_score'")
    residual = float(score_metrics_row["mae"].fillna(12).iloc[0]) if not score_metrics_row.empty else 12.0
    score = predict_score_interval(models["score"], score_payload, residual_mae=residual)

    wcol, scol = st.columns([1, 1])
    with wcol:
        st.markdown(
            f'<div class="winner-card">'
            f'<div class="win-label">PREDICTED WINNER</div>'
            f'<div class="win-team">{winner}</div>'
            f'<div class="win-sub">{max(team1_prob, team2_prob):.1f}% model confidence</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with scol:
        st.markdown(
            f'<div class="score-card">'
            f'<div class="score-label">EXPECTED FIRST-INNINGS SCORE</div>'
            f'<div class="score-range">{score["low_score"]} – {score["high_score"]}</div>'
            f'<div class="score-avg">point estimate ≈ {score["expected_score"]} · {batting_first} batting first</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("##### Win Probability Breakdown")
    st.markdown(_prob_bars([(team1, team1_prob, "p1"), (team2, team2_prob, "p2")]), unsafe_allow_html=True)

    pcol, hcol = st.columns([1, 1])
    with pcol:
        st.plotly_chart(_win_gauge(team1, team1_prob), width="stretch")
    with hcol:
        h2h_payload = data["h2h"][
            ((data["h2h"]["team"] == team1) & (data["h2h"]["opponent"] == team2))
            | ((data["h2h"]["team"] == team2) & (data["h2h"]["opponent"] == team1))
        ]
        st.subheader("Head-to-Head")
        if not h2h_payload.empty:
            st.dataframe(
                h2h_payload[["team", "opponent", "matches", "wins", "losses", "win_rate", "avg_run_margin"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No direct head-to-head history found for this pairing.")

    st.subheader("Players to Watch")
    top_performers = _top_performers(data["player_matches"], [team1, team2], venue)
    if not top_performers.empty:
        st.dataframe(top_performers, width="stretch", hide_index=True)
    else:
        st.info("No recent player form data available for these teams.")

    st.markdown(
        '<div class="stat-note">💡 Pre-match T20 outcomes are genuinely near-coin-flip; '
        "the model is trained with chronological validation so confidence bands stay honest.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Live Chase Win Predictor")
    live_left, live_right = st.columns([1, 1])
    with live_left:
        chasing_team = st.selectbox("Chasing Team", [team1, team2], key="live_chasing_team")
        defending_team = team2 if chasing_team == team1 else team1
        target = st.number_input("Target", min_value=1, max_value=300, value=180, step=1)
        runs_needed = st.number_input("Runs Needed", min_value=0, max_value=300, value=60, step=1)
    with live_right:
        balls_remaining = st.slider("Balls Remaining", min_value=0, max_value=120, value=36, step=1)
        wickets_remaining = st.slider("Wickets Remaining", min_value=0, max_value=10, value=6, step=1)
        current_score = max(target - runs_needed, 0)
        overs_remaining = balls_remaining / 6
        balls_bowled = max(120 - balls_remaining, 1)
        current_run_rate = current_score / (balls_bowled / 6)
        required_run_rate = runs_needed / overs_remaining if overs_remaining > 0 else 99.0

    live_payload = pd.DataFrame(
        [
            {
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
        ]
    )
    chase_prob = float(predict_win_probability(models["live"], live_payload)[0]) * 100
    live_cols = st.columns(4)
    live_cols[0].metric(f"{chasing_team} Chase Win", f"{chase_prob:.1f}%")
    live_cols[1].metric(f"{defending_team} Defend Win", f"{100 - chase_prob:.1f}%")
    live_cols[2].metric("Current Score", f"{current_score}/{10 - wickets_remaining}")
    live_cols[3].metric("Required RR", f"{required_run_rate:.2f}")
    st.markdown(
        _prob_bars(
            [
                (f"{chasing_team} · chase", chase_prob, "p1"),
                (f"{defending_team} · defend", 100 - chase_prob, "p2"),
            ]
        ),
        unsafe_allow_html=True,
    )

# ===================== TEAM COMPARISON =====================
with tab_team:
    a, b = st.columns(2)
    team_a = a.selectbox("Team A", teams, index=0, key="team_a")
    team_b = b.selectbox("Team B", [team for team in teams if team != team_a], index=0, key="team_b")
    h2h_summary, h2h_top = build_head_to_head(bundle.matches, bundle.deliveries, team_a, team_b)

    if h2h_summary.empty:
        st.warning("These two teams have no head-to-head record in the dataset.")
    else:
        hcol1, hcol2 = st.columns([1, 1])
        with hcol1:
            st.subheader("Head-to-Head Record")
            st.dataframe(h2h_summary, width="stretch", hide_index=True)
        with hcol2:
            st.subheader("Win Distribution")
            fig = px.bar(
                h2h_summary,
                x="winner",
                y="wins",
                color="winner",
                color_discrete_sequence=["#8b5cf6", "#f5b004", "#38bdf8"],
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(_chart_theme(fig, height=340), width="stretch")

    if not h2h_top.empty:
        st.subheader("Head-to-Head Top Performers")
        st.dataframe(h2h_top, width="stretch", hide_index=True)

    st.subheader("Overall Team Records")
    trows = data["team_results"]
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
    st.dataframe(
        overall.sort_values("win_rate", ascending=False).round(1),
        width="stretch",
        hide_index=True,
    )

# ===================== PLAYER ANALYTICS =====================
with tab_player:
    season = st.selectbox("Season", seasons, index=len(seasons) - 1, key="player_season")
    player_view = data["player_dashboard"][data["player_dashboard"]["season"].astype(int) == int(season)]

    st.subheader(f"Top Fantasy Scorers · {season}")
    st.dataframe(
        player_view.sort_values("fantasy_points", ascending=False).head(20)[
            ["player", "batting_runs", "strike_rate", "batting_average", "bowling_wickets", "economy", "fantasy_points"]
        ].round(2),
        width="stretch",
        hide_index=True,
    )

    p1c, p2c = st.columns(2)
    with p1c:
        st.subheader("Best Batsmen")
        batting_cols = [c for c in ["player", "batting_runs", "batting_average", "strike_rate"] if c in player_view.columns]
        if batting_cols:
            st.dataframe(
                player_view.sort_values("batting_runs", ascending=False).head(15)[batting_cols].round(2),
                width="stretch",
                hide_index=True,
            )
    with p2c:
        st.subheader("Best Bowlers")
        bowl_cols = [c for c in ["player", "bowling_wickets", "economy"] if c in player_view.columns]
        if bowl_cols:
            st.dataframe(
                player_view.sort_values("bowling_wickets", ascending=False).head(15)[bowl_cols].round(2),
                width="stretch",
                hide_index=True,
            )

    st.subheader("Strike Rate vs Batting Average")
    scatter_cols = ["player", "strike_rate", "batting_average", "batting_runs", "bowling_wickets"]
    if all(col in player_view.columns for col in scatter_cols):
        scatter = px.scatter(
            player_view,
            x="strike_rate",
            y="batting_average",
            size="batting_runs",
            color="bowling_wickets",
            hover_name="player",
            color_continuous_scale=["#f5b004", "#8b5cf6", "#22d3ee"],
        )
        st.plotly_chart(_chart_theme(scatter, height=480, title=f"Season {season} · batting profile"), width="stretch")

# ===================== VENUE ANALYTICS =====================
with tab_venue:
    st.subheader("Venue Performance Heatmap")
    heatmap_metrics = ["avg_first_innings_score", "chase_win_rate", "defend_win_rate"]
    available_heatmap_metrics = [metric for metric in heatmap_metrics if metric in data["venue_stats"].columns]
    venue_heatmap = data["venue_stats"].copy().sort_values("matches", ascending=False).head(30)
    venue_heatmap_long = venue_heatmap.melt(
        id_vars=["venue", "matches"],
        value_vars=available_heatmap_metrics,
        var_name="metric",
        value_name="value",
    )
    heatmap_fig = px.density_heatmap(
        venue_heatmap_long,
        x="metric",
        y="venue",
        z="value",
        histfunc="avg",
        color_continuous_scale="Blues",
        labels={"value": "avg value", "metric": "metric", "venue": "venue"},
    )
    heatmap_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=620,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(heatmap_fig, width="stretch")

    vc1, vc2 = st.columns(2)
    with vc1:
        st.subheader("Highest Scoring Venues")
        vs = data["venue_stats"].sort_values("avg_first_innings_score", ascending=False).head(12)
        fig = px.bar(
            vs,
            x="avg_first_innings_score",
            y="venue",
            orientation="h",
            color="avg_first_innings_score",
            color_continuous_scale=["#38bdf8", "#8b5cf6", "#f5b004"],
        )
        st.plotly_chart(_chart_theme(fig, height=460), width="stretch")
    with vc2:
        st.subheader("Chase vs Defend Success")
        vs2 = data["venue_stats"].sort_values("matches", ascending=False).head(12)
        fig = px.bar(
            vs2.melt(id_vars="venue", value_vars=["chase_win_rate", "defend_win_rate"], var_name="kind", value_name="rate"),
            x="venue",
            y="rate",
            color="kind",
            barmode="group",
            color_discrete_sequence=["#8b5cf6", "#f5b004"],
        )
        st.plotly_chart(_chart_theme(fig, height=460), width="stretch")

    venue_table_columns = [
        "venue",
        "matches",
        "avg_first_innings_score",
        "avg_second_innings_score",
        "chase_win_rate",
        "defend_win_rate",
        "toss_win_match_win_rate",
    ]
    st.subheader("Venue Statistics")
    st.dataframe(
        data["venue_stats"][venue_table_columns].sort_values("matches", ascending=False),
        width="stretch",
        hide_index=True,
    )

# ===================== MODEL INSIGHTS =====================
with tab_models:
    st.subheader("Model Performance Dashboard")
    metrics_df = data["metrics"]

    for purpose, title in [
        ("match_winner", "Match Winner Classification"),
        ("live_chase_win_probability", "Live Chase Win Probability"),
        ("first_innings_score", "First-Innings Score Regression"),
        ("player_runs", "Player Performance Regression"),
    ]:
        subset = metrics_df[metrics_df["purpose"] == purpose]
        if subset.empty:
            continue
        st.markdown(f"##### {title}")
        display_cols = [c for c in subset.columns if c not in {"purpose", "tn", "fp", "fn", "tp", "cv_roc_auc_std", "cv_mae_std"}]
        st.dataframe(subset[display_cols].round(3), width="stretch", hide_index=True)

    st.markdown("##### Top Winning Factors (XGBoost)")
    fi = data["feature_importance"]
    xgb_fi = fi[fi["model"] == "xgboost"].sort_values("importance", ascending=False).head(15)
    if not xgb_fi.empty:
        xgb_fi["feature"] = xgb_fi["feature"].str.replace(r"^(venue|team1|team2|toss_decision)_", "category·", regex=True)
        fig = px.bar(
            xgb_fi,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#38bdf8", "#8b5cf6", "#f5b004"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=520,
            margin=dict(l=30, r=20, t=20, b=20),
            yaxis=dict(title=None),
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown(
        '<div class="stat-note">📈 Metrics use both a random 80/20 split (generalisation) and a '
        "chronological split (train on past, predict future). Chronological numbers are the honest "
        "estimate of real forward-looking performance.</div>",
        unsafe_allow_html=True,
    )
