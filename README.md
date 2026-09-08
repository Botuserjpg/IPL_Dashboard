# IPL Match Prediction System

Analytics-driven IPL forecasting built on verified historical ball-by-ball data. Provides pre-match win probabilities, live (in-play) chase win probabilities, first-innings score forecasts, and player/team/venue analytics through an interactive Streamlit dashboard and a FastAPI web app.

Trained exclusively on authentic IPL `matches.csv` and `deliveries.csv` (2008–2024). No synthetic or dummy data is used, and placeholder player names are rejected at validation.

## 🚀 Live Demo

> **Live demo link:** <!-- TODO: add your Streamlit Cloud URL here, e.g. https://<your-app>.streamlit.app once deployed -->

## ⚙️ Tech Stack

- **Frontend / Dashboard:** Streamlit, Plotly, HTML/CSS
- **ML:** XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression, KMeans clustering
- **Data:** pandas, NumPy, scikit-learn
- **Web app / API:** FastAPI, Uvicorn, Pydantic
- **Reporting:** ReportLab (PDF), Power BI (CSV exports), matplotlib / seaborn

## 🎯 Key Verified Results

Forward-looking (held-out / chronological) evaluation — train on past, predict future. No leakage.

| Task | Best model | Accuracy | ROC-AUC |
|---|---|---|---|
| **Live chase win probability** (held-out season 2024) | XGBoost | **81.4%** | **0.898** |
| Match winner (pre-match) | XGBoost | ~57% | ~0.59 |

- **XGBoost is the best-performing model** for the live chase task, verified on a strictly temporal holdout.
- Pre-match T20 winner prediction sits near the coin-flip noise floor and is reported honestly as such.
- First-innings score is forecast as an interval (cross-validated MAE ≈ 24 runs).
- Full evaluation log: `reports/win_probability_evaluation_summary.md` + `evaluation_results.csv` (byte-for-byte reproducible with fixed seed 42).

## 🏏 What It Does

- **Match Prediction** — pick teams, venue, toss → win probability, expected score interval, live chase win probability.
- **Live Chase Predictor** — in-play win probability from current score, wickets, and balls remaining.
- **Team Comparison** — head-to-head records, form, toss impact, squad strength.
- **Player Analytics** — season leaderboards, fantasy scorers, player clusters.
- **Venue Analytics** — average scores, chase/defend success, toss impact per ground.
- **Model Insights** — per-model metrics, chronological vs random-split, feature importance.

## 📁 Project Structure

```
app.py                  # Streamlit dashboard (main entry point)
server.py               # FastAPI web app + JSON API
main.py                 # Model training entry point
src/ipl_analytics/      # Core package (data, features, models, analytics, web_api)
scripts/                # Training, data prep, evaluation, exports
data/raw/               # matches.csv + deliveries.csv (verified IPL ball-by-ball)
data/processed/         # Feature-engineered datasets
models/                 # Trained .joblib artifacts
powerbi/                # Power BI ready CSVs
reports/                # Evaluation reports & PDFs
web/index.html          # FastAPI-served browser UI
requirements.txt        # Pinned dependencies
```

## 🖥️ Run Locally

```bash
pip install -r requirements.txt

# Streamlit dashboard
streamlit run app.py

# FastAPI web app (http://127.0.0.1:8000, API docs at /docs)
python server.py

# Retrain / regenerate models & processed data
python main.py
```

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. At https://share.streamlit.io, create a new app pointing at `app.py`.
3. **Secrets:** none are required. There are no hardcoded credentials or API keys in the codebase.

> **Note:** The first time the app runs on Streamlit Cloud it will retrain all models from `data/raw/` (see `app.py::ensure_assets`), which is slow and files-dependent. To avoid long cold starts, you can commit pre-trained `models/*.joblib` artifacts (currently gitignored) instead.

## 📊 Data Sources

- Kaggle: `patrickb1912/ipl-complete-dataset-20082020`
- Kaggle: `manasgarg/ipl`
- Cricsheet: Indian Premier League JSON download from `cricsheet.org/downloads`

Place verified CSVs in the project and prepare via:

```bash
python scripts/prepare_data.py --format kaggle --source path/to/folder
python scripts/prepare_data.py --format cricsheet-json --source path/to/ipl_json.zip
```

## 🔍 Evaluation & Reproducibility

```bash
python scripts/run_full_evaluation.py   # held-out season evaluation -> evaluation_results.csv
```

Seeds fixed (42), estimator hyperparameters pinned, and library versions recorded in `environment.txt` so results are fully reproducible.
