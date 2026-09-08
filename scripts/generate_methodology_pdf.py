"""Generate a PDF: evaluation methodology walkthrough + interview Q&A.

Run: python scripts/generate_methodology_pdf.py
Output: reports/IPL_WinProbability_Evaluation_Guide.pdf
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = PROJECT_ROOT / "reports" / "IPL_WinProbability_Evaluation_Guide.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Title"], fontSize=17, spaceAfter=4, textColor=colors.HexColor("#312e81"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#312e81"))
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=9.3, leading=13, alignment=TA_LEFT)
STEP = ParagraphStyle("STEP", parent=BODY, leftIndent=14, bulletIndent=2, spaceAfter=3)
CODE = ParagraphStyle("CODE", parent=BODY, fontName="Courier", fontSize=8.2, leading=11.5, backColor=colors.HexColor("#f3f4f6"), borderPadding=4, leftIndent=8, spaceAfter=4)
Q = ParagraphStyle("Q", parent=BODY, fontName="Helvetica-Bold", textColor=colors.HexColor("#1e1b4b"), spaceBefore=7, spaceAfter=2)
A = ParagraphStyle("A", parent=BODY, leftIndent=12, spaceAfter=3)

ACCENT = colors.HexColor("#c7d2fe")
HEADER_BG = colors.HexColor("#312e81")


def step(num: int, title: str, body: str) -> list:
    return [Paragraph(f"<bullet>&bull;</bullet><b>Step {num} - {title}</b>", STEP), Paragraph(body, STEP)]


def code(text: str) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>").replace(" ", "&nbsp;"), CODE)


def results_table() -> Table:
    data = [
        ["Model", "Accuracy", "ROC-AUC", "Precision", "Recall"],
        ["XGBoost", "0.8137", "0.8977", "0.8651", "0.7209"],
        ["Random Forest", "0.8077", "0.8935", "0.8516", "0.7217"],
        ["Logistic Regression", "0.7796", "0.8587", "0.8196", "0.6883"],
    ]
    t = Table(data, colWidths=[38 * mm, 24 * mm, 24 * mm, 26 * mm, 24 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ACCENT]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


story = []

# ---------- Cover / Part 1 ----------
story.append(Paragraph("IPL Win-Probability Models - Held-Out Evaluation Guide", H1))
story.append(Paragraph("Part 1: How the evaluation was performed (repeatable steps) &bull; Part 2: Likely interview questions with model answers. All numbers were actually computed from the project's ball-by-ball dataset and are byte-for-byte reproducible.", BODY))
story.append(Spacer(1, 6))
story.append(Paragraph("Results reference (held-out season 2024, 8,185 second-innings states)", H2))
story.append(results_table())
story.append(Spacer(1, 4))

story.append(Paragraph("Part 1 - Repeatable Methodology", H2))

steps = [
    ("Explore the repository first",
     "Locate three things before writing any evaluation code: the raw data (data/raw/matches.csv - 1 row per match; "
     "data/raw/deliveries.csv - 1 row per ball), the training code (src/ipl_analytics/models.py, scripts/train_models.py) "
     "and any saved artifacts (models/*.joblib). The training code is the source of truth for features, hyperparameters and target."),
    ("Identify the correct models and target column",
     "The project has two classifier families. The win-probability models are live_random_forest / live_xgboost / "
     "live_logistic_regression trained by train_live_chase_classifier on build_match_context_dataset output: one row per ball of "
     "every second innings (125,393 rows). Target column 'won' is already int 0/1 (1 = chasing team wins), so no label mapping is needed."),
    ("Choose a temporal (season-based) split, not a random one",
     "Rows are time-ordered; a random shuffle lets future matches influence training and inflates metrics. Use: test_season = "
     "int(context['season'].max()) = 2024; train = seasons < 2024 (117,208 rows); test = season == 2024 (8,185 rows). "
     "Persist the test set to data/test_set.csv for auditing."),
    ("Do NOT evaluate the saved artifacts directly - retrain instead",
     "The saved .joblib models were fitted on ALL seasons including 2024 (random split inside _classify_score). Scoring them on "
     "season 2024 means many test rows were seen during training = leakage. Fix: rebuild pipelines with identical recipes and seeds, "
     "fit only on seasons < 2024. This mirrors the repo's own _chrono_evaluate_classifier pattern."),
    ("Replicate preprocessing exactly",
     "From _preprocessor in src/ipl_analytics/models.py: categorical ['batting_team','bowling_team','venue'] -> "
     "SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore'); numeric (target, current_score, runs_needed, "
     "overs_remaining, balls_remaining, wickets_remaining, current_run_rate, required_run_rate, run_rate_gap, balls_per_wicket, pace_ratio) "
     "-> SimpleImputer(strategy='median'); StandardScaler ONLY for Logistic Regression (matches original pipeline; trees are scale-invariant)."),
    ("Pin reproducibility controls",
     "np.random.seed(42); random.seed(42); every estimator gets random_state=42 and n_jobs=1. XGBoost uses eval_metric='logloss'. "
     "(use_label_encoder=False is obsolete in xgboost >= 2.0.) Classification threshold fixed at 0.5 everywhere, matching repo code."),
    ("Fit, predict, score",
     "pipeline.fit(X_train, y_train); y_proba = pipeline.predict_proba(X_test)[:, 1]; y_pred = (y_proba >= 0.5).astype(int); then "
     "accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_proba), precision_score(y_test, y_pred, zero_division=0), "
     "recall_score(y_test, y_pred, zero_division=0). Round to 4 decimals into evaluation_results.csv."),
    ("Persist evidence and verify determinism",
     "Write evaluation_steps.txt (exact commands, calls, hyperparameters), environment.txt (python/pip freeze versions). "
     "Re-run the whole pipeline a second time and compare SHA-256 hashes of the outputs - they matched byte-for-byte."),
]
for i, (title, body) in enumerate(steps, 1):
    story.extend(step(i, title, body))

story.append(Paragraph("Exact reproduction", H2))
story.append(code(
    "# from project root\n"
    "python scripts/run_full_evaluation.py\n\n"
    "# key files produced\n"
    "evaluation_results.csv   # model,accuracy,roc_auc,precision,recall\n"
    "evaluation_steps.txt     # full audit log of every call made\n"
    "environment.txt          # pinned package versions\n"
    "data/test_set.csv        # held-out season 2024 rows\n"
    "scripts/preprocess_test.py          # feature lists + pipeline builder\n"
    "scripts/run_full_evaluation.py      # end-to-end orchestrator"))

story.append(PageBreak())

# ---------- Part 2: Q&A ----------
story.append(Paragraph("Part 2 - Interview Questions &amp; Answers", H1))
story.append(Paragraph("Every question below maps to a real decision made in this evaluation. Answering these fluently makes the resume numbers fully defensible.", BODY))

qa = [
    ("Why did you use a season-based split instead of train_test_split?",
     "The rows are time-ordered sports events. A random stratified split would place balls from the same match - even the same innings - "
     "on both sides of the split, so the model effectively memorises outcomes it should predict. Holding out the most recent season (2024) "
     "simulates real deployment: train on history, predict matches that happen later. It also exposes concept drift (rule changes, venue shifts, "
     "player turnover between eras)."),
    ("Your saved models existed - why did you retrain them?",
     "The artifacts had been fitted on the full dataset, which includes the held-out season. Evaluating them there would leak training rows into "
     "the test set and overstate performance. I rebuilt the pipelines with identical hyperparameters and seeds but fitted only on seasons < 2024 - "
     "exactly the refit-on-prior-data pattern the repo itself uses in _chrono_evaluate_classifier."),
    ("ROC-AUC ~0.90 seems very high for cricket. Is something wrong?",
     "No - and it's important to explain why. This is a LIVE win-probability task: at over 18 with 20 needed off 12, the outcome is largely determined "
     "by the state itself, so probability estimates naturally sharpen as the chase progresses. Suspicious signals would be near-perfect accuracy or "
     "identical train/test scores; instead we see realistic error rates (accuracy ~0.81) and a drop versus in-sample metrics. The pre-match winner "
     "models in the same repo sit near 0.50 AUC, which shows the pipeline isn't leaking."),
    ("Interpret precision and recall for this model.",
     "Precision 0.865: when XGBoost says the chasing team wins, it's right ~87% of the time - relevant if you act on positive predictions. "
     "Recall 0.721: it catches ~72% of all actual chase wins. The gap comes from the class mix and hard late-swing states; you'd tune the 0.5 "
     "threshold depending on whether false positives or false negatives cost more."),
    ("Why OneHotEncoder(handle_unknown='ignore')?",
     "Season 2024 can contain venues or team pairings never seen in training. With handle_unknown='error' the transform would crash at inference; "
     "'ignore' encodes an unseen category as all zeros, letting the numeric match-state features carry the prediction. Median imputation keeps "
     "occasional missing numerics from breaking the batch."),
    ("Why scale features only for Logistic Regression?",
     "That mirrors the original training pipeline exactly. Tree ensembles split on thresholds and are invariant to monotone scaling; logistic "
     "regression benefits from standardised features for stable convergence and fair regularisation. Reproducing preprocessing faithfully matters "
     "more than debating the choice - otherwise you're evaluating a different model than the repo ships."),
    ("How did you guarantee reproducibility?",
     "Seeds fixed for Python random, NumPy and each estimator (random_state=42), n_jobs=1 to avoid thread-order nondeterminism, pinned library "
     "versions recorded (Python 3.14.6, scikit-learn 1.9.0, xgboost 3.4.0, pandas 3.0.5), and proof by execution: two independent runs produced "
     "byte-identical result files (SHA-256 verified)."),
    ("What are the limitations / what would you improve?",
     "Three things: (1) calibration - probabilities should be checked with Brier score / reliability curves before any decision-making use, since "
     "XGB probabilities aren't guaranteed calibrated; (2) rolling-origin cross-validation across several seasons instead of a single holdout to "
     "quantify variance across eras; (3) explainability - SHAP values per state to show which factors drive swings, plus richer context features "
     "(pitch, toss, player availability)."),
    ("How is this different from predicting the match winner before the game?",
     "Different problem, different difficulty. Pre-match prediction only uses historical form and conditions (~coin-flip territory in this repo, "
     "AUC ~0.50-0.59). Live win-probability conditioning on the current state (score, wickets, balls left, required rate) is far more informative - "
     "which is exactly why broadcasters show these curves. Conflating the two on a resume is the classic mistake this project avoids."),
]
for q, a in qa:
    story.append(KeepTogether([Paragraph(f"Q: {q}", Q), Paragraph(f"A: {a}", A)]))

doc = SimpleDocTemplate(
    str(OUT_PDF),
    pagesize=A4,
    leftMargin=16 * mm,
    rightMargin=16 * mm,
    topMargin=14 * mm,
    bottomMargin=14 * mm,
    title="IPL Win-Probability Evaluation Guide",
    author="SRAJAN_IPL_FORECASTING",
)
doc.build(story)
print(f"PDF written: {OUT_PDF}")
