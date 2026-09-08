# IPL Live Win-Probability Models — Evaluation Methodology Summary

**Project:** SRAJAN_IPL_FORECASTING · **Evaluation date:** 2026-08-24 · **All metrics computed from real ball-by-ball data; fully reproducible (fixed seed 42).**

## Objective

Estimate the probability that the chasing team wins an IPL match at any point of the second innings, using live match-state features, and benchmark three classifiers under a strict held-out protocol.

## Data

| Item | Value |
|---|---|
| Source | Official-style IPL ball-by-ball dataset (`data/raw/deliveries.csv`, 260,920 deliveries; `data/raw/matches.csv`, 1,095 matches) |
| Coverage | Seasons 2008–2024 |
| Unit of analysis | One row per legal delivery of every second innings → 125,393 match-state observations |
| Target | `won` ∈ {0,1}: 1 = chasing team wins |
| Validation | Team-name whitelist, fake-player detection, duplicate checks all passed |

## Methodology (leakage-free by design)

1. **Temporal split:** train on seasons 2008–2023 (117,208 rows); test on the most recent season 2024 (8,185 rows). Simulates deployment: models never see future matches.
2. **No artifact reuse:** pre-trained artifacts had been fitted on all seasons including 2024; refitting identical pipelines on the training partition only avoids training-on-test contamination.
3. **Preprocessing:** categorical features (batting team, bowling team, venue) → mode imputation + one-hot encoding with `handle_unknown='ignore'`; numeric features → median imputation; standard scaling applied only where the original pipeline used it (Logistic Regression).
4. **Metrics** on class predictions use threshold 0.5; ROC-AUC uses predicted probabilities.

## Models & Held-Out Results (season 2024)

| Model | Key hyperparameters | Accuracy | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| XGBoost | 120 trees, depth 4, lr 0.05 | **0.8137** | **0.8977** | **0.8651** | 0.7209 |
| Random Forest | 100 trees, depth 9 | 0.8077 | 0.8935 | 0.8516 | **0.7217** |
| Logistic Regression | max_iter 1500, scaled | 0.7796 | 0.8587 | 0.8196 | 0.6883 |

## Reproducibility

Seeds fixed for Python `random`, NumPy, and every estimator (`random_state=42`). Environment pinned: Python 3.14.6, scikit-learn 1.9.0, xgboost 3.4.0, pandas 3.0.5. Two independent runs produced byte-identical outputs (SHA-256 verified).

Re-run end-to-end:

```
python scripts/run_full_evaluation.py
```

## Key Takeaways

- Gradient boosting (XGBoost) edges out Random Forest on both discrimination (AUC) and precision; linear baseline trails by ~0.04 AUC.
- High AUC is expected for late-innings states: score/wicket/run-rate information converges naturally as the chase nears its end.
- Temporal evaluation reports honest, deployment-realistic performance versus optimistic random-split estimates.
