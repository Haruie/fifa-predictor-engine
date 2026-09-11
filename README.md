# FIFA World Cup Match Predictor

A player-aware machine learning framework for predicting FIFA World Cup match outcomes,
built following the methodology in *"From Players to Champions: A Generalizable Machine
Learning Approach for Match Outcome Prediction with Insights from the FIFA World Cup"*
(Al-Bustami & Ghazal, IEEE eIT 2025), with data scope defined in the project proposal.

## Idea

Most existing models predict match winners using only team-level stats (rankings, past
results). National squads are rebuilt every World Cup, so team reputation alone is a weak
signal. This project fuses:

- **Player-level data** — technical, physical, mental, and goalkeeping attributes per
  player, aggregated into **year-specific team profiles** (rosters change every cycle).
- **Team-level data** — historical results, head-to-head record, tournament context.

...into a 5-model ensemble (Logistic Regression, Random Forest, XGBoost, AdaBoost, KNN)
combined via majority voting, benchmarked against a Weighted Win Ratio (WWR) baseline.

## Project structure

```
fifa-match-predictor/
├── config/
│   └── config.yaml          # all tunable settings in one place
├── data/
│   ├── raw/                 # untouched downloaded data (gitignored)
│   ├── external/            # reference data (rankings, tournament calendars)
│   └── processed/           # cleaned team-year profiles, model-ready match features
├── notebooks/                # exploratory analysis (EDA, ad-hoc checks)
├── src/
│   ├── config.py             # config loader + seeding
│   ├── data/
│   │   ├── collect_players.py   # pull player attribute data
│   │   ├── collect_matches.py   # pull match history data
│   │   └── clean.py             # cleaning + team-year profile assembly
│   ├── features/
│   │   └── build_features.py    # symmetric Team A/B feature vectors, scaling, PCA
│   ├── models/
│   │   ├── baseline.py          # Weighted Win Ratio (WWR) baseline — paper §IV
│   │   ├── ensemble.py          # 5-model ensemble + majority voting
│   │   └── train.py             # CV hyperparameter search + training pipeline
│   └── evaluation/
│       └── evaluate.py          # accuracy, high/low-scoring split, challenging cases
├── tests/                     # unit tests
├── outputs/
│   ├── models/                # cached trained-model bundles (api/model_cache.py)
│   └── figures/               # plots for the report
├── api/
│   ├── main.py                # FastAPI app: /teams, /predict, /evaluation
│   └── model_cache.py         # trains the pipeline once, caches to outputs/models/
├── frontend/                  # React + Vite web app (predictor + results dashboard)
├── requirements.txt
└── config/config.yaml
```

## Web app

A small full-stack demo sits on top of the pipeline: pick two teams and a FIFA
edition year for a live ensemble prediction, or browse the evaluation results
as interactive charts.

The quickest way in — one command starts both halves, waits for the backend to
finish training/loading, and opens the browser:

```bash
python start.py
```

`start.py --backend-only` / `--frontend-only` start one half, `--no-browser`
skips opening a window, and anything already listening on `:8000` / `:5173` is
reused rather than started twice. Ctrl+C stops whatever it started.

Because of that reuse, a second `python start.py` finds both ports taken,
opens the browser and exits right away — leaving no process to Ctrl+C. To
shut the servers down in that case:

```bash
python stop.py
```

It kills by port (`--backend-only` / `--frontend-only` for one half,
`--dry-run` to see what it would kill first).

To run the two halves by hand instead:

```bash
# Terminal 1 — backend (trains the pipeline on first run, ~4 min; then caches to
# outputs/models/ and loads from cache instantly on subsequent starts)
python -m uvicorn api.main:app --reload

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Open the Vite dev server URL (default `http://localhost:5173`) — it proxies
`/api/*` to the backend on `http://localhost:8000`.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If pulling datasets from Kaggle, place your API token at `~/.kaggle/kaggle.json`
(see https://www.kaggle.com/docs/api) — `collect_players.py` and `collect_matches.py`
use the `kaggle` CLI for this.

## Phase 2: Data collection (confirmed sources)

Two datasets, confirmed to fit this project's scope:

| | Dataset | Slug | File |
|---|---|---|---|
| Player-level | FIFA Complete Player Dataset (per edition, Stefano Leone) | `stefanoleone992/fifa-{edition}-complete-player-dataset` | `players_{edition}.csv` |
| Match-level | International football results from 1872–present | `martj42/international-football-results-from-1872-to-2017` | `results.csv` |

The player dataset is versioned per FIFA game edition; this project maps roster
`year` → `edition = year - 2000` (e.g. 2021 → FIFA 21), matching the paper's
2015–2023 scope. It includes a `nation_position` field (non-null only for
players actually in that edition's national squad) — `clean.filter_national_team_squad()`
uses this to build accurate squads, rather than filtering on nationality alone
(which would include every player of that nationality, most of whom never get
capped).

To pull the data:

```bash
# 1. Get a Kaggle API token: kaggle.com/settings -> API -> Create New Token
#    Save it to ~/.kaggle/kaggle.json (chmod 600 on macOS/Linux)

# 2. Run the collectors
python -m src.data.collect_players   # downloads one dataset per configured year
python -m src.data.collect_matches   # downloads the match history dataset
```

Both scripts skip files that are already present in `data/raw/`, so they're
safe to re-run.

## How the phases map to this repo

| Phase | Task | Where it lives |
|---|---|---|
| 1 | Problem definition & literature survey | *(done — see project docs)* |
| 2 | Data collection & cleaning | `src/data/` |
| 3 | Feature engineering (team profiles, scaling, PCA) | `src/features/build_features.py` |
| 4 | Model development (5-model ensemble + majority voting) | `src/models/` |
| 5 | Model evaluation & hyperparameter tuning | `src/models/train.py`, `src/evaluation/` |
| 6 | Testing against baseline | `src/models/baseline.py` + `src/evaluation/evaluate.py` |
| 7 | Report writing | `notebooks/` + `outputs/figures/` |
| 8 | Final presentation / demo | `frontend/` + `api/` (live predictor + results dashboard) |

## Status

- [x] Repo scaffolding
- [x] Data collection (real Kaggle datasets connected: player attributes 2015–2025 — FIFA 15–23,
  EA FC 24 from the same author, EA FC 25 mapped via `src/data/adapters.py` — match history 1872–present)
- [x] Feature engineering (1749 team-year profiles, 111 raw features -> 26–27 PCA components at 95% variance, depending on the split)
- [x] Baseline (WWR) — fit on full World Cup history (finals + qualifiers), test-leakage safe
- [x] Ensemble models — 5-model majority vote, trained end-to-end on real data (1495 usable matches)
- [x] Evaluation — accuracy, high/low-scoring split, challenging cases, validated across 5 random seeds
- [x] Report figures (`outputs/figures/`: accuracy comparison, seed variance, confusion matrices,
  PCA variance, ROC curve, per-model accuracy, feature importance)
- [x] Report writing (Phase 7) — `report/report.md` / `report/report.pdf`
- [x] Web app (Phase 8) — FastAPI backend (`api/`) + React frontend (`frontend/`): live team-vs-team prediction and an interactive results dashboard

**Latest results** (1495 World Cup matches incl. qualifiers, 5-seed average): ensemble
75.5% ± 2.5% overall accuracy vs. baseline 74.8% ± 2.8% — see `outputs/figures/`.
The ensemble is ahead on only 2 of the 5 seeds; per-seed margins range from -2.3 to +4.3 points,
so the mean advantage rests on two good seeds rather than a consistent edge.
Run `python -m src.pipeline` to reproduce, or `python -m src.make_figures` to
regenerate the figures.

### Held-out test: the 2026 World Cup knockout bracket

`results.csv` contains the 2026 World Cup, but `data.years` deliberately stops at **2025**, so
no 2026 match reaches the ensemble or the WWR baseline. Team strength comes from EA FC 25
squad ratings, published before the tournament began on 2026-06-11. That makes the knockout
bracket a genuine out-of-sample test rather than a cross-validation fold:

```bash
python -m src.backtest_wc2026
```

| Round | Ties | Ensemble | Baseline |
|---|---|---|---|
| Round of 32 | 16 | 75% | 50% |
| Round of 16 | 8 | 75% | 88% |
| Quarter-finals | 4 | 100% | 100% |
| Semi-finals | 2 | 0% | 50% |
| **Overall** | **30** | **73.3%** | **66.7%** |

73.3% out of sample sits within the cross-validated 75.5% ± 2.5%, which is the main evidence
that the model generalises rather than fitting its own test splits. Two caveats keep it honest:
with n=30 the 95% interval is ±15.8%, so the 6.7-point margin over the baseline is **not**
statistically significant; and 4 of the 30 ties were decided on penalties, which the model has
no way to represent — it predicts a 90-minute winner and is scored against the shootout result.

The model got both semi-finals wrong, picking France over Spain and England over Argentina.

Neither the third-place playoff nor the final has a recorded score in this dataset, so both are
genuine forward predictions: **France** to finish third (55%), and **Argentina** to beat Spain
in the final (52%) — barely above a coin flip, which is the correct amount of conviction for a
World Cup final between those two.

**Beyond overall accuracy**, `python -m src.pipeline` also reports (console + `report.attrs`):
- Per-model accuracy for each of the 5 base classifiers, before the majority vote
- Precision / recall / F1 / ROC-AUC (accuracy alone hides the home-win class imbalance)
- McNemar's test on paired baseline-vs-ensemble predictions — the ~0.7-point accuracy
  gap is **not statistically significant** at p<0.05 on any single seed (per-seed p-values
  range from 0.12 to 1.00; honest finding, not swept under the rug — see the report's
  Discussion section)
- Ensemble accuracy broken down by vote agreement (3/5, 4/5, 5/5 of the base models agreeing)
- Feature importance (Random Forest + XGBoost, mapped back from PCA-component space to
  the original 108 engineered features) — an approximation, since PCA components have no
  direct real-world meaning

## Reference

Al-Bustami, A. & Ghazal, Z. "From Players to Champions: A Generalizable Machine Learning
Approach for Match Outcome Prediction with Insights from the FIFA World Cup." 2025 IEEE
International Conference on Electro Information Technology (eIT), pp. 574–578.
DOI: 10.1109/EIT64391.2025.11103598
