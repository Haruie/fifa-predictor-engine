"""FastAPI backend for the FIFA match predictor frontend.

Run with: uvicorn api.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix, roc_curve

from api.model_cache import load_or_train
from src.config import load_config
from src.features.build_features import build_single_match_features

SEEDS = [0, 1, 17, 42, 123]

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    if cfg["project"]["random_state"] not in SEEDS:
        raise ValueError(
            f"config.yaml project.random_state ({cfg['project']['random_state']}) must be "
            f"one of the hardcoded evaluation SEEDS {SEEDS} (api/main.py) so a default "
            "report exists to serve -- add it to SEEDS or pick one of the existing values."
        )
    bundle = load_or_train(cfg, SEEDS)
    state["profiles"] = bundle["profiles"]
    state["reports_by_seed"] = bundle["reports_by_seed"]
    state["default_seed"] = bundle["default_seed"]
    state["default_report"] = bundle["reports_by_seed"][bundle["default_seed"]]
    yield


app = FastAPI(title="FIFA Match Predictor API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class PredictRequest(BaseModel):
    team_a: str
    team_b: str
    # Which FIFA edition's squad ratings to use. Omit for the most recent
    # edition both teams appear in -- that's what a *future* fixture needs,
    # since there is no player dataset for a tournament that hasn't happened.
    year: int | None = None
    # World Cup finals are played at neutral venues, but the model is trained
    # on home_team/away_team rows where 62.8% of non-neutral matches are home
    # wins (vs 49.6% on neutral ground). It therefore favours whichever team is
    # passed as team_a. With neutral=True the match is scored in both
    # orientations and averaged, so the answer no longer depends on argument
    # order. Set False only for a fixture with a genuine home side.
    neutral: bool = True


def _latest_shared_year(team_a: str, team_b: str) -> int:
    """Most recent edition for which BOTH teams have a squad profile."""
    profiles = state["profiles"]
    years_a = set(profiles.loc[profiles["team"] == team_a, "year"])
    years_b = set(profiles.loc[profiles["team"] == team_b, "year"])
    shared = years_a & years_b
    if not shared:
        raise HTTPException(
            status_code=400,
            detail=f"No FIFA edition has squad data for both {team_a!r} and {team_b!r}.",
        )
    return int(max(shared))


@app.get("/teams")
def teams() -> list[dict]:
    profiles = state["profiles"]
    by_team: dict[str, list[int]] = {}
    for team, year in zip(profiles["team"], profiles["year"]):
        by_team.setdefault(team, []).append(int(year))
    return [{"team": t, "years": sorted(years)} for t, years in sorted(by_team.items())]


def _proba_home_win(team_a: str, team_b: str, year: int) -> dict[str, float]:
    """Per-model P(team_a wins) for one orientation of a matchup."""
    attrs = state["default_report"].attrs
    try:
        row = build_single_match_features(team_a, team_b, year, state["profiles"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    X = row[attrs["feature_cols"]].fillna(attrs["train_mean"])
    X_scaled = attrs["scaler"].transform(X)
    X_final = attrs["pca"].transform(X_scaled) if attrs["pca"] is not None else X_scaled

    ensemble = attrs["ensemble"]
    return {name: float(est.predict_proba(X_final)[0][1])
            for name, est in ensemble.named_estimators_.items()}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if req.team_a == req.team_b:
        raise HTTPException(status_code=400, detail="Pick two different teams.")

    year = req.year if req.year is not None else _latest_shared_year(req.team_a, req.team_b)

    per_model_proba_a = _proba_home_win(req.team_a, req.team_b, year)
    if req.neutral:
        # Score the reversed fixture too and average. P(a wins) from the
        # reversed run is 1 - P(b wins as home), so the home-side bias the model
        # learned cancels instead of landing on whichever team was passed first.
        reversed_proba_b = _proba_home_win(req.team_b, req.team_a, year)
        per_model_proba_a = {
            name: (p + (1.0 - reversed_proba_b[name])) / 2.0
            for name, p in per_model_proba_a.items()
        }

    votes = {name: (req.team_a if p >= 0.5 else req.team_b) for name, p in per_model_proba_a.items()}
    # Each model's confidence in its OWN pick (always >= 0.5).
    model_confidence = {name: (p if p >= 0.5 else 1 - p) for name, p in per_model_proba_a.items()}

    # Winner and confidence both derive from the same soft-vote average, so
    # they can never disagree in direction (a hard-vote winner combined with a
    # separately-computed soft-vote confidence can point different ways when
    # the vote is a narrow, low-conviction majority). The offline evaluation
    # figures (README/report) still use the paper's hard majority vote via
    # ensemble.predict() -- this only affects the live single-match endpoint.
    mean_proba_a = sum(per_model_proba_a.values()) / len(per_model_proba_a)
    winner_is_a = mean_proba_a >= 0.5
    confidence = mean_proba_a if winner_is_a else 1 - mean_proba_a

    return {
        "team_a": req.team_a,
        "team_b": req.team_b,
        # The edition actually used, which may differ from what was requested
        # (or have been chosen here when the caller omitted it).
        "year": year,
        "neutral": req.neutral,
        "winner": req.team_a if winner_is_a else req.team_b,
        "confidence": confidence,
        "model_votes": votes,
        "model_confidence": model_confidence,
    }


@app.get("/evaluation")
def evaluation() -> dict:
    report = state["default_report"]
    attrs = report.attrs

    seed_variance = {
        label: [
            {"seed": s, "accuracy": float(state["reports_by_seed"][s].loc[label, "Overall Accuracy"])}
            for s in SEEDS
        ]
        for label in ("Proposed Method", "Baseline Model")
    }

    pca_variance = attrs["pca_explained_variance"]

    fpr, tpr, _ = roc_curve(attrs["y_test"], attrs["ml_proba"])
    roc_points = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)]

    return {
        "comparison": report.reset_index().to_dict(orient="records"),
        "seed_variance": seed_variance,
        "confusion_matrix": {
            "proposed": confusion_matrix(attrs["y_test"], attrs["ml_pred"]).tolist(),
            "baseline": confusion_matrix(attrs["y_test"], attrs["baseline_pred"]).tolist(),
        },
        "pca_cumulative_variance": np.cumsum(pca_variance).tolist() if pca_variance is not None else None,
        "per_model_accuracy": attrs["per_model_accuracy"].to_dict(),
        "feature_importance": attrs["feature_importance"].to_dict() if attrs["feature_importance"] is not None else None,
        "roc": {"auc": float(attrs["ensemble_auc"]), "points": roc_points},
        # Every block above except `seed_variance` comes from ONE seed's report
        # (the default seed). The frontend labels those panels with this value so
        # they can't be mistaken for the across-seed averages in `seed_variance`
        # -- on some seeds the baseline beats the ensemble even though the mean
        # goes the other way.
        "default_seed": int(state["default_seed"]),
        "seeds": [int(s) for s in SEEDS],
        "dataset": {
            "n_team_profiles": int(len(state["profiles"])),
            "n_train_matches": int(attrs["n_train"]),
            "n_test_matches": int(attrs["n_test"]),
            "n_raw_features": int(len(attrs["feature_cols"])),
            "n_pca_components": int(len(pca_variance)) if pca_variance is not None else None,
        },
    }
