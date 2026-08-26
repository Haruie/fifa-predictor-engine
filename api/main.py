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
    bundle = load_or_train(cfg, SEEDS)
    state["profiles"] = bundle["profiles"]
    state["reports_by_seed"] = bundle["reports_by_seed"]
    state["default_report"] = bundle["reports_by_seed"][bundle["default_seed"]]
    yield


app = FastAPI(title="FIFA Match Predictor API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class PredictRequest(BaseModel):
    team_a: str
    team_b: str
    year: int


@app.get("/teams")
def teams() -> list[dict]:
    profiles = state["profiles"]
    by_team: dict[str, list[int]] = {}
    for team, year in zip(profiles["team"], profiles["year"]):
        by_team.setdefault(team, []).append(int(year))
    return [{"team": t, "years": sorted(years)} for t, years in sorted(by_team.items())]


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    attrs = state["default_report"].attrs
    try:
        row = build_single_match_features(req.team_a, req.team_b, req.year, state["profiles"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    X = row[attrs["feature_cols"]].fillna(attrs["train_mean"])
    X_scaled = attrs["scaler"].transform(X)
    X_final = attrs["pca"].transform(X_scaled) if attrs["pca"] is not None else X_scaled

    ensemble = attrs["ensemble"]
    winner_is_a = bool(ensemble.predict(X_final)[0])
    per_model_proba_a = {name: float(est.predict_proba(X_final)[0][1]) for name, est in ensemble.named_estimators_.items()}
    votes = {name: (req.team_a if p >= 0.5 else req.team_b) for name, p in per_model_proba_a.items()}
    # Each model's confidence in its OWN pick (always >= 0.5).
    model_confidence = {name: (p if p >= 0.5 else 1 - p) for name, p in per_model_proba_a.items()}

    # Ensemble confidence: mean predicted probability across the 5 models,
    # continuous (not the 60/80/100%-only artifact of counting hard votes).
    mean_proba_a = sum(per_model_proba_a.values()) / len(per_model_proba_a)
    confidence = mean_proba_a if winner_is_a else 1 - mean_proba_a

    return {
        "team_a": req.team_a,
        "team_b": req.team_b,
        "year": req.year,
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
        "dataset": {
            "n_team_profiles": int(len(state["profiles"])),
            "n_train_matches": int(attrs["n_train"]),
            "n_test_matches": int(attrs["n_test"]),
            "n_raw_features": int(len(attrs["feature_cols"])),
            "n_pca_components": int(len(pca_variance)) if pca_variance is not None else None,
        },
    }
