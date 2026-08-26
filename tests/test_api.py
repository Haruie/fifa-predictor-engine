"""Wiring test for the FastAPI backend (api/main.py).

Run with: pytest tests/test_api.py -v
Note: first run trains the full pipeline (same cost as `python -m src.pipeline`)
and caches it to outputs/models/; later runs load the cache and are fast.
"""
from fastapi.testclient import TestClient

from api.main import app


def test_teams_and_predict_roundtrip():
    with TestClient(app) as client:
        teams_resp = client.get("/teams")
        assert teams_resp.status_code == 200
        teams = teams_resp.json()
        assert len(teams) > 0

        team_a, team_b = teams[0], teams[1]
        year = sorted(set(team_a["years"]) & set(team_b["years"]))[0] if set(team_a["years"]) & set(team_b["years"]) else team_a["years"][0]

        predict_resp = client.post("/predict", json={
            "team_a": team_a["team"], "team_b": team_b["team"], "year": year,
        })
        # Either a clean prediction, or a 400 if that exact pair/year has no
        # overlapping profile data -- both are valid wiring outcomes.
        assert predict_resp.status_code in (200, 400)
        if predict_resp.status_code == 200:
            body = predict_resp.json()
            assert body["winner"] in (team_a["team"], team_b["team"])
            assert len(body["model_votes"]) == 5


def test_evaluation_endpoint():
    with TestClient(app) as client:
        resp = client.get("/evaluation")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["comparison"]) == 2
        assert "Proposed Method" in body["seed_variance"]
