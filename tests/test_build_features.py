"""Tests for feature engineering (src/features/build_features.py).

This module decides the labels, so a silent bug here moves every reported
number without breaking anything visibly.

Run with: pytest tests/test_build_features.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    build_match_features,
    build_single_match_features,
    get_feature_columns,
)


@pytest.fixture
def profiles() -> pd.DataFrame:
    return pd.DataFrame([
        {"team": "A", "year": 2018, "overall": 80.0, "pace": 70.0, "squad_size": 23},
        {"team": "B", "year": 2018, "overall": 70.0, "pace": 75.0, "squad_size": 23},
    ])


def _match(home, away, hs, aws, neutral=False, date="2018-06-14"):
    return {"date": pd.Timestamp(date), "home_team": home, "away_team": away,
            "home_score": hs, "away_score": aws, "year": 2018,
            "tournament": "FIFA World Cup", "neutral": neutral}


def test_draws_are_dropped(profiles):
    matches = pd.DataFrame([
        _match("A", "B", 2, 1),
        _match("A", "B", 1, 1),   # draw -> excluded per the paper's framing
        _match("A", "B", 0, 3),
    ])
    feat = build_match_features(matches, profiles)
    assert len(feat) == 2


def test_label_one_means_the_home_side_won(profiles):
    matches = pd.DataFrame([_match("A", "B", 2, 1), _match("A", "B", 0, 3)])
    feat = build_match_features(matches, profiles).reset_index(drop=True)

    assert feat.loc[0, "label"] == 1  # home scored more
    assert feat.loc[1, "label"] == 0  # away scored more


def test_unplayed_fixtures_are_dropped_not_labelled_as_away_wins(profiles):
    """`NaN != NaN` is True, so an unplayed row survives the draw filter and
    then gets label = (NaN > NaN) = 0. It must be dropped before either step."""
    matches = pd.DataFrame([
        _match("A", "B", 2, 1),
        _match("A", "B", np.nan, np.nan),  # scheduled, not yet played
    ])
    feat = build_match_features(matches, profiles)

    assert len(feat) == 1
    assert feat["label"].tolist() == [1]


def test_diff_is_a_minus_b(profiles):
    matches = pd.DataFrame([_match("A", "B", 2, 1)])
    feat = build_match_features(matches, profiles).reset_index(drop=True)

    assert feat.loc[0, "overall_a"] == 80.0
    assert feat.loc[0, "overall_b"] == 70.0
    assert feat.loc[0, "overall_diff"] == pytest.approx(10.0)
    assert feat.loc[0, "pace_diff"] == pytest.approx(-5.0)


def test_neutral_site_is_carried_through_as_a_numeric_feature(profiles):
    matches = pd.DataFrame([_match("A", "B", 2, 1, neutral=True),
                            _match("A", "B", 2, 1, neutral=False)])
    feat = build_match_features(matches, profiles).reset_index(drop=True)

    assert feat["neutral_site"].tolist() == [1.0, 0.0]
    assert "neutral_site" in get_feature_columns(feat)


def test_feature_columns_exclude_ids_labels_and_outcomes(profiles):
    matches = pd.DataFrame([_match("A", "B", 2, 1)])
    cols = get_feature_columns(build_match_features(matches, profiles))

    for leaked in ("label", "home_score", "away_score", "home_team", "away_team",
                   "date", "year", "tournament", "squad_size_a", "squad_size_b"):
        assert leaked not in cols


def test_single_match_features_match_the_training_schema(profiles):
    """Inference must produce every column training selected, or /predict dies
    on a KeyError the moment the schema changes."""
    matches = pd.DataFrame([_match("A", "B", 2, 1)])
    train_cols = get_feature_columns(build_match_features(matches, profiles))

    row = build_single_match_features("A", "B", 2018, profiles)
    missing = [c for c in train_cols if c not in row.columns]
    assert missing == []


def test_single_match_features_default_to_a_neutral_venue(profiles):
    row = build_single_match_features("A", "B", 2018, profiles)
    assert row["neutral_site"].iloc[0] == 1.0

    row = build_single_match_features("A", "B", 2018, profiles, neutral=False)
    assert row["neutral_site"].iloc[0] == 0.0


def test_unknown_team_or_year_raises(profiles):
    with pytest.raises(ValueError):
        build_single_match_features("A", "Nowhere", 2018, profiles)
    with pytest.raises(ValueError):
        build_single_match_features("A", "B", 1998, profiles)
