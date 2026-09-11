"""Tests for the as-of-date match-history features (src/features/history_features.py).

The whole value of this module is that a feature for a match on date D is built
only from matches before D. That property is invisible in the output -- a leaky
implementation produces a perfectly plausible-looking frame -- so it is asserted
directly here rather than inferred from a headline accuracy number.

Run with: pytest tests/test_history_features.py -v
"""
import pandas as pd
import pytest

from src.features.build_features import mirror_feature_frame
from src.features.history_features import (
    ELO_START,
    HistoryState,
    build_history_features,
    build_history_state,
    history_feature_columns,
)
from src.models.baseline import WeightedWinRatioBaseline


def _one_sided_history() -> pd.DataFrame:
    """A beats B in all four meetings, one per year."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2010-01-01", "2011-01-01", "2012-01-01", "2013-01-01"]),
        "home_team": ["A"] * 4,
        "away_team": ["B"] * 4,
        "home_score": [3, 2, 4, 1],
        "away_score": [0, 1, 0, 0],
        "neutral": [False] * 4,
    })


def test_first_match_sees_an_empty_history():
    history = _one_sided_history()
    feats, _ = build_history_features(history, history)
    first = feats.iloc[0]

    assert first["elo_a"] == ELO_START
    assert first["elo_b"] == ELO_START
    assert first["played_a"] == 0.0
    assert first["h2h_winrate_a"] == 0.5  # the shrinkage prior, nothing observed


def test_no_row_sees_its_own_result_or_any_later_one():
    history = _one_sided_history()
    feats, _ = build_history_features(history, history)

    # A wins every meeting, so a leak-free Elo must rise strictly and lag by one
    # match: row k reflects exactly the k matches before it.
    assert feats["elo_a"].is_monotonic_increasing
    assert feats["played_a"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert feats["played_b"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_same_day_matches_cannot_see_each_other():
    """Two fixtures on one date must both see the state from before that date."""
    same_day = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "home_team": ["A", "A"],
        "away_team": ["B", "B"],
        "home_score": [5, 5],
        "away_score": [0, 0],
        "neutral": [False, False],
    })
    feats, _ = build_history_features(same_day, same_day)

    assert (feats["elo_a"] == ELO_START).all()
    assert feats["played_a"].tolist() == [0.0, 0.0]


def test_diff_columns_are_a_minus_b():
    history = _one_sided_history()
    feats, _ = build_history_features(history, history)

    for name in ("elo", "wwr", "played", "form_win", "form_gd", "h2h_winrate"):
        expected = feats[f"{name}_a"] - feats[f"{name}_b"]
        pd.testing.assert_series_equal(feats[f"{name}_diff"], expected, check_names=False)


def test_swapping_sides_negates_every_diff():
    """Mirror-safety: the row-mirroring in the pipeline relies on this."""
    state = HistoryState()
    state.update("A", "B", 3, 0)
    state.update("A", "C", 1, 0)
    state.update("B", "C", 0, 2)

    ab = state.snapshot("A", "B")
    ba = state.snapshot("B", "A")
    for col in [c for c in ab if c.endswith("_diff")]:
        assert ab[col] == pytest.approx(-ba[col])

    # And the a/b values are genuinely swapped, not merely symmetric.
    assert ab["elo_a"] == pytest.approx(ba["elo_b"])
    assert ab["elo_b"] == pytest.approx(ba["elo_a"])


def test_wwr_feature_matches_the_baseline_formula():
    """The point of wwr_* is to hand the ensemble the baseline's own signal, so
    a divergence here would quietly weaken the comparison it exists to settle."""
    history = _one_sided_history()

    baseline = WeightedWinRatioBaseline(wwr_m=10.0).fit(history)
    state = build_history_state(history, wwr_m=10.0)

    for team in ("A", "B"):
        assert state._wwr(team) == pytest.approx(baseline._wwr(team))


def test_build_history_state_matches_the_end_of_a_full_pass():
    history = _one_sided_history()
    _, streamed = build_history_features(history, history)
    standalone = build_history_state(history)

    assert standalone.elo == pytest.approx(streamed.elo)
    assert standalone.played == streamed.played
    assert standalone.wins == streamed.wins


def test_history_cut_by_date_ignores_later_matches():
    """What the 2026 backtest depends on: slicing history by date really does
    keep later results out of the state."""
    history = _one_sided_history()
    cut = build_history_state(history[history["date"] < "2012-01-01"])
    full = build_history_state(history)

    assert cut.played["A"] == 2
    assert full.played["A"] == 4
    assert cut.elo["A"] < full.elo["A"]  # A kept winning after the cut


def test_unknown_team_gets_neutral_defaults_not_zeros():
    """An unseen team should look average, not like a team that loses always."""
    state = HistoryState()
    snap = state.snapshot("Nowhere", "Elsewhere")

    assert snap["elo_a"] == ELO_START
    assert snap["form_win_a"] == 0.5
    assert snap["h2h_winrate_a"] == 0.5
    assert snap["played_a"] == 0.0


def test_mirror_feature_frame_swaps_sides_and_leaves_context_alone():
    X = pd.DataFrame({
        "elo_a": [1600.0], "elo_b": [1400.0], "elo_diff": [200.0],
        "neutral_site": [1.0],
    })
    mirrored = mirror_feature_frame(X)

    assert list(mirrored.columns) == list(X.columns)  # order preserved
    assert mirrored["elo_a"].iloc[0] == 1400.0
    assert mirrored["elo_b"].iloc[0] == 1600.0
    assert mirrored["elo_diff"].iloc[0] == -200.0
    # Venue is a property of the match, not of a side -- it must not flip.
    assert mirrored["neutral_site"].iloc[0] == 1.0


def test_history_feature_columns_are_all_present():
    history = _one_sided_history()
    feats, _ = build_history_features(history, history)
    assert list(feats.columns) == history_feature_columns()
