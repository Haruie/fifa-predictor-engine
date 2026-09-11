"""Feature engineering: turn (team-year profiles + match history) into a
model-ready X, y for the classification task described in the paper
(Section III.B.1 — binary Team A win / Team B win, draws excluded).

Since each match has two inputs (Team A, Team B), every profile attribute is
represented twice per row: `<attr>_a` and `<attr>_b`. We also add the
difference `<attr>_diff = <attr>_a - <attr>_b`, which tends to help linear
models pick up on relative strength without needing interaction terms.
"""
from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def _attach_team_profiles(pairs_df: pd.DataFrame, team_profiles: pd.DataFrame) -> pd.DataFrame:
    """Join team-year profiles onto each (home_team, away_team, year) row,
    producing `_a` / `_b` / `_diff` suffixed columns. Shared by
    build_match_features (training, needs scores/labels) and
    build_single_match_features (inference, doesn't)."""
    profile_cols = [c for c in team_profiles.columns if c not in ("team", "year", "squad_size")]

    a = team_profiles.add_suffix("_a").rename(columns={"team_a": "home_team", "year_a": "year"})
    b = team_profiles.add_suffix("_b").rename(columns={"team_b": "away_team", "year_b": "year"})

    merged = pairs_df.merge(a, on=["home_team", "year"], how="inner")
    merged = merged.merge(b, on=["away_team", "year"], how="inner")

    for col in profile_cols:
        merged[f"{col}_diff"] = merged[f"{col}_a"] - merged[f"{col}_b"]

    return merged


def build_match_features(matches_df: pd.DataFrame, team_profiles: pd.DataFrame) -> pd.DataFrame:
    """Join team-year profiles onto each match for both the home and away team,
    producing one row per match with `_a` / `_b` / `_diff` suffixed columns.

    Args:
        matches_df: must contain home_team, away_team, year (or a date to derive
            year from), and the outcome columns needed to build the label.
        team_profiles: output of clean.build_all_team_year_profiles().
    """
    merged = _attach_team_profiles(matches_df, team_profiles)

    # Drop fixtures with no recorded result BEFORE the draw filter. results.csv
    # carries scheduled-but-unplayed matches with NaN scores, and `NaN != NaN`
    # is True in pandas -- so they'd survive the draw filter and then get
    # `label = (NaN > NaN) = False`, i.e. silently train as away wins.
    merged = merged.dropna(subset=["home_score", "away_score"])

    # Binary label: exclude draws per the paper's classification framing.
    merged = merged[merged["home_score"] != merged["away_score"]].copy()
    merged["label"] = (merged["home_score"] > merged["away_score"]).astype(int)  # 1 = Team A (home) wins

    return merged


def build_single_match_features(team_a: str, team_b: str, year: int, team_profiles: pd.DataFrame) -> pd.DataFrame:
    """One-row feature frame for an arbitrary team pair with no known outcome,
    for live prediction. Raises ValueError if either team has no profile for
    that year (or they've never played, which can't happen for a single pair)."""
    pairs_df = pd.DataFrame([{"home_team": team_a, "away_team": team_b, "year": year}])
    merged = _attach_team_profiles(pairs_df, team_profiles)
    if merged.empty:
        raise ValueError(f"No profile data for {team_a!r} and/or {team_b!r} in {year}")
    return merged


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the *_a, *_b, *_diff engineered columns, excluding IDs/labels."""
    exclude = {"home_team", "away_team", "year", "date", "tournament", "label",
               "home_score", "away_score", "net_score", "neutral_site",
               "squad_size_a", "squad_size_b"}
    return [c for c in df.columns if c not in exclude and (
        c.endswith("_a") or c.endswith("_b") or c.endswith("_diff")
    )]


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_type: str = "standard"):
    """Fit a scaler on the training set only, then transform both splits."""
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def apply_pca(X_train_scaled, X_test_scaled, variance_threshold: float = 0.95):
    """Fit PCA on the training set to retain `variance_threshold` of the
    explained variance, then transform both splits. Per the paper, PCA is
    applied "when beneficial" -- compare downstream CV accuracy with/without
    this step during Phase 5 model selection."""
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca, pca
