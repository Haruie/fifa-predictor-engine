"""Cleaning and team-year profile assembly.

Implements the paper's Feature Selection principles (Section III.A.2):
  1. Relevance   -- keep only attributes that quantify offensive/defensive/overall
                    proficiency (finishing, tackling, positioning, etc.)
  2. Completeness -- keep attributes available for a sufficient number of players
                     across teams/years, to avoid sample bias
  3. Non-redundancy -- drop highly correlated attributes to avoid multicollinearity
"""
from __future__ import annotations

import pandas as pd

from src.data.collect_players import ALL_COLUMNS

# `nationality_name` values that differ between the FIFA player dataset and
# the international match-results dataset -- normalized here so team-year
# profiles join cleanly against match data (confirmed: FIFA calls it "Korea
# Republic", results.csv calls it "South Korea").
NAME_ALIASES = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "Brunei Darussalam": "Brunei",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Guinea Bissau": "Guinea-Bissau",
    "Curacao": "Curaçao",
    "São Tomé e Príncipe": "São Tomé and Príncipe",
    "Swaziland": "Eswatini",
    "Holland": "Netherlands",
}


def filter_national_team_squad(df: pd.DataFrame, top_n: int = 23) -> pd.DataFrame:
    """Build a national-squad subset per (nationality, year).

    `nation_position` (non-null = called up) is EA's per-edition national-team
    licensing tag, not a real squad-membership record -- EA only licenses a
    subset of federations per FIFA edition (confirmed: 50 nations in FIFA 18,
    33 in FIFA 22). Unlicensed federations (Croatia, Japan, Nigeria, Senegal,
    etc.) never get a nation_position tag even though their players are real
    internationals, which would silently drop 13 of 40 2018+2022 World Cup
    teams if `nation_position` were used alone (35/128 matches survive vs.
    118/128 with this fallback).

    So: use `nation_position` where a (nationality, year) pair has it (real
    squad list, most accurate), and fall back to the `top_n` players by
    `overall` rating for that nationality/year (a proxy squad) where it
    doesn't.
    """
    df = df.copy()
    df["nationality_name"] = df["nationality_name"].replace(NAME_ALIASES)

    licensed = df[df["nation_position"].notna()]
    licensed_pairs = set(map(tuple, licensed[["nationality_name", "year"]].drop_duplicates().values))
    is_licensed_pair = pd.Series(list(zip(df["nationality_name"], df["year"])), index=df.index).isin(licensed_pairs)

    squad_licensed = df[is_licensed_pair & df["nation_position"].notna()]
    unlicensed_pool = df[~is_licensed_pair]
    squad_fallback = (
        unlicensed_pool.sort_values("overall", ascending=False)
        .groupby(["nationality_name", "year"], group_keys=False)
        .head(top_n)
    )

    return pd.concat([squad_licensed, squad_fallback], ignore_index=True)


def clean_player_data(df: pd.DataFrame, completeness_threshold: float = 0.8) -> pd.DataFrame:
    """Drop columns with too many missing values (completeness principle) and
    rows missing critical identifiers.

    Call this AFTER filter_national_team_squad(), not before: nation_position
    is null for ~94% of the full player pool by design (most players are never
    capped), so running this first on the unfiltered pool drops the column
    before the squad filter can use it.

    Args:
        df: raw concatenated player data.
        completeness_threshold: minimum fraction of non-null values required to
            keep a column.
    """
    df = df.dropna(subset=["short_name", "nationality_name", "year"])

    keep_cols = [c for c in df.columns if df[c].isna().mean() <= (1 - completeness_threshold)]
    df = df[keep_cols]

    return df


def drop_redundant_features(df: pd.DataFrame, corr_threshold: float = 0.9) -> pd.DataFrame:
    """Non-redundancy principle: drop one of each pair of numeric columns whose
    pairwise correlation exceeds corr_threshold."""
    import numpy as np

    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr().abs()
    upper_mask = pd.DataFrame(
        np.triu(np.ones(corr.shape), k=1).astype(bool),
        index=corr.index, columns=corr.columns,
    )
    upper = corr.where(upper_mask)

    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    return df.drop(columns=to_drop)


def build_team_year_profile(players_df: pd.DataFrame, team: str, year: int) -> pd.Series:
    """Aggregate individual player rows into a single team-year profile.

    The paper builds *year-specific* team profiles to account for evolving
    rosters -- this is the core "player-to-team" aggregation step. Expects
    `players_df` to already be filtered to actual national-squad players (see
    filter_national_team_squad()) so the mean reflects the squad, not every
    player of that nationality. Default aggregation is the squad mean per
    numeric attribute; worth revisiting in Phase 3 (e.g. weighting by minutes
    played, or a top-11-by-position aggregation instead of full squad mean).
    """
    squad = players_df[(players_df["nationality_name"] == team) & (players_df["year"] == year)]
    if squad.empty:
        raise ValueError(f"No players found for {team} in {year}.")

    numeric_cols = [c for c in ALL_COLUMNS if c in squad.columns and pd.api.types.is_numeric_dtype(squad[c])]
    profile = squad[numeric_cols].mean()
    profile["team"] = team
    profile["year"] = year
    profile["squad_size"] = len(squad)
    return profile


def build_all_team_year_profiles(players_df: pd.DataFrame) -> pd.DataFrame:
    """Run build_team_year_profile() for every (team, year) combination present
    in the data."""
    combos = players_df[["nationality_name", "year"]].drop_duplicates()
    rows = [
        build_team_year_profile(players_df, row.nationality_name, row.year)
        for row in combos.itertuples(index=False)
    ]
    return pd.DataFrame(rows).reset_index(drop=True)
