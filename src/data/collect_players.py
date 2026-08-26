"""Player-level data collection.

CONFIRMED DATA SOURCE (researched Aug 2026): Stefano Leone's "FIFA Complete
Player Dataset" series on Kaggle -- one dataset per game edition, each
containing a `players_XX.csv` (e.g. players_21.csv for FIFA 21), scraped from
sofifa.com. Slugs follow the pattern:

    stefanoleone992/fifa-{edition}-complete-player-dataset

e.g. stefanoleone992/fifa-21-complete-player-dataset -> players_21.csv

Edition-to-year mapping: the paper's "year-specific" rosters (2015-2022, plus
partial 2023) line up directly with FIFA {edition} = year - 2000, e.g.
year 2021 -> FIFA 21 -> players_21.csv. This project uses that 1:1 mapping.

Real column groups confirmed from the dataset (prefixed, matching the paper's
categories almost exactly):
  - Metadata: short_name, long_name, age, height_cm, weight_kg,
    nationality_name, club_name, preferred_foot, player_positions
  - Attacking: attacking_crossing, attacking_finishing, attacking_heading_accuracy,
    attacking_short_passing, attacking_volleys
  - Skill: skill_dribbling, skill_curve, skill_fk_accuracy, skill_long_passing,
    skill_ball_control
  - Movement: movement_acceleration, movement_sprint_speed, movement_agility,
    movement_reactions, movement_balance
  - Power: power_shot_power, power_jumping, power_stamina, power_strength,
    power_long_shots
  - Mentality: mentality_aggression, mentality_interceptions,
    mentality_positioning, mentality_vision, mentality_penalties, mentality_composure
  - Defending: defending_marking_awareness, defending_standing_tackle,
    defending_sliding_tackle
  - Goalkeeping: goalkeeping_diving, goalkeeping_handling, goalkeeping_kicking,
    goalkeeping_positioning, goalkeeping_reflexes

Important modeling note: `nationality_name` alone is NOT the same as "plays for
the senior national team" -- most players with a given nationality never get
called up. The dataset also has `nation_position` (non-null only for players
currently in that edition's national squad) and `nation_jersey_number`. This
project filters on `nation_position.notna()` to build actual national-team
rosters, which is a meaningfully better signal than nationality_name alone for
World Cup squad composition. See clean.filter_national_team_squad().

TODO (Phase 2):
  - Install the Kaggle CLI (`pip install kaggle`, already in requirements.txt)
    and place your API token at ~/.kaggle/kaggle.json (or %USERPROFILE%\\.kaggle\\
    kaggle.json on Windows) -- get it from kaggle.com/settings -> API ->
    "Create New Token".
  - Run `python -m src.data.collect_players` once credentials are set up to
    pull all configured years into data/raw/.
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path

# Columns we retain, grouped per the paper's structure. These are the REAL
# column names from the stefanoleone992 dataset (confirmed via research).
METADATA_COLUMNS = ["short_name", "long_name", "age", "height_cm", "weight_kg",
                     "nationality_name", "club_name", "preferred_foot",
                     "player_positions", "nation_position", "nation_jersey_number"]
ATTACKING_COLUMNS = ["attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
                      "attacking_short_passing", "attacking_volleys"]
SKILL_COLUMNS = ["skill_dribbling", "skill_curve", "skill_fk_accuracy",
                  "skill_long_passing", "skill_ball_control"]
MOVEMENT_COLUMNS = ["movement_acceleration", "movement_sprint_speed", "movement_agility",
                     "movement_reactions", "movement_balance"]
POWER_COLUMNS = ["power_shot_power", "power_jumping", "power_stamina",
                  "power_strength", "power_long_shots"]
MENTALITY_COLUMNS = ["mentality_aggression", "mentality_interceptions", "mentality_positioning",
                      "mentality_vision", "mentality_penalties", "mentality_composure"]
DEFENDING_COLUMNS = ["defending_marking_awareness", "defending_standing_tackle",
                      "defending_sliding_tackle"]
GOALKEEPING_COLUMNS = ["goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
                        "goalkeeping_positioning", "goalkeeping_reflexes"]

ALL_COLUMNS = (
    METADATA_COLUMNS + ATTACKING_COLUMNS + SKILL_COLUMNS + MOVEMENT_COLUMNS
    + POWER_COLUMNS + MENTALITY_COLUMNS + DEFENDING_COLUMNS + GOALKEEPING_COLUMNS
)


def _edition_for_year(year: int) -> int:
    """Map a roster year (e.g. 2021) to its FIFA game edition number (e.g. 21)."""
    return year - 2000


def download_player_data(cfg: dict | None = None) -> None:
    """Download one FIFA-edition dataset per configured year via the Kaggle
    CLI, unzip it, and keep only the players_XX.csv file in data/raw/.

    Requires the `kaggle` package (pip install kaggle) and a valid API token
    at ~/.kaggle/kaggle.json.
    """
    cfg = cfg or load_config()
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    for year in cfg["data"]["years"]:
        edition = _edition_for_year(year)
        slug = f"stefanoleone992/fifa-{edition}-complete-player-dataset"
        target_csv = raw_dir / f"players_{year}.csv"

        if target_csv.exists():
            print(f"[{year}] already downloaded -> {target_csv}")
            continue

        print(f"[{year}] downloading {slug} ...")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(raw_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[{year}] FAILED: {result.stderr.strip()}")
            print("  (Older editions like FIFA 15-19 may need the legacy filename "
                  "players_YY.csv inside the zip -- check the dataset's file list "
                  "on Kaggle if this keeps failing.)")
            continue

        zip_path = raw_dir / f"fifa-{edition}-complete-player-dataset.zip"
        with zipfile.ZipFile(zip_path) as zf:
            # The zip contains multiple CSVs (players, teams, coaches); we only
            # want the players file for this edition.
            member = f"players_{edition}.csv"
            if member in zf.namelist():
                zf.extract(member, raw_dir)
                (raw_dir / member).rename(target_csv)
            else:
                print(f"[{year}] '{member}' not found in zip; contents: {zf.namelist()}")
        zip_path.unlink(missing_ok=True)
        print(f"[{year}] saved -> {target_csv}")


def load_player_data(cfg: dict | None = None) -> pd.DataFrame:
    """Load raw player data from data/raw/ into a single DataFrame, tagging
    each row with its `year` (since the paper builds year-specific profiles)."""
    cfg = cfg or load_config()
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    years = cfg["data"]["years"]

    frames = []
    for year in years:
        candidate = raw_dir / f"players_{year}.csv"
        if candidate.exists():
            df = pd.read_csv(candidate, low_memory=False)
            df["year"] = year
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No player data files found in {raw_dir}. Run download_player_data() "
            "first (needs a Kaggle API token), or manually place players_<year>.csv "
            "files there."
        )

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    download_player_data()

