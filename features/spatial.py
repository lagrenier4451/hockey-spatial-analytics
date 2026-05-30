"""
features/spatial.py

Coordinate handling and spatial feature engineering.

COORDINATE SYSTEM
-----------------
BDC 2025 event data uses the SAME absolute rink coordinate system
as the tracking data — no transformation required:

  X ∈ [-100, +100]  (centre ice = 0)
  Y ∈ [-42.5, +42.5]

Confirmed by landmark positions in the data:
  Faceoff wins cluster at X = {-69, -20, 0, +20, +69}  ← exact rink dots
  Zone entries cluster at  X ≈ {-25, +25}               ← exact blue lines
  Y ranges within ±42.5                                 ← exact board width

ZONE CLASSIFICATION (absolute position)
----------------------------------------
  Left Zone    : abs_x < -25
  Neutral      : -25 ≤ abs_x ≤ +25
  Right Zone   : abs_x > +25

Because both teams share the rink, Left/Right Zone describes where on
the ice an event occurred, independent of which team was attacking there.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

GOAL_X = 89.0   # absolute rink goal-line X (feet from centre)


# ── Clock parsing ─────────────────────────────────────────────────────────────

def _clock_to_seconds(clock_str) -> "int | None":
    """Convert 'MM:SS' clock string → integer seconds remaining."""
    try:
        parts = str(clock_str).strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return None


# ── Spatial geometry (vectorised) ─────────────────────────────────────────────

def _dist_to_nearest_goal(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Euclidean distance (feet) to the nearer goal for each (x, y) pair."""
    d_right = np.sqrt((xs - GOAL_X) ** 2 + ys ** 2)
    d_left  = np.sqrt((xs + GOAL_X) ** 2 + ys ** 2)
    return np.minimum(d_right, d_left)


def _angle_to_nearest_goal(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Shot angle (degrees) to the nearest goal.
    0° = straight on; 90° = directly on the goal-line extended.
    """
    use_right = np.abs(xs - GOAL_X) < np.abs(xs + GOAL_X)
    gx  = np.where(use_right, GOAL_X, -GOAL_X)
    dx  = np.abs(xs - gx)
    dy  = np.abs(ys)
    return np.degrees(np.arctan2(dy, dx))


# ── Public scalar helpers (kept for backwards-compatibility) ──────────────────

def distance_to_goal(abs_x: float, abs_y: float, goal_x: float = GOAL_X) -> float:
    if pd.isna(abs_x) or pd.isna(abs_y):
        return np.nan
    return float(_dist_to_nearest_goal(
        np.array([abs_x]), np.array([abs_y])
    )[0])


def angle_to_goal(abs_x: float, abs_y: float, goal_x: float = GOAL_X) -> float:
    if pd.isna(abs_x) or pd.isna(abs_y):
        return np.nan
    return float(_angle_to_nearest_goal(
        np.array([abs_x]), np.array([abs_y])
    )[0])


# ── Event processing ──────────────────────────────────────────────────────────

def normalise_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add abs_x/abs_y/abs_x2/abs_y2 columns.

    BDC 2025 event x/y coordinates are already in absolute rink space
    (origin = centre ice), so no mathematical transform is needed.
    """
    df = df.copy()
    df["abs_x"]  = df["x"]
    df["abs_y"]  = df["y"]
    df["abs_x2"] = df["x2"]
    df["abs_y2"] = df["y2"]
    return df


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dist_to_goal, angle_to_goal, and zone to a normalised events DataFrame.
    Requires abs_x and abs_y columns.
    """
    df = df.copy()

    valid = df["abs_x"].notna() & df["abs_y"].notna()

    dist   = np.full(len(df), np.nan)
    angle  = np.full(len(df), np.nan)

    if valid.any():
        xs = df.loc[valid, "abs_x"].values.astype(float)
        ys = df.loc[valid, "abs_y"].values.astype(float)
        dist[valid]  = _dist_to_nearest_goal(xs, ys)
        angle[valid] = _angle_to_nearest_goal(xs, ys)

    df["dist_to_goal"]  = dist
    df["angle_to_goal"] = angle
    df["zone"] = pd.cut(
        df["abs_x"],
        bins=[-105, -25, 25, 105],
        labels=["Left Zone", "Neutral", "Right Zone"],
    )
    return df


# ── Convenience loader ────────────────────────────────────────────────────────

def load_all_events() -> pd.DataFrame:
    """
    Load all Events CSVs, add coordinates and spatial features.
    Returns a single DataFrame covering all three games.
    """
    dfs = []
    for ev_path in sorted(config.DATA_RAW.glob("*Events.csv")):
        stem    = ev_path.stem.replace("-.Events", "").replace(".", " ")
        game_id = stem.strip()

        df = pd.read_csv(ev_path)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df.rename(columns={
            "x_coordinate":      "x",
            "y_coordinate":      "y",
            "x_coordinate_2":    "x2",
            "y_coordinate_2":    "y2",
            "home_team_skaters": "home_skaters",
            "away_team_skaters": "away_skaters",
            "home_team_goals":   "home_goals",
            "away_team_goals":   "away_goals",
        })
        df["game_id"] = game_id
        dfs.append(df)

    all_events = pd.concat(dfs, ignore_index=True)

    # Parse clock string "MM:SS" → integer seconds remaining
    all_events["clock_seconds"] = all_events["clock"].apply(_clock_to_seconds)

    all_events = normalise_events(all_events)
    all_events = add_spatial_features(all_events)
    return all_events


if __name__ == "__main__":
    df = load_all_events()
    print(f"Loaded {len(df):,} events across {df['game_id'].nunique()} games")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nclock_seconds: {df['clock_seconds'].notna().sum()} parsed, "
          f"range {df['clock_seconds'].min()}–{df['clock_seconds'].max()}")
    print(f"\nShot distance stats:")
    shots = df[df["event"].isin(["Shot", "Goal"])]
    print(shots["dist_to_goal"].describe().round(1))
    print(f"\nZone counts:")
    print(df["zone"].value_counts().to_string())
