"""
features/spatial.py

Coordinate normalisation and spatial feature engineering.

THE COORDINATE PROBLEM
----------------------
The BDC event data uses a per-team perspective:
  X = 0   → eventing team's own goal
  X = 100 → opponent's goal
  Y ∈ [-42.5, +42.5]

The tracking data uses absolute rink coordinates:
  X ∈ [-100, +100]  (centre ice = 0)
  Y ∈ [-42.5, +42.5]

To overlay events on the rink, or to join events with tracking frames,
we must normalise event coordinates to absolute rink coordinates.

HOW TO NORMALISE
----------------
1. camera_orientations.csv tells us which side the HOME goalie is on
   in period 1 (i.e. which end the home team defends in period 1).

2. If Home goalie is on the RIGHT side in P1:
     - Home team DEFENDS the right end  → Home team ATTACKS LEFT (neg X)
     - Away team ATTACKS RIGHT (pos X)

3. Directions FLIP every period:
     P1 → base direction
     P2 → flipped
     P3 → same as P1
     OT (P4+) → same as P2 (teams pick ends but convention is flip)

4. For a team attacking in the POSITIVE direction:
     abs_x = -89 + (event_x / 100) * 178
     (X=0 maps to -89 [own goal], X=100 maps to +89 [opp goal])

5. For a team attacking in the NEGATIVE direction:
     abs_x = +89 - (event_x / 100) * 178
     (X=0 maps to +89 [own goal], X=100 maps to -89 [opp goal])

Y coordinates are already in absolute rink space — no transform needed.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Absolute rink goal-line positions (feet from centre)
GOAL_X = 89.0


# ── Camera orientation loader ──────────────────────────────────────────────────

def load_camera_orientations() -> dict:
    """
    Returns {game_id: home_attacks_positive_in_p1} where:
        True  → home team attacks in the +X direction in period 1
        False → home team attacks in the -X direction in period 1
    """
    cam_path = config.DATA_RAW / "camera_orientations.csv"
    cam = pd.read_csv(cam_path)

    orientations = {}
    for _, row in cam.iterrows():
        game_id = str(row["GameID"]).strip()
        right_col = "Goalie Team on Right Side of Rink in 1st Period"
        goalie_side = str(row.get(right_col, "")).strip()

        # If HOME goalie is on the RIGHT, home DEFENDS right → attacks LEFT (neg X)
        # If AWAY goalie is on the RIGHT, home attacks RIGHT (pos X)
        home_attacks_positive = (goalie_side == "Away")
        orientations[game_id] = home_attacks_positive

    return orientations


_ORIENTATIONS = None   # cached after first load


def get_orientations() -> dict:
    global _ORIENTATIONS
    if _ORIENTATIONS is None:
        _ORIENTATIONS = load_camera_orientations()
    return _ORIENTATIONS


# ── Direction resolver ─────────────────────────────────────────────────────────

def team_attacks_positive(game_id: str, team: str, period: int) -> bool:
    """
    Return True if `team` attacks in the +X direction in `period`.

    Parameters
    ----------
    game_id : e.g. "2024-10-25 Team H @ Team G"
    team    : team label from the events file (e.g. "Team G")
    period  : 1, 2, 3, 4 (OT)
    """
    orientations = get_orientations()
    home_pos_p1 = orientations.get(game_id, True)  # default: home attacks positive

    # Parse home team from game_id: "DATE Away @ Home"
    parts = game_id.split(" ")
    try:
        at_idx = parts.index("@")
        home_team = " ".join(parts[at_idx + 1:])
    except ValueError:
        home_team = ""

    is_home = (team.strip() == home_team.strip())

    # Direction in period 1
    base_positive = home_pos_p1 if is_home else (not home_pos_p1)

    # Flip on even periods (P2, OT=P4)
    if period % 2 == 0:
        return not base_positive
    return base_positive


# ── Coordinate normalisation ───────────────────────────────────────────────────

def normalise_event_x(event_x: float, attacks_positive: bool) -> float:
    """
    Convert a single event X coordinate (0-100, team perspective)
    to absolute rink X (−89 to +89).

    Parameters
    ----------
    event_x        : raw X from events CSV (0 = own goal, 100 = opp goal)
    attacks_positive : True if this team attacks in the +X direction
    """
    if pd.isna(event_x):
        return np.nan
    frac = float(event_x) / 100.0          # 0.0 → own goal, 1.0 → opp goal
    if attacks_positive:
        return -GOAL_X + frac * 2 * GOAL_X  # -89 → +89
    else:
        return  GOAL_X - frac * 2 * GOAL_X  # +89 → -89


def normalise_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add absolute rink coordinate columns to an events DataFrame.

    Input columns required: game_id, team, period, x, y, x2, y2
    Added columns:
        abs_x   absolute X for event location
        abs_y   same as y (no transform needed)
        abs_x2  absolute X for secondary location (pass target, etc.)
        abs_y2  same as y2
        attacks_positive  direction flag for this row
    """
    df = df.copy()

    attacks_pos = []
    abs_x_vals  = []
    abs_x2_vals = []

    for _, row in df.iterrows():
        pos = team_attacks_positive(
            str(row.get("game_id", "")),
            str(row.get("team", "")),
            int(row["period"]) if pd.notna(row.get("period")) else 1,
        )
        attacks_pos.append(pos)
        abs_x_vals.append(normalise_event_x(row.get("x"),  pos))
        abs_x2_vals.append(normalise_event_x(row.get("x2"), pos))

    df["attacks_positive"] = attacks_pos
    df["abs_x"]  = abs_x_vals
    df["abs_y"]  = df["y"]
    df["abs_x2"] = abs_x2_vals
    df["abs_y2"] = df["y2"]

    return df


# ── Spatial feature engineering ───────────────────────────────────────────────

def distance_to_goal(abs_x: float, abs_y: float, goal_x: float = GOAL_X) -> float:
    """
    Euclidean distance (feet) from a position to the NEAREST goal.

    WHY nearest goal?
        In the offensive zone, the relevant goal is always the opponent's.
        But without knowing which end the team is attacking, using the
        nearest goal is a safe proxy for shot-quality calculations.
    """
    if pd.isna(abs_x) or pd.isna(abs_y):
        return np.nan
    dist_right = np.sqrt((abs_x - goal_x) ** 2 + abs_y ** 2)
    dist_left  = np.sqrt((abs_x + goal_x) ** 2 + abs_y ** 2)
    return float(min(dist_right, dist_left))


def angle_to_goal(abs_x: float, abs_y: float, goal_x: float = GOAL_X) -> float:
    """
    Shot angle (degrees) relative to the goal line at the NEAREST goal.

    0° = directly in front of goal (centre of ice)
    90° = directly on the goal line extended

    WHY angle matters for xG?
        The goalkeeper covers the net proportionally to the shot angle.
        A shot from a tight angle faces a much smaller visible net area.
    """
    if pd.isna(abs_x) or pd.isna(abs_y):
        return np.nan
    # Use nearest goal
    if abs(abs_x - goal_x) < abs(abs_x + goal_x):
        gx = goal_x
    else:
        gx = -goal_x
    dx = abs(abs_x - gx)
    dy = abs(abs_y)
    return float(np.degrees(np.arctan2(dy, dx)))


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add distance_to_goal, angle_to_goal, and zone columns to a normalised
    events DataFrame (must already have abs_x, abs_y columns).

    Zone classification (from attacking team's perspective):
        defensive : abs_x < -25  (own blue line to own goal)
        neutral   : -25 ≤ abs_x ≤ 25
        offensive : abs_x > 25   (opponent's blue line to their goal)

    Note: zone is based on absolute position, not team direction — useful
    for heatmaps.  For team-relative analysis, use attacks_positive column.
    """
    df = df.copy()
    df["dist_to_goal"]  = df.apply(
        lambda r: distance_to_goal(r.get("abs_x"), r.get("abs_y")), axis=1
    )
    df["angle_to_goal"] = df.apply(
        lambda r: angle_to_goal(r.get("abs_x"), r.get("abs_y")), axis=1
    )
    df["zone"] = pd.cut(
        df["abs_x"],
        bins=[-105, -25, 25, 105],
        labels=["Defensive", "Neutral", "Offensive"],
    )
    return df


# ── Convenience loader ────────────────────────────────────────────────────────

def load_all_events() -> pd.DataFrame:
    """
    Load all Events CSVs, normalise coordinates, and add spatial features.
    Returns a single DataFrame covering all three games.
    """
    dfs = []
    for ev_path in sorted(config.DATA_RAW.glob("*Events.csv")):
        # Derive game_id from filename
        stem = ev_path.stem.replace("-.Events", "").replace(".", " ")
        game_id = stem.strip()

        df = pd.read_csv(ev_path)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df.rename(columns={
            "x_coordinate":   "x",   "y_coordinate":   "y",
            "x_coordinate_2": "x2",  "y_coordinate_2": "y2",
            "player_id":      "player_id",
            "player_id_2":    "player_id_2",
            "home_team_skaters": "home_skaters",
            "away_team_skaters": "away_skaters",
            "home_team_goals":   "home_goals",
            "away_team_goals":   "away_goals",
        })
        df["game_id"] = game_id
        dfs.append(df)

    all_events = pd.concat(dfs, ignore_index=True)
    all_events = normalise_events(all_events)
    all_events = add_spatial_features(all_events)
    return all_events


if __name__ == "__main__":
    df = load_all_events()
    print(f"Loaded {len(df):,} events across {df['game_id'].nunique()} games")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nShot distance stats:")
    shots = df[df["event"].isin(["Shot", "Goal"])]
    print(shots["dist_to_goal"].describe().round(1))
