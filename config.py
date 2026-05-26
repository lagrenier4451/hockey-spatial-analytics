"""
config.py — central constants for the hockey spatial analytics project.
"""
from pathlib import Path

BASE_DIR       = Path(__file__).parent
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATABASE_PATH  = DATA_PROCESSED / "hockey_spatial.db"
LOGS_DIR       = BASE_DIR / "logs"

# ── Rink dimensions (feet, tracking coordinate system) ────────────────────────
# Origin = centre ice.  X runs along the long axis, Y along the short axis.
RINK_LENGTH   = 200   # feet  (X: -100 to +100)
RINK_WIDTH    = 85    # feet  (Y: -42.5 to +42.5)
GOAL_LINE_X   = 89    # feet from centre (goal lines at ±89)
BLUE_LINE_X   = 25    # feet from centre (blue lines at ±25)
CREASE_RADIUS = 6     # feet
FACEOFF_RADIUS = 15   # feet (large faceoff circles)
FACEOFF_DOT_X = 69    # feet from centre (offensive zone dots)
FACEOFF_DOT_Y = 22    # feet from centre line

# ── Event coordinate system ───────────────────────────────────────────────────
# Events use a per-team perspective: X 0-100 (0=own goal, 100=opp goal),
# Y -42.5 to +42.5.  We normalise to the tracking system when joining.
EVENT_X_MAX = 100
EVENT_Y_HALF = 42.5

# ── Game IDs ─────────────────────────────────────────────────────────────────
GAMES = {
    "2024-10-25 Team H @ Team G": {
        "date": "2024-10-25",
        "home": "Team G",
        "away": "Team H",
    },
    "2024-11-15 Team D @ Team C": {
        "date": "2024-11-15",
        "home": "Team C",
        "away": "Team D",
    },
    "2024-11-16 Team F @ Team E": {
        "date": "2024-11-16",
        "home": "Team E",
        "away": "Team F",
    },
}
