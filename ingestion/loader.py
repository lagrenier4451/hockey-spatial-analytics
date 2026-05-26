"""
ingestion/loader.py

Loads all Big Data Cup 2025 CSV files into the SQLite database.

Run with:
    python ingestion/loader.py

What it does:
    1. Reads schema.sql and creates all tables
    2. Loads camera_orientations.csv → games table
    3. For each game:
       - Events CSV  → events table
       - Tracking CSV → tracking table  (large — ~70 MB each)
       - Shifts CSV  → shifts table
    4. Derives clock_seconds for every row (so time-based queries are fast)

WHY convert clock to seconds?
    The clock format "19:55" means 19 minutes 55 seconds remaining in the
    period.  To ask "give me all events in the final 2 minutes", you'd need
    SUBSTR tricks on a string.  Storing as an integer (1195 seconds) makes
    that a simple WHERE clock_seconds <= 120.
"""

import sys
import sqlite3
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def clock_to_seconds(clock_str) -> int:
    """
    Convert "MM:SS" clock string to integer seconds remaining.

    "19:55" → 1195    "00:00" → 0    NaN → 0
    """
    try:
        parts = str(clock_str).strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0


def get_connection() -> sqlite3.Connection:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Run schema.sql to create all tables (safe to re-run — uses IF NOT EXISTS)."""
    schema_path = Path(__file__).parent / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    conn = get_connection()
    with conn:
        conn.executescript(sql)
    conn.close()
    logger.info("Database schema ready: %s", config.DATABASE_PATH)


# ── Game registry ─────────────────────────────────────────────────────────────

def load_games():
    """
    Populate the games table from camera_orientations.csv.

    The camera orientation file tells us which side the home goalie starts on
    in period 1 — we need this to normalise tracking coordinates across periods.
    """
    cam_path = config.DATA_RAW / "camera_orientations.csv"
    cam = pd.read_csv(cam_path)

    conn = get_connection()
    inserted = 0
    with conn:
        for _, row in cam.iterrows():
            game_id = str(row["GameID"]).strip()

            # Parse home/away from game_id string: "2024-10-25 Team H @ Team G"
            # Format: "DATE Away @ Home"
            parts = game_id.split(" ")
            # Find "@" separator
            try:
                at_idx = parts.index("@")
                away = " ".join(parts[1:at_idx])
                home = " ".join(parts[at_idx + 1:])
                date = parts[0]
            except ValueError:
                away, home, date = "Unknown", "Unknown", "Unknown"

            right_col = "Goalie Team on Right Side of Rink in 1st Period"
            home_right = 1 if str(row.get(right_col, "")).strip() == "Home" else 0

            conn.execute("""
                INSERT OR REPLACE INTO games
                    (game_id, date, home_team, away_team, home_goalie_right_in_p1)
                VALUES (?, ?, ?, ?, ?)
            """, (game_id, date, home, away, home_right))
            inserted += 1

    conn.close()
    logger.info("Loaded %d games into games table", inserted)


# ── Events ────────────────────────────────────────────────────────────────────

def load_events(game_id: str, events_path: Path):
    """Load one game's Events CSV into the events table."""
    df = pd.read_csv(events_path)
    logger.info("  Events: %d rows from %s", len(df), events_path.name)

    conn = get_connection()
    inserted = 0
    with conn:
        # Delete existing rows for this game so re-runs are safe
        conn.execute("DELETE FROM events WHERE game_id = ?", (game_id,))

        for _, row in df.iterrows():
            conn.execute("""
                INSERT INTO events
                    (game_id, date, period, clock, clock_seconds,
                     home_skaters, away_skaters, home_goals, away_goals,
                     team, player_id, event_type, x, y,
                     detail_1, detail_2, detail_3, detail_4,
                     player_id_2, x2, y2)
                VALUES (?,?,?,?,?, ?,?,?,?, ?,?,?,?,?, ?,?,?,?, ?,?,?)
            """, (
                game_id,
                str(row.get("Date", "")),
                int(row["Period"]) if pd.notna(row.get("Period")) else None,
                str(row.get("Clock", "")),
                clock_to_seconds(row.get("Clock")),
                int(row["Home_Team_Skaters"]) if pd.notna(row.get("Home_Team_Skaters")) else None,
                int(row["Away_Team_Skaters"]) if pd.notna(row.get("Away_Team_Skaters")) else None,
                int(row["Home_Team_Goals"])   if pd.notna(row.get("Home_Team_Goals"))   else None,
                int(row["Away_Team_Goals"])   if pd.notna(row.get("Away_Team_Goals"))   else None,
                str(row.get("Team", "")),
                str(row.get("Player_Id", "")) if pd.notna(row.get("Player_Id")) else None,
                str(row.get("Event", "")),
                float(row["X_Coordinate"])   if pd.notna(row.get("X_Coordinate"))   else None,
                float(row["Y_Coordinate"])   if pd.notna(row.get("Y_Coordinate"))   else None,
                str(row["Detail_1"]) if pd.notna(row.get("Detail_1")) else None,
                str(row["Detail_2"]) if pd.notna(row.get("Detail_2")) else None,
                str(row["Detail_3"]) if pd.notna(row.get("Detail_3")) else None,
                str(row["Detail_4"]) if pd.notna(row.get("Detail_4")) else None,
                str(row["Player_Id_2"]) if pd.notna(row.get("Player_Id_2")) else None,
                float(row["X_Coordinate_2"]) if pd.notna(row.get("X_Coordinate_2")) else None,
                float(row["Y_Coordinate_2"]) if pd.notna(row.get("Y_Coordinate_2")) else None,
            ))
            inserted += 1

    conn.close()
    logger.info("  Events: inserted %d rows", inserted)


# ── Tracking ──────────────────────────────────────────────────────────────────

def load_tracking(game_id: str, tracking_path: Path, chunk_size: int = 50_000):
    """
    Load one game's Tracking CSV into the tracking table.

    Tracking files are ~70 MB with millions of rows — we read in chunks
    rather than loading the whole file into memory at once.

    WHY chunked reading?
        pd.read_csv on a 70 MB file loads ~500 MB into RAM (pandas overhead).
        Reading 50k rows at a time keeps memory flat regardless of file size.
    """
    logger.info("  Tracking: loading %s (this takes ~30s)...", tracking_path.name)

    conn = get_connection()
    conn.execute("DELETE FROM tracking WHERE game_id = ?", (game_id,))
    conn.commit()

    total = 0
    for chunk in pd.read_csv(tracking_path, chunksize=chunk_size):
        rows = []
        for _, row in chunk.iterrows():
            image_id = str(row.get("Image Id", ""))
            rows.append((
                game_id,
                image_id,
                int(row["Period"]) if pd.notna(row.get("Period")) else None,
                str(row.get("Game Clock", "")),
                clock_to_seconds(row.get("Game Clock")),
                str(row.get("Player or Puck", "")),
                str(row.get("Team", "")) if pd.notna(row.get("Team")) else None,
                str(row["Player Id"]) if pd.notna(row.get("Player Id")) else None,
                float(row["Rink Location X (Feet)"]) if pd.notna(row.get("Rink Location X (Feet)")) else None,
                float(row["Rink Location Y (Feet)"]) if pd.notna(row.get("Rink Location Y (Feet)")) else None,
                float(row["Rink Location Z (Feet)"]) if pd.notna(row.get("Rink Location Z (Feet)")) else None,
                1 if str(row.get("Goal Score", "")).strip() == "G" else 0,
            ))

        conn.executemany("""
            INSERT INTO tracking
                (game_id, image_id, period, clock, clock_seconds,
                 entity_type, team, player_id, x_feet, y_feet, z_feet, is_goal)
            VALUES (?,?,?,?,?, ?,?,?,?,?,?,?)
        """, rows)
        conn.commit()
        total += len(rows)
        logger.info("    ... %d rows loaded", total)

    conn.close()
    logger.info("  Tracking: %d total rows", total)


# ── Shifts ────────────────────────────────────────────────────────────────────

def load_shifts(game_id: str, shifts_path: Path):
    """Load one game's Shifts CSV into the shifts table."""
    df = pd.read_csv(shifts_path)
    logger.info("  Shifts: %d rows from %s", len(df), shifts_path.name)

    conn = get_connection()
    inserted = 0
    with conn:
        conn.execute("DELETE FROM shifts WHERE game_id = ?", (game_id,))
        for _, row in df.iterrows():
            conn.execute("""
                INSERT INTO shifts
                    (game_id, team, player_id, shift_number, period,
                     start_clock, end_clock, start_seconds, end_seconds, shift_length)
                VALUES (?,?,?,?,?, ?,?,?,?,?)
            """, (
                game_id,
                str(row.get("team_name", "")),
                str(row.get("Player_Id", "")) if pd.notna(row.get("Player_Id")) else None,
                int(row["shift_number"]) if pd.notna(row.get("shift_number")) else None,
                int(row["period"])       if pd.notna(row.get("period"))       else None,
                str(row.get("start_clock", "")),
                str(row.get("end_clock", "")),
                clock_to_seconds(row.get("start_clock")),
                clock_to_seconds(row.get("end_clock")),
                str(row.get("shift_length", "")),
            ))
            inserted += 1

    conn.close()
    logger.info("  Shifts: inserted %d rows", inserted)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_database():
    logger.info("=" * 60)
    logger.info("Hockey Spatial Analytics — Database Builder")
    logger.info("=" * 60)

    init_db()
    load_games()

    # Each game has three files named: DATE.Away.@.Home.-.TYPE.csv
    # We glob for Events files and derive the other two names from each.
    event_files = sorted(config.DATA_RAW.glob("*.Events.csv"))
    logger.info("Found %d game event files", len(event_files))

    for ev_path in event_files:
        # Derive game_id from filename:
        #   "2024-10-25.Team.H.@.Team.G.-.Events.csv"
        #   → "2024-10-25 Team H @ Team G"
        stem = ev_path.stem.replace("-.Events", "").replace(".", " ")
        game_id = stem.strip()

        tr_path = ev_path.parent / ev_path.name.replace("Events", "Tracking")
        sh_path = ev_path.parent / ev_path.name.replace("Events", "Shifts")

        logger.info("")
        logger.info("Loading game: %s", game_id)

        load_events(game_id, ev_path)

        if tr_path.exists():
            load_tracking(game_id, tr_path)
        else:
            logger.warning("  Tracking file not found: %s", tr_path.name)

        if sh_path.exists():
            load_shifts(game_id, sh_path)
        else:
            logger.warning("  Shifts file not found: %s", sh_path.name)

    logger.info("")
    logger.info("Database build complete: %s", config.DATABASE_PATH)

    # Quick summary
    conn = get_connection()
    for table in ["games", "events", "tracking", "shifts"]:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info("  %-12s %8d rows", table, n)
    conn.close()


if __name__ == "__main__":
    build_database()
