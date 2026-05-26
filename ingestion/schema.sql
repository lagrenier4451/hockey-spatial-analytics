-- =============================================================================
-- Hockey Spatial Analytics — Database Schema
-- =============================================================================
--
-- Three source files per game → four tables here.
-- Tracking is stored raw (it's huge) and joined to events in queries.
--
-- Coordinate conventions:
--   events table  → per-team perspective  (X: 0-100, Y: -42.5 to +42.5)
--   tracking table → absolute rink coords (X: -100 to +100, Y: -42.5 to +42.5)
--   When joining, use normalise_event_x() logic from features/spatial.py.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- =============================================================================
-- 1. GAMES — one row per game
-- =============================================================================
CREATE TABLE IF NOT EXISTS games (
    game_id         TEXT PRIMARY KEY,   -- "2024-10-25 Team H @ Team G"
    date            TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    home_goalie_right_in_p1  INTEGER    -- 1 = home goalie on right side in P1
);

-- =============================================================================
-- 2. EVENTS — one row per tracked event
-- =============================================================================
--
-- WHY store clock_seconds separately?
--   "19:55" is a string; arithmetic requires seconds.  We derive it on load
--   so queries like "events in last 2 minutes" don't need string parsing.
--
CREATE TABLE IF NOT EXISTS events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         TEXT    NOT NULL,
    date            TEXT,
    period          INTEGER,
    clock           TEXT,
    clock_seconds   INTEGER,            -- seconds remaining in period
    home_skaters    INTEGER,
    away_skaters    INTEGER,
    home_goals      INTEGER,
    away_goals      INTEGER,
    team            TEXT,
    player_id       TEXT,
    event_type      TEXT,
    x               REAL,               -- eventing-team perspective
    y               REAL,
    detail_1        TEXT,
    detail_2        TEXT,
    detail_3        TEXT,
    detail_4        TEXT,
    player_id_2     TEXT,               -- pass target, shot on goalie, etc.
    x2              REAL,               -- secondary location (target)
    y2              REAL,

    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE INDEX IF NOT EXISTS idx_events_game    ON events(game_id);
CREATE INDEX IF NOT EXISTS idx_events_type    ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_period  ON events(game_id, period, clock_seconds);

-- =============================================================================
-- 3. TRACKING — frame-by-frame positions of every player and the puck
-- =============================================================================
--
-- WHY NOT foreign key to events?
--   Tracking frames are continuous (25-30 fps); events are discrete.
--   We join them by (game_id, period, clock_seconds) in queries.
--
-- WHY store image_id?
--   It uniquely identifies a single tracking frame and encodes the game ID.
--   Useful for debugging exactly which frame a position came from.
--
CREATE TABLE IF NOT EXISTS tracking (
    tracking_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         TEXT    NOT NULL,
    image_id        TEXT,
    period          INTEGER,
    clock           TEXT,
    clock_seconds   INTEGER,
    entity_type     TEXT,               -- 'Player' or 'Puck'
    team            TEXT,               -- 'Home', 'Away', or null for puck
    player_id       TEXT,
    x_feet          REAL,               -- absolute rink coords
    y_feet          REAL,
    z_feet          REAL,               -- always 1 for players, ~0 for puck
    is_goal         INTEGER DEFAULT 0,  -- 1 if goal scored on this frame

    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE INDEX IF NOT EXISTS idx_tracking_game   ON tracking(game_id);
CREATE INDEX IF NOT EXISTS idx_tracking_frame  ON tracking(game_id, period, clock_seconds);
CREATE INDEX IF NOT EXISTS idx_tracking_player ON tracking(player_id, game_id);

-- =============================================================================
-- 4. SHIFTS — who was on the ice during each shift
-- =============================================================================
CREATE TABLE IF NOT EXISTS shifts (
    shift_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         TEXT    NOT NULL,
    team            TEXT,
    player_id       TEXT,
    shift_number    INTEGER,
    period          INTEGER,
    start_clock     TEXT,
    end_clock       TEXT,
    start_seconds   INTEGER,
    end_seconds     INTEGER,
    shift_length    TEXT,

    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE INDEX IF NOT EXISTS idx_shifts_game   ON shifts(game_id);
CREATE INDEX IF NOT EXISTS idx_shifts_player ON shifts(player_id, game_id);
