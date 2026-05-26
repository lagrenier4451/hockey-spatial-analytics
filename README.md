# Hockey Spatial Analytics

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://lagrenier4451-hockey-spatial-analytics-dashboardapp-PLACEHOLDER.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Data](https://img.shields.io/badge/Data-Big_Data_Cup_2025-1A3A5C)](https://stathletes.com)

An end-to-end spatial analytics pipeline for professional hockey — normalising tracking coordinates, engineering spatial features, and visualising decision-making through interactive rink plots.

---

## What This Project Does

Professional hockey generates two complementary data streams:

| Stream | Coordinate System | Content |
|--------|------------------|---------|
| **Event data** | Team perspective (X: 0–100) | Shot, pass, zone entry, faceoff … |
| **Tracking data** | Absolute rink (X: −100 → +100) | Player + puck position every 0.04 s |

The challenge: *they don't share the same coordinate system.* This project solves that, then answers:

- **Where** do shots come from? Which zones produce the most dangerous attempts?
- **How** do passes flow? Complete vs incomplete, direct vs indirect, where attacks break down?
- **Which** zone-entry method (carry vs dump) leads to better scoring chances?

---

## Dashboard Pages

| Page | What You See |
|------|-------------|
| **Overview** | All events on a full rink; event-type counts; zone breakdown by game |
| **Shot Analysis** | Shot/goal scatter; distance & angle histograms; shot-type bar; danger-zone heatmap |
| **Pass Analysis** | Arrow plot (complete = navy, incomplete = red); origin heatmap; direct vs indirect pie |
| **Zone Entries** | Carry vs dump scatter; density heatmaps; next-event analysis after each entry type |

---

## Data Source

**Big Data Cup 2025** — provided by [Stathletes](https://stathletes.com)

Three regular-season NHL games (anonymised team labels: Team C through Team H):
- `2024-10-25 Team H @ Team G`
- `2024-11-15 Team D @ Team C`
- `2024-11-16 Team F @ Team E`

Each game includes:
- **Events CSV** (~165 KB) — in-repo, loaded by the dashboard
- **Tracking CSV** (~70 MB) — excluded from repo (`.gitignore`), used only for local DB build
- **Shifts CSV** — excluded from repo

> **License**: Non-commercial / portfolio use only. Data provided by Stathletes for the Big Data Cup competition.

---

## Project Structure

```
hockey-spatial-analytics/
├── config.py                   # Rink constants, game metadata, file paths
│
├── ingestion/
│   ├── schema.sql              # SQLite schema (games, events, tracking, shifts)
│   └── loader.py               # Chunked CSV ingestion → hockey_spatial.db
│
├── features/
│   └── spatial.py              # Coordinate normalisation + spatial feature engineering
│
├── visualization/
│   └── rink.py                 # matplotlib rink renderer (scatter, heatmap)
│
├── dashboard/
│   └── app.py                  # Streamlit 4-page dashboard
│
├── data/
│   ├── raw/                    # Events CSVs + camera_orientations.csv (committed)
│   │                           # Tracking + Shifts CSVs (gitignored — too large)
│   └── processed/              # SQLite DB output (gitignored)
│
├── .streamlit/
│   └── config.toml             # Dark-navy theme
│
└── requirements.txt
```

---

## The Coordinate Problem

Event data is recorded from the **eventing team's perspective**:

```
X = 0   → own goal
X = 100 → opponent's goal
Y ∈ [−42.5, +42.5]
```

Tracking data uses **absolute rink coordinates**:

```
X ∈ [−100, +100]  (centre ice = 0)
Y ∈ [−42.5, +42.5]
```

### How We Normalise

`camera_orientations.csv` records which side the home goalie defends in Period 1. From that:

1. If home goalie is on the **right** → home defends right → home attacks **left** (−X)
2. Direction **flips every period**: P1 = base, P2 = flipped, P3 = base, OT = flipped

For a team attacking in the **positive** direction:
```
abs_x = −89 + (event_x / 100) × 178
```

For a team attacking in the **negative** direction:
```
abs_x = +89 − (event_x / 100) × 178
```

---

## Spatial Features

After normalisation, `features/spatial.py` adds:

| Feature | Description |
|---------|-------------|
| `dist_to_goal` | Euclidean distance to nearest goal (feet) |
| `angle_to_goal` | Shot angle in degrees (0° = straight on, 90° = side angle) |
| `zone` | Defensive / Neutral / Offensive (based on absolute X vs blue lines) |
| `attacks_positive` | Direction flag for the eventing team in that period |

---

## Running Locally

### 1 — Clone & install

```bash
git clone https://github.com/lagrenier4451/hockey-spatial-analytics.git
cd hockey-spatial-analytics
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2 — Add raw data (optional — for full DB build)

Download the Big Data Cup 2025 data files from [Stathletes](https://stathletes.com/big-data-cup) and place them in `data/raw/`.

```bash
python -m ingestion.loader      # builds data/processed/hockey_spatial.db
```

> The dashboard runs without this step — it loads directly from the committed Events CSVs.

### 3 — Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## Architecture

```
data/raw/
  *Events.csv ──────► features/spatial.py ──► normalised DataFrame
  camera_orientations.csv                          │
                                                   ▼
                                          dashboard/app.py
                                       (Streamlit 4-page UI)

  *Tracking.csv ────► ingestion/loader.py ──► hockey_spatial.db
  *Shifts.csv                               (local analysis only)
```

---

## Roadmap

- [x] Phase 1 — Data ingestion pipeline, SQLite DB, rink visualisation
- [x] Phase 2 — Coordinate normalisation, spatial features, interactive dashboard
- [ ] Phase 3 — Zone entry model (carry vs dump success probability)
- [ ] Phase 4 — Shot vs pass decision model (xG-weighted decision quality)
- [ ] Phase 5 — Pass selection model (recipient choice given spatial context)
- [ ] Phase 6 — Decision Quality Score → integration with [NHL Playoff Analytics](https://github.com/lagrenier4451/nhl-playoff-db)

---

## Related Project

**[NHL Playoff Analytics](https://github.com/lagrenier4451/nhl-playoff-db)** — a separate repo tracking Composite Value Scores for playoff performers since 2006. Phase 6 above will feed spatial Decision Quality Scores into that system's CVS formula.
