"""
features/decision_iq.py

Decision IQ — scores every discrete player decision as ΔSpatial Value.

FRAMEWORK
---------
Hockey IQ = consistently choosing the option that moves the puck
toward higher-danger (more threatening) space.

For every scoreable event:

    ΔV  =  xV(destination)  −  xV(origin)

    Positive ΔV  →  puck moved to more dangerous position  ✅
    Negative ΔV  →  puck moved to less dangerous position  ❌
    Near zero    →  lateral / positionally neutral move    ➖

DECISION TYPES SCORED
---------------------
  Pass       (Play / Incomplete Play)
      ΔV = xV(abs_x2, abs_y2) − xV(abs_x, abs_y)
      Measures: did the passer advance the puck toward danger?

  Shot / Goal
      ΔV = xV(abs_x, abs_y)
      The shot IS the endpoint — its spatial value is the xG at that position.
      Measures: did the player shoot from a high-quality location?

  Zone Entry
      ΔV = xV(abs_x, abs_y) − NEUTRAL_BASELINE
      Measures: how much danger did the entry gain over neutral-zone possession?

AGGREGATION
-----------
  Team IQ   = mean(ΔV) across all team decisions  [primary — robust with 3 games]
  Player IQ = mean(ΔV) per player                 [secondary — directional only]

Both are scaled 0–100 within the dataset for readability.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.xg_model import get_model

# Approximate P(goal) from neutral-zone possession.
# Everything is measured as gain above this baseline for zone entries.
NEUTRAL_BASELINE = 0.03

# Events that generate a scoreable ΔV
SCORED_EVENTS = {"Play", "Incomplete Play", "Shot", "Goal", "Zone Entry"}


# ── Core scorer ───────────────────────────────────────────────────────────────

def score_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate every event with a Decision Quality score (delta_v).

    Input  : normalised events DataFrame (must have abs_x, abs_y, abs_x2,
             abs_y2, dist_to_goal, angle_to_goal, event, player_id, team)
    Returns: copy of input with new columns:
        xv_origin     float  xV at event origin
        xv_dest       float  xV at event destination (NaN if no destination)
        decision_type str    "Pass" | "Shot" | "Zone Entry" | NaN
        delta_v       float  ΔSpatial Value for this decision (NaN if unscorable)
    """
    model = get_model(events_df)   # train / return cached model
    df    = events_df.copy()

    # ── Origin xV (computed for all events) ──────────────────────────────────
    df["xv_origin"] = model.score_series(df["abs_x"], df["abs_y"])

    # ── Destination xV (only for events with x2/y2) ──────────────────────────
    has_dest = df["abs_x2"].notna() & df["abs_y2"].notna()
    df["xv_dest"] = np.nan
    if has_dest.any():
        dest_vals = model.score_series(
            df.loc[has_dest, "abs_x2"], df.loc[has_dest, "abs_y2"]
        )
        df.loc[has_dest, "xv_dest"] = dest_vals.values

    # ── Decision type + ΔV ───────────────────────────────────────────────────
    df["decision_type"] = pd.NA
    df["delta_v"]       = np.nan

    # Passes ― Play (complete) and Incomplete Play
    mask_pass = df["event"].isin(["Play", "Incomplete Play"])
    df.loc[mask_pass, "decision_type"] = "Pass"
    passable = mask_pass & has_dest
    df.loc[passable, "delta_v"] = (
        df.loc[passable, "xv_dest"].values
        - df.loc[passable, "xv_origin"].values
    )

    # Shots ― the shot position IS the decision endpoint (xG of the attempt)
    mask_shot = df["event"].isin(["Shot", "Goal"])
    df.loc[mask_shot, "decision_type"] = "Shot"
    df.loc[mask_shot, "delta_v"] = df.loc[mask_shot, "xv_origin"].values

    # Zone entries ― gain over neutral-zone baseline
    mask_entry = df["event"] == "Zone Entry"
    df.loc[mask_entry, "decision_type"] = "Zone Entry"
    df.loc[mask_entry, "delta_v"] = (
        df.loc[mask_entry, "xv_origin"].values - NEUTRAL_BASELINE
    )

    return df


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _scale_0_100(series: pd.Series) -> pd.Series:
    """Min-max scale a Series to [0, 100]."""
    lo, hi = series.min(), series.max()
    if hi > lo:
        return (series - lo) / (hi - lo) * 100.0
    return pd.Series(50.0, index=series.index)


def team_iq(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Decision IQ per team.

    Returns
    -------
    DataFrame with columns:
        team, n_decisions, mean_dv, decision_iq (0–100),
        pass_dv, shot_dv, entry_dv  (mean ΔV broken down by type)
    """
    scored = scored_df.dropna(subset=["delta_v", "team"]).copy()

    overall = (
        scored.groupby("team")
        .agg(
            n_decisions=("delta_v", "count"),
            mean_dv    =("delta_v", "mean"),
        )
        .reset_index()
    )

    # Per-type breakdown
    for dtype, col in [
        ("Pass",       "pass_dv"),
        ("Shot",       "shot_dv"),
        ("Zone Entry", "entry_dv"),
    ]:
        sub = (
            scored[scored["decision_type"] == dtype]
            .groupby("team")["delta_v"]
            .mean()
            .reset_index(name=col)
        )
        overall = overall.merge(sub, on="team", how="left")

    overall["decision_iq"] = _scale_0_100(overall["mean_dv"])
    return overall.sort_values("decision_iq", ascending=False).reset_index(drop=True)


def player_iq(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Decision IQ per player.

    NOTE: With 3 games (~15–25 decisions per player) this is directional only.
    Use confidence intervals or large-sample caveats when displaying.

    Returns
    -------
    DataFrame with columns:
        player_id, team, n_decisions, mean_dv, std_dv, decision_iq (0–100)
    """
    scored = scored_df.dropna(subset=["delta_v", "player_id"]).copy()

    agg = (
        scored.groupby(["player_id", "team"])
        .agg(
            n_decisions=("delta_v", "count"),
            mean_dv    =("delta_v", "mean"),
            std_dv     =("delta_v", "std"),
        )
        .reset_index()
    )

    agg["decision_iq"] = _scale_0_100(agg["mean_dv"])
    return agg.sort_values("decision_iq", ascending=False).reset_index(drop=True)


def top_decisions(scored_df: pd.DataFrame, n: int = 5) -> tuple:
    """
    Return (best_n, worst_n) — the highest and lowest ΔV individual decisions.

    Returns
    -------
    (best_df, worst_df) — each is a DataFrame sorted by delta_v
    """
    cols = [
        "player_id", "team", "event", "decision_type",
        "delta_v", "xv_origin", "xv_dest",
        "abs_x", "abs_y", "game_id", "period",
    ]
    available = [c for c in cols if c in scored_df.columns]
    scored = (
        scored_df.dropna(subset=["delta_v", "player_id"])
        [available]
        .sort_values("delta_v", ascending=False)
        .reset_index(drop=True)
    )
    best  = scored.head(n)
    worst = scored.tail(n).sort_values("delta_v").reset_index(drop=True)
    return best, worst


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from features.spatial import load_all_events

    events = load_all_events()
    scored = score_events(events)

    n_scored = scored["delta_v"].notna().sum()
    print(f"Scored {n_scored:,} decisions out of {len(scored):,} events")
    print("\nMean dV by decision type:")
    print(
        scored.dropna(subset=["delta_v"])
        .groupby("decision_type")["delta_v"]
        .agg(["mean", "count"])
        .round(5)
    )

    print("\nTeam Decision IQ:")
    print(team_iq(scored)[["team", "n_decisions", "mean_dv", "decision_iq"]].to_string(index=False))

    print("\nTop 5 individual decisions (dV):")
    best, _ = top_decisions(scored, n=5)
    print(best[["player_id", "team", "decision_type", "delta_v"]].to_string(index=False))
