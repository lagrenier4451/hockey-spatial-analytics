"""
dashboard/app.py — Hockey Spatial Analytics Dashboard

Five pages built on BDC 2025 tracking event data:
  1. Overview    — event breakdown, rink map, game selector
  2. Shots       — shot scatter, heatmap, distance/angle analysis
  3. Passes      — pass arrows, direct vs indirect, zone flows
  4. Zone Entries — carry vs dump, entry success rates
  5. Decision IQ  — spatial value surface, ΔV scores, team/player IQ

Run locally:
    streamlit run dashboard/app.py --server.port 8551

Live:
    Deployed at Streamlit Community Cloud (see README)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.spatial import load_all_events
from visualization.rink import draw_rink, draw_heatmap
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hockey Spatial Analytics",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour constants ──────────────────────────────────────────────────────────
NAVY  = "#1A3A5C"
BLUE  = "#2E6DA4"
LBLUE = "#5B9BD5"
GOLD  = "#F4D03F"
RED   = "#C00000"

EVENT_COLORS = {
    "Shot":            "#1A3A5C",
    "Goal":            "#F4D03F",
    "Play":            "#2E6DA4",
    "Incomplete Play": "#C00000",
    "Zone Entry":      "#27AE60",
    "Dump In/Out":     "#E67E22",
    "Puck Recovery":   "#8E44AD",
    "Takeaway":        "#E74C3C",
    "Faceoff Win":     "#95A5A6",
    "Penalty Taken":   "#E74C3C",
}

# ── Data loader ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading event data...")
def get_events() -> pd.DataFrame:
    """Load and normalise all events once; cache across page interactions."""
    return load_all_events()


def fig_to_streamlit(fig):
    """Render a matplotlib figure in Streamlit then close it."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


@st.cache_data(show_spinner="Computing Decision IQ scores...")
def get_scored_events() -> pd.DataFrame:
    """
    Score all events with ΔSpatial Value.
    Cached once — reused across all Decision IQ page interactions.
    """
    from features.decision_iq import score_events
    return score_events(get_events())


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏒 Hockey Spatial Analytics")
st.sidebar.markdown("**Data:** Big Data Cup 2025 — Stathletes")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Page",
    ["Overview", "Shot Analysis", "Pass Analysis", "Zone Entries", "Decision IQ"],
)

all_events = get_events()
games = ["All Games"] + sorted(all_events["game_id"].unique().tolist())

st.sidebar.markdown("### Filter")
selected_game = st.sidebar.selectbox("Game", games)

if selected_game == "All Games":
    df = all_events.copy()
else:
    df = all_events[all_events["game_id"] == selected_game].copy()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Coordinates normalised to absolute rink space.\n\n"
    "X: −100 (left end) → +100 (right end)\n"
    "Y: −42.5 (bottom board) → +42.5 (top board)"
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("Hockey Spatial Analytics — Overview")
    st.caption(
        "Big Data Cup 2025 · Stathletes tracking data · 3 professional games"
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    total     = len(df)
    shots     = len(df[df["event"].isin(["Shot", "Goal"])])
    goals     = len(df[df["event"] == "Goal"])
    passes    = len(df[df["event"].isin(["Play", "Incomplete Play"])])
    entries   = len(df[df["event"] == "Zone Entry"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Events",   f"{total:,}")
    c2.metric("Shots",          f"{shots}")
    c3.metric("Goals",          f"{goals}")
    c4.metric("Pass Attempts",  f"{passes}")
    c5.metric("Zone Entries",   f"{entries}")

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("All Events — Rink Map")
        fig, ax = draw_rink(figsize=(12, 5))

        for etype, color in EVENT_COLORS.items():
            subset = df[df["event"] == etype]
            if subset.empty or subset["abs_x"].isna().all():
                continue
            ax.scatter(
                subset["abs_x"], subset["abs_y"],
                c=color, s=20, alpha=0.55, label=etype, zorder=5,
                edgecolors="none",
            )

        ax.legend(
            loc="lower center",
            ncol=5,
            fontsize=7,
            framealpha=0.85,
            bbox_to_anchor=(0.5, -0.08),
        )
        fig_to_streamlit(fig)

    with col_right:
        st.subheader("Events by Type")
        counts = (
            df["event"].value_counts()
            .reset_index()
            .rename(columns={"index": "Event", "event": "Count",
                             "count": "Count"})
        )
        # plotly bar for interactivity
        import plotly.express as px
        fig_bar = px.bar(
            counts, x="count" if "count" in counts.columns else "Count",
            y="event" if "event" in counts.columns else "Event",
            orientation="h",
            color="event" if "event" in counts.columns else "Event",
            color_discrete_map=EVENT_COLORS,
            labels={"event": "", "count": "# Events"},
        )
        fig_bar.update_layout(
            showlegend=False,
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Zone breakdown ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Event Distribution by Zone")
    zone_counts = df.groupby(["zone", "event"]).size().reset_index(name="n")
    import plotly.express as px
    fig_zone = px.bar(
        zone_counts, x="zone", y="n", color="event",
        color_discrete_map=EVENT_COLORS,
        labels={"zone": "Zone", "n": "Events", "event": "Type"},
        title="",
        barmode="stack",
    )
    fig_zone.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Event",
        height=320,
    )
    st.plotly_chart(fig_zone, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SHOT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Shot Analysis":
    st.title("Shot Analysis")
    st.caption(
        "All shots and goals normalised to absolute rink coordinates. "
        "Distance and angle computed from the nearest goal."
    )

    shots_df = df[df["event"].isin(["Shot", "Goal"])].copy()
    goals_df = shots_df[shots_df["event"] == "Goal"]

    if shots_df.empty:
        st.warning("No shot data for the selected game.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Shots",      len(shots_df))
    c2.metric("Goals",            len(goals_df))
    c3.metric("Conversion Rate",  f"{len(goals_df)/len(shots_df)*100:.1f}%")
    c4.metric("Avg Distance",     f"{shots_df['dist_to_goal'].mean():.1f} ft")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Shot Locations")
        fig, ax = draw_rink(figsize=(8, 4))

        non_goals = shots_df[shots_df["event"] == "Shot"]
        ax.scatter(non_goals["abs_x"], non_goals["abs_y"],
                   c=NAVY, s=45, alpha=0.6, zorder=5,
                   edgecolors="white", linewidths=0.4, label="Shot")
        ax.scatter(goals_df["abs_x"], goals_df["abs_y"],
                   c=GOLD, s=120, alpha=0.9, zorder=6,
                   edgecolors="black", linewidths=0.7, label="Goal", marker="*")
        ax.legend(fontsize=9, loc="upper left")
        fig_to_streamlit(fig)

    with col2:
        st.subheader("Shot Density Heatmap")
        valid = shots_df.dropna(subset=["abs_x", "abs_y"])
        # Mirror all shots to right side for cleaner heatmap
        mirror_x = valid["abs_x"].abs()
        fig2, ax2 = draw_heatmap(
            mirror_x, valid["abs_y"],
            half=True, half_side="right",
            bins=25, cmap="YlOrRd",
        )
        ax2.set_title("Shot Volume (mirrored to one end)", fontsize=10,
                      fontweight="bold")
        fig_to_streamlit(fig2)

    # ── Distance + angle distributions ────────────────────────────────────────
    st.markdown("---")
    st.subheader("Shot Distance & Angle")
    import plotly.express as px

    col3, col4 = st.columns(2)
    with col3:
        fig_dist = px.histogram(
            shots_df.dropna(subset=["dist_to_goal"]),
            x="dist_to_goal", color="event",
            color_discrete_map={"Shot": NAVY, "Goal": GOLD},
            nbins=30,
            labels={"dist_to_goal": "Distance to Goal (ft)", "count": "Shots"},
            title="Shot Distance Distribution",
            barmode="overlay",
            opacity=0.75,
        )
        fig_dist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col4:
        fig_angle = px.histogram(
            shots_df.dropna(subset=["angle_to_goal"]),
            x="angle_to_goal", color="event",
            color_discrete_map={"Shot": NAVY, "Goal": GOLD},
            nbins=30,
            labels={"angle_to_goal": "Shot Angle (°)", "count": "Shots"},
            title="Shot Angle Distribution",
            barmode="overlay",
            opacity=0.75,
        )
        fig_angle.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
        )
        st.plotly_chart(fig_angle, use_container_width=True)

    # ── Shot type breakdown ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Shot Type (Detail)")
    shot_types = shots_df["detail_1"].value_counts().reset_index()
    fig_type = px.bar(
        shot_types, x="detail_1", y="count",
        color_discrete_sequence=[NAVY],
        labels={"detail_1": "Shot Type", "count": "Count"},
        title="",
    )
    fig_type.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_type, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PASS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Pass Analysis":
    st.title("Pass Analysis")
    st.caption(
        "Completed passes (Play) and incomplete passes shown with sender → receiver arrows. "
        "Each arrow goes from the passer's position to the intended receiver."
    )

    passes_df = df[df["event"].isin(["Play", "Incomplete Play"])].copy()
    complete   = passes_df[passes_df["event"] == "Play"]
    incomplete = passes_df[passes_df["event"] == "Incomplete Play"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Completed Passes",   len(complete))
    c2.metric("Incomplete Passes",  len(incomplete))
    total_p = len(passes_df)
    c3.metric("Completion Rate",
              f"{len(complete)/total_p*100:.1f}%" if total_p else "—")

    st.markdown("---")

    # ── Sidebar sample size ────────────────────────────────────────────────────
    max_arrows = st.sidebar.slider("Max pass arrows shown", 50, 500, 200, step=50)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Completed Passes")
        fig, ax = draw_rink(figsize=(8, 4))
        sample = complete.dropna(subset=["abs_x", "abs_x2"]).sample(
            min(max_arrows, len(complete)), random_state=42
        )
        for _, row in sample.iterrows():
            ax.annotate(
                "", xy=(row["abs_x2"], row["abs_y2"]),
                xytext=(row["abs_x"], row["abs_y"]),
                arrowprops=dict(
                    arrowstyle="->", color=NAVY, alpha=0.35, lw=0.8,
                ),
            )
        ax.set_title(f"Completed Passes (n={len(sample)} sampled)",
                     fontsize=10, fontweight="bold")
        fig_to_streamlit(fig)

    with col2:
        st.subheader("Incomplete Passes")
        fig2, ax2 = draw_rink(figsize=(8, 4))
        sample_inc = incomplete.dropna(subset=["abs_x", "abs_x2"]).sample(
            min(max_arrows, len(incomplete)), random_state=42
        )
        for _, row in sample_inc.iterrows():
            ax2.annotate(
                "", xy=(row["abs_x2"], row["abs_y2"]),
                xytext=(row["abs_x"], row["abs_y"]),
                arrowprops=dict(
                    arrowstyle="->", color=RED, alpha=0.35, lw=0.8,
                ),
            )
        ax2.set_title(f"Incomplete Passes (n={len(sample_inc)} sampled)",
                      fontsize=10, fontweight="bold")
        fig_to_streamlit(fig2)

    # ── Direct vs Indirect ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Direct vs Indirect Passes")
    import plotly.express as px

    col3, col4 = st.columns(2)
    with col3:
        detail_counts = complete["detail_1"].value_counts().reset_index()
        fig_detail = px.pie(
            detail_counts, values="count", names="detail_1",
            color_discrete_sequence=[NAVY, LBLUE, GOLD],
            title="Completed — Pass Type",
            hole=0.4,
        )
        fig_detail.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_detail, use_container_width=True)

    with col4:
        # Pass origin zone breakdown
        zone_pass = complete.groupby("zone").size().reset_index(name="count")
        fig_zone = px.bar(
            zone_pass, x="zone", y="count",
            color="zone",
            color_discrete_map={
                "Defensive": RED, "Neutral": LBLUE, "Offensive": NAVY,
            },
            labels={"zone": "Zone of Origin", "count": "Passes"},
            title="Completed Passes by Zone",
        )
        fig_zone.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_zone, use_container_width=True)

    # ── Passer location heatmap ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Where Do Passes Originate?")
    valid_passes = complete.dropna(subset=["abs_x", "abs_y"])
    fig3, ax3 = draw_heatmap(
        valid_passes["abs_x"], valid_passes["abs_y"],
        title="Pass Origin Density", bins=30, cmap="Blues", half=False,
    )
    fig_to_streamlit(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ZONE ENTRIES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Zone Entries":
    st.title("Zone Entries")
    st.caption(
        "A zone entry is the moment a team crosses the opposing blue line. "
        "Carry-ins (controlled entry) vs dump-ins significantly affect "
        "whether the team generates a shot — this is the foundation of our model."
    )

    entries = df[df["event"] == "Zone Entry"].copy()
    dumps   = df[df["event"] == "Dump In/Out"].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Controlled Entries",  len(entries))
    c2.metric("Dump Ins/Outs",       len(dumps))
    total_e = len(entries) + len(dumps)
    c3.metric("Carry % of Total",
              f"{len(entries)/total_e*100:.1f}%" if total_e else "—")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Entry Locations — Controlled vs Dump")
        fig, ax = draw_rink(figsize=(8, 4))
        valid_e = entries.dropna(subset=["abs_x"])
        valid_d = dumps.dropna(subset=["abs_x"])
        ax.scatter(valid_e["abs_x"], valid_e["abs_y"],
                   c=NAVY, s=50, alpha=0.65, zorder=5,
                   edgecolors="white", linewidths=0.4, label="Carry-in")
        ax.scatter(valid_d["abs_x"], valid_d["abs_y"],
                   c=GOLD, s=50, alpha=0.65, zorder=5,
                   edgecolors="black", linewidths=0.4, label="Dump In/Out",
                   marker="^")
        ax.legend(fontsize=9, loc="upper left")
        fig_to_streamlit(fig)

    with col2:
        st.subheader("Entry Type Breakdown")
        import plotly.express as px

        entry_types = entries["detail_1"].value_counts().reset_index()
        fig_e = px.bar(
            entry_types, x="count", y="detail_1",
            orientation="h",
            color_discrete_sequence=[NAVY],
            labels={"detail_1": "Entry Type", "count": "Count"},
            title="Zone Entry — Detail Type",
        )
        fig_e.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_e, use_container_width=True)

    # ── Entry heatmap at blue line ─────────────────────────────────────────────
    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Carry-In Density")
        valid_e2 = entries.dropna(subset=["abs_x", "abs_y"])
        if not valid_e2.empty:
            fig2, _ = draw_heatmap(
                valid_e2["abs_x"].abs(), valid_e2["abs_y"],
                title="Carry Entry Locations", bins=20, cmap="Blues",
                half=True, half_side="right",
            )
            fig_to_streamlit(fig2)
        else:
            st.info("No valid entry coordinates in selected game.")

    with col4:
        st.subheader("Dump-In Density")
        valid_d2 = dumps.dropna(subset=["abs_x", "abs_y"])
        if not valid_d2.empty:
            fig3, _ = draw_heatmap(
                valid_d2["abs_x"].abs(), valid_d2["abs_y"],
                title="Dump Entry Locations", bins=20, cmap="YlOrRd",
                half=True, half_side="right",
            )
            fig_to_streamlit(fig3)
        else:
            st.info("No valid dump coordinates in selected game.")

    # ── What happens after an entry ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("What Happens After Entry? (Next Event)")
    st.caption(
        "After a zone entry, the most valuable outcome is a shot attempt. "
        "This preview shows the event immediately following each entry — "
        "the foundation for the zone entry outcome model (Phase 3)."
    )

    all_sorted = all_events.sort_values(
        ["game_id", "period", "clock_seconds"], ascending=[True, True, False]
    ).reset_index(drop=True)

    entry_indices = all_sorted[
        (all_sorted["game_id"].isin(df["game_id"].unique())) &
        (all_sorted["event"] == "Zone Entry")
    ].index.tolist()

    next_events = []
    for idx in entry_indices:
        if idx + 1 < len(all_sorted):
            next_events.append(all_sorted.loc[idx + 1, "event"])

    if next_events:
        next_df = pd.Series(next_events).value_counts().reset_index()
        next_df.columns = ["Next Event", "Count"]
        fig_next = px.bar(
            next_df, x="Count", y="Next Event",
            orientation="h",
            color="Next Event",
            color_discrete_map=EVENT_COLORS,
            title="Event Immediately After Zone Entry",
        )
        fig_next.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_next, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DECISION IQ
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Decision IQ":
    import plotly.express as px
    from features.xg_model import get_model
    from features.decision_iq import team_iq, player_iq, top_decisions

    st.title("Decision IQ")
    st.caption(
        "Hockey IQ measured as ΔSpatial Value — did each decision move the puck "
        "toward more dangerous space? A logistic regression danger model is trained "
        "from shot locations in this dataset, then every pass, shot, and zone entry "
        "is scored as the change in expected-goal probability it produced."
    )

    # Score all events (cached) then filter for display
    scored_all = get_scored_events()
    scored = (
        scored_all.copy() if selected_game == "All Games"
        else scored_all[scored_all["game_id"] == selected_game].copy()
    )

    # Pre-compute aggregations (always on full dataset for stability)
    team_iq_df   = team_iq(scored_all)
    player_iq_df = player_iq(scored_all)
    best_dec, worst_dec = top_decisions(scored_all, n=5)

    n_scored  = scored["delta_v"].notna().sum()
    avg_dv    = float(scored["delta_v"].mean()) if n_scored > 0 else 0.0
    best_team = team_iq_df.iloc[0]["team"] if not team_iq_df.empty else "—"

    # ── Metrics ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Decisions Scored",      f"{n_scored:,}")
    c2.metric("Avg ΔSpatial Value",    f"{avg_dv:+.4f}")
    c3.metric("Highest Team IQ",       best_team)

    st.markdown("---")

    # ── Section 1: Danger Surface ─────────────────────────────────────────────
    st.subheader("Ice Danger Surface  (Spatial Value)")
    st.caption(
        "Every point on the rink has a Spatial Value xV — the probability a shot "
        "from here results in a goal.  Brighter = more dangerous.  "
        "This surface is the reference for all ΔV calculations below."
    )

    model = get_model(scored_all)
    xs_surf, ys_surf, vals_surf = model.danger_surface(nx=80, ny=40)

    fig_surf, ax_surf = draw_rink(figsize=(13, 5))
    ax_surf.imshow(
        vals_surf,
        origin="lower",
        extent=[-100, 100, -42.5, 42.5],
        cmap="hot",
        alpha=0.60,
        aspect="auto",
        zorder=2,
        interpolation="bilinear",
    )
    ax_surf.set_title(
        "xV(x, y) = P(goal | shot taken from here)  ·  brighter = more dangerous",
        fontsize=9, style="italic",
    )
    fig_to_streamlit(fig_surf)

    st.caption(
        f"Model: logistic regression · trained on **{model.n_shots} shots** "
        f"and **{model.n_goals} goals** · features: distance + angle to nearest goal"
    )

    st.markdown("---")

    # ── Section 2: Decision Quality Map ──────────────────────────────────────
    st.subheader("Pass Decision Quality Map")
    st.caption(
        "Each arrow = one pass. "
        "🟢 Green arrows moved the puck toward danger (positive ΔV). "
        "🔴 Red arrows moved it away from danger (negative ΔV). "
        "Yellow = roughly neutral."
    )

    pass_dec = scored[
        (scored["decision_type"] == "Pass") &
        scored["delta_v"].notna() &
        scored["abs_x"].notna() &
        scored["abs_x2"].notna()
    ].copy()

    max_arrows_diq = st.sidebar.slider(
        "Max decision arrows", 50, 400, 150, step=50, key="diq_arrows"
    )
    if len(pass_dec) > max_arrows_diq:
        pass_dec = pass_dec.sample(max_arrows_diq, random_state=42)

    fig_dmap, ax_dmap = draw_rink(figsize=(13, 5))

    norm_dv = TwoSlopeNorm(vmin=-0.12, vcenter=0.0, vmax=0.15)
    cmap_dv = cm.RdYlGn

    for _, row in pass_dec.iterrows():
        color = cmap_dv(norm_dv(float(row["delta_v"])))
        ax_dmap.annotate(
            "",
            xy=(row["abs_x2"], row["abs_y2"]),
            xytext=(row["abs_x"], row["abs_y"]),
            arrowprops=dict(
                arrowstyle="->", color=color, alpha=0.50, lw=0.9,
            ),
        )
    ax_dmap.set_title(
        f"n = {len(pass_dec)} pass decisions sampled",
        fontsize=9, style="italic",
    )
    fig_to_streamlit(fig_dmap)

    st.markdown("---")

    # ── Section 3: Team IQ ────────────────────────────────────────────────────
    st.subheader("Team Decision IQ")
    st.caption("Primary view — always computed across all three games for statistical stability.")

    col_t1, col_t2 = st.columns([1, 2])

    with col_t1:
        fig_team = px.bar(
            team_iq_df,
            x="decision_iq",
            y="team",
            orientation="h",
            color="decision_iq",
            color_continuous_scale=[[0.0, "#C00000"], [0.5, "#5B9BD5"], [1.0, "#1A3A5C"]],
            text=team_iq_df["decision_iq"].round(1).astype(str),
            labels={"decision_iq": "Decision IQ (0–100)", "team": "Team"},
            title="Overall Decision IQ",
        )
        fig_team.update_traces(textposition="outside")
        fig_team.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, showlegend=False, height=360,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_team, use_container_width=True)

    with col_t2:
        breakdown_cols = [c for c in ["pass_dv", "shot_dv", "entry_dv"]
                          if c in team_iq_df.columns]
        if breakdown_cols:
            melt_df = team_iq_df[["team"] + breakdown_cols].melt(
                id_vars="team", var_name="Type", value_name="Mean ΔV"
            )
            melt_df["Type"] = melt_df["Type"].map({
                "pass_dv":  "Pass",
                "shot_dv":  "Shot",
                "entry_dv": "Zone Entry",
            })
            fig_breakdown = px.bar(
                melt_df, x="team", y="Mean ΔV", color="Type",
                barmode="group",
                color_discrete_map={
                    "Pass": BLUE, "Shot": GOLD, "Zone Entry": "#27AE60",
                },
                labels={"team": "Team", "Mean ΔV": "Mean ΔV"},
                title="Decision IQ Breakdown by Type",
            )
            fig_breakdown.add_hline(
                y=0, line_dash="dash", line_color="white",
                opacity=0.4, annotation_text="neutral",
            )
            fig_breakdown.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=360,
            )
            st.plotly_chart(fig_breakdown, use_container_width=True)

    st.markdown("---")

    # ── Section 4: Player IQ ─────────────────────────────────────────────────
    st.subheader("Player Decision IQ")
    st.info(
        "⚠️ **Small sample caveat** — 3 games yields ~15–25 decisions per player. "
        "Treat individual scores as directional, not definitive.  "
        "Bubble size = number of decisions scored."
    )

    player_iq_df["label"] = "P-" + player_iq_df["player_id"].astype(str)

    fig_player = px.scatter(
        player_iq_df,
        x="n_decisions",
        y="decision_iq",
        color="team",
        size="n_decisions",
        size_max=28,
        text="label",
        hover_data={
            "label": True,
            "mean_dv": ":.4f",
            "n_decisions": True,
            "team": True,
        },
        labels={
            "n_decisions": "Decisions Scored (sample size)",
            "decision_iq": "Decision IQ (0–100)",
            "team": "Team",
        },
        title="Player Decision IQ vs Sample Size",
    )
    fig_player.update_traces(textposition="top center", textfont_size=8)
    fig_player.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=440,
    )
    st.plotly_chart(fig_player, use_container_width=True)

    st.markdown("---")

    # ── Section 5: Best & Worst Decisions ────────────────────────────────────
    st.subheader("Best & Worst Individual Decisions")
    st.caption("Ranked by ΔSpatial Value — how much danger a single decision created or surrendered.")

    def _fmt_decision_table(dec_df: pd.DataFrame) -> pd.DataFrame:
        out = dec_df[[
            "player_id", "team", "decision_type", "delta_v", "game_id"
        ]].copy()
        out["player_id"]  = "P-" + out["player_id"].astype(str)
        out["delta_v"]    = out["delta_v"].round(4)
        out.columns       = ["Player", "Team", "Decision", "ΔV", "Game"]
        return out

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("**🟢 Top 5 — Highest ΔV (Best Decisions)**")
        st.dataframe(_fmt_decision_table(best_dec), hide_index=True)
    with col_b2:
        st.markdown("**🔴 Bottom 5 — Lowest ΔV (Poorest Decisions)**")
        st.dataframe(_fmt_decision_table(worst_dec), hide_index=True)
