"""
dashboard/app.py — Hockey Spatial Analytics Dashboard

Four pages built on BDC 2025 tracking event data:
  1. Overview    — event breakdown, rink map, game selector
  2. Shots       — shot scatter, heatmap, distance/angle analysis
  3. Passes      — pass arrows, direct vs indirect, zone flows
  4. Zone Entries — carry vs dump, entry success rates

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


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏒 Hockey Spatial Analytics")
st.sidebar.markdown("**Data:** Big Data Cup 2025 — Stathletes")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Page",
    ["Overview", "Shot Analysis", "Pass Analysis", "Zone Entries"],
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
