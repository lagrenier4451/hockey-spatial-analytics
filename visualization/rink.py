"""
visualization/rink.py

Draws a to-scale NHL ice rink using matplotlib patches and lines.
All measurements are in feet using the tracking coordinate system:
  - Origin at centre ice
  - X: -100 (left goal line) to +100 (right goal line)
  - Y: -42.5 (bottom board) to +42.5 (top board)

Usage
-----
    from visualization.rink import draw_rink
    fig, ax = draw_rink()
    ax.scatter(x_positions, y_positions)
    plt.show()

    # Half-rink (offensive zone only):
    fig, ax = draw_rink(half=True)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Circle, FancyArrowPatch
import numpy as np

# ── Rink colour palette ───────────────────────────────────────────────────────
ICE         = "#FFFFFF"
BOARDS      = "#000000"
GOAL_LINE   = "#FF0000"
BLUE_LINE   = "#0000FF"
RED_LINE    = "#FF0000"
CREASE      = "#CCE5FF"
CIRCLE      = "#FF0000"
GOAL_FILL   = "#C0C0C0"
DOT         = "#FF0000"


def _add_faceoff_circle(ax, cx, cy, radius=15, dot_radius=0.5):
    """Draw a faceoff circle with centre dot."""
    circle = Circle((cx, cy), radius, fill=False,
                    edgecolor=CIRCLE, linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    dot = Circle((cx, cy), dot_radius, color=DOT, zorder=4)
    ax.add_patch(dot)


def _add_goal(ax, goal_x, facing_right=True):
    """
    Draw a goal (net) at goal_x.
    The net extends 2 feet behind the goal line, 6 feet wide.
    """
    net_depth = 2
    net_width = 3   # half-width from centre
    dx = net_depth if not facing_right else -net_depth

    goal = plt.Polygon(
        [
            (goal_x,      -net_width),
            (goal_x + dx, -net_width),
            (goal_x + dx,  net_width),
            (goal_x,       net_width),
        ],
        closed=True, fill=True,
        facecolor=GOAL_FILL, edgecolor=BOARDS, linewidth=1.5, zorder=3,
    )
    ax.add_patch(goal)


def _add_crease(ax, goal_x, facing_right=True):
    """Draw the crease arc in front of each goal."""
    crease_radius = 6
    # The crease is a semicircle in front of the goal line
    theta1, theta2 = (0, 180) if facing_right else (180, 360)
    crease = Arc(
        (goal_x, 0), crease_radius * 2, crease_radius * 2,
        angle=0, theta1=theta1, theta2=theta2,
        color=GOAL_LINE, linewidth=1.5, zorder=3,
    )
    ax.add_patch(crease)
    # Fill the crease
    angles = np.linspace(np.radians(theta1), np.radians(theta2), 100)
    xs = goal_x + crease_radius * np.cos(angles)
    ys = crease_radius * np.sin(angles)
    xs = np.append(xs, goal_x)
    ys = np.append(ys, 0)
    ax.fill(xs, ys, color=CREASE, alpha=0.5, zorder=2)


def draw_rink(
    ax=None,
    half: bool = False,
    half_side: str = "right",
    figsize: tuple = (14, 6),
    title: str = None,
) -> tuple:
    """
    Draw a full or half NHL rink.

    Parameters
    ----------
    ax        : existing matplotlib Axes to draw on (creates new fig if None)
    half      : if True, draw only one half of the rink
    half_side : 'right' or 'left' — which half to show
    figsize   : figure size in inches (only used when ax is None)
    title     : optional chart title

    Returns
    -------
    (fig, ax) — the matplotlib Figure and Axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ── Ice surface ───────────────────────────────────────────────────────────
    rink = mpatches.FancyBboxPatch(
        (-100, -42.5), 200, 85,
        boxstyle="round,pad=0,rounding_size=28",
        facecolor=ICE, edgecolor=BOARDS, linewidth=3, zorder=1,
    )
    ax.add_patch(rink)

    # ── Centre line (red) ─────────────────────────────────────────────────────
    ax.plot([0, 0], [-42.5, 42.5], color=RED_LINE, linewidth=3, zorder=3)

    # ── Centre faceoff dot ────────────────────────────────────────────────────
    ax.add_patch(Circle((0, 0), 0.75, color=DOT, zorder=4))
    ax.add_patch(Circle((0, 0), 15, fill=False,
                         edgecolor=CIRCLE, linewidth=1.5, zorder=3))

    # ── Blue lines ────────────────────────────────────────────────────────────
    for x in [-25, 25]:
        ax.plot([x, x], [-42.5, 42.5], color=BLUE_LINE, linewidth=4, zorder=3)

    # ── Goal lines ────────────────────────────────────────────────────────────
    for x in [-89, 89]:
        ax.plot([x, x], [-42.5, 42.5], color=GOAL_LINE, linewidth=2, zorder=3)

    # ── Goals + creases ───────────────────────────────────────────────────────
    _add_goal(ax,  89, facing_right=False)
    _add_goal(ax, -89, facing_right=True)
    _add_crease(ax,  89, facing_right=False)
    _add_crease(ax, -89, facing_right=True)

    # ── Faceoff circles (4 in zone, 2 in neutral) ─────────────────────────────
    for side in [-1, 1]:   # left/right side
        # Offensive zone circles
        for zone_x in [-69, 69]:
            _add_faceoff_circle(ax, zone_x, side * 22)
        # Neutral zone dots (no circle, just dot)
        for zone_x in [-20, 20]:
            ax.add_patch(Circle((zone_x, side * 22), 0.75, color=DOT, zorder=4))

    # ── Referee crease (centre ice, bottom) ───────────────────────────────────
    ref = Arc((0, -42.5), 20, 20, angle=0, theta1=0, theta2=180,
              color=RED_LINE, linewidth=1, zorder=3, linestyle="--")
    ax.add_patch(ref)

    # ── Axes formatting ───────────────────────────────────────────────────────
    if half:
        if half_side == "right":
            ax.set_xlim(-5, 105)
        else:
            ax.set_xlim(-105, 5)
    else:
        ax.set_xlim(-105, 105)

    ax.set_ylim(-47, 47)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    return fig, ax


def draw_event_scatter(
    xs, ys,
    colors=None,
    sizes=None,
    alpha=0.6,
    title: str = None,
    half: bool = False,
    ax=None,
    label: str = None,
) -> tuple:
    """
    Draw a scatter of events on a rink.

    Parameters
    ----------
    xs, ys   : event coordinates (tracking system: -100 to 100, -42.5 to 42.5)
    colors   : colour array or single colour
    sizes    : marker size array or single int
    alpha    : transparency
    title    : chart title
    half     : show half rink
    """
    fig, ax = draw_rink(ax=ax, half=half, title=title)
    ax.scatter(
        xs, ys,
        c=colors if colors is not None else "#1A3A5C",
        s=sizes  if sizes  is not None else 40,
        alpha=alpha, zorder=5, edgecolors="white", linewidths=0.4,
        label=label,
    )
    if label:
        ax.legend(loc="upper right", fontsize=9)
    return fig, ax


def draw_heatmap(
    xs, ys,
    title: str = None,
    half: bool = False,
    half_side: str = "right",
    bins: int = 40,
    cmap: str = "hot",
    ax=None,
) -> tuple:
    """
    Draw a 2D density heatmap of events on a rink.

    Uses a 2D histogram to count events per grid cell, then overlays the
    rink lines on top.  The hottest cells show where events are most dense.
    """
    fig, ax = draw_rink(ax=ax, half=half, half_side=half_side, title=title)

    if len(xs) == 0:
        return fig, ax

    # Draw heatmap UNDER rink markings — set zorder=1
    h, xedges, yedges = np.histogram2d(
        xs, ys,
        bins=bins,
        range=[[-100, 100], [-42.5, 42.5]],
    )
    h = h.T   # transpose so rows=Y, cols=X (matplotlib imshow convention)

    ax.imshow(
        h,
        origin="lower",
        extent=[-100, 100, -42.5, 42.5],
        cmap=cmap,
        alpha=0.55,
        aspect="auto",
        zorder=2,
        interpolation="bilinear",
    )
    return fig, ax


if __name__ == "__main__":
    # Quick test — draw an empty rink and save it
    fig, ax = draw_rink(title="NHL Rink — Hockey Spatial Analytics")
    out = "data/processed/rink_test.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Rink saved to {out}")
    plt.close(fig)
