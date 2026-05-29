"""
features/xg_model.py

Spatial Value Model — learns an ice-danger surface from shot data.

    Spatial Value  xV(x, y)  =  P(goal | shot taken from this position)

Fit once from all available shot/goal events using logistic regression
on distance-to-goal + angle-to-goal.  Every position on the rink
receives a danger score 0–1 that serves as the currency for Decision IQ.

Usage
-----
    from features.xg_model import get_model
    model = get_model(events_df)           # trains + caches
    v = model.spatial_value(abs_x, abs_y)  # scalar
    xs, ys, grid = model.danger_surface()  # 2-D visualisation grid
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

GOAL_X = 89.0  # absolute goal-line position (feet)


# ── Vectorised geometry ───────────────────────────────────────────────────────
# These mirror features/spatial.py but operate on NumPy arrays for speed.

def _dist_vec(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Euclidean distance to nearest goal (vectorised)."""
    d_right = np.sqrt((xs - GOAL_X) ** 2 + ys ** 2)
    d_left  = np.sqrt((xs + GOAL_X) ** 2 + ys ** 2)
    return np.minimum(d_right, d_left)


def _angle_vec(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Shot angle in degrees relative to nearest goal (vectorised)."""
    use_right = np.abs(xs - GOAL_X) < np.abs(xs + GOAL_X)
    gx  = np.where(use_right, GOAL_X, -GOAL_X)
    dx  = np.abs(xs - gx)
    dy  = np.abs(ys)
    return np.degrees(np.arctan2(dy, dx))


# ── Model ─────────────────────────────────────────────────────────────────────

class SpatialValueModel:
    """
    Logistic regression danger model.

    Spatial Value = P(goal | distance_to_goal, angle_to_goal)

    Parameters are fit from Shot + Goal events. Any (abs_x, abs_y) on the
    rink can then be scored via spatial_value() or score_series().

    Attributes
    ----------
    fitted   : bool — True after fit() is called
    n_shots  : int  — number of shot events used for training
    n_goals  : int  — number of goal events used for training
    """

    def __init__(self):
        self._lr     = LogisticRegression(max_iter=1_000, random_state=42)
        self._scaler = StandardScaler()
        self.fitted  = False
        self.n_shots = 0
        self.n_goals = 0

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, events_df: pd.DataFrame) -> "SpatialValueModel":
        """
        Train the model on shot/goal events.

        Parameters
        ----------
        events_df : DataFrame with columns event, dist_to_goal, angle_to_goal
        """
        shots = (
            events_df[events_df["event"].isin(["Shot", "Goal"])]
            .dropna(subset=["dist_to_goal", "angle_to_goal"])
            .copy()
        )
        shots["is_goal"] = (shots["event"] == "Goal").astype(int)

        X = shots[["dist_to_goal", "angle_to_goal"]].values
        y = shots["is_goal"].values

        self._scaler.fit(X)
        self._lr.fit(self._scaler.transform(X), y)

        self.fitted  = True
        self.n_shots = len(shots)
        self.n_goals = int(y.sum())
        return self

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _predict_batch(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> np.ndarray:
        """Internal: score flat arrays of (x, y) → P(goal)."""
        dists  = _dist_vec(xs.ravel(), ys.ravel())
        angles = _angle_vec(xs.ravel(), ys.ravel())
        X = self._scaler.transform(np.column_stack([dists, angles]))
        return self._lr.predict_proba(X)[:, 1]

    def spatial_value(self, abs_x: float, abs_y: float) -> float:
        """
        Danger probability (0–1) for a single rink position.

        Uses the nearest goal — both ends of the rink are symmetric.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before spatial_value()")
        if pd.isna(abs_x) or pd.isna(abs_y):
            return np.nan
        return float(self._predict_batch(
            np.array([abs_x]), np.array([abs_y])
        )[0])

    def score_series(
        self, xs: pd.Series, ys: pd.Series
    ) -> pd.Series:
        """
        Batch-score two pandas Series of rink coordinates.

        Returns a Series of xV values (NaN where either input is NaN).
        """
        result = pd.Series(np.nan, index=xs.index, dtype=float)
        valid  = xs.notna() & ys.notna()
        if valid.any():
            result[valid] = self._predict_batch(
                xs[valid].values.astype(float),
                ys[valid].values.astype(float),
            )
        return result

    # ── Visualisation surface ─────────────────────────────────────────────────

    def danger_surface(self, nx: int = 80, ny: int = 40) -> tuple:
        """
        Generate a 2-D grid of spatial values for rink visualisation.

        Returns
        -------
        xs   : 1-D array, shape (nx,)  — X coordinates
        ys   : 1-D array, shape (ny,)  — Y coordinates
        vals : 2-D array, shape (ny, nx)  — xV at each grid point
               (rows=Y, cols=X — matplotlib imshow convention)
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before danger_surface()")
        xs   = np.linspace(-100, 100, nx)
        ys   = np.linspace(-42.5, 42.5, ny)
        XX, YY = np.meshgrid(xs, ys)
        vals = self._predict_batch(XX.ravel(), YY.ravel()).reshape(ny, nx)
        return xs, ys, vals


# ── Module-level singleton ────────────────────────────────────────────────────

_MODEL: "SpatialValueModel | None" = None


def get_model(events_df: "pd.DataFrame | None" = None) -> SpatialValueModel:
    """
    Return the cached SpatialValueModel, fitting on first call.

    Parameters
    ----------
    events_df : if provided, (re)fit the model from scratch.
                If None and model already exists, return the cached model.
                If None and no model exists, load events automatically.
    """
    global _MODEL
    if _MODEL is None or events_df is not None:
        if events_df is None:
            from features.spatial import load_all_events
            events_df = load_all_events()
        _MODEL = SpatialValueModel().fit(events_df)
    return _MODEL


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from features.spatial import load_all_events
    events = load_all_events()
    model  = SpatialValueModel().fit(events)
    print(f"Model fit on {model.n_shots} shots, {model.n_goals} goals")
    print(f"xV(slot, centre)     = {model.spatial_value(15, 0):.4f}")
    print(f"xV(point, left wing) = {model.spatial_value(55, 20):.4f}")
    print(f"xV(neutral zone)     = {model.spatial_value(0, 0):.4f}")
    print(f"xV(behind net)       = {model.spatial_value(92, 5):.4f}")
