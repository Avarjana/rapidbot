"""Faithful(er) implementation of the Tori Trades "Top-Down Trendline" strategy.
NO I/O. Prototype only — not wired into main.py.

Key differences from trendline_strategy.py (v1), which was NOT faithful to the video:
  - Trendlines are built as the LOWER/UPPER CONVEX HULL of price points from an
    anchored extreme. This is the only construction that guarantees "price can
    never poke through the line" — the video's most important structural rule.
    Walking the hull's edges left-to-right IS the "Point B becomes new Point A,
    each line steeper than the last" chaining rule (a documented convexity property).
  - Two-timeframe top-down: a Daily chain sets directional bias (only take the
    1H break that agrees with which daily line — support or resistance — is
    currently nearest to price); the 1H chain supplies the Action Line (entry
    trigger) and Safety Line (trailing stop). Full Monthly/Weekly/4H context from
    the video is NOT implemented — this is a documented scope cut, not an oversight.
  - Lines are recomputed once per UTC day ("daily maintenance" per the video),
    then the frozen slope/intercept is projected forward bar-by-bar, which is
    itself a continuously trailing stop even though the line is only refit daily.
  - Stop distance = distance from entry to the Safety Line at entry, NOT an ATR
    multiple. Risk sizing defaults to 1.5% (video specifies 1-2%).
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

from strategy import quantize, quantize_price


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1h OHLC into a coarser timeframe (e.g. '1D')."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    out = df[["open", "high", "low", "close"]].resample(rule).agg(agg).dropna()
    return out


def _cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def lower_hull_chain(xs: np.ndarray, ys: np.ndarray) -> list:
    """Lower convex hull (support boundary: all points lie on/above the chain).
    xs must be sorted ascending. Returns hull vertices [(x, y), ...] left to right.
    """
    hull: list = []
    for x, y in zip(xs, ys):
        p = (float(x), float(y))
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return hull


def upper_hull_chain(xs: np.ndarray, ys: np.ndarray) -> list:
    """Upper convex hull (resistance boundary: all points lie on/below the chain)."""
    hull: list = []
    for x, y in zip(xs, ys):
        p = (float(x), float(y))
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    return hull


def _last_edge_with_min_span(hull: list, min_span_bars: float) -> Optional[Tuple[float, float, float]]:
    """Walk back from the newest hull vertex to find the last edge spanning at
    least `min_span_bars`. A 1-bar-wide edge (two adjacent candles) isn't a
    trendline a human would draw — it's noise, and its slope, extrapolated
    forward, is usually near-vertical. Returns None if no edge qualifies.
    """
    for i in range(len(hull) - 1, 0, -1):
        x0, y0 = hull[i - 1]
        x1, y1 = hull[i]
        if x1 - x0 >= min_span_bars:
            slope = (y1 - y0) / (x1 - x0)
            intercept = y1 - slope * x1
            return slope, intercept, x1
    return None


def active_support_line(xs: np.ndarray, lows: np.ndarray, min_span_bars: float = 1.0) -> Optional[Tuple[float, float, float]]:
    """Ascending trendline: anchor at the absolute lowest low in the window, chain
    forward via the lower convex hull, return the most recent edge that spans at
    least `min_span_bars` as (slope, intercept, anchor_x).
    """
    if len(xs) < 2:
        return None
    anchor_i = int(np.argmin(lows))
    fx, fy = xs[anchor_i:], lows[anchor_i:]
    if len(fx) < 2:
        return None
    hull = lower_hull_chain(fx, fy)
    return _last_edge_with_min_span(hull, min_span_bars)


def active_resistance_line(xs: np.ndarray, highs: np.ndarray, min_span_bars: float = 1.0) -> Optional[Tuple[float, float, float]]:
    """Descending trendline: anchor at the absolute highest high, chain forward via
    the upper convex hull, return the most recent edge spanning at least
    `min_span_bars` as (slope, intercept, anchor_x).
    """
    if len(xs) < 2:
        return None
    anchor_i = int(np.argmax(highs))
    fx, fy = xs[anchor_i:], highs[anchor_i:]
    if len(fx) < 2:
        return None
    hull = upper_hull_chain(fx, fy)
    return _last_edge_with_min_span(hull, min_span_bars)


def project_line(line: Optional[Tuple[float, float, float]], x: float) -> float:
    if line is None:
        return float("nan")
    slope, intercept, _anchor_x = line
    return slope * x + intercept


def size_position_by_stop_distance(
    equity_usdt: float,
    close_price: float,
    stop_distance: float,
    risk_frac: float = 0.015,
    max_leverage: float = 5.0,
    qty_step: float = 0.001,
    min_order_qty: float = 0.001,
    min_notional: float = 5.0,
) -> Tuple[float, bool, str]:
    """Position sizing off the Safety Line's distance rather than ATR (video's
    stop is 'just beyond the safety line', not an indicator-derived stop).
    """
    if stop_distance is None or not np.isfinite(stop_distance) or stop_distance <= 0 or close_price <= 0 or equity_usdt <= 0:
        return 0.0, False, "Invalid stop distance"

    qty_risk = (equity_usdt * risk_frac) / stop_distance
    qty_lev = (equity_usdt * max_leverage) / close_price
    qty_raw = min(qty_risk, qty_lev)
    qty = quantize(qty_raw, qty_step)

    if qty < min_order_qty:
        return qty, False, f"Qty {qty} below min_order_qty {min_order_qty}"
    notional = qty * close_price
    if notional < min_notional:
        return qty, False, f"Notional {notional:.2f} below min_notional {min_notional}"
    return qty, True, "OK"
