"""Prototype diagonal-trendline strategy for rapid_bot, built for comparison against
the production horizontal-channel (Donchian) breakout strategy in strategy.py.

NO I/O. Mirrors strategy.py's structure so backtest_trendline.py can reuse the same
position sizing / ATR stop / fee model, isolating the entry-signal logic as the only
real difference between the two bots.

Pivot detection and trendline fitting:
  - A pivot high at bar i is confirmed once `window` bars have closed on both sides
    (fractal high). It becomes KNOWN to the strategy only at bar i + window (no lookahead).
  - The resistance line is fit through the last `n_points` confirmed pivot highs;
    the support line through the last `n_points` confirmed pivot lows.
  - Signal mirrors evaluate_signal(): while flat, close breaking above the projected
    resistance line = long; close breaking below the projected support line = short.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def compute_pivots(high: pd.Series, low: pd.Series, window: int = 24) -> Tuple[pd.Series, pd.Series]:
    """Fractal pivot detection. A pivot at bar i is the max/min over [i-window, i+window].

    Returns two boolean Series, TRUE at bar i if i is a raw pivot (not yet confirmed —
    caller must shift by `window` to know it without lookahead).
    """
    full_window = 2 * window + 1
    roll_max = high.rolling(window=full_window, center=True).max()
    roll_min = low.rolling(window=full_window, center=True).min()
    pivot_high = high == roll_max
    pivot_low = low == roll_min
    return pivot_high.fillna(False), pivot_low.fillna(False)


def compute_trendlines(
    df: pd.DataFrame,
    pivot_window: int = 24,
    n_points: int = 2,
) -> pd.DataFrame:
    """Compute resistance/support trendlines, confirmed without lookahead.

    resistance_line[k] = value of the line fit through the last n_points confirmed
                          pivot highs, projected (extrapolated) to bar k's x-position.
    support_line[k]    = same, using confirmed pivot lows.
    Both are NaN until n_points pivots of that type have been confirmed.
    """
    res = df.copy()
    n = len(res)
    high = res["high"].to_numpy(dtype=float)
    low = res["low"].to_numpy(dtype=float)

    raw_pivot_high, raw_pivot_low = compute_pivots(res["high"], res["low"], window=pivot_window)
    raw_pivot_high = raw_pivot_high.to_numpy()
    raw_pivot_low = raw_pivot_low.to_numpy()

    resistance_line = np.full(n, np.nan, dtype=float)
    support_line = np.full(n, np.nan, dtype=float)

    recent_highs: list = []  # list of (x, price), most recent last
    recent_lows: list = []

    for k in range(n):
        # A pivot raw-flagged at index (k - pivot_window) is confirmed as of bar k.
        confirm_idx = k - pivot_window
        if confirm_idx >= 0:
            if raw_pivot_high[confirm_idx]:
                recent_highs.append((confirm_idx, high[confirm_idx]))
                if len(recent_highs) > n_points:
                    recent_highs.pop(0)
            if raw_pivot_low[confirm_idx]:
                recent_lows.append((confirm_idx, low[confirm_idx]))
                if len(recent_lows) > n_points:
                    recent_lows.pop(0)

        if len(recent_highs) >= n_points:
            xs = np.array([p[0] for p in recent_highs], dtype=float)
            ys = np.array([p[1] for p in recent_highs], dtype=float)
            if xs[-1] != xs[0]:
                slope, intercept = np.polyfit(xs, ys, 1)
                resistance_line[k] = slope * k + intercept
            else:
                resistance_line[k] = ys[-1]

        if len(recent_lows) >= n_points:
            xs = np.array([p[0] for p in recent_lows], dtype=float)
            ys = np.array([p[1] for p in recent_lows], dtype=float)
            if xs[-1] != xs[0]:
                slope, intercept = np.polyfit(xs, ys, 1)
                support_line[k] = slope * k + intercept
            else:
                support_line[k] = ys[-1]

    res["resistance_line"] = resistance_line
    res["support_line"] = support_line
    return res


def evaluate_trendline_signal(
    close_val: float,
    resistance_line_val: float,
    support_line_val: float,
    mom30_val: Optional[float] = None,
    allow_long: bool = True,
    allow_short: bool = True,
    require_momentum_confirmation: bool = False,
) -> Optional[str]:
    """Entry signal on a closed bar (flat only, checked by caller).

    long_signal  = close[k] > resistance_line[k]   (breaks above descending/resistance line)
    short_signal = close[k] < support_line[k]       (breaks below ascending/support line)
    Optional momentum filter mirrors the channel-bot's trend_veto behavior.
    """
    if pd.isna(resistance_line_val) or pd.isna(support_line_val):
        return None

    long_sig = close_val > resistance_line_val
    short_sig = close_val < support_line_val

    if require_momentum_confirmation and mom30_val is not None and not pd.isna(mom30_val):
        long_sig = long_sig and (mom30_val >= 0)
        short_sig = short_sig and (mom30_val <= 0)

    if long_sig and allow_long:
        return "Buy"
    elif short_sig and allow_short:
        return "Sell"
    return None


def check_trendline_exit(
    side: str,
    close_price: float,
    resistance_line: float,
    support_line: float,
) -> bool:
    """Exit when price falls back through the opposite trendline (failed breakout),
    mirroring check_channel_exit()'s role in strategy.py.

    long  exits when close[k] < support_line[k]
    short exits when close[k] > resistance_line[k]
    """
    if side in ("Buy", "long", "Long"):
        if pd.isna(support_line):
            return False
        return close_price < support_line
    elif side in ("Sell", "short", "Short"):
        if pd.isna(resistance_line):
            return False
        return close_price > resistance_line
    return False
