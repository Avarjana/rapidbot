"""ICT (Inner Circle Trader) concept filters, tested as optional confluence gates
layered on top of the EXISTING production channel-breakout signal — not a new
strategy. NO I/O. Does not modify strategy.py/backtest.py/main.py.

Only the mechanically well-defined ICT concepts are implemented (order blocks are
left out — "valid displacement" has no objective definition and would repeat the
overfitting risk we already found with the trendline prototype):

  - Kill zones: session-time filter (London / New York hours, UTC).
  - Fair Value Gaps (FVG): a precise 3-candle imbalance definition.
  - Liquidity sweeps: a wick through a recent swing point that closes back inside
    (the "stop hunt then reversal" pattern), all causal / no lookahead.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

# Commonly cited ICT kill zone windows (UTC). Documented here, not tuned.
LONDON_KILLZONE = (7, 10)   # 07:00-10:00 UTC
NEWYORK_KILLZONE = (12, 15)  # 12:00-15:00 UTC


def compute_killzone_mask(index: pd.DatetimeIndex) -> np.ndarray:
    hours = index.hour
    in_london = (hours >= LONDON_KILLZONE[0]) & (hours < LONDON_KILLZONE[1])
    in_ny = (hours >= NEWYORK_KILLZONE[0]) & (hours < NEWYORK_KILLZONE[1])
    return np.asarray(in_london | in_ny)


def compute_fvg_events(df: pd.DataFrame, fill_horizon_bars: int = 720) -> pd.DataFrame:
    """Bullish FVG at bar i: low[i] > high[i-2]   -> zone (high[i-2], low[i])
    Bearish FVG at bar i: high[i] < low[i-2]      -> zone (high[i], low[i-2])
    fill_bar = first later bar whose range re-enters the zone (capped at
    fill_horizon_bars ahead; treated as unfilled beyond that for our purposes).
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    events = []
    for i in range(2, n):
        if low[i] > high[i - 2]:
            events.append((i, "bull", high[i - 2], low[i]))
        elif high[i] < low[i - 2]:
            events.append((i, "bear", high[i], low[i - 2]))

    out = []
    for formed_at, direction, zlo, zhi in events:
        fill_bar = None
        end = min(n, formed_at + 1 + fill_horizon_bars)
        for j in range(formed_at + 1, end):
            if low[j] <= zhi and high[j] >= zlo:
                fill_bar = j
                break
        out.append({"formed_at": formed_at, "direction": direction, "zone_low": zlo, "zone_high": zhi, "fill_bar": fill_bar})
    return pd.DataFrame(out)


def fvg_confluence_mask(n: int, events: pd.DataFrame, lookback_bars: int, direction: str) -> np.ndarray:
    """True at bar k if an unfilled FVG of `direction` ('bull'/'bear') formed
    within [k - lookback_bars, k].
    """
    mask = np.zeros(n, dtype=bool)
    sub = events[events["direction"] == direction]
    if sub.empty:
        return mask
    for _, ev in sub.iterrows():
        formed_at = int(ev["formed_at"])
        fill_bar = ev["fill_bar"]
        fill_bar = int(fill_bar) if pd.notna(fill_bar) else n
        active_end = min(n, fill_bar)
        window_end = min(n, formed_at + lookback_bars + 1)
        lo = formed_at
        hi = min(active_end, window_end)
        if hi > lo:
            mask[lo:hi] = True
    return mask


def compute_swing_points(df: pd.DataFrame, window: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Confirmed (causal) swing high/low flags, same fractal technique used
    elsewhere in this repo's trendline prototypes. Confirmed `window` bars later.
    """
    full_window = 2 * window + 1
    roll_max = df["high"].rolling(window=full_window, center=True).max()
    roll_min = df["low"].rolling(window=full_window, center=True).min()
    raw_swing_high = (df["high"] == roll_max).fillna(False).to_numpy()
    raw_swing_low = (df["low"] == roll_min).fillna(False).to_numpy()
    return raw_swing_high, raw_swing_low


def compute_sweep_events(df: pd.DataFrame, window: int = 12) -> List[Tuple[int, str]]:
    """Liquidity sweep: price wicks beyond the most recent confirmed swing
    high/low and closes back inside it. Returns [(bar_idx, 'sweep_low'|'sweep_high'), ...].
    """
    raw_swing_high, raw_swing_low = compute_swing_points(df, window=window)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    events = []
    last_swing_low = None
    last_swing_high = None
    for k in range(n):
        confirm_idx = k - window
        if confirm_idx >= 0:
            if raw_swing_low[confirm_idx]:
                last_swing_low = low[confirm_idx]
            if raw_swing_high[confirm_idx]:
                last_swing_high = high[confirm_idx]

        if last_swing_low is not None and low[k] < last_swing_low and close[k] > last_swing_low:
            events.append((k, "sweep_low"))
        if last_swing_high is not None and high[k] > last_swing_high and close[k] < last_swing_high:
            events.append((k, "sweep_high"))
    return events


def sweep_confluence_mask(n: int, events: List[Tuple[int, str]], lookback_bars: int, kind: str) -> np.ndarray:
    """True at bar k if a sweep of `kind` ('sweep_low'/'sweep_high') occurred
    within [k - lookback_bars, k]."""
    mask = np.zeros(n, dtype=bool)
    idxs = [i for i, k in events if k == kind]
    for i in idxs:
        lo = i
        hi = min(n, i + lookback_bars + 1)
        mask[lo:hi] = True
    return mask
