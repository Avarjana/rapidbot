"""Pure strategy logic for rapid_bot.
NO I/O, NO exchange calls.
Shared identically by backtest.py and main.py to prevent strategy drift.
"""

import math
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def compute_wilder_atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's 14-period Average True Range (ATR).

    TR[i]  = max(high[i] - low[i], |high[i] - close[i-1]|, |low[i] - close[i-1]|)
    ATR[14] = mean(TR[1..14])
    ATR[i]  = (ATR[i-1] * 13 + TR[i]) / 14 for i > 14
    """
    n = len(close)
    if n < period:
        return pd.Series(index=close.index, dtype=float)

    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)

    tr = np.zeros(n, dtype=float)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    atr = np.full(n, np.nan, dtype=float)
    if n >= period:
        # First ATR is simple mean of first `period` TRs (index 0 to period-1)
        atr[period - 1] = np.mean(tr[:period])
        alpha = period - 1
        for i in range(period, n):
            atr[i] = (atr[i - 1] * alpha + tr[i]) / period

    return pd.Series(atr, index=close.index, name="atr")


def compute_indicators(
    df: pd.DataFrame,
    entry_channel_hours: int = 960,
    exit_channel_hours: int = 480,
    mom_hours: int = 720,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Compute indicators with strict no-lookahead guarantee (k-1 upper bound / shift(1)).

    entry_high[k] = max(high[k-960 .. k-1])
    entry_low[k]  = min(low [k-960 .. k-1])
    exit_low[k]   = min(low [k-480 .. k-1])
    exit_high[k]  = max(high[k-480 .. k-1])
    mom30[k]      = close[k] / close[k-720] - 1
    ATR[k]        = Wilder ATR(14)
    """
    res = df.copy()

    # Channels: rolling over previous bars, EXCLUDING bar k via shift(1)
    res["entry_high"] = res["high"].rolling(window=entry_channel_hours).max().shift(1)
    res["entry_low"] = res["low"].rolling(window=entry_channel_hours).min().shift(1)

    res["exit_high"] = res["high"].rolling(window=exit_channel_hours).max().shift(1)
    res["exit_low"] = res["low"].rolling(window=exit_channel_hours).min().shift(1)

    # Momentum 30-day (720 hours)
    if mom_hours > 0:
        res["mom30"] = res["close"] / res["close"].shift(mom_hours) - 1.0
    else:
        res["mom30"] = 0.0

    # Wilder ATR(14) on 1h bars
    res["atr"] = compute_wilder_atr_series(res["high"], res["low"], res["close"], period=atr_period)

    return res


def evaluate_signal(
    close_val: float,
    entry_high_val: float,
    entry_low_val: float,
    mom30_val: float,
    allow_long: bool = True,
    allow_short: bool = True,
) -> Optional[str]:
    """Evaluate entry signal on a closed bar.

    long_signal  = close[k] > entry_high[k] AND mom30[k] >= 0
    short_signal = close[k] < entry_low[k]  AND mom30[k] <= 0
    """
    if pd.isna(entry_high_val) or pd.isna(entry_low_val) or pd.isna(mom30_val):
        return None

    long_sig = (close_val > entry_high_val) and (mom30_val >= 0)
    short_sig = (close_val < entry_low_val) and (mom30_val <= 0)

    if long_sig and allow_long:
        return "Buy"
    elif short_sig and allow_short:
        return "Sell"
    return None


def quantize(value: float, step: float) -> float:
    """Quantize to step size using floor for qty or round to nearest tick for price."""
    if step <= 0:
        return value
    precision = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    # Floor to step
    quantized = math.floor(value / step + 1e-9) * step
    return round(quantized, precision)


def quantize_price(price: float, tick_size: float) -> float:
    """Quantize price to nearest tick size."""
    if tick_size <= 0:
        return price
    precision = max(0, -int(math.floor(math.log10(tick_size)))) if tick_size < 1 else 0
    quantized = round(price / tick_size) * tick_size
    return round(quantized, precision)


def compute_position_size(
    float_usdt: float,
    close_price: float,
    atr_val: float,
    risk_frac: float = 0.05,
    max_leverage: float = 5.0,
    atr_mult: float = 3.0,
    qty_step: float = 0.001,
    min_order_qty: float = 0.001,
    min_notional: float = 5.0,
) -> Tuple[float, float, bool, str]:
    """Calculate position size according to Section 1.3.

    stop_distance = 3.0 * ATR[k]
    qty_risk      = (float_usdt * 0.05) / stop_distance
    qty_lev       = (float_usdt * 5.0)  / close[k]
    qty_raw       = min(qty_risk, qty_lev)
    qty           = floor(qty_raw / qty_step) * qty_step

    Returns: (qty, stop_distance, is_valid, reason)
    """
    if pd.isna(atr_val) or atr_val <= 0 or close_price <= 0 or float_usdt <= 0:
        return 0.0, 0.0, False, "Invalid inputs for sizing"

    stop_distance = atr_mult * atr_val
    qty_risk = (float_usdt * risk_frac) / stop_distance
    qty_lev = (float_usdt * max_leverage) / close_price
    qty_raw = min(qty_risk, qty_lev)

    qty = quantize(qty_raw, qty_step)

    if qty < min_order_qty:
        return qty, stop_distance, False, f"Qty {qty} below min_order_qty {min_order_qty}"

    notional = qty * close_price
    if notional < min_notional:
        return qty, stop_distance, False, f"Notional {notional:.2f} below min_notional {min_notional}"

    return qty, stop_distance, True, "OK"


def compute_stop_price(side: str, entry_price: float, atr_at_entry: float, atr_mult: float = 3.0, tick_size: float = 0.1) -> float:
    """Calculate fixed exchange-side stop loss price at entry.

    long:  stop_price = entry_price - 3.0 * ATR[at entry]
    short: stop_price = entry_price + 3.0 * ATR[at entry]
    """
    stop_distance = atr_mult * atr_at_entry
    if side in ("Buy", "long", "Long"):
        raw_stop = entry_price - stop_distance
    else:
        raw_stop = entry_price + stop_distance

    return quantize_price(raw_stop, tick_size)


def check_channel_exit(
    side: str,
    close_price: float,
    exit_high: float,
    exit_low: float,
) -> bool:
    """Check channel exit condition on closed bar (Section 1.5):

    long  exits when close[k] < exit_low[k]
    short exits when close[k] > exit_high[k]
    """
    if pd.isna(exit_high) or pd.isna(exit_low):
        return False

    if side in ("Buy", "long", "Long"):
        return close_price < exit_low
    elif side in ("Sell", "short", "Short"):
        return close_price > exit_high
    return False
