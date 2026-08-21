"""Backtest engine for the faithful Top-Down Trendline strategy (trendline_v2_strategy.py).

Daily bias (from a Daily convex-hull chain) filters direction; the 1H convex-hull
chain supplies the Action Line (entry) and Safety Line (trailing stop). Lines are
refit once per UTC day ("daily maintenance"), matching the source strategy, and are
projected forward bar-by-bar in between (a continuously trailing stop from a
daily-refit line). Same fee/funding/capital-policy model as backtest.py for comparability.
"""

import argparse
from typing import Dict, Optional
import numpy as np
import pandas as pd

from backtest import load_dataset, print_summary
from risk import apply_capital_policy
from trendline_v2_strategy import (
    resample_ohlc,
    active_support_line,
    active_resistance_line,
    project_line,
    size_position_by_stop_distance,
)


def run_trendline_v2_backtest(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    base_capital: float = 100.0,
    capital_policy: str = "skim_refill",
    taker_fee_rate: float = 0.00055,
    n_1h_window: int = 500,
    n_daily_window: int = 365,
    min_span_1h: float = 24.0,
    min_span_daily: float = 5.0,
    risk_frac: float = 0.015,
    max_leverage: float = 5.0,
) -> Dict:
    data = df.copy()
    if start_date:
        data = data.loc[start_date:]
    if end_date:
        data = data.loc[:end_date]

    # Daily series is resampled from the FULL history (so early bars in `data`
    # still have real daily context), then we slice to what's usable.
    daily_full = resample_ohlc(df, "1D")
    daily_full["x"] = np.arange(len(daily_full))
    daily_low = daily_full["low"].to_numpy()
    daily_high = daily_full["high"].to_numpy()
    daily_close = daily_full["close"].to_numpy()
    daily_x = daily_full["x"].to_numpy()
    daily_dates = daily_full.index

    hourly_x = np.arange(len(df))  # global position, consistent across the full series
    hourly_low = df["low"].to_numpy()
    hourly_high = df["high"].to_numpy()
    # map data's rows back to their global position in df
    start_pos = df.index.get_indexer([data.index[0]])[0]

    float_usdt = base_capital
    bank_usdt = 0.0
    pos_side: Optional[str] = None
    pos_qty = 0.0
    pos_entry_price = 0.0
    pos_entry_time = None
    cum_fees = 0.0
    cum_funding = 0.0

    trades = []
    equity_curve = []

    current_date = None
    support_1h = None
    resistance_1h = None
    daily_bias = None

    for offset, (ts, row) in enumerate(data.iterrows()):
        k = start_pos + offset
        close_p = row["close"]
        high_p = row["high"]
        low_p = row["low"]
        funding_rate = row["fundingRate"]
        today = ts.normalize()

        # --- Daily maintenance: refit chains once per UTC day, causal only ---
        if today != current_date:
            current_date = today

            # 1H chain: trailing window strictly before today's first bar (k)
            start_1h = max(0, k - n_1h_window)
            if k - start_1h >= 2:
                support_1h = active_support_line(hourly_x[start_1h:k], hourly_low[start_1h:k], min_span_bars=min_span_1h)
                resistance_1h = active_resistance_line(hourly_x[start_1h:k], hourly_high[start_1h:k], min_span_bars=min_span_1h)
            else:
                support_1h = None
                resistance_1h = None

            # Daily chain: trailing window strictly before today
            d_end = int(daily_dates.searchsorted(today))
            d_start = max(0, d_end - n_daily_window)
            if d_end - d_start >= 2:
                d_sup = active_support_line(daily_x[d_start:d_end], daily_low[d_start:d_end], min_span_bars=min_span_daily)
                d_res = active_resistance_line(daily_x[d_start:d_end], daily_high[d_start:d_end], min_span_bars=min_span_daily)
                last_x = daily_x[d_end - 1]
                last_close = daily_close[d_end - 1]
                sup_val = project_line(d_sup, last_x)
                res_val = project_line(d_res, last_x)
                if np.isfinite(sup_val) and np.isfinite(res_val):
                    daily_bias = "long" if abs(last_close - sup_val) < abs(last_close - res_val) else "short"
                elif np.isfinite(sup_val):
                    daily_bias = "long"
                elif np.isfinite(res_val):
                    daily_bias = "short"
                else:
                    daily_bias = None
            else:
                daily_bias = None

        if pos_side is not None and funding_rate != 0.0:
            pos_notional = pos_qty * close_p
            f_cost = pos_notional * funding_rate if pos_side == "Buy" else -pos_notional * funding_rate
            float_usdt -= f_cost
            cum_funding += f_cost

        # --- Exit: safety line (opposite of the entry action line), trailed daily ---
        stopped_out = False
        exit_price = 0.0
        exit_reason = None
        if pos_side == "Buy":
            stop_now = project_line(support_1h, k)
            if np.isfinite(stop_now) and low_p <= stop_now:
                stopped_out = True
                exit_price = stop_now
                exit_reason = "safety_line"
        elif pos_side == "Sell":
            stop_now = project_line(resistance_1h, k)
            if np.isfinite(stop_now) and high_p >= stop_now:
                stopped_out = True
                exit_price = stop_now
                exit_reason = "safety_line"

        if pos_side is not None and stopped_out:
            raw_pnl = pos_qty * (exit_price - pos_entry_price) if pos_side == "Buy" else pos_qty * (pos_entry_price - exit_price)
            exit_fee = pos_qty * exit_price * taker_fee_rate
            cum_fees += exit_fee
            net_trade_pnl = raw_pnl - exit_fee
            float_usdt += net_trade_pnl

            if capital_policy == "skim_refill":
                float_usdt, bank_usdt = apply_capital_policy(float_usdt, bank_usdt, base_capital=base_capital, policy="skim_refill")

            trades.append({
                "entry_time": str(pos_entry_time), "exit_time": str(ts), "side": pos_side, "qty": pos_qty,
                "entry_price": pos_entry_price, "exit_price": exit_price, "raw_pnl": raw_pnl,
                "net_pnl": net_trade_pnl, "exit_reason": exit_reason, "float_after": float_usdt, "bank_after": bank_usdt,
            })
            pos_side = None
            pos_qty = 0.0
            pos_entry_price = 0.0
            pos_entry_time = None

        # --- Entry: Action Line break, filtered by Daily bias ---
        if pos_side is None and daily_bias is not None:
            res_val = project_line(resistance_1h, k)
            sup_val = project_line(support_1h, k)
            sig = None
            if daily_bias == "long" and np.isfinite(res_val) and close_p > res_val:
                sig = "Buy"
            elif daily_bias == "short" and np.isfinite(sup_val) and close_p < sup_val:
                sig = "Sell"

            if sig is not None:
                safety_val = sup_val if sig == "Buy" else res_val
                stop_distance = (close_p - safety_val) if sig == "Buy" else (safety_val - close_p)
                qty, valid, reason = size_position_by_stop_distance(
                    equity_usdt=float_usdt, close_price=close_p, stop_distance=stop_distance,
                    risk_frac=risk_frac, max_leverage=max_leverage,
                )
                if valid:
                    pos_side = sig
                    pos_qty = qty
                    pos_entry_price = close_p
                    pos_entry_time = ts
                    entry_fee = pos_qty * pos_entry_price * taker_fee_rate
                    float_usdt -= entry_fee
                    cum_fees += entry_fee

        total_equity = float_usdt + bank_usdt + (
            pos_qty * (close_p - pos_entry_price) if pos_side == "Buy"
            else pos_qty * (pos_entry_price - close_p) if pos_side == "Sell"
            else 0.0
        )
        equity_curve.append({"timestamp": ts, "float": float_usdt, "bank": bank_usdt, "total_equity": total_equity})

    eq_df = pd.DataFrame(equity_curve).set_index("timestamp")
    trades_df = pd.DataFrame(trades)

    total_return_pct = ((eq_df["total_equity"].iloc[-1] / base_capital) - 1.0) * 100.0 if not eq_df.empty else 0.0
    peak = eq_df["total_equity"].cummax()
    drawdown = (eq_df["total_equity"] - peak) / peak
    max_dd_pct = drawdown.min() * 100.0 if not drawdown.empty else 0.0
    win_trades = trades_df[trades_df["net_pnl"] > 0] if not trades_df.empty else pd.DataFrame()
    win_rate = (len(win_trades) / len(trades_df)) * 100.0 if not trades_df.empty else 0.0

    return {
        "capital_policy": capital_policy,
        "start_date": str(data.index[0]), "end_date": str(data.index[-1]),
        "initial_capital": base_capital, "final_float": float_usdt, "final_bank": bank_usdt,
        "final_total": float_usdt + bank_usdt, "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct, "total_trades": len(trades_df), "win_rate_pct": win_rate,
        "cum_fees": cum_fees, "cum_funding": cum_funding, "trades": trades_df, "equity_curve": eq_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Top-down trendline (v2, faithful) backtest")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--policy", choices=["skim_refill", "compound"], default="skim_refill")
    parser.add_argument("--n-1h-window", type=int, default=500)
    parser.add_argument("--n-daily-window", type=int, default=365)
    parser.add_argument("--risk-frac", type=float, default=0.015)
    args = parser.parse_args()

    df = load_dataset()
    res = run_trendline_v2_backtest(
        df, start_date=args.start, end_date=args.end, capital_policy=args.policy,
        n_1h_window=args.n_1h_window, n_daily_window=args.n_daily_window, risk_frac=args.risk_frac,
    )
    print_summary(res)


if __name__ == "__main__":
    main()
