"""Backtest replacing the entry threshold (entry_high/entry_low, the single
40-day extreme) with the AVERAGE of the top-N highs / bottom-N lows within the
same 40-day window -- smooths out a single anomalous wick's influence while
still tracking genuine resistance/support structure. N=1 reproduces the
production single-extreme threshold exactly.

Everything else is the currently-deployed exit stack, untouched: entries'
momentum filter, position sizing, initial ATR stop, channel exit ("bottom
bar", unchanged single 20-day extreme), 2R breakeven, 3R->1R profit lock.
"""

import argparse
from typing import Dict, Optional
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from backtest import load_dataset, print_summary
from strategy import (
    compute_wilder_atr_series,
    evaluate_signal,
    compute_position_size,
    compute_stop_price,
    check_channel_exit,
    check_breakeven_trigger,
    compute_breakeven_stop_price,
    compute_locked_profit_stop_price,
)
from risk import apply_capital_policy


def rolling_top_n_mean(values: np.ndarray, window: int, n: int) -> np.ndarray:
    """out[i] = mean of the top n values in values[i-window+1 : i+1]. NaN before that."""
    out = np.full(len(values), np.nan)
    if len(values) < window:
        return out
    sw = sliding_window_view(values, window)
    top_n = np.partition(sw, window - n, axis=1)[:, window - n:]
    out[window - 1:] = top_n.mean(axis=1)
    return out


def rolling_bottom_n_mean(values: np.ndarray, window: int, n: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) < window:
        return out
    sw = sliding_window_view(values, window)
    bottom_n = np.partition(sw, n - 1, axis=1)[:, :n]
    out[window - 1:] = bottom_n.mean(axis=1)
    return out


def compute_avg_extreme_indicators(
    df: pd.DataFrame,
    entry_channel_hours: int = 960,
    exit_channel_hours: int = 480,
    mom_hours: int = 720,
    atr_period: int = 14,
    entry_top_n: int = 1,
) -> pd.DataFrame:
    res = df.copy()

    high = res["high"].to_numpy(dtype=float)
    low = res["low"].to_numpy(dtype=float)

    top_n_high = rolling_top_n_mean(high, entry_channel_hours, entry_top_n)
    bottom_n_low = rolling_bottom_n_mean(low, entry_channel_hours, entry_top_n)
    # shift(1): no-lookahead, same convention as the production single-extreme version
    res["entry_high"] = pd.Series(top_n_high, index=res.index).shift(1)
    res["entry_low"] = pd.Series(bottom_n_low, index=res.index).shift(1)

    # Channel exit ("bottom bar") stays the untouched single-extreme version
    res["exit_high"] = res["high"].rolling(window=exit_channel_hours).max().shift(1)
    res["exit_low"] = res["low"].rolling(window=exit_channel_hours).min().shift(1)

    if mom_hours > 0:
        res["mom30"] = res["close"] / res["close"].shift(mom_hours) - 1.0
    else:
        res["mom30"] = 0.0

    res["atr"] = compute_wilder_atr_series(res["high"], res["low"], res["close"], period=atr_period)
    return res


def run_avg_extreme_backtest(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    base_capital: float = 100.0,
    capital_policy: str = "skim_refill",
    taker_fee_rate: float = 0.00055,
    entry_channel_hours: int = 960,
    exit_channel_hours: int = 480,
    mom_hours: int = 720,
    atr_period: int = 14,
    atr_mult: float = 3.0,
    risk_frac: float = 0.05,
    max_leverage: float = 5.0,
    entry_top_n: int = 1,
    breakeven_trigger_r: float = 2.0,
    stage2_trigger_r: float = 3.0,
    stage2_lock_r: float = 1.0,
) -> Dict:
    data = compute_avg_extreme_indicators(
        df, entry_channel_hours=entry_channel_hours, exit_channel_hours=exit_channel_hours,
        mom_hours=mom_hours, atr_period=atr_period, entry_top_n=entry_top_n,
    )
    if start_date:
        data = data.loc[start_date:]
    if end_date:
        data = data.loc[:end_date]

    float_usdt = base_capital
    bank_usdt = 0.0
    pos_side: Optional[str] = None
    pos_qty = 0.0
    pos_entry_price = 0.0
    pos_stop_price = 0.0
    pos_initial_stop = 0.0
    pos_stage = 0
    pos_entry_time = None
    cum_fees = 0.0
    cum_funding = 0.0

    trades = []
    equity_curve = []

    for ts, row in data.iterrows():
        close_p = row["close"]
        high_p = row["high"]
        low_p = row["low"]
        atr_val = row["atr"]
        funding_rate = row["fundingRate"]

        if pos_side is not None and funding_rate != 0.0:
            pos_notional = pos_qty * close_p
            f_cost = pos_notional * funding_rate if pos_side == "Buy" else -pos_notional * funding_rate
            float_usdt -= f_cost
            cum_funding += f_cost

        if pos_side is not None and pos_stage == 0:
            if check_breakeven_trigger(pos_side, pos_entry_price, pos_initial_stop, high_p, low_p, breakeven_trigger_r):
                pos_stop_price = compute_breakeven_stop_price(pos_side, pos_entry_price, taker_fee_rate=taker_fee_rate)
                pos_stage = 1

        if pos_side is not None and pos_stage == 1:
            if check_breakeven_trigger(pos_side, pos_entry_price, pos_initial_stop, high_p, low_p, stage2_trigger_r):
                r = abs(pos_entry_price - pos_initial_stop)
                lock_price = pos_entry_price + stage2_lock_r * r if pos_side == "Buy" else pos_entry_price - stage2_lock_r * r
                pos_stop_price = max(pos_stop_price, lock_price) if pos_side == "Buy" else min(pos_stop_price, lock_price)
                pos_stage = 2

        stopped_out = False
        exit_price = 0.0
        exit_reason = None
        if pos_side == "Buy":
            if low_p <= pos_stop_price:
                stopped_out, exit_price, exit_reason = True, pos_stop_price, f"stop_stage{pos_stage}"
        elif pos_side == "Sell":
            if high_p >= pos_stop_price:
                stopped_out, exit_price, exit_reason = True, pos_stop_price, f"stop_stage{pos_stage}"

        if pos_side is not None and not stopped_out:
            if check_channel_exit(pos_side, close_p, row["exit_high"], row["exit_low"]):
                stopped_out, exit_price, exit_reason = True, close_p, "channel_exit"

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
            pos_side, pos_qty, pos_entry_price, pos_stop_price = None, 0.0, 0.0, 0.0
            pos_initial_stop, pos_stage, pos_entry_time = 0.0, 0, None

        if pos_side is None:
            sig = evaluate_signal(
                close_val=close_p, entry_high_val=row["entry_high"], entry_low_val=row["entry_low"],
                mom30_val=row["mom30"], allow_long=True, allow_short=True,
            )
            if sig is not None:
                qty, stop_dist, valid, reason = compute_position_size(
                    float_usdt=float_usdt, close_price=close_p, atr_val=atr_val,
                    risk_frac=risk_frac, max_leverage=max_leverage, atr_mult=atr_mult,
                )
                if valid:
                    pos_side = sig
                    pos_qty = qty
                    pos_entry_price = close_p
                    pos_stop_price = compute_stop_price(sig, pos_entry_price, atr_val, atr_mult=atr_mult)
                    pos_initial_stop = pos_stop_price
                    pos_stage = 0
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
        "capital_policy": capital_policy, "start_date": str(data.index[0]), "end_date": str(data.index[-1]),
        "initial_capital": base_capital, "final_float": float_usdt, "final_bank": bank_usdt,
        "final_total": float_usdt + bank_usdt, "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct, "total_trades": len(trades_df), "win_rate_pct": win_rate,
        "cum_fees": cum_fees, "cum_funding": cum_funding, "trades": trades_df, "equity_curve": eq_df,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--policy", choices=["skim_refill", "compound"], default="skim_refill")
    parser.add_argument("--entry-top-n", type=int, default=5)
    args = parser.parse_args()

    df = load_dataset()
    res = run_avg_extreme_backtest(df, start_date=args.start, end_date=args.end, capital_policy=args.policy, entry_top_n=args.entry_top_n)
    print_summary(res)


if __name__ == "__main__":
    main()
