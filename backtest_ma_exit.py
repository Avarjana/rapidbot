"""Backtest the production strategy (entries, sizing, initial ATR stop -- all
untouched) with the channel exit REPLACED by a moving-average-cross exit:
exit long when close crosses below its N-period SMA (short: crosses above).

Same window length as the production channel exit (480h = 20 days) by default,
so this is an apples-to-apples test of indicator SHAPE (rolling-low breach vs.
MA cross) rather than a different lookback horizon.
"""

import argparse
from typing import Dict, Optional
import pandas as pd

from backtest import load_dataset, print_summary
from strategy import (
    compute_indicators,
    evaluate_signal,
    compute_position_size,
    compute_stop_price,
)
from risk import apply_capital_policy


def run_ma_exit_backtest(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    base_capital: float = 100.0,
    capital_policy: str = "skim_refill",
    taker_fee_rate: float = 0.00055,
    entry_channel_hours: int = 960,
    mom_hours: int = 720,
    atr_period: int = 14,
    atr_mult: float = 3.0,
    risk_frac: float = 0.05,
    max_leverage: float = 5.0,
    ma_hours: int = 480,  # 20 days, matching the production channel exit's window
) -> Dict:
    data = compute_indicators(
        df, entry_channel_hours=entry_channel_hours, exit_channel_hours=480,
        mom_hours=mom_hours, atr_period=atr_period,
    )
    # No-lookahead MA, same convention as exit_low/exit_high (shift(1)): the
    # threshold at bar k is computed from bars strictly before k.
    data["ma_exit"] = data["close"].rolling(window=ma_hours).mean().shift(1)

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
        ma_val = row["ma_exit"]

        if pos_side is not None and funding_rate != 0.0:
            pos_notional = pos_qty * close_p
            f_cost = pos_notional * funding_rate if pos_side == "Buy" else -pos_notional * funding_rate
            float_usdt -= f_cost
            cum_funding += f_cost

        stopped_out = False
        exit_price = 0.0
        exit_reason = None
        if pos_side == "Buy":
            if low_p <= pos_stop_price:
                stopped_out, exit_price, exit_reason = True, pos_stop_price, "stop_loss"
        elif pos_side == "Sell":
            if high_p >= pos_stop_price:
                stopped_out, exit_price, exit_reason = True, pos_stop_price, "stop_loss"

        if pos_side is not None and not stopped_out and pd.notna(ma_val):
            if pos_side == "Buy" and close_p < ma_val:
                stopped_out, exit_price, exit_reason = True, close_p, "ma_cross"
            elif pos_side == "Sell" and close_p > ma_val:
                stopped_out, exit_price, exit_reason = True, close_p, "ma_cross"

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
            pos_side, pos_qty, pos_entry_price, pos_stop_price, pos_entry_time = None, 0.0, 0.0, 0.0, None

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
    parser = argparse.ArgumentParser(description="Production strategy with MA-cross exit instead of channel exit")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--policy", choices=["skim_refill", "compound"], default="skim_refill")
    parser.add_argument("--ma-hours", type=int, default=480)
    args = parser.parse_args()

    df = load_dataset()
    res = run_ma_exit_backtest(df, start_date=args.start, end_date=args.end, capital_policy=args.policy, ma_hours=args.ma_hours)
    print_summary(res)


if __name__ == "__main__":
    main()
