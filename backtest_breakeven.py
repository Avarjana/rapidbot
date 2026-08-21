"""Backtest the production strategy (entries, sizing, initial ATR stop, channel
exit — all untouched) with ONE addition: once a trade's favorable excursion
reaches `trigger_r` multiples of its initial risk (R = |entry - initial_stop|),
the stop is moved up to breakeven (entry price + a small buffer covering the
round-trip taker fee, so a "breakeven" exit doesn't quietly become a small loss).

This is a one-time ratchet, not continuous trailing — different mechanism from
backtest_trailing.py. The channel exit stays active throughout; this only changes
what the HARD stop does after the trigger fires.
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
    check_channel_exit,
)
from risk import apply_capital_policy


def run_breakeven_backtest(
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
    trigger_r: Optional[float] = 1.0,  # None = disabled (reproduces production exactly)
) -> Dict:
    data = compute_indicators(
        df, entry_channel_hours=entry_channel_hours, exit_channel_hours=exit_channel_hours,
        mom_hours=mom_hours, atr_period=atr_period,
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
    pos_initial_r = 0.0
    pos_breakeven_done = False
    pos_entry_time = None
    cum_fees = 0.0
    cum_funding = 0.0

    trades = []
    equity_curve = []
    breakeven_triggers = 0

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

        # --- Break-even ratchet: fires once, using this bar's favorable extreme ---
        if pos_side is not None and trigger_r is not None and not pos_breakeven_done:
            breakeven_price_buf = pos_entry_price * (1 + 2 * taker_fee_rate) if pos_side == "Buy" else pos_entry_price * (1 - 2 * taker_fee_rate)
            if pos_side == "Buy":
                trigger_level = pos_entry_price + trigger_r * pos_initial_r
                if high_p >= trigger_level:
                    pos_stop_price = max(pos_stop_price, breakeven_price_buf)
                    pos_breakeven_done = True
                    breakeven_triggers += 1
            else:
                trigger_level = pos_entry_price - trigger_r * pos_initial_r
                if low_p <= trigger_level:
                    pos_stop_price = min(pos_stop_price, breakeven_price_buf)
                    pos_breakeven_done = True
                    breakeven_triggers += 1

        stopped_out = False
        exit_price = 0.0
        exit_reason = None
        if pos_side == "Buy":
            if low_p <= pos_stop_price:
                stopped_out, exit_price, exit_reason = True, pos_stop_price, ("breakeven_stop" if pos_breakeven_done else "stop_loss")
        elif pos_side == "Sell":
            if high_p >= pos_stop_price:
                stopped_out, exit_price, exit_reason = True, pos_stop_price, ("breakeven_stop" if pos_breakeven_done else "stop_loss")

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
            pos_initial_r, pos_breakeven_done, pos_entry_time = 0.0, False, None

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
                    pos_initial_r = abs(pos_entry_price - pos_stop_price)
                    pos_breakeven_done = False
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
    scratch_trades = trades_df[trades_df["exit_reason"] == "breakeven_stop"] if not trades_df.empty else pd.DataFrame()

    return {
        "capital_policy": capital_policy, "start_date": str(data.index[0]), "end_date": str(data.index[-1]),
        "initial_capital": base_capital, "final_float": float_usdt, "final_bank": bank_usdt,
        "final_total": float_usdt + bank_usdt, "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct, "total_trades": len(trades_df), "win_rate_pct": win_rate,
        "cum_fees": cum_fees, "cum_funding": cum_funding, "trades": trades_df, "equity_curve": eq_df,
        "breakeven_triggers": breakeven_triggers, "scratched_trades": len(scratch_trades),
    }


def main():
    parser = argparse.ArgumentParser(description="Production strategy with an optional move-to-breakeven stop")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--policy", choices=["skim_refill", "compound"], default="skim_refill")
    parser.add_argument("--trigger-r", type=float, default=1.0)
    parser.add_argument("--disable", action="store_true", help="disable breakeven move (reproduces production)")
    args = parser.parse_args()

    df = load_dataset()
    res = run_breakeven_backtest(
        df, start_date=args.start, end_date=args.end, capital_policy=args.policy,
        trigger_r=None if args.disable else args.trigger_r,
    )
    print_summary(res)
    print(f"Breakeven triggers: {res['breakeven_triggers']}  Trades scratched at breakeven: {res['scratched_trades']}")


if __name__ == "__main__":
    main()
