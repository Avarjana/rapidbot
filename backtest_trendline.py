"""Backtesting engine for the prototype diagonal-trendline strategy.
Mirrors backtest.py's engine (same fee model, funding, position sizing, ATR stop,
capital policy) so results are directly comparable to the production channel-breakout bot.
"""

import argparse
from typing import Dict, Optional
import pandas as pd

from backtest import load_dataset, print_summary
from strategy import (
    compute_wilder_atr_series,
    compute_position_size,
    compute_stop_price,
)
from trendline_strategy import (
    compute_trendlines,
    evaluate_trendline_signal,
    check_trendline_exit,
)
from risk import apply_capital_policy


def run_trendline_backtest(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    base_capital: float = 100.0,
    capital_policy: str = "compound",
    taker_fee_rate: float = 0.00055,
    pivot_window: int = 24,
    n_points: int = 2,
    require_momentum_confirmation: bool = False,
    mom_hours: int = 720,
    atr_period: int = 14,
    atr_mult: float = 3.0,
    risk_frac: float = 0.05,
    max_leverage: float = 5.0,
) -> Dict:
    """Event-driven hourly backtest for the trendline-breakout strategy."""
    data = compute_trendlines(df, pivot_window=pivot_window, n_points=n_points)
    data["atr"] = compute_wilder_atr_series(data["high"], data["low"], data["close"], period=atr_period)
    data["mom30"] = data["close"] / data["close"].shift(mom_hours) - 1.0 if mom_hours > 0 else 0.0

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

        if pos_side is not None and funding_rate != 0.0:
            pos_notional = pos_qty * close_p
            if pos_side == "Buy":
                f_cost = pos_notional * funding_rate
            else:
                f_cost = -pos_notional * funding_rate
            float_usdt -= f_cost
            cum_funding += f_cost

        stopped_out = False
        exit_reason = None
        exit_price = 0.0

        if pos_side == "Buy":
            if low_p <= pos_stop_price:
                stopped_out = True
                exit_price = pos_stop_price
                exit_reason = "stop_loss"
        elif pos_side == "Sell":
            if high_p >= pos_stop_price:
                stopped_out = True
                exit_price = pos_stop_price
                exit_reason = "stop_loss"

        if pos_side is not None and not stopped_out:
            if check_trendline_exit(pos_side, close_p, row["resistance_line"], row["support_line"]):
                stopped_out = True
                exit_price = close_p
                exit_reason = "trendline_exit"

        if pos_side is not None and stopped_out:
            if pos_side == "Buy":
                raw_pnl = pos_qty * (exit_price - pos_entry_price)
            else:
                raw_pnl = pos_qty * (pos_entry_price - exit_price)

            exit_fee = pos_qty * exit_price * taker_fee_rate
            cum_fees += exit_fee
            net_trade_pnl = raw_pnl - exit_fee
            float_usdt += net_trade_pnl

            if capital_policy == "skim_refill":
                float_usdt, bank_usdt = apply_capital_policy(
                    float_usdt, bank_usdt, base_capital=base_capital, policy="skim_refill"
                )

            trades.append({
                "entry_time": str(pos_entry_time),
                "exit_time": str(ts),
                "side": pos_side,
                "qty": pos_qty,
                "entry_price": pos_entry_price,
                "exit_price": exit_price,
                "stop_price": pos_stop_price,
                "raw_pnl": raw_pnl,
                "net_pnl": net_trade_pnl,
                "exit_reason": exit_reason,
                "float_after": float_usdt,
                "bank_after": bank_usdt,
            })

            pos_side = None
            pos_qty = 0.0
            pos_entry_price = 0.0
            pos_stop_price = 0.0
            pos_entry_time = None

        if pos_side is None:
            sig = evaluate_trendline_signal(
                close_val=close_p,
                resistance_line_val=row["resistance_line"],
                support_line_val=row["support_line"],
                mom30_val=row["mom30"],
                allow_long=True,
                allow_short=True,
                require_momentum_confirmation=require_momentum_confirmation,
            )
            if sig is not None:
                qty, stop_dist, valid, reason = compute_position_size(
                    float_usdt=float_usdt,
                    close_price=close_p,
                    atr_val=atr_val,
                    risk_frac=risk_frac,
                    max_leverage=max_leverage,
                    atr_mult=atr_mult,
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
        equity_curve.append({
            "timestamp": ts,
            "float": float_usdt,
            "bank": bank_usdt,
            "total_equity": total_equity,
        })

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
        "start_date": str(data.index[0]),
        "end_date": str(data.index[-1]),
        "initial_capital": base_capital,
        "final_float": float_usdt,
        "final_bank": bank_usdt,
        "final_total": float_usdt + bank_usdt,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct,
        "total_trades": len(trades_df),
        "win_rate_pct": win_rate,
        "cum_fees": cum_fees,
        "cum_funding": cum_funding,
        "trades": trades_df,
        "equity_curve": eq_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Trendline-breakout prototype backtest")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--policy", choices=["skim_refill", "compound"], default="compound")
    parser.add_argument("--pivot-window", type=int, default=24)
    parser.add_argument("--n-points", type=int, default=2)
    parser.add_argument("--momentum-filter", action="store_true")
    args = parser.parse_args()

    df = load_dataset()
    res = run_trendline_backtest(
        df,
        start_date=args.start,
        end_date=args.end,
        capital_policy=args.policy,
        pivot_window=args.pivot_window,
        n_points=args.n_points,
        require_momentum_confirmation=args.momentum_filter,
    )
    print_summary(res)


if __name__ == "__main__":
    main()
