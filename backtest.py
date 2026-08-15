"""Backtesting engine for rapid_bot using pure strategy.py.
Computes performance, trade log, funding costs, and verifies canary benchmarks.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from strategy import (
    compute_indicators,
    evaluate_signal,
    compute_position_size,
    compute_stop_price,
    check_channel_exit,
)
from risk import apply_capital_policy


def load_dataset(
    kline_path: str = "data/BTC_USDT_ext_1h.parquet",
    funding_path: str = "data/BTC_USDT_ext_funding.parquet",
) -> pd.DataFrame:
    """Load and merge klines with 8-hourly funding rates (fill_value=0.0)."""
    k_df = pd.read_parquet(kline_path)
    if not isinstance(k_df.index, pd.DatetimeIndex):
        k_df["datetime"] = pd.to_datetime(k_df["timestamp"], unit="ms", utc=True)
        k_df = k_df.set_index("datetime")
    k_df = k_df.sort_index()

    f_df = pd.read_parquet(funding_path)
    if not isinstance(f_df.index, pd.DatetimeIndex):
        f_df["datetime"] = pd.to_datetime(f_df["timestamp"], unit="ms", utc=True)
        f_df = f_df.set_index("datetime")
    f_df = f_df.sort_index()

    # Merge funding onto 1h bars with fill_value=0.0
    # Funding happens at 00:00, 08:00, 16:00 UTC
    merged = k_df.join(f_df[["fundingRate"]], how="left")
    merged["fundingRate"] = merged["fundingRate"].fillna(0.0)

    return merged


def run_backtest(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    base_capital: float = 100.0,
    capital_policy: str = "skim_refill",  # skim_refill | compound
    taker_fee_rate: float = 0.00055,  # 0.055% taker fee
    entry_channel_hours: int = 960,
    exit_channel_hours: int = 480,
    mom_hours: int = 720,
    atr_period: int = 14,
    atr_mult: float = 3.0,
    risk_frac: float = 0.05,
    max_leverage: float = 5.0,
) -> Dict:
    """Execute event-driven hourly backtest using pure strategy.py."""
    data = compute_indicators(
        df,
        entry_channel_hours=entry_channel_hours,
        exit_channel_hours=exit_channel_hours,
        mom_hours=mom_hours,
        atr_period=atr_period,
    )

    if start_date:
        data = data.loc[start_date:]
    if end_date:
        data = data.loc[:end_date]

    float_usdt = base_capital
    bank_usdt = 0.0
    pos_side: Optional[str] = None  # 'Buy' (long) or 'Sell' (short)
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

        # Track equity before bar actions
        unrealized_pnl = 0.0
        if pos_side == "Buy":
            unrealized_pnl = pos_qty * (close_p - pos_entry_price)
        elif pos_side == "Sell":
            unrealized_pnl = pos_qty * (pos_entry_price - close_p)

        # 1. If in position, charge funding if this bar has a funding rate
        if pos_side is not None and funding_rate != 0.0:
            # Long pays funding if rate > 0, short receives
            pos_notional = pos_qty * close_p
            if pos_side == "Buy":
                f_cost = pos_notional * funding_rate  # positive means cost
            else:
                f_cost = -pos_notional * funding_rate

            float_usdt -= f_cost
            cum_funding += f_cost

        # 2. Check Intrabar Stop Loss
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

        # 3. If not stopped out, check Channel Exit on closed bar k
        if pos_side is not None and not stopped_out:
            if check_channel_exit(pos_side, close_p, row["exit_high"], row["exit_low"]):
                stopped_out = True
                exit_price = close_p
                exit_reason = "channel_exit"

        # 4. Process Exit
        if pos_side is not None and stopped_out:
            # Calculate realized PnL
            if pos_side == "Buy":
                raw_pnl = pos_qty * (exit_price - pos_entry_price)
            else:
                raw_pnl = pos_qty * (pos_entry_price - exit_price)

            exit_fee = pos_qty * exit_price * taker_fee_rate
            cum_fees += exit_fee
            net_trade_pnl = raw_pnl - exit_fee

            float_usdt += net_trade_pnl

            # Apply capital policy
            float_before_policy = float_usdt
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

        # 5. Check Entry on closed bar k (if flat)
        if pos_side is None:
            sig = evaluate_signal(
                close_val=close_p,
                entry_high_val=row["entry_high"],
                entry_low_val=row["entry_low"],
                mom30_val=row["mom30"],
                allow_long=True,
                allow_short=True,
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

        # Log equity point
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

    # Performance stats
    total_return_pct = ((eq_df["total_equity"].iloc[-1] / base_capital) - 1.0) * 100.0 if not eq_df.empty else 0.0
    
    # Max Drawdown
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


def print_summary(res: Dict):
    print("=" * 65)
    print(f"RAPID_BOT BACKTEST SUMMARY ({res['capital_policy'].upper()})")
    print("=" * 65)
    print(f"Period:             {res['start_date'][:10]} -> {res['end_date'][:10]}")
    print(f"Initial Capital:    ${res['initial_capital']:.2f} USDT")
    print(f"Final Float:        ${res['final_float']:.2f} USDT")
    print(f"Final Bank:         ${res['final_bank']:.2f} USDT")
    print(f"Final Total Equity: ${res['final_total']:.2f} USDT")
    print(f"Total Return:       {res['total_return_pct']:+.2f}%")
    print(f"Max Drawdown:       {res['max_drawdown_pct']:.2f}%")
    print(f"Total Trades:       {res['total_trades']}")
    print(f"Win Rate:           {res['win_rate_pct']:.1f}%")
    print(f"Cumulative Fees:    ${res['cum_fees']:.2f} USDT")
    print(f"Cumulative Funding: ${res['cum_funding']:.2f} USDT")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="rapid_bot backtest engine")
    parser.add_argument("--eval-2026", action="store_true", help="Run 2026-01-01 to 2026-08-13 canary benchmark")
    parser.add_argument("--policy", choices=["skim_refill", "compound"], default="skim_refill", help="Capital policy")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    df = load_dataset()

    if args.eval_2026:
        print("\n--- 2026 Canary Benchmark (2026-01-01 -> 2026-08-13) ---")
        res_sr = run_backtest(df, start_date="2026-01-01", end_date="2026-08-13", capital_policy="skim_refill")
        print_summary(res_sr)

        res_cp = run_backtest(df, start_date="2026-01-01", end_date="2026-08-13", capital_policy="compound")
        print_summary(res_cp)
    else:
        res = run_backtest(df, start_date=args.start, end_date=args.end, capital_policy=args.policy)
        print_summary(res)


if __name__ == "__main__":
    main()
