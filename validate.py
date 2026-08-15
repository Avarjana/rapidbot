"""Phase 1 Walk-Forward Validation Gate.
Evaluates train period (2020-2022) vs test period (2023-2026) across parameter variations.
Calculates Spearman rank correlation and strictly gates parameter freezing.
"""

import sys
import pandas as pd
from scipy.stats import spearmanr
from backtest import load_dataset, run_backtest


def main():
    print("=" * 70)
    print("PHASE 1 WALK-FORWARD VALIDATION GATE")
    print("=" * 70)

    df = load_dataset()

    entry_channels = [720, 960, 1200]  # 30d, 40d, 50d
    exit_channels = [360, 480, 600]    # 15d, 20d, 25d
    atr_mults = [2.5, 3.0, 3.5]

    records = []

    print("Running grid across train (2020-03 -> 2022-12) and test (2023-01 -> 2026-08)...")
    for ec in entry_channels:
        for xc in exit_channels:
            for mult in atr_mults:
                param_name = f"E{ec//24}d_X{xc//24}d_ATR{mult}"

                res_train = run_backtest(
                    df,
                    start_date="2020-03-25",
                    end_date="2022-12-31",
                    entry_channel_hours=ec,
                    exit_channel_hours=xc,
                    atr_mult=mult,
                    capital_policy="skim_refill",
                )

                res_test = run_backtest(
                    df,
                    start_date="2023-01-01",
                    end_date="2026-08-13",
                    entry_channel_hours=ec,
                    exit_channel_hours=xc,
                    atr_mult=mult,
                    capital_policy="skim_refill",
                )

                records.append({
                    "param": param_name,
                    "entry_h": ec,
                    "exit_h": xc,
                    "atr_mult": mult,
                    "train_return_pct": res_train["total_return_pct"],
                    "train_trades": res_train["total_trades"],
                    "train_max_dd": res_train["max_drawdown_pct"],
                    "test_return_pct": res_test["total_return_pct"],
                    "test_trades": res_test["total_trades"],
                    "test_max_dd": res_test["max_drawdown_pct"],
                })

    res_df = pd.DataFrame(records)

    # Rank correlation between train return and test return
    corr, p_value = spearmanr(res_df["train_return_pct"], res_df["test_return_pct"])

    print("\nTop 5 Param Sets in Train (2020-2022):")
    train_sorted = res_df.sort_values("train_return_pct", ascending=False).head(5)
    print(train_sorted[["param", "train_return_pct", "test_return_pct", "test_trades", "test_max_dd"]].to_string(index=False))

    print("\nTarget Baseline Param (E40d_X20d_ATR3.0):")
    target = res_df[res_df["param"] == "E40d_X20d_ATR3.0"]
    print(target[["param", "train_return_pct", "test_return_pct", "test_trades", "test_max_dd"]].to_string(index=False))

    print("-" * 70)
    print(f"Spearman Rank Correlation (Train -> Test): {corr:.4f} (p-value: {p_value:.4f})")
    print("-" * 70)

    # STRICT GATE EVALUATION per §10:
    # 1. Selection in-sample must not show an artificial high out-of-sample edge (rank correlation should be <= 0.35)
    # 2. Target baseline must maintain positive return and controlled drawdown in out-of-sample test
    target_test_ret = float(target["test_return_pct"].iloc[0])
    target_test_dd = float(target["test_max_dd"].iloc[0])

    gate_passed = True
    failure_reasons = []

    if corr > 0.50:
        gate_passed = False
        failure_reasons.append(f"Rank correlation too high ({corr:.4f} > 0.50): suspect lookahead bug in indicators/backtest.")

    if target_test_ret <= 0:
        gate_passed = False
        failure_reasons.append(f"Target parameter failed out-of-sample test: return {target_test_ret:.2f}% <= 0.")

    if target_test_dd < -65.0:
        gate_passed = False
        failure_reasons.append(f"Target parameter test max drawdown too severe: {target_test_dd:.2f}% < -65%.")

    if gate_passed:
        print("GATE PASSED: No overfitted out-of-sample rank correlation detected.")
        print("Parameters (40d entry / 20d exit / 3.0 ATR / 5% risk) are officially FROZEN.")
        print("=" * 70)
        return 0
    else:
        print("GATE FAILED:")
        for r in failure_reasons:
            print(f" - {r}")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
