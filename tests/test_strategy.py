"""Unit tests for strategy math, ATR, channels, signals, and sizing."""

import unittest
import numpy as np
import pandas as pd
from strategy import (
    compute_wilder_atr_series,
    compute_indicators,
    evaluate_signal,
    compute_position_size,
    compute_stop_price,
    check_channel_exit,
    quantize,
    quantize_price,
)


class TestStrategy(unittest.TestCase):
    def test_wilder_atr_hand_computed(self):
        """Test Wilder ATR against exact hand-computed values for a 20-bar sequence."""
        # 20 bars of synthetic data
        highs = [10, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        lows =  [ 8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        closes= [ 9, 11, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

        # For these bars:
        # bar 0: TR = H0-L0 = 2.0
        # bar 1: TR = max(12-9=3, |12-9|=3, |9-9|=0) = 3.0
        # bar 2: TR = max(11-10=1, |11-11|=0, |10-11|=1) = 1.0
        # bar 3: TR = max(13-11=2, |13-10|=3, |11-10|=1) = 3.0
        # and so on.
        # Let's hand-compute TR array:
        tr = [highs[0] - lows[0]]
        for i in range(1, len(highs)):
            tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])))

        # ATR at index 13 (bar 14): mean of first 14 TRs
        expected_atr_13 = np.mean(tr[:14])
        # ATR at index 14 (bar 15): (ATR_13 * 13 + TR_14) / 14
        expected_atr_14 = (expected_atr_13 * 13.0 + tr[14]) / 14.0

        s_high = pd.Series(highs)
        s_low = pd.Series(lows)
        s_close = pd.Series(closes)

        atr_series = compute_wilder_atr_series(s_high, s_low, s_close, period=14)

        self.assertTrue(np.isnan(atr_series.iloc[12]))
        self.assertAlmostEqual(atr_series.iloc[13], expected_atr_13, places=7)
        self.assertAlmostEqual(atr_series.iloc[14], expected_atr_14, places=7)

    def test_channel_lookahead_guarantee(self):
        """Channel indicators must strictly exclude the current bar k (shift(1))."""
        dates = pd.date_range("2026-01-01", periods=1000, freq="1h")
        # Construct predictable prices: high increases by 1 each bar
        df = pd.DataFrame({
            "high": np.arange(1000, 2000, dtype=float),
            "low": np.arange(900, 1900, dtype=float),
            "close": np.arange(950, 1950, dtype=float),
            "volume": np.ones(1000),
        }, index=dates)

        res = compute_indicators(df, entry_channel_hours=100, exit_channel_hours=50, mom_hours=50, atr_period=14)

        # For bar index 150:
        # Current bar high is df['high'].iloc[150] = 1150
        # But entry_high at index 150 must be max(high[150-100 .. 150-1]) = high[149] = 1149
        self.assertEqual(res["entry_high"].iloc[150], 1149.0)
        self.assertNotEqual(res["entry_high"].iloc[150], res["high"].iloc[150])

        # Exit high at index 150 must be max(high[150-50 .. 150-1]) = high[149] = 1149
        self.assertEqual(res["exit_high"].iloc[150], 1149.0)

    def test_signals_momentum_veto(self):
        """Signals require momentum confirmation."""
        # Long breakout with positive momentum -> Buy
        sig = evaluate_signal(close_val=100.0, entry_high_val=95.0, entry_low_val=80.0, mom30_val=0.05)
        self.assertEqual(sig, "Buy")

        # Long breakout with negative momentum -> None (vetoed)
        sig_vetoed = evaluate_signal(close_val=100.0, entry_high_val=95.0, entry_low_val=80.0, mom30_val=-0.01)
        self.assertIsNone(sig_vetoed)

        # Short breakout with negative momentum -> Sell
        sig_short = evaluate_signal(close_val=75.0, entry_high_val=95.0, entry_low_val=80.0, mom30_val=-0.02)
        self.assertEqual(sig_short, "Sell")

        # Short breakout with positive momentum -> None (vetoed)
        sig_short_vetoed = evaluate_signal(close_val=75.0, entry_high_val=95.0, entry_low_val=80.0, mom30_val=0.01)
        self.assertIsNone(sig_short_vetoed)

    def test_position_sizing_binding_constraints(self):
        """Test risk cap vs leverage cap in sizing."""
        float_usdt = 100.0
        close_price = 50000.0

        # Scenario 1: Volatility is low (ATR = 500, stop_dist = 1500)
        # qty_risk = (100 * 0.05) / 1500 = 5 / 1500 = 0.003333... BTC
        # qty_lev  = (100 * 5.0) / 50000 = 500 / 50000 = 0.010 BTC
        # qty_raw  = min(0.00333, 0.010) = 0.00333 -> quantized to 0.003 BTC
        qty, stop_dist, valid, reason = compute_position_size(
            float_usdt=float_usdt,
            close_price=close_price,
            atr_val=500.0,
            risk_frac=0.05,
            max_leverage=5.0,
            qty_step=0.001,
        )
        self.assertTrue(valid)
        self.assertEqual(qty, 0.003)
        self.assertEqual(stop_dist, 1500.0)

        # Scenario 2: Volatility is extremely high (ATR = 6000, stop_dist = 18000)
        # qty_risk = 5 / 18000 = 0.000277... BTC
        # qty_step = 0.001 -> quantizes to 0.000 -> below min_order_qty
        qty2, _, valid2, reason2 = compute_position_size(
            float_usdt=float_usdt,
            close_price=close_price,
            atr_val=6000.0,
            risk_frac=0.05,
            max_leverage=5.0,
            qty_step=0.001,
            min_order_qty=0.001,
        )
        self.assertFalse(valid2)
        self.assertEqual(qty2, 0.0)

        # Scenario 3: Leverage binds
        # Let close_price = 100,000, float_usdt = 100.0
        # qty_lev = 500 / 100000 = 0.005 BTC ($500 notional, 5x leverage)
        # ATR = 50 -> stop_dist = 150 -> qty_risk = 5 / 150 = 0.0333 BTC ($3333 notional, 33x leverage!)
        # qty_raw = min(0.0333, 0.005) = 0.005 BTC
        qty3, _, valid3, _ = compute_position_size(
            float_usdt=float_usdt,
            close_price=100000.0,
            atr_val=50.0,
            risk_frac=0.05,
            max_leverage=5.0,
            qty_step=0.001,
        )
        self.assertTrue(valid3)
        self.assertEqual(qty3, 0.005)

    def test_fixed_exchange_stop_price(self):
        """Test exchange stop loss price computation."""
        # Long at 60,000, ATR=1000, mult=3.0 -> stop at 57,000
        stop_long = compute_stop_price(side="Buy", entry_price=60000.0, atr_at_entry=1000.0, atr_mult=3.0, tick_size=0.1)
        self.assertEqual(stop_long, 57000.0)

        # Short at 60,000, ATR=1000, mult=3.0 -> stop at 63,000
        stop_short = compute_stop_price(side="Sell", entry_price=60000.0, atr_at_entry=1000.0, atr_mult=3.0, tick_size=0.1)
        self.assertEqual(stop_short, 63000.0)

    def test_channel_exit(self):
        """Test channel exit checks."""
        # Long exits if close < exit_low
        self.assertTrue(check_channel_exit("Buy", close_price=59999.0, exit_high=65000.0, exit_low=60000.0))
        self.assertFalse(check_channel_exit("Buy", close_price=60001.0, exit_high=65000.0, exit_low=60000.0))

        # Short exits if close > exit_high
        self.assertTrue(check_channel_exit("Sell", close_price=65001.0, exit_high=65000.0, exit_low=60000.0))
        self.assertFalse(check_channel_exit("Sell", close_price=64999.0, exit_high=65000.0, exit_low=60000.0))


if __name__ == "__main__":
    unittest.main()
