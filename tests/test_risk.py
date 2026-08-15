"""Unit tests for risk limits and capital management policy."""

import unittest
from risk import (
    apply_capital_policy,
    check_account_floor,
    check_trade_loss_limit,
    check_data_staleness,
    check_api_failures,
)


class TestRisk(unittest.TestCase):
    def test_capital_policy_win_skim(self):
        """Win banks profits above base capital (100 USDT)."""
        # Starting with float 100, bank 0. Closed trade made +25 USDT -> float 125
        flt, bnk = apply_capital_policy(float_usdt=125.0, bank_usdt=0.0, base_capital=100.0)
        self.assertEqual(flt, 100.0)
        self.assertEqual(bnk, 25.0)

        # Another win of +10 USDT -> float 110, bank was 25 -> float 100, bank 35
        flt, bnk = apply_capital_policy(float_usdt=110.0, bank_usdt=bnk, base_capital=100.0)
        self.assertEqual(flt, 100.0)
        self.assertEqual(bnk, 35.0)

    def test_capital_policy_loss_refill(self):
        """Loss refills float from bank up to base capital."""
        # Float was 100, bank is 35. Loss of -15 -> float is 85
        flt, bnk = apply_capital_policy(float_usdt=85.0, bank_usdt=35.0, base_capital=100.0)
        self.assertEqual(flt, 100.0)
        self.assertEqual(bnk, 20.0)

        # Loss of -25 -> float is 75, bank is 20 -> bank fully depleted to 0, float refills to 95
        flt, bnk = apply_capital_policy(float_usdt=75.0, bank_usdt=bnk, base_capital=100.0)
        self.assertEqual(flt, 95.0)
        self.assertEqual(bnk, 0.0)

        # Further loss with 0 in bank -> float is 90, cannot refill
        flt, bnk = apply_capital_policy(float_usdt=90.0, bank_usdt=bnk, base_capital=100.0)
        self.assertEqual(flt, 90.0)
        self.assertEqual(bnk, 0.0)

    def test_capital_policy_no_ratchet_down(self):
        """Confirm the ratchet-down variant is absent (float does not permanently shrink if bank > 0)."""
        flt, bnk = apply_capital_policy(float_usdt=80.0, bank_usdt=50.0, base_capital=100.0)
        self.assertEqual(flt, 100.0)
        self.assertEqual(bnk, 30.0)

    def test_kill_switches(self):
        """Test account floor, single trade loss, and stale data kill switches."""
        # Floor kill switch
        tripped, msg = check_account_floor(float_usdt=30.0, bank_usdt=5.0, floor_threshold=40.0)
        self.assertTrue(tripped)
        self.assertIn("KILL SWITCH", msg)

        tripped, msg = check_account_floor(float_usdt=35.0, bank_usdt=10.0, floor_threshold=40.0)
        self.assertFalse(tripped)

        # Single trade loss > 15% (stop failure detector)
        # Lost $16 on a $100 float = -16% -> kill switch tripped
        tripped, msg = check_trade_loss_limit(trade_pnl=-16.0, pre_trade_float=100.0, max_loss_pct=0.15)
        self.assertTrue(tripped)
        self.assertIn("Single trade loss", msg)

        # Lost $5 on a $100 float = -5% (normal stop loss) -> ok
        tripped, msg = check_trade_loss_limit(trade_pnl=-5.0, pre_trade_float=100.0, max_loss_pct=0.15)
        self.assertFalse(tripped)

        # Stale data > 180 min
        tripped, msg = check_data_staleness(minutes_since_last_bar=181.0, max_stale_min=180)
        self.assertTrue(tripped)

        tripped, msg = check_data_staleness(minutes_since_last_bar=60.0, max_stale_min=180)
        self.assertFalse(tripped)

        # Consecutive API failures
        tripped, msg = check_api_failures(consec_failures=10, max_allowed=10)
        self.assertTrue(tripped)


if __name__ == "__main__":
    unittest.main()
