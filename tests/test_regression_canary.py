"""Regression test verifying that the backtest engine reproduces the exact 2026 canary metrics.
Guards against accidental strategy drift (§11).
"""

import unittest
from backtest import load_dataset, run_backtest


class TestCanaryRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()

    def test_eval_2026_canary_skim_refill(self):
        """2026-01-01 -> 2026-08-13 must reproduce +27.1% skim_refill, 8 trades, max DD ~-29.5%."""
        res = run_backtest(
            self.df,
            start_date="2026-01-01",
            end_date="2026-08-13",
            capital_policy="skim_refill",
        )
        self.assertEqual(res["total_trades"], 8)
        self.assertAlmostEqual(res["total_return_pct"], 27.14, delta=0.5)
        self.assertAlmostEqual(res["max_drawdown_pct"], -29.47, delta=1.0)
        self.assertAlmostEqual(res["final_bank"], 27.14, delta=0.5)
        self.assertAlmostEqual(res["final_float"], 100.0, delta=0.01)

    def test_eval_2026_canary_compound(self):
        """2026-01-01 -> 2026-08-13 must reproduce +23.2% compound, 8 trades, max DD ~-32.3%."""
        res = run_backtest(
            self.df,
            start_date="2026-01-01",
            end_date="2026-08-13",
            capital_policy="compound",
        )
        self.assertEqual(res["total_trades"], 8)
        self.assertAlmostEqual(res["total_return_pct"], 23.24, delta=0.5)
        self.assertAlmostEqual(res["max_drawdown_pct"], -32.27, delta=1.0)
        self.assertAlmostEqual(res["final_float"], 123.24, delta=0.5)


if __name__ == "__main__":
    unittest.main()
