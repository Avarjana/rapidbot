"""Unit tests for config loading and validation."""

import unittest
from pathlib import Path
from config import _strip_inline_comment, parse_env_file, load_config


class TestConfig(unittest.TestCase):
    def test_strip_inline_comment(self):
        self.assertEqual(_strip_inline_comment("60 # minutes; 1h bars"), "60")
        self.assertEqual(_strip_inline_comment("960 # 40 days"), "960")
        self.assertEqual(_strip_inline_comment('"value # with hash" # trailing comment'), '"value # with hash"')
        self.assertEqual(_strip_inline_comment("BTCUSDT"), "BTCUSDT")

    def test_load_default_config(self):
        cfg = load_config(".env.rapid")
        self.assertEqual(cfg.symbol, "BTCUSDT")
        self.assertEqual(cfg.interval, 60)
        self.assertEqual(cfg.entry_channel_hours, 960)
        self.assertEqual(cfg.exit_channel_hours, 480)
        self.assertEqual(cfg.atr_period, 14)
        self.assertEqual(cfg.atr_mult, 3.0)
        self.assertEqual(cfg.base_capital, 100.0)
        self.assertEqual(cfg.risk_frac, 0.05)
        self.assertEqual(cfg.max_leverage, 5.0)
        self.assertEqual(cfg.capital_policy, "skim_refill")


if __name__ == "__main__":
    unittest.main()
