"""Integration tests for rapid_bot lifecycle, commands, foreign positions, and dry-run stops."""

import os
import tempfile
import unittest
from pathlib import Path
from exchange import PositionInfo
from main import RapidBot
from state import PositionState


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_file = Path(self.temp_dir.name) / ".env.rapid.test"
        self.state_file = Path(self.temp_dir.name) / "state.json"
        self.log_file = Path(self.temp_dir.name) / "bot.log"

        content = f"""
BOT_NAME=rapid_bot_test
SYMBOL=BTCUSDT
CATEGORY=linear
INTERVAL=60
BYBIT_API_KEY=
BYBIT_API_SECRET=
BYBIT_TESTNET=1
DRY_RUN=1
ENTRY_CHANNEL_HOURS=960
EXIT_CHANNEL_HOURS=480
ATR_PERIOD=14
ATR_MULT=3.0
TREND_VETO_HOURS=720
ALLOW_LONG=1
ALLOW_SHORT=1
BASE_CAPITAL=100.0
RISK_FRAC=0.05
MAX_LEVERAGE=5.0
CAPITAL_POLICY=skim_refill
KILL_TOTAL_BELOW=40.0
MAX_SINGLE_TRADE_LOSS_PCT=0.15
STALE_DATA_HALT_MIN=180
MAX_CONSEC_API_FAILURES=10
WARMUP_BARS=1200
LOOP_INTERVAL_SEC=1
STATE_FILE={self.state_file.as_posix()}
LOG_FILE={self.log_file.as_posix()}
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
"""
        with open(self.env_file, "w", encoding="utf-8") as f:
            f.write(content)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_startup_and_telegram_commands(self):
        bot = RapidBot(config_path=str(self.env_file))
        bot.startup()

        # Status command
        status = bot._cmd_status()
        self.assertIn("rapid_bot_test Status", status)
        self.assertIn("Float: $100.00", status)

        # Pause command
        pause_resp = bot._cmd_pause()
        self.assertIn("paused", pause_resp)
        self.assertTrue(bot.state.paused)

        # Resume command
        resume_resp = bot._cmd_resume()
        self.assertIn("resumed", resume_resp)
        self.assertFalse(bot.state.paused)

        bot.stop()

    def test_foreign_position_detection(self):
        """Foreign position (bot was flat, but position exists on exchange) halts without closing per §7."""
        bot = RapidBot(config_path=str(self.env_file))
        bot.startup()

        # Simulate foreign position found on exchange while bot state.position is None
        bot.exchange._mock_position = PositionInfo(
            side="Buy",
            size=0.005,
            avg_price=60000.0,
            stop_loss=0.0,
            unrealised_pnl=0.0,
            liq_price=0.0,
            leverage=5.0,
        )

        bot.reconcile()

        # Bot must halt on foreign position and NOT auto-close
        self.assertTrue(bot.state.halted)
        self.assertIn("FOREIGN POSITION DETECTED", bot.state.halt_reason)
        self.assertEqual(bot.exchange.get_position().size, 0.005)

        bot.stop()

    def test_dry_run_intrabar_stop_simulation(self):
        """Dry run mode simulates stop out when price reaches stop, skipping the entry bar."""
        bot = RapidBot(config_path=str(self.env_file))
        bot.startup()

        entry_ts = 1700000000000
        bot.state.position = PositionState(
            side="Buy",
            qty=0.003,
            entry_price=60000.0,
            stop_price=57000.0,
            entry_bar_ts=entry_ts,
            entry_order_id="dry-1",
        )
        bot.exchange._mock_position = PositionInfo(
            side="Buy",
            size=0.003,
            avg_price=60000.0,
            stop_loss=57000.0,
            unrealised_pnl=0.0,
            liq_price=0.0,
            leverage=5.0,
        )
        bot.state_mgr.save(bot.state)

        # 1. On the entry bar itself (bar_ts == entry_ts), stop check must NOT trigger
        hit_same_bar, _ = bot.exchange.check_dry_run_intrabar_stop(
            high=61000.0, low=56000.0, bar_ts=entry_ts, entry_bar_ts=entry_ts
        )
        self.assertFalse(hit_same_bar)

        # 2. On the subsequent bar (bar_ts > entry_ts), stop check DOES trigger
        next_bar_ts = entry_ts + 3600 * 1000
        stop_hit, stop_p = bot.exchange.check_dry_run_intrabar_stop(
            high=59000.0, low=56900.0, bar_ts=next_bar_ts, entry_bar_ts=entry_ts
        )
        self.assertTrue(stop_hit)
        self.assertEqual(stop_p, 57000.0)

        # Handle position closed
        bot._handle_position_closed(exit_reason="stop_loss", exit_price_hint=stop_p)

        self.assertIsNone(bot.state.position)
        self.assertEqual(len(bot.state.trades), 1)
        self.assertEqual(bot.state.trades[0]["why"], "stop_loss")
        self.assertEqual(bot.state.trades[0]["exit"], 57000.0)
        # Net PnL should be -$9.19 (-$9.00 raw minus ~$0.19 fees), float is ~$90.81
        self.assertAlmostEqual(bot.state.float_usdt, 90.81, places=2)
        self.assertAlmostEqual(bot.state.cum_fees, 0.19, delta=0.02)

        bot.stop()


if __name__ == "__main__":
    unittest.main()
