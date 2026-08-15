"""Unit tests for atomic state management."""

import os
import tempfile
import unittest
from pathlib import Path
from state import BotState, PositionState, StateManager


class TestState(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_file = Path(self.temp_dir.name) / "test_state.json"
        self.mgr = StateManager(str(self.state_file))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_fresh_state(self):
        state = self.mgr.load(default_float=100.0)
        self.assertEqual(state.float_usdt, 100.0)
        self.assertEqual(state.bank_usdt, 0.0)
        self.assertIsNone(state.position)
        self.assertTrue(self.state_file.exists())

    def test_save_and_reload(self):
        state = self.mgr.load(default_float=100.0)
        state.float_usdt = 100.0
        state.bank_usdt = 35.5
        state.position = PositionState(
            side="Buy",
            qty=0.003,
            entry_price=60000.0,
            stop_price=57000.0,
            entry_bar_ts=1700000000000,
            entry_order_id="test-order-1",
        )
        self.mgr.save(state)

        reloaded = self.mgr.load()
        self.assertEqual(reloaded.float_usdt, 100.0)
        self.assertEqual(reloaded.bank_usdt, 35.5)
        self.assertIsNotNone(reloaded.position)
        self.assertEqual(reloaded.position.side, "Buy")
        self.assertEqual(reloaded.position.qty, 0.003)
        self.assertEqual(reloaded.position.entry_price, 60000.0)


if __name__ == "__main__":
    unittest.main()
