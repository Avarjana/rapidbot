"""Atomic state persistence for rapid_bot.
Ensures crash-safe state updates using temporary file + os.replace.
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PositionState:
    side: str  # "Buy" or "Sell"
    qty: float
    entry_price: float
    stop_price: float
    entry_bar_ts: int
    entry_order_id: str
    initial_stop_price: float = 0.0  # frozen at entry; stop_price may move (breakeven), this doesn't
    breakeven_triggered: bool = False


@dataclass
class BotState:
    version: int = 1
    float_usdt: float = 100.0
    bank_usdt: float = 0.0
    last_processed_bar_ts: int = 0
    last_processed_funding_ts: int = 0
    position: Optional[PositionState] = None
    trades: List[Dict[str, Any]] = field(default_factory=list)
    cum_fees: float = 0.0
    cum_funding: float = 0.0
    halted: bool = False
    halt_reason: Optional[str] = None
    paused: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotState":
        pos_data = data.get("position")
        pos = PositionState(**pos_data) if pos_data else None
        return cls(
            version=data.get("version", 1),
            float_usdt=float(data.get("float_usdt", 100.0)),
            bank_usdt=float(data.get("bank_usdt", 0.0)),
            last_processed_bar_ts=int(data.get("last_processed_bar_ts", 0)),
            last_processed_funding_ts=int(data.get("last_processed_funding_ts", 0)),
            position=pos,
            trades=data.get("trades", []),
            cum_fees=float(data.get("cum_fees", 0.0)),
            cum_funding=float(data.get("cum_funding", 0.0)),
            halted=bool(data.get("halted", False)),
            halt_reason=data.get("halt_reason"),
            paused=bool(data.get("paused", False)),
        )


class StateManager:
    def __init__(self, filepath: str = "rapid_state.json"):
        self.filepath = Path(filepath)

    def load(self, default_float: float = 100.0) -> BotState:
        """Load state from JSON file or initialize fresh state."""
        if not self.filepath.exists():
            state = BotState(float_usdt=default_float)
            self.save(state)
            return state

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BotState.from_dict(data)
        except Exception as e:
            print(f"[State] Error loading state from {self.filepath}: {e}. Creating fresh backup.")
            backup_path = self.filepath.with_suffix(".corrupt.bak")
            if self.filepath.exists():
                self.filepath.rename(backup_path)
            state = BotState(float_usdt=default_float)
            self.save(state)
            return state

    def save(self, state: BotState) -> None:
        """Atomically persist state using tempfile + os.replace."""
        data = state.to_dict()
        dir_name = self.filepath.parent
        dir_name.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
            json.dump(data, tf, indent=2)
            temp_path = Path(tf.name)

        os.replace(temp_path, self.filepath)
