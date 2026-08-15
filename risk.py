"""Risk management and capital policy for rapid_bot."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CapitalState:
    float_usdt: float
    bank_usdt: float
    base_capital: float = 100.0


def apply_capital_policy(
    float_usdt: float,
    bank_usdt: float,
    base_capital: float = 100.0,
    policy: str = "skim_refill",
) -> Tuple[float, float]:
    """Apply capital policy after every closed trade (Section 1.6).

    Policies:
    1. 'skim_refill':
       if float > base:
           bank  += float - base
           float  = base
       elif float < base and bank > 0:
           top    = min(base - float, bank)
           float += top
           bank  -= top

    2. 'compound':
       no skimming or refill, float runs freely.
    """
    if policy == "compound":
        return float_usdt, bank_usdt

    if policy != "skim_refill":
        raise ValueError(f"Unknown capital policy: {policy}")

    # skim_refill logic
    if float_usdt > base_capital:
        bank_usdt += float_usdt - base_capital
        float_usdt = base_capital
    elif float_usdt < base_capital and bank_usdt > 0:
        top = min(base_capital - float_usdt, bank_usdt)
        float_usdt += top
        bank_usdt -= top

    return round(float_usdt, 4), round(bank_usdt, 4)


def check_account_floor(float_usdt: float, bank_usdt: float, floor_threshold: float = 40.0) -> Tuple[bool, str]:
    """Check if total capital (float + bank) is below the kill floor."""
    total = float_usdt + bank_usdt
    if total < floor_threshold:
        return True, f"KILL SWITCH: Total capital ({total:.2f} USDT) below floor ({floor_threshold:.2f} USDT)"
    return False, "OK"


def check_trade_loss_limit(trade_pnl: float, pre_trade_float: float, max_loss_pct: float = 0.15) -> Tuple[bool, str]:
    """Check if a single trade lost more than max_loss_pct of float (stop failure detector)."""
    if pre_trade_float <= 0:
        return True, "Pre-trade float <= 0"
    loss_pct = -trade_pnl / pre_trade_float
    if loss_pct > max_loss_pct:
        return True, f"KILL SWITCH: Single trade loss {loss_pct*100:.1f}% exceeded limit {max_loss_pct*100:.1f}% (stop failure detector)"
    return False, "OK"


def check_data_staleness(minutes_since_last_bar: float, max_stale_min: int = 180) -> Tuple[bool, str]:
    """Check if bar data has become stale."""
    if minutes_since_last_bar > max_stale_min:
        return True, f"KILL SWITCH: Data stale ({minutes_since_last_bar:.1f}m > {max_stale_min}m)"
    return False, "OK"


def check_api_failures(consec_failures: int, max_allowed: int = 10) -> Tuple[bool, str]:
    """Check consecutive API failures limit."""
    if consec_failures >= max_allowed:
        return True, f"KILL SWITCH: {consec_failures} consecutive API failures (limit {max_allowed})"
    return False, "OK"
