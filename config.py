"""Configuration loader and validator for rapid_bot."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # Identity
    bot_name: str
    symbol: str
    category: str
    interval: int  # minutes

    # Credentials & Mode
    bybit_api_key: str
    bybit_api_secret: str
    bybit_testnet: bool
    dry_run: bool

    # Strategy Parameters (FROZEN)
    entry_channel_hours: int
    exit_channel_hours: int
    atr_period: int
    atr_mult: float
    trend_veto_hours: int
    allow_long: bool
    allow_short: bool
    breakeven_enabled: bool
    breakeven_trigger_r: float

    # Sizing
    base_capital: float
    risk_frac: float
    max_leverage: float
    capital_policy: str  # skim_refill | compound

    # Risk & Kill Switches
    kill_total_below: float
    max_single_trade_loss_pct: float
    stale_data_halt_min: int
    max_consec_api_failures: int

    # Operations
    warmup_bars: int
    loop_interval_sec: int
    state_file: str
    log_file: str
    telegram_bot_token: str
    telegram_chat_id: str


def _strip_inline_comment(value: str) -> str:
    """Strip trailing inline comments starting with # while preserving quotes if any."""
    in_quote = False
    quote_char = ""
    result = []
    for char in value:
        if char in ('"', "'"):
            if not in_quote:
                in_quote = True
                quote_char = char
            elif quote_char == char:
                in_quote = False
        elif char == "#" and not in_quote:
            break
        result.append(char)
    return "".join(result).strip()


def parse_env_file(filepath: str | Path) -> dict[str, str]:
    """Parse .env file with support for comments and whitespace."""
    env_vars = {}
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = _strip_inline_comment(val)
                # Strip wrapping quotes
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                env_vars[key] = val
    return env_vars


def load_config(filepath: str | Path = ".env.rapid") -> Config:
    """Load, validate, and return Config dataclass."""
    env = parse_env_file(filepath)

    # Read variables with fallbacks or type conversions
    bot_name = env.get("BOT_NAME", "rapid_bot")
    symbol = env.get("SYMBOL", "BTCUSDT")
    category = env.get("CATEGORY", "linear")
    interval = int(env.get("INTERVAL", "60"))

    bybit_api_key = env.get("BYBIT_API_KEY", "")
    bybit_api_secret = env.get("BYBIT_API_SECRET", "")
    bybit_testnet = env.get("BYBIT_TESTNET", "1").lower() in ("1", "true", "yes")
    dry_run = env.get("DRY_RUN", "1").lower() in ("1", "true", "yes")

    entry_channel_hours = int(env.get("ENTRY_CHANNEL_HOURS", "960"))
    exit_channel_hours = int(env.get("EXIT_CHANNEL_HOURS", "480"))
    atr_period = int(env.get("ATR_PERIOD", "14"))
    atr_mult = float(env.get("ATR_MULT", "3.0"))
    trend_veto_hours = int(env.get("TREND_VETO_HOURS", "720"))
    allow_long = env.get("ALLOW_LONG", "1").lower() in ("1", "true", "yes")
    allow_short = env.get("ALLOW_SHORT", "1").lower() in ("1", "true", "yes")

    breakeven_enabled = env.get("BREAKEVEN_ENABLED", "1").lower() in ("1", "true", "yes")
    breakeven_trigger_r = float(env.get("BREAKEVEN_TRIGGER_R", "2.0"))

    base_capital = float(env.get("BASE_CAPITAL", "100.0"))
    risk_frac = float(env.get("RISK_FRAC", "0.05"))
    max_leverage = float(env.get("MAX_LEVERAGE", "5.0"))
    capital_policy = env.get("CAPITAL_POLICY", "skim_refill").lower()

    kill_total_below = float(env.get("KILL_TOTAL_BELOW", "40.0"))
    max_single_trade_loss_pct = float(env.get("MAX_SINGLE_TRADE_LOSS_PCT", "0.15"))
    stale_data_halt_min = int(env.get("STALE_DATA_HALT_MIN", "180"))
    max_consec_api_failures = int(env.get("MAX_CONSEC_API_FAILURES", "10"))

    warmup_bars = int(env.get("WARMUP_BARS", "1200"))
    loop_interval_sec = int(env.get("LOOP_INTERVAL_SEC", "60"))
    state_file = env.get("STATE_FILE", "rapid_state.json")
    log_file = env.get("LOG_FILE", "rapid_bot.log")
    telegram_bot_token = env.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = env.get("TELEGRAM_CHAT_ID", "")

    # Validation rules
    assert interval == 60, f"INTERVAL must be 60 (1-hour bars), got {interval}"
    assert entry_channel_hours == 960, f"ENTRY_CHANNEL_HOURS must be 960, got {entry_channel_hours}"
    assert exit_channel_hours == 480, f"EXIT_CHANNEL_HOURS must be 480, got {exit_channel_hours}"
    assert atr_period == 14, f"ATR_PERIOD must be 14, got {atr_period}"
    assert atr_mult == 3.0, f"ATR_MULT must be 3.0, got {atr_mult}"
    assert warmup_bars >= entry_channel_hours + atr_period, (
        f"WARMUP_BARS ({warmup_bars}) must be >= {entry_channel_hours + atr_period}"
    )
    assert breakeven_trigger_r > 0, f"BREAKEVEN_TRIGGER_R must be > 0, got {breakeven_trigger_r}"
    assert capital_policy in ("skim_refill", "compound"), f"Invalid capital policy: {capital_policy}"
    assert base_capital > 0, "BASE_CAPITAL must be positive"
    assert 0 < risk_frac <= 0.20, f"RISK_FRAC must be in (0, 0.20], got {risk_frac}"
    assert max_leverage >= 1.0, f"MAX_LEVERAGE must be >= 1.0, got {max_leverage}"

    # In live trading (DRY_RUN=0), credentials must be present
    if not dry_run:
        if not bybit_api_key or not bybit_api_secret:
            raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set when DRY_RUN=0")

    return Config(
        bot_name=bot_name,
        symbol=symbol,
        category=category,
        interval=interval,
        bybit_api_key=bybit_api_key,
        bybit_api_secret=bybit_api_secret,
        bybit_testnet=bybit_testnet,
        dry_run=dry_run,
        entry_channel_hours=entry_channel_hours,
        exit_channel_hours=exit_channel_hours,
        atr_period=atr_period,
        atr_mult=atr_mult,
        trend_veto_hours=trend_veto_hours,
        allow_long=allow_long,
        allow_short=allow_short,
        breakeven_enabled=breakeven_enabled,
        breakeven_trigger_r=breakeven_trigger_r,
        base_capital=base_capital,
        risk_frac=risk_frac,
        max_leverage=max_leverage,
        capital_policy=capital_policy,
        kill_total_below=kill_total_below,
        max_single_trade_loss_pct=max_single_trade_loss_pct,
        stale_data_halt_min=stale_data_halt_min,
        max_consec_api_failures=max_consec_api_failures,
        warmup_bars=warmup_bars,
        loop_interval_sec=loop_interval_sec,
        state_file=state_file,
        log_file=log_file,
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id,
    )
