"""Live data manager for rapid_bot.
Handles warmup kline history, closed-bar detection on 1h boundaries, and 8h funding rate tracking.
"""

import logging
import time
from typing import Optional, Tuple
import pandas as pd
from exchange import BybitExchange

logger = logging.getLogger("rapid_bot.data")


class DataManager:
    def __init__(self, exchange: BybitExchange, warmup_bars: int = 1200, interval: str = "60"):
        self.exchange = exchange
        self.warmup_bars = warmup_bars
        self.interval = interval
        self.df_klines: pd.DataFrame = pd.DataFrame()
        self.last_committed_bar_ts: int = 0
        self.last_funding_ts: int = 0
        self.current_funding_rate: float = 0.0

    def initialize_warmup(self) -> pd.DataFrame:
        """Fetch warmup klines at startup. Assert at least 974 bars."""
        logger.info(f"Fetching {self.warmup_bars} warmup klines from Bybit...")
        df = self.exchange.fetch_klines(count=self.warmup_bars, interval=self.interval)

        if df.empty or len(df) < 974:
            raise RuntimeError(
                f"Insufficient warmup bars: fetched {len(df)}, required at least 974 (960 entry + 14 ATR). Refusing to trade."
            )

        now_ms = int(time.time() * 1000)
        latest_ts = df["timestamp"].iloc[-1]
        if now_ms < latest_ts + 3600 * 1000:
            df = df.iloc[:-1].copy()

        self.df_klines = df
        self.last_committed_bar_ts = int(self.df_klines["timestamp"].iloc[-1])
        logger.info(
            f"Warmup complete. {len(self.df_klines)} closed bars loaded. Latest closed bar: {self.df_klines.index[-1]} (ts: {self.last_committed_bar_ts})"
        )
        return self.df_klines

    def check_for_new_closed_bar(self) -> Tuple[bool, Optional[pd.Series], pd.DataFrame]:
        """Poll latest klines and check if a new 1-hour bar has closed.
        Does not mutate committed timestamp until caller confirms processing.
        Returns: (is_new_bar, closed_bar_series, full_dataframe)
        """
        recent_df = self.exchange.fetch_klines(count=5, interval=self.interval)
        if recent_df.empty:
            return False, None, self.df_klines

        now_ms = int(time.time() * 1000)
        # Bar closed if now >= bar_start_ts + 3600_000
        closed_candidates = recent_df[now_ms >= recent_df["timestamp"] + 3600 * 1000]
        if closed_candidates.empty:
            return False, None, self.df_klines

        newest_closed_bar = closed_candidates.iloc[-1]
        newest_closed_ts = int(newest_closed_bar["timestamp"])

        if newest_closed_ts > self.last_committed_bar_ts:
            combined = pd.concat([self.df_klines, closed_candidates])
            combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            if len(combined) > self.warmup_bars + 100:
                combined = combined.iloc[-self.warmup_bars:]

            self.df_klines = combined
            return True, newest_closed_bar, self.df_klines

        return False, None, self.df_klines

    def commit_bar(self, bar_ts: int):
        """Commit bar timestamp only after successful strategy processing."""
        self.last_committed_bar_ts = bar_ts

    def fetch_latest_funding_rate(self, symbol: str = "BTCUSDT") -> Tuple[float, int]:
        """Fetch current 8h funding rate from Bybit linear. Returns (rate, timestamp)."""
        try:
            resp = self.exchange.session.get_funding_rate_history(
                category="linear",
                symbol=symbol,
                limit=1,
            )
            if resp.get("retCode") == 0:
                items = resp.get("result", {}).get("list", [])
                if items:
                    self.current_funding_rate = float(items[0].get("fundingRate", 0.0))
                    self.last_funding_ts = int(items[0].get("fundingRateTimestamp", 0))
                    return self.current_funding_rate, self.last_funding_ts
        except Exception as e:
            logger.warning(f"Error querying latest funding rate: {e}")
        return self.current_funding_rate, self.last_funding_ts
