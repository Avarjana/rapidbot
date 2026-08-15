"""Exchange layer for rapid_bot wrapping Bybit V5 Unified Trading Account (UTA).
Handles linear perpetuals, exchange-side stop placement, order reconciliation, and DRY_RUN simulation.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import pybit._helpers as pybit_helpers
from pybit.unified_trading import HTTP
from strategy import compute_stop_price

logger = logging.getLogger("rapid_bot.exchange")


@dataclass
class InstrumentFilters:
    tick_size: float
    qty_step: float
    min_order_qty: float
    min_notional: float


@dataclass
class PositionInfo:
    side: str  # "None", "Buy", "Sell"
    size: float
    avg_price: float
    stop_loss: float
    unrealised_pnl: float
    liq_price: float
    leverage: float


@dataclass
class ClosedPnlResult:
    closed_pnl: float  # Net realized PnL (already net of fees)
    avg_entry_price: float
    avg_exit_price: float
    closed_size: float
    cum_fee: float  # Derived for reporting only
    exec_time: str


class BybitExchange:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        dry_run: bool = True,
        symbol: str = "BTCUSDT",
        category: str = "linear",
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.dry_run = dry_run
        self.symbol = symbol
        self.category = category

        # Initialize Pybit HTTP client with generous recv_window
        self.session = HTTP(
            testnet=self.testnet,
            api_key=self.api_key if not self.dry_run else "",
            api_secret=self.api_secret if not self.dry_run else "",
            recv_window=10000,
        )

        # Synchronize pybit timestamp with Bybit server time to avoid ErrCode 10002
        self._sync_pybit_time()

        # Mock state for DRY_RUN
        self._mock_position = PositionInfo(
            side="None",
            size=0.0,
            avg_price=0.0,
            stop_loss=0.0,
            unrealised_pnl=0.0,
            liq_price=0.0,
            leverage=5.0,
        )
        self._mock_last_closed_pnl: Optional[ClosedPnlResult] = None
        self._last_market_price = 0.0

    def _sync_pybit_time(self) -> None:
        """Synchronize pybit timestamp generation with Bybit server time to eliminate ErrCode 10002 errors."""
        try:
            res = self.session.get_server_time()
            if res.get("retCode") == 0:
                server_time_ms = int(res["result"]["timeNano"]) // 1_000_000
                local_time_ms = int(time.time() * 1000)
                offset_ms = server_time_ms - local_time_ms
                pybit_helpers.generate_timestamp = lambda: int(time.time() * 1000) + offset_ms
                if abs(offset_ms) > 100:
                    logger.info(f"Synchronized pybit timestamp offset with Bybit server: {offset_ms:+d} ms")
        except Exception as e:
            logger.warning(f"Failed to synchronize server time with Bybit: {e}")

    def sync_pybit_time(self) -> None:
        """Public alias to re-synchronize pybit timestamp generation."""
        self._sync_pybit_time()

    def get_instruments_info(self) -> InstrumentFilters:
        """Fetch tick size, lot step, min order qty, and min notional from Bybit V5.
        Fails fast if API call fails (never hardcodes filters per §1.3).
        """
        try:
            resp = self.session.get_instruments_info(category=self.category, symbol=self.symbol)
            if resp.get("retCode") == 0:
                item = resp["result"]["list"][0]
                price_filter = item.get("priceFilter", {})
                lot_size_filter = item.get("lotSizeFilter", {})

                tick_size = float(price_filter.get("tickSize"))
                qty_step = float(lot_size_filter.get("qtyStep"))
                min_qty = float(lot_size_filter.get("minOrderQty"))
                min_notional = float(lot_size_filter.get("minNotionalValue", "5.0"))

                return InstrumentFilters(
                    tick_size=tick_size,
                    qty_step=qty_step,
                    min_order_qty=min_qty,
                    min_notional=min_notional,
                )
            else:
                raise RuntimeError(f"Bybit API error getting instruments info: {resp.get('retMsg')}")
        except Exception as e:
            if not self.dry_run:
                logger.critical(f"FATAL: Could not fetch instrument filters from exchange: {e}")
                raise RuntimeError(f"Failed to fetch live instrument filters for {self.symbol}: {e}") from e
            logger.warning(f"[DRY_RUN] Using standard BTCUSDT filters on error: {e}")
            return InstrumentFilters(tick_size=0.10, qty_step=0.001, min_order_qty=0.001, min_notional=5.0)

    def assert_one_way_mode(self, max_retries: int = 3):
        """Assert account is in One-Way Position Mode (positionIdx=0) per §5."""
        if self.dry_run:
            logger.info("[DRY_RUN] One-way mode assertion passed.")
            return

        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.session.get_positions(category=self.category, symbol=self.symbol)
                if resp.get("retCode") == 0:
                    pos_list = resp.get("result", {}).get("list", [])
                    for p in pos_list:
                        pos_idx = p.get("positionIdx", 0)
                        if pos_idx != 0:
                            raise RuntimeError(f"Account is in Hedge Mode (positionIdx={pos_idx}). rapid_bot requires One-Way Mode (positionIdx=0).")
                    logger.info("One-way position mode (positionIdx=0) verified on exchange.")
                    return
                else:
                    last_err = RuntimeError(f"Failed to verify position mode: {resp.get('retMsg')}")
            except Exception as e:
                if "Hedge Mode" in str(e):
                    logger.critical(f"FATAL: One-way position mode check failed: {e}")
                    raise
                last_err = e
                logger.warning(f"Position mode check retry {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(1.0)

        logger.critical(f"FATAL: One-way position mode check failed after retries: {last_err}")
        raise RuntimeError(f"One-way position mode check failed: {last_err}") from last_err

    def set_leverage(self, leverage: int = 5) -> bool:
        """Set leverage to 5x. Tolerate 'leverage not modified' response."""
        if self.dry_run:
            logger.info(f"[DRY_RUN] Leverage set to {leverage}x")
            return True

        try:
            resp = self.session.set_leverage(
                category=self.category,
                symbol=self.symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            if resp.get("retCode") == 0:
                logger.info(f"Leverage set to {leverage}x successfully")
                return True
        except Exception as e:
            err_str = str(e).lower()
            if "not modified" in err_str or "110043" in err_str:
                logger.info(f"Leverage is already {leverage}x (not modified)")
                return True
            logger.error(f"Failed to set leverage: {e}")
            return False
        return True

    def fetch_klines(self, count: int = 1200, interval: str = "60") -> pd.DataFrame:
        """Fetch backward-paginated klines (Section 3.2)."""
        all_bars = []
        end_time_ms = int(time.time() * 1000)

        limit = 1000
        while len(all_bars) < count:
            fetch_limit = min(limit, count - len(all_bars))
            try:
                resp = self.session.get_kline(
                    category=self.category,
                    symbol=self.symbol,
                    interval=interval,
                    end=end_time_ms,
                    limit=fetch_limit,
                )
                if resp.get("retCode") != 0:
                    break
                k_list = resp.get("result", {}).get("list", [])
                if not k_list:
                    break

                for item in k_list:
                    all_bars.append({
                        "timestamp": int(item[0]),
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                    })

                oldest_ts = min(int(item[0]) for item in k_list)
                end_time_ms = oldest_ts - 1
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"Error fetching klines: {e}")
                break

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame(all_bars).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime")
        return df

    def get_position(self) -> PositionInfo:
        """Fetch current open position on exchange."""
        if self.dry_run:
            return self._mock_position

        try:
            resp = self.session.get_positions(category=self.category, symbol=self.symbol)
            if resp.get("retCode") == 0:
                pos_list = resp.get("result", {}).get("list", [])
                for p in pos_list:
                    size = float(p.get("size", 0.0))
                    if size > 0:
                        side = p.get("side")  # "Buy" or "Sell"
                        avg_price = float(p.get("avgPrice", 0.0))
                        stop_loss = float(p.get("stopLoss", 0.0) or 0.0)
                        unrealised_pnl = float(p.get("unrealisedPnl", 0.0) or 0.0)
                        liq_price = float(p.get("liqPrice", 0.0) or 0.0)
                        leverage = float(p.get("leverage", 5.0) or 5.0)
                        return PositionInfo(
                            side=side,
                            size=size,
                            avg_price=avg_price,
                            stop_loss=stop_loss,
                            unrealised_pnl=unrealised_pnl,
                            liq_price=liq_price,
                            leverage=leverage,
                        )
        except Exception as e:
            logger.error(f"Error fetching position: {e}")

        return PositionInfo(side="None", size=0.0, avg_price=0.0, stop_loss=0.0, unrealised_pnl=0.0, liq_price=0.0, leverage=5.0)

    def check_dry_run_intrabar_stop(self, high: float, low: float, bar_ts: int, entry_bar_ts: int) -> Tuple[bool, float]:
        """Check if intrabar high/low triggers stop loss in DRY_RUN mode.
        Strictly skips the entry bar itself (bar_ts <= entry_bar_ts) to match backtest.py.
        """
        if not self.dry_run or self._mock_position.size == 0:
            return False, 0.0

        if bar_ts <= entry_bar_ts:
            return False, 0.0

        pos = self._mock_position
        stop_price = pos.stop_loss
        stopped = False
        if pos.side == "Buy" and low <= stop_price:
            stopped = True
        elif pos.side == "Sell" and high >= stop_price:
            stopped = True

        if stopped:
            logger.info(f"[DRY_RUN] Intrabar stop-out triggered at ${stop_price:,.2f}")
            # Calculate mock closed pnl (already net of fees)
            if pos.side == "Buy":
                raw_pnl = pos.size * (stop_price - pos.avg_price)
            else:
                raw_pnl = pos.size * (pos.avg_price - stop_price)
            fee = pos.size * (pos.avg_price + stop_price) * 0.00055
            net_pnl = raw_pnl - fee

            self._mock_last_closed_pnl = ClosedPnlResult(
                closed_pnl=net_pnl,
                avg_entry_price=pos.avg_price,
                avg_exit_price=stop_price,
                closed_size=pos.size,
                cum_fee=fee,
                exec_time=str(time.time()),
            )
            self._mock_position = PositionInfo(
                side="None", size=0.0, avg_price=0.0, stop_loss=0.0, unrealised_pnl=0.0, liq_price=0.0, leverage=5.0
            )
            return True, stop_price

        return False, 0.0

    def place_entry_with_stop(
        self,
        side: str,
        qty: float,
        atr_val: float,
        atr_mult: float,
        bar_ts: int,
        tick_size: float = 0.10,
    ) -> Tuple[bool, Optional[str], float, float]:
        """Execute Section 6.1 Order Lifecycle:
        1. Place market entry order
        2. Poll actual average fill price from exchange
        3. Recompute stop price strictly from actual average fill price (§6.1 Steps 4–5)
        4. Attach exchange stop via set_trading_stop
        5. Verify stop is present.
        If stop placement/verification fails -> CLOSE IMMEDIATELY (flatten).

        Returns: (success, order_id, fill_price, actual_stop_price)
        """
        order_link_id = f"rapid-{bar_ts}-entry"

        if self.dry_run:
            fill_price = self._last_market_price if self._last_market_price > 0 else 60000.0
            actual_stop = compute_stop_price(side, fill_price, atr_val, atr_mult, tick_size)
            logger.info(f"[DRY_RUN] Entry {side} {qty} BTC @ ${fill_price:,.2f}, Stop: ${actual_stop:,.2f}")
            self._mock_position = PositionInfo(
                side=side,
                size=qty,
                avg_price=fill_price,
                stop_loss=actual_stop,
                unrealised_pnl=0.0,
                liq_price=0.0,
                leverage=5.0,
            )
            return True, f"dry-{bar_ts}", fill_price, actual_stop

        # 1. Place Market Order
        order_id = None
        try:
            resp = self.session.place_order(
                category=self.category,
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                reduceOnly=False,
                positionIdx=0,
                orderLinkId=order_link_id,
            )
            if resp.get("retCode") == 0:
                order_id = resp.get("result", {}).get("orderId")
                logger.info(f"Market entry placed: {order_id} ({order_link_id})")
            else:
                logger.error(f"Market order placement failed: {resp.get('retMsg')}")
                return False, None, 0.0, 0.0
        except Exception as e:
            logger.error(f"Exception placing market entry: {e}")
            return False, None, 0.0, 0.0

        # 2. Poll for fill & actual average price
        avg_fill_price = 0.0
        for _ in range(10):
            time.sleep(0.5)
            pos = self.get_position()
            if pos.side == side and pos.size >= qty:
                avg_fill_price = pos.avg_price
                break

        if avg_fill_price <= 0:
            logger.warning("Could not obtain fill price from position. Checking order history...")
            try:
                hist = self.session.get_order_history(category=self.category, symbol=self.symbol, orderId=order_id)
                if hist.get("retCode") == 0:
                    orders = hist.get("result", {}).get("list", [])
                    if orders:
                        avg_fill_price = float(orders[0].get("avgPrice", 0.0))
            except Exception as e:
                logger.error(f"Error querying order history: {e}")

        if avg_fill_price <= 0:
            logger.critical("Failed to confirm entry fill! Flattening position for safety.")
            self.place_reduce_only_market(side="Sell" if side == "Buy" else "Buy", qty=qty, bar_ts=bar_ts)
            return False, order_id, 0.0, 0.0

        # 3. Recompute exact stop price strictly from actual average fill price (§6.1 Steps 4–5)
        actual_stop_price = compute_stop_price(side, avg_fill_price, atr_val, atr_mult, tick_size)

        # 4. Attach stop loss on exchange with retries
        stop_set = False
        for attempt in range(3):
            try:
                s_resp = self.session.set_trading_stop(
                    category=self.category,
                    symbol=self.symbol,
                    stopLoss=str(actual_stop_price),
                    positionIdx=0,
                )
                if s_resp.get("retCode") == 0:
                    stop_set = True
                    break
                logger.warning(f"Retry {attempt+1} set_trading_stop: {s_resp.get('retMsg')}")
            except Exception as e:
                logger.warning(f"Retry {attempt+1} set_trading_stop exception: {e}")
            time.sleep(0.5)

        # 5. Verify stop exists on position
        stop_verified = False
        if stop_set:
            time.sleep(0.5)
            pos = self.get_position()
            if pos.stop_loss > 0 and abs(pos.stop_loss - actual_stop_price) <= tick_size:
                stop_verified = True

        # 6. Mandatory Guard (Section 6.1 Step 8): If stop fails, FLATTEN IMMEDIATELY
        if not stop_verified:
            logger.critical("GUARD TRIPPED: Stop loss could not be set/verified on exchange! Flattening position immediately.")
            self.place_reduce_only_market(side="Sell" if side == "Buy" else "Buy", qty=qty, bar_ts=bar_ts)
            return False, order_id, avg_fill_price, 0.0

        logger.info(f"Entry confirmed: {side} {qty} BTC @ ${avg_fill_price:,.2f}, Stop: ${actual_stop_price:,.2f} verified on exchange")
        return True, order_id, avg_fill_price, actual_stop_price

    def ensure_trading_stop(self, symbol: str, expected_stop_price: float, tick_size: float = 0.1) -> bool:
        """Verify exchange stop exists and matches expected stop. Re-places with retries, or signals flatten if unfixable (§6.3)."""
        if self.dry_run:
            return True

        pos = self.get_position()
        if pos.size == 0:
            return True

        # Check if stop matches within tick tolerance
        if pos.stop_loss > 0 and abs(pos.stop_loss - expected_stop_price) <= tick_size:
            return True

        logger.warning(f"Stop loss missing/divergent (found: {pos.stop_loss}, expected: {expected_stop_price}). Attempting re-placement...")
        for attempt in range(3):
            try:
                resp = self.session.set_trading_stop(
                    category=self.category,
                    symbol=symbol,
                    stopLoss=str(expected_stop_price),
                    positionIdx=0,
                )
                if resp.get("retCode") == 0:
                    time.sleep(0.5)
                    verified_pos = self.get_position()
                    if verified_pos.stop_loss > 0 and abs(verified_pos.stop_loss - expected_stop_price) <= tick_size:
                        logger.info("Stop loss successfully restored on exchange.")
                        return True
            except Exception as e:
                logger.error(f"Retry {attempt+1} restoring stop failed: {e}")
            time.sleep(0.5)

        logger.critical("Stop loss could NOT be restored on exchange after retries!")
        return False

    def place_reduce_only_market(self, side: str, qty: float, bar_ts: int = 0) -> Tuple[bool, float]:
        """Place reduce-only market order to close position."""
        order_link_id = f"rapid-{bar_ts}-exit"

        if self.dry_run:
            exit_price = self._last_market_price if self._last_market_price > 0 else 60000.0
            pos = self._mock_position
            if pos.side == "Buy":
                raw_pnl = qty * (exit_price - pos.avg_price)
            else:
                raw_pnl = qty * (pos.avg_price - exit_price)
            fee = qty * (pos.avg_price + exit_price) * 0.00055
            net_pnl = raw_pnl - fee

            self._mock_last_closed_pnl = ClosedPnlResult(
                closed_pnl=net_pnl,
                avg_entry_price=pos.avg_price,
                avg_exit_price=exit_price,
                closed_size=qty,
                cum_fee=fee,
                exec_time=str(time.time()),
            )
            self._mock_position = PositionInfo(
                side="None", size=0.0, avg_price=0.0, stop_loss=0.0, unrealised_pnl=0.0, liq_price=0.0, leverage=5.0
            )
            logger.info(f"[DRY_RUN] Exit {side} {qty} BTC @ ${exit_price:,.2f} (reduce-only)")
            return True, exit_price

        try:
            resp = self.session.place_order(
                category=self.category,
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                reduceOnly=True,
                positionIdx=0,
                orderLinkId=order_link_id,
            )
            if resp.get("retCode") == 0:
                logger.info(f"Exit order placed: {resp.get('result', {}).get('orderId')}")
                # Confirm position is actually filled and size drops to 0 on exchange
                for attempt in range(10):
                    time.sleep(0.5)
                    pos = self.get_position()
                    if pos.size == 0.0:
                        return True, 0.0
                logger.error("Exit order placed, but position size did not reduce to 0 on exchange within timeout.")
                return False, 0.0
            else:
                logger.error(f"Exit order failed: {resp.get('retMsg')}")
                return False, 0.0
        except Exception as e:
            logger.error(f"Exception placing exit order: {e}")
            return False, 0.0

    def get_last_closed_pnl(
        self,
        symbol: str,
        min_entry_ts: int = 0,
        expected_qty: float = 0.0,
        timeout_sec: float = 5.0,
    ) -> Optional[ClosedPnlResult]:
        """Fetch the most recent closed trade from Bybit V5 get_closed_pnl.
        Polls until the published record is newer than entry_ts and matches closedSize,
        preventing booking the previous trade.
        """
        if self.dry_run:
            return self._mock_last_closed_pnl

        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            try:
                resp = self.session.get_closed_pnl(category=self.category, symbol=symbol, limit=3)
                if resp.get("retCode") == 0:
                    items = resp.get("result", {}).get("list", [])
                    for item in items:
                        up_time = int(item.get("updatedTime", item.get("createdTime", 0)))
                        size = float(item.get("closedSize", 0.0))

                        # Verify record is fresh (updated after trade entry/exit order)
                        if min_entry_ts > 0 and up_time < min_entry_ts:
                            continue

                        # Verify size matches
                        if expected_qty > 0 and abs(size - expected_qty) > 1e-4:
                            continue

                        closed_pnl = float(item.get("closedPnl", 0.0))
                        entry_p = float(item.get("avgEntryPrice", 0.0))
                        exit_p = float(item.get("avgExitPrice", 0.0))
                        # Derive fee for reporting from cumEntryValue / cumExitValue
                        cum_entry_val = float(item.get("cumEntryValue", entry_p * size))
                        cum_exit_val = float(item.get("cumExitValue", exit_p * size))
                        derived_fee = (cum_entry_val + cum_exit_val) * 0.00055

                        return ClosedPnlResult(
                            closed_pnl=closed_pnl,  # Already net of both fees
                            avg_entry_price=entry_p,
                            avg_exit_price=exit_p,
                            closed_size=size,
                            cum_fee=derived_fee,
                            exec_time=str(up_time),
                        )
            except Exception as e:
                logger.error(f"Error querying closed PnL: {e}")
            time.sleep(0.5)

        logger.warning("Could not find matching fresh closed PnL record within timeout.")
        return None

    def get_funding_history(self, symbol: str, start_time_ms: int = 0) -> List[Dict[str, Any]]:
        """Fetch funding rate history from Bybit V5 for backfilling missed funding payments."""
        if self.dry_run:
            return []
        try:
            kwargs = {"category": self.category, "symbol": symbol, "limit": 50}
            if start_time_ms > 0:
                kwargs["startTime"] = start_time_ms
                kwargs["endTime"] = int(time.time() * 1000)
            resp = self.session.get_funding_rate_history(**kwargs)
            if resp.get("retCode") == 0:
                return resp.get("result", {}).get("list", [])
        except Exception as e:
            logger.warning(f"Failed to fetch funding rate history: {e}")
        return []

    def set_last_market_price(self, price: float):
        """Update last known market price for dry run tracking."""
        self._last_market_price = price
