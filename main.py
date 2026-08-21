"""Main executable for rapid_bot.
Orchestrates data, pure strategy evaluation, exchange execution, risk checks, atomic state, and Telegram alerting.
"""

import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import requests
from pybit.exceptions import FailedRequestError, InvalidRequestError

from config import Config, load_config
from data import DataManager
from exchange import BybitExchange, ClosedPnlResult, InstrumentFilters, PositionInfo
from notify import TelegramNotifier
from risk import (
    apply_capital_policy,
    check_account_floor,
    check_api_failures,
    check_data_staleness,
    check_trade_loss_limit,
)
from state import BotState, PositionState, StateManager
from strategy import (
    check_breakeven_trigger,
    check_channel_exit,
    compute_breakeven_stop_price,
    compute_indicators,
    compute_position_size,
    compute_stop_price,
    evaluate_signal,
)

try:
    import msvcrt
except ImportError:
    msvcrt = None

try:
    import fcntl
except ImportError:
    fcntl = None


class ProcessLock:
    def __init__(self, lock_file: str = "rapid_bot.lock"):
        self.lock_file = lock_file
        self._file = None

    def acquire(self) -> bool:
        try:
            self._file = open(self.lock_file, "a+")
            self._file.seek(0)
            if msvcrt:
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            elif fcntl:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._file.seek(0)
            self._file.truncate()
            self._file.write(str(os.getpid()))
            self._file.flush()
            return True
        except (IOError, OSError):
            if self._file:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None
            return False

    def release(self):
        if self._file:
            try:
                if msvcrt:
                    self._file.seek(0)
                    msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
                elif fcntl:
                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
                self._file.close()
                self._file = None
            except (IOError, OSError):
                pass
            try:
                if os.path.exists(self.lock_file):
                    os.remove(self.lock_file)
            except (IOError, OSError):
                pass


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("rapid_bot")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (INFO+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (DEBUG+)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class RapidBot:
    def __init__(self, config_path: str = ".env.rapid"):
        self.cfg: Config = load_config(config_path)
        self.logger = setup_logger(self.cfg.log_file)

        self.lock_file = f"{self.cfg.bot_name}.lock"
        self.proc_lock = ProcessLock(self.lock_file)
        if not self.proc_lock.acquire():
            self.logger.critical(f"FATAL: Another instance of {self.cfg.bot_name} is already running (lock file '{self.lock_file}' held). Exiting.")
            sys.exit(1)

        self.logger.info(f"Initializing {self.cfg.bot_name} (Dry Run: {self.cfg.dry_run}, Testnet: {self.cfg.bybit_testnet})")

        self.lock = threading.RLock()
        self.state_mgr = StateManager(self.cfg.state_file)
        self.state: BotState = self.state_mgr.load(default_float=self.cfg.base_capital)

        self.exchange = BybitExchange(
            api_key=self.cfg.bybit_api_key,
            api_secret=self.cfg.bybit_api_secret,
            testnet=self.cfg.bybit_testnet,
            dry_run=self.cfg.dry_run,
            symbol=self.cfg.symbol,
            category=self.cfg.category,
        )

        self.data_mgr = DataManager(
            exchange=self.exchange,
            warmup_bars=self.cfg.warmup_bars,
            interval=str(self.cfg.interval),
        )

        self.notifier = TelegramNotifier(
            token=self.cfg.telegram_bot_token,
            chat_id=self.cfg.telegram_chat_id,
            bot_name=self.cfg.bot_name,
        )

        self.filters: InstrumentFilters = InstrumentFilters(0.1, 0.001, 0.001, 5.0)
        self.running = False
        self.consec_api_failures = 0
        self.last_funding_check_hour = -1

        self._setup_telegram_commands()

    def _setup_telegram_commands(self):
        self.notifier.register_command("/status", self._cmd_status)
        self.notifier.register_command("/pause", self._cmd_pause)
        self.notifier.register_command("/resume", self._cmd_resume)
        self.notifier.register_command("/flatten", self._cmd_flatten)

    def _cmd_status(self) -> str:
        with self.lock:
            pos = self.exchange.get_position()
            pos_str = "Flat"
            if pos.size > 0:
                dist_to_stop = abs(pos.avg_price - pos.stop_loss) if pos.stop_loss > 0 else 0.0
                dist_pct = (dist_to_stop / pos.avg_price * 100.0) if pos.avg_price > 0 else 0.0
                pos_str = (
                    f"Side: {pos.side}, Qty: {pos.size} BTC\n"
                    f"Entry: ${pos.avg_price:,.2f}, Stop: ${pos.stop_loss:,.2f} ({dist_pct:.2f}% dist)\n"
                    f"Unrealised PnL: ${pos.unrealised_pnl:+.2f} USDT"
                )
            status_msg = (
                f"🤖 *{self.cfg.bot_name} Status*\n"
                f"• Float: ${self.state.float_usdt:.2f} USDT\n"
                f"• Bank: ${self.state.bank_usdt:.2f} USDT\n"
                f"• Total Equity: ${self.state.float_usdt + self.state.bank_usdt:.2f} USDT\n"
                f"• Paused: {self.state.paused} | Halted: {self.state.halted}\n"
                f"• Position:\n{pos_str}\n"
                f"• Next order:\n{self._describe_next_order(pos)}\n"
                f"• Cum Fees: ${self.state.cum_fees:.2f} | Cum Funding: ${self.state.cum_funding:.2f}\n"
                f"• Trades: {len(self.state.trades)}"
            )
            return status_msg

    def _describe_next_order(self, pos: PositionInfo) -> str:
        """Describe the order the bot is waiting to place and how far price is from it.

        Flat  -> the entry triggers (40h channel breaks), with the mom30 veto applied.
        In position -> the channel exit level and the exchange stop, whichever comes first.
        Levels are recomputed from the same closed bars the strategy uses, so they match
        exactly what _process_closed_bar will evaluate on the next bar close.
        """
        if self.state.halted:
            return f"  None - bot HALTED ({self.state.halt_reason})"

        df = self.data_mgr.df_klines
        min_bars = self.cfg.entry_channel_hours + self.cfg.atr_period
        if df.empty or len(df) < min_bars:
            return f"  Unavailable - only {len(df)} bars loaded, need {min_bars}"

        ind = compute_indicators(
            df,
            entry_channel_hours=self.cfg.entry_channel_hours,
            exit_channel_hours=self.cfg.exit_channel_hours,
            mom_hours=self.cfg.trend_veto_hours,
            atr_period=self.cfg.atr_period,
        )
        row = ind.iloc[-1]
        close_p = float(row["close"])
        if pd.isna(row["entry_high"]) or pd.isna(row["atr"]):
            return "  Unavailable - indicators still warming up"

        def away(level: float) -> str:
            return f"{(level / close_p - 1.0) * 100.0:+.2f}%"

        lines = [f"  Last close: ${close_p:,.2f} (bar {ind.index[-1]:%Y-%m-%d %H:%M} UTC)"]

        if pos.size > 0:
            # In a trade: the next order is an exit. Two ways out, whichever hits first.
            xh, xl = float(row["exit_high"]), float(row["exit_low"])
            if pos.side == "Buy":
                lines.append(f"  SELL to close on 1h close < ${xl:,.2f} ({away(xl)})")
            else:
                lines.append(f"  BUY to close on 1h close > ${xh:,.2f} ({away(xh)})")
            if pos.stop_loss > 0:
                lines.append(f"  Stop (on exchange, intrabar) ${pos.stop_loss:,.2f} ({away(pos.stop_loss)})")
            else:
                lines.append("  ⚠️ No stop detected on the exchange position")

            ps = self.state.position
            if self.cfg.breakeven_enabled and ps is not None:
                if ps.breakeven_triggered:
                    lines.append(f"  Breakeven: already triggered, stop is at breakeven+fees")
                elif ps.initial_stop_price > 0:
                    r = abs(ps.entry_price - ps.initial_stop_price)
                    trig = ps.entry_price + self.cfg.breakeven_trigger_r * r if pos.side == "Buy" else ps.entry_price - self.cfg.breakeven_trigger_r * r
                    lines.append(f"  Breakeven: stop moves to ~${ps.entry_price:,.2f} on 1h high/low reaching ${trig:,.2f} ({away(trig)})")
                else:
                    lines.append("  Breakeven: not tracked for this position (opened before this feature)")
            return "\n".join(lines)

        # Flat: the next order is an entry, if the mom30 veto allows that direction.
        eh, el = float(row["entry_high"]), float(row["entry_low"])
        mom, atr = float(row["mom30"]), float(row["atr"])

        if self.cfg.allow_long and mom >= 0:
            lines.append(f"  BUY on 1h close > ${eh:,.2f} ({away(eh)})")
        else:
            reason = "ALLOW_LONG=0" if not self.cfg.allow_long else f"mom30 {mom*100:+.2f}% < 0"
            lines.append(f"  BUY vetoed ({reason}) - would trigger > ${eh:,.2f} ({away(eh)})")

        if self.cfg.allow_short and mom <= 0:
            lines.append(f"  SELL on 1h close < ${el:,.2f} ({away(el)})")
        else:
            reason = "ALLOW_SHORT=0" if not self.cfg.allow_short else f"mom30 {mom*100:+.2f}% > 0"
            lines.append(f"  SELL vetoed ({reason}) - would trigger < ${el:,.2f} ({away(el)})")

        qty, stop_dist, valid, reason = compute_position_size(
            float_usdt=self.state.float_usdt,
            close_price=close_p,
            atr_val=atr,
            risk_frac=self.cfg.risk_frac,
            max_leverage=self.cfg.max_leverage,
            atr_mult=self.cfg.atr_mult,
            qty_step=self.filters.qty_step,
            min_order_qty=self.filters.min_order_qty,
            min_notional=self.filters.min_notional,
        )
        if valid:
            notional = qty * close_p
            lines.append(
                f"  Size if it fires: {qty} BTC = ${notional:,.2f} "
                f"({notional / self.state.float_usdt:.2f}x), stop {stop_dist / close_p * 100:.2f}% "
                f"= ${qty * stop_dist:.2f} risk"
            )
        else:
            lines.append(f"  ⚠️ Would be REJECTED: {reason}")

        if self.state.paused:
            lines.append("  ⏸️ PAUSED - no entry will be taken until /resume")
        return "\n".join(lines)

    def _cmd_pause(self) -> str:
        with self.lock:
            self.state.paused = True
            self.state_mgr.save(self.state)
            self.logger.warning("Bot paused via Telegram command. No new entries will be taken.")
            return "⏸️ *rapid_bot paused*: No new entries will be taken. Existing position/stop untouched."

    def _cmd_resume(self) -> str:
        with self.lock:
            self.state.paused = False
            self.state_mgr.save(self.state)
            self.logger.warning("Bot resumed via Telegram command.")
            return "▶️ *rapid_bot resumed*: Normal strategy operation active."

    def _cmd_flatten(self) -> str:
        with self.lock:
            self.logger.warning("Manual flatten command received via Telegram.")
            pos = self.exchange.get_position()
            if pos.size > 0:
                exit_side = "Sell" if pos.side == "Buy" else "Buy"
                exit_order_time_ms = int(time.time() * 1000)
                success, exit_price = self.exchange.place_reduce_only_market(side=exit_side, qty=pos.size)
                if success:
                    self._handle_position_closed(
                        exit_reason="telegram_flatten",
                        exit_price_hint=exit_price,
                        exit_order_time_ms=exit_order_time_ms,
                    )
            self.state.paused = True
            self.state_mgr.save(self.state)
            return "🚨 *rapid_bot flattened & paused*: Closed open position and paused new entries."

    def startup(self):
        """Execute Section 5 startup sequence."""
        with self.lock:
            self.logger.info("Executing startup sequence...")
            self.notifier.start()

            # 1. Fetch instrument filters (fail fast if invalid per §1.3)
            self.filters = self.exchange.get_instruments_info()
            self.logger.info(
                f"Instrument filters for {self.cfg.symbol}: tick={self.filters.tick_size}, "
                f"step={self.filters.qty_step}, minQty={self.filters.min_order_qty}, minNotional={self.filters.min_notional}"
            )

            # 2. Assert one-way position mode (positionIdx=0) per §5
            self.exchange.assert_one_way_mode()

            # 3. Set leverage to 5x
            self.exchange.set_leverage(int(self.cfg.max_leverage))

            # 4. Fetch warmup klines
            df_klines = self.data_mgr.initialize_warmup()

            # 5. Exchange reconciliation
            self.reconcile()

            # 6. Log viability sanity check with real liquidation distance (§5, §12)
            last_close = df_klines["close"].iloc[-1]
            self.exchange.set_last_market_price(last_close)
            indicators = compute_indicators(
                df_klines,
                entry_channel_hours=self.cfg.entry_channel_hours,
                exit_channel_hours=self.cfg.exit_channel_hours,
                mom_hours=self.cfg.trend_veto_hours,
                atr_period=self.cfg.atr_period,
            )
            last_atr = indicators["atr"].iloc[-1]
            qty, stop_dist, valid, reason = compute_position_size(
                float_usdt=self.state.float_usdt,
                close_price=last_close,
                atr_val=last_atr,
                risk_frac=self.cfg.risk_frac,
                max_leverage=self.cfg.max_leverage,
                atr_mult=self.cfg.atr_mult,
                qty_step=self.filters.qty_step,
                min_order_qty=self.filters.min_order_qty,
                min_notional=self.filters.min_notional,
            )

            # Real linear liquidation distance at 5x leverage (MMR ~0.5%)
            liq_dist_pct = (1.0 / self.cfg.max_leverage - 0.005) * 100.0
            stop_dist_pct = (stop_dist / last_close * 100.0) if last_close > 0 else 0.0

            self.logger.info(
                f"Startup viability check: Close=${last_close:,.2f}, ATR=${last_atr:.2f}, "
                f"StopDist=${stop_dist:.2f} ({stop_dist_pct:.2f}%), LiqDistEst={liq_dist_pct:.2f}%, "
                f"Qty={qty} BTC, Valid={valid} ({reason})"
            )

            self.running = True

    def reconcile(self):
        """Reconcile internal state with exchange positions (Section 5, 6, 7)."""
        pos = self.exchange.get_position()

        # Case 1: Position existed in state, but is now closed on exchange (stop-loss or manual exit)
        if pos.size == 0 and self.state.position is not None:
            self.logger.warning("Reconciliation: Position in state is closed on exchange (stop-loss fill detected).")
            self._handle_position_closed(exit_reason="stop_loss_fill")

        # Case 2: Position exists on exchange that the bot did not open (§7) -> HALT & ALERT, DO NOT AUTO-CLOSE
        elif pos.size > 0 and self.state.position is None:
            msg = (
                f"FOREIGN POSITION DETECTED: Exchange has active position ({pos.side} {pos.size} BTC @ ${pos.avg_price:,.2f}) "
                f"that the bot did not open. Halting bot immediately per §7 (will NOT auto-close)."
            )
            self._trigger_kill_switch(msg, auto_flatten=False)

    def _handle_position_closed(
        self,
        exit_reason: str,
        exit_price_hint: float = 0.0,
        exit_order_time_ms: int = 0,
    ):
        """Accurately record realized PnL, deduct fees, apply capital policy, and log trade per §12."""
        if self.state.position is None:
            return

        pos = self.state.position
        min_entry_ts = exit_order_time_ms if exit_order_time_ms > 0 else pos.entry_bar_ts
        closed_pnl_res: Optional[ClosedPnlResult] = self.exchange.get_last_closed_pnl(
            symbol=self.cfg.symbol,
            min_entry_ts=min_entry_ts,
            expected_qty=pos.qty,
        )

        if closed_pnl_res is not None and closed_pnl_res.avg_exit_price > 0:
            exit_price = closed_pnl_res.avg_exit_price
            net_trade_pnl = closed_pnl_res.closed_pnl  # Already net of both entry & exit fees
            total_fee = closed_pnl_res.cum_fee
        else:
            exit_price = exit_price_hint if exit_price_hint > 0 else (
                pos.stop_price if "stop" in exit_reason.lower() else self.data_mgr.df_klines["close"].iloc[-1]
            )
            if pos.side in ("Buy", "Long"):
                raw_pnl = pos.qty * (exit_price - pos.entry_price)
            else:
                raw_pnl = pos.qty * (pos.entry_price - exit_price)
            total_fee = pos.qty * (pos.entry_price + exit_price) * 0.00055
            net_trade_pnl = raw_pnl - total_fee

        pre_float = self.state.float_usdt

        # Accumulate fees for reporting and add net PnL to float
        self.state.cum_fees += total_fee
        self.state.float_usdt += net_trade_pnl

        # Check single trade loss kill switch (>15%) per §7
        tripped, msg = check_trade_loss_limit(net_trade_pnl, pre_float, self.cfg.max_single_trade_loss_pct)
        if tripped:
            self._trigger_kill_switch(msg, auto_flatten=False)

        # Apply capital policy (skim_refill)
        self.state.float_usdt, self.state.bank_usdt = apply_capital_policy(
            float_usdt=self.state.float_usdt,
            bank_usdt=self.state.bank_usdt,
            base_capital=self.cfg.base_capital,
            policy=self.cfg.capital_policy,
        )

        realised_leverage = (pos.qty * pos.entry_price) / pre_float if pre_float > 0 else 0.0
        distance_to_stop_pct = (abs(pos.entry_price - pos.stop_price) / pos.entry_price * 100.0) if pos.entry_price > 0 else 0.0

        trade_record = {
            "dir": "long" if pos.side in ("Buy", "Long") else "short",
            "in": datetime.fromtimestamp(pos.entry_bar_ts / 1000, tz=timezone.utc).isoformat(),
            "out": datetime.now(timezone.utc).isoformat(),
            "entry": pos.entry_price,
            "exit": exit_price,
            "stop": pos.stop_price,
            "qty": pos.qty,
            "realised_leverage": round(realised_leverage, 2),
            "distance_to_stop_pct": round(distance_to_stop_pct, 2),
            "pnl": round(net_trade_pnl, 2),
            "fee": round(total_fee, 2),
            "why": exit_reason,
            "float_after": self.state.float_usdt,
            "bank_after": self.state.bank_usdt,
        }
        self.state.trades.append(trade_record)
        self.state.position = None
        self.state_mgr.save(self.state)

        # Trade log per Section 12
        trade_log_msg = (
            f"TRADE CLOSED [{trade_record['why']}]: {trade_record['dir'].upper()} {trade_record['qty']} BTC | "
            f"Entry: ${trade_record['entry']:,.2f} -> Exit: ${trade_record['exit']:,.2f} | "
            f"Dist-to-Stop: {trade_record['distance_to_stop_pct']}% | Leverage: {trade_record['realised_leverage']}x | "
            f"Net PnL: ${trade_record['pnl']:+.2f} USDT (Fee: ${trade_record['fee']:.2f}) | "
            f"Float: ${self.state.float_usdt:.2f}, Bank: ${self.state.bank_usdt:.2f}"
        )
        self.logger.info(trade_log_msg)
        self.notifier.send_alert(f"📉 *Trade Closed ({exit_reason})*\n{trade_log_msg}", level="WARNING")

    def _trigger_kill_switch(self, reason: str, auto_flatten: bool = True):
        """Halt bot and flatten positions on hard kill switch trigger (Section 7)."""
        self.logger.critical(f"HALTING BOT: {reason}")
        self.state.halted = True
        self.state.halt_reason = reason
        self.state_mgr.save(self.state)

        if auto_flatten:
            pos = self.exchange.get_position()
            if pos.size > 0:
                exit_side = "Sell" if pos.side == "Buy" else "Buy"
                exit_order_time_ms = int(time.time() * 1000)
                success, exit_price = self.exchange.place_reduce_only_market(side=exit_side, qty=pos.size)
                if success:
                    self._handle_position_closed(
                        exit_reason="kill_switch_flatten",
                        exit_price_hint=exit_price,
                        exit_order_time_ms=exit_order_time_ms,
                    )

        self.notifier.send_alert(f"🚨 *CRITICAL KILL SWITCH TRIPPED*\n{reason}\nBot has halted.", level="CRITICAL")

    def run_loop(self):
        """Main execution loop every LOOP_INTERVAL_SEC."""
        self.startup()

        while self.running:
            try:
                with self.lock:
                    # 1. Kill switches check
                    if self.state.halted:
                        self.logger.warning(f"Bot is HALTED: {self.state.halt_reason}. Sleeping...")
                        time.sleep(self.cfg.loop_interval_sec)
                        continue

                    floor_tripped, floor_msg = check_account_floor(
                        self.state.float_usdt, self.state.bank_usdt, self.cfg.kill_total_below
                    )
                    if floor_tripped:
                        self._trigger_kill_switch(floor_msg, auto_flatten=True)
                        continue

                    # 2. Check for new closed bar
                    new_bar, closed_row, df_klines = self.data_mgr.check_for_new_closed_bar()

                    # Always update last market price for dry run tracking
                    if not df_klines.empty:
                        current_price = df_klines["close"].iloc[-1]
                        self.exchange.set_last_market_price(current_price)

                    # 3. Check data staleness (>180 min measured from bar close)
                    if not df_klines.empty:
                        last_bar_ts = df_klines["timestamp"].iloc[-1]
                        bar_close_ms = last_bar_ts + 3600 * 1000
                        mins_since_close = max(0.0, (time.time() * 1000 - bar_close_ms) / (60 * 1000))
                        stale_tripped, stale_msg = check_data_staleness(mins_since_close, self.cfg.stale_data_halt_min)
                        if stale_tripped:
                            self._trigger_kill_switch(stale_msg, auto_flatten=True)
                            continue

                    # 4. Check Intrabar Position, Dry-Run Stops (skipping entry bar), and Stop Loss Integrity (§6.3)
                    if self.cfg.dry_run and not df_klines.empty and self.state.position is not None:
                        last_row = df_klines.iloc[-1]
                        stop_hit, stop_p = self.exchange.check_dry_run_intrabar_stop(
                            high=last_row["high"],
                            low=last_row["low"],
                            bar_ts=int(last_row["timestamp"]),
                            entry_bar_ts=self.state.position.entry_bar_ts,
                        )
                        if stop_hit:
                            exit_order_time_ms = int(time.time() * 1000)
                            self._handle_position_closed(
                                exit_reason="stop_loss",
                                exit_price_hint=stop_p,
                                exit_order_time_ms=exit_order_time_ms,
                            )

                    self.reconcile()
                    pos = self.exchange.get_position()

                    # Verify active exchange stop integrity (§6.3)
                    if pos.size > 0 and self.state.position is not None:
                        stop_ok = self.exchange.ensure_trading_stop(
                            self.cfg.symbol, self.state.position.stop_price, self.filters.tick_size
                        )
                        if not stop_ok:
                            self._trigger_kill_switch("Exchange stop loss missing and unfixable (§6.3)", auto_flatten=True)
                            continue

                    # 5. Periodic 8-hour funding log & fee accumulation (00:00, 08:00, 16:00 UTC)
                    now_utc = datetime.now(timezone.utc)
                    if now_utc.hour in (0, 8, 16) and now_utc.hour != self.last_funding_check_hour:
                        self.last_funding_check_hour = now_utc.hour
                        self.exchange.sync_pybit_time()

                        if not self.cfg.dry_run and self.state.last_processed_funding_ts > 0:
                            history = self.exchange.get_funding_history(
                                self.cfg.symbol, start_time_ms=self.state.last_processed_funding_ts + 1
                            )
                            for item in reversed(history):
                                f_ts = int(item.get("fundingRateTimestamp", 0))
                                if f_ts > self.state.last_processed_funding_ts:
                                    rate = float(item.get("fundingRate", 0.0))
                                    if pos.size > 0:
                                        notional = pos.size * pos.avg_price
                                        cost = notional * rate if pos.side == "Buy" else -notional * rate
                                        self.state.float_usdt -= cost
                                        self.state.cum_funding += cost
                                        self.logger.info(
                                            f"Funding settled: Rate={rate*100:.4f}%, Cost=${cost:+.2f} USDT, CumFunding=${self.state.cum_funding:.2f}"
                                        )
                                    self.state.last_processed_funding_ts = f_ts
                                    self.state_mgr.save(self.state)
                        else:
                            rate, funding_ts = self.data_mgr.fetch_latest_funding_rate(self.cfg.symbol)
                            if pos.size > 0 and funding_ts > self.state.last_processed_funding_ts:
                                notional = pos.size * pos.avg_price
                                cost = notional * rate if pos.side == "Buy" else -notional * rate
                                self.state.float_usdt -= cost
                                self.state.cum_funding += cost
                                self.state.last_processed_funding_ts = funding_ts
                                self.state_mgr.save(self.state)
                                self.logger.info(
                                    f"8-hour Funding applied: Rate={rate*100:.4f}%, Cost=${cost:+.2f} USDT, CumFunding=${self.state.cum_funding:.2f}"
                                )
                            else:
                                self.logger.info(f"8-hour Funding check ({now_utc.strftime('%H:00 UTC')}): Current Rate = {rate*100:.4f}%")

                    # 6. Act on new closed bar only
                    if new_bar and closed_row is not None:
                        bar_ts = int(closed_row["timestamp"])

                        if bar_ts <= self.state.last_processed_bar_ts:
                            self.logger.debug(f"Bar ts {bar_ts} already processed. Skipping.")
                        else:
                            self._process_closed_bar(df_klines, closed_row, bar_ts)
                            # Commit bar only after successful processing
                            self.data_mgr.commit_bar(bar_ts)
                            self.state.last_processed_bar_ts = bar_ts
                            self.state_mgr.save(self.state)

                    self.consec_api_failures = 0

            except (requests.exceptions.RequestException, InvalidRequestError, FailedRequestError) as api_err:
                if isinstance(api_err, InvalidRequestError) or "10002" in str(api_err):
                    self.exchange.sync_pybit_time()
                self.consec_api_failures += 1
                self.logger.error(f"API failure ({self.consec_api_failures}/{self.cfg.max_consec_api_failures}): {api_err}")
                api_tripped, api_msg = check_api_failures(self.consec_api_failures, self.cfg.max_consec_api_failures)
                if api_tripped:
                    self._trigger_kill_switch(api_msg, auto_flatten=True)

            except Exception as e:
                self.logger.error(f"Error in main loop iteration: {e}", exc_info=True)

            time.sleep(self.cfg.loop_interval_sec)

    def _process_closed_bar(self, df: pd.DataFrame, bar: pd.Series, bar_ts: int):
        """Compute pure indicators and execute entry/exit logic on closed bar with same-bar reversal support (§4)."""
        indicators = compute_indicators(
            df,
            entry_channel_hours=self.cfg.entry_channel_hours,
            exit_channel_hours=self.cfg.exit_channel_hours,
            mom_hours=self.cfg.trend_veto_hours,
            atr_period=self.cfg.atr_period,
        )

        row = indicators.iloc[-1]
        close_p = row["close"]
        eh = row["entry_high"]
        el = row["entry_low"]
        xh = row["exit_high"]
        xl = row["exit_low"]
        mom = row["mom30"]
        atr = row["atr"]

        self.logger.debug(
            f"Bar {bar.name} [ts:{bar_ts}]: Close={close_p}, EntryH={eh}, EntryL={el}, "
            f"ExitH={xh}, ExitL={xl}, Mom30={mom:.4f}, ATR={atr:.2f}"
        )

        pos = self.exchange.get_position()

        # Step 0: If In Position -> Check Breakeven Trigger (one-time stop ratchet)
        # initial_stop_price==0 means this position pre-dates this feature (legacy
        # state, no R data) -- skip it rather than guess, matches only NEW trades.
        if pos.size > 0 and self.cfg.breakeven_enabled and self.state.position is not None:
            ps = self.state.position
            if ps.initial_stop_price > 0 and not ps.breakeven_triggered:
                if check_breakeven_trigger(
                    side=ps.side,
                    entry_price=ps.entry_price,
                    initial_stop_price=ps.initial_stop_price,
                    high_price=row["high"],
                    low_price=row["low"],
                    trigger_r=self.cfg.breakeven_trigger_r,
                ):
                    new_stop = compute_breakeven_stop_price(
                        ps.side, ps.entry_price, tick_size=self.filters.tick_size
                    )
                    stop_ok = self.exchange.ensure_trading_stop(
                        self.cfg.symbol, new_stop, self.filters.tick_size
                    )
                    if stop_ok:
                        ps.stop_price = new_stop
                        ps.breakeven_triggered = True
                        self.state_mgr.save(self.state)
                        self.logger.info(f"Breakeven triggered for {ps.side}: stop moved to ${new_stop:,.2f}")
                        self.notifier.send_alert(
                            f"🔒 *Breakeven*: stop moved to ${new_stop:,.2f} "
                            f"(reached {self.cfg.breakeven_trigger_r}R from entry ${ps.entry_price:,.2f})",
                            level="INFO",
                        )
                    else:
                        self.logger.warning("Breakeven stop move failed to verify on exchange; will retry next bar.")

        # Step 1: If In Position -> Check Channel Exit
        if pos.size > 0:
            if check_channel_exit(pos.side, close_p, xh, xl):
                self.logger.info(f"Channel exit condition met for {pos.side} at {close_p}")
                exit_side = "Sell" if pos.side == "Buy" else "Buy"
                exit_order_time_ms = int(time.time() * 1000)
                success, exit_price = self.exchange.place_reduce_only_market(
                    side=exit_side, qty=pos.size, bar_ts=bar_ts
                )
                if success:
                    self._handle_position_closed(
                        exit_reason="channel_exit",
                        exit_price_hint=exit_price,
                        exit_order_time_ms=exit_order_time_ms,
                    )

        # Re-fetch position: If now flat (either from previous stop-out or same-bar channel exit), check Entry
        pos = self.exchange.get_position()
        if pos.size == 0 and not self.state.paused:
            sig = evaluate_signal(
                close_val=close_p,
                entry_high_val=eh,
                entry_low_val=el,
                mom30_val=mom,
                allow_long=self.cfg.allow_long,
                allow_short=self.cfg.allow_short,
            )
            if sig is not None:
                qty, stop_dist, valid, reason = compute_position_size(
                    float_usdt=self.state.float_usdt,
                    close_price=close_p,
                    atr_val=atr,
                    risk_frac=self.cfg.risk_frac,
                    max_leverage=self.cfg.max_leverage,
                    atr_mult=self.cfg.atr_mult,
                    qty_step=self.filters.qty_step,
                    min_order_qty=self.filters.min_order_qty,
                    min_notional=self.filters.min_notional,
                )
                if valid:
                    self.logger.info(f"Entry signal {sig} triggered: Qty={qty} BTC, ATR={atr:.2f}")

                    success, order_id, fill_price, actual_stop = self.exchange.place_entry_with_stop(
                        side=sig,
                        qty=qty,
                        atr_val=atr,
                        atr_mult=self.cfg.atr_mult,
                        bar_ts=bar_ts,
                        tick_size=self.filters.tick_size,
                    )

                    if success:
                        self.state.position = PositionState(
                            side=sig,
                            qty=qty,
                            entry_price=fill_price,
                            stop_price=actual_stop,
                            entry_bar_ts=bar_ts,
                            entry_order_id=order_id or "dry_run",
                            initial_stop_price=actual_stop,
                            breakeven_triggered=False,
                        )
                        self.state_mgr.save(self.state)

                        dist_to_stop_pct = abs(fill_price - actual_stop) / fill_price * 100.0
                        entry_alert = (
                            f"🚀 *New Entry*: {sig.upper()} {qty} BTC @ ${fill_price:,.2f}\n"
                            f"Stop: ${actual_stop:,.2f} (Dist: {dist_to_stop_pct:.2f}%)\n"
                            f"Float: ${self.state.float_usdt:.2f}"
                        )
                        self.notifier.send_alert(entry_alert, level="WARNING")
                else:
                    self.logger.info(f"Signal {sig} rejected by sizing filter: {reason}")

    def stop(self):
        """Clean shutdown."""
        self.logger.info("Stopping rapid_bot...")
        self.running = False
        self.notifier.stop()
        if hasattr(self, "proc_lock"):
            self.proc_lock.release()
        for h in list(self.logger.handlers):
            h.close()
            self.logger.removeHandler(h)


def main():
    bot = RapidBot()

    def sig_handler(sig, frame):
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    bot.run_loop()


if __name__ == "__main__":
    main()
