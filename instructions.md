# rapid_bot — implementation specification

Build spec. `RAPID_BOT_PLAN.md` holds the evidence and the reasoning for every
choice here; this document is what you build from. Where the two disagree, the
plan's §-references are authoritative on *why* and this document on *what*.

**Decided and closed** (do not re-open without new out-of-sample evidence —
`REMEDIATION_PLAN.md` Track D):

| | |
|---|---|
| strategy | Donchian channel breakout, both directions |
| instrument | BTCUSDT Bybit **linear perpetual** |
| capital | $100 trading float, skim + refill |
| expectation | +10–20%/yr base case, −30% to −50% drawdown (plan §"The answer") |

---

## 1. Strategy specification

All logic evaluates on **closed 1-hour bars only**. Never act on a forming bar.

### 1.1 Indicators

Let `k` be the index of the most recently *closed* hourly bar.

```
entry_high[k] = max(high[k-960 .. k-1])      # 40 days, EXCLUDING bar k
entry_low[k]  = min(low [k-960 .. k-1])
exit_low[k]   = min(low [k-480 .. k-1])      # 20 days
exit_high[k]  = max(high[k-480 .. k-1])
mom30[k]      = close[k] / close[k-720] - 1  # 30 days
ATR[k]        = Wilder ATR, period 14, on 1h bars
```

Wilder ATR, stated explicitly because the smoothing is easy to get wrong:

```
TR[i]  = max(high[i]-low[i], |high[i]-close[i-1]|, |low[i]-close[i-1]|)
ATR[14] = mean(TR[1..14])
ATR[i]  = (ATR[i-1] * 13 + TR[i]) / 14        for i > 14
```

The `k-1` upper bound on the channels is the no-lookahead guarantee. It matches
`.shift(1)` in every backtest in this folder. **Getting this wrong makes the
backtest unreproducible and the bot worse than modelled.**

### 1.2 Entry

Flat, and on a closed bar:

```
long_signal  = close[k] > entry_high[k]  AND  mom30[k] >= 0
short_signal = close[k] < entry_low[k]   AND  mom30[k] <= 0
```

If both are false, do nothing. Both cannot be true simultaneously.

The `mom30` term is the trend veto (plan §10.5). It was **non-binding across
all 8 trades in the 2026 evaluation** because a 40-day breakout is already a
momentum signal — it is kept as a cheap guard, not as an edge. Do not tune it.

### 1.3 Position sizing

```
stop_distance = 3.0 * ATR[k]                       # in price units
qty_risk      = (float_usdt * 0.05) / stop_distance
qty_lev       = (float_usdt * 5.0)  / close[k]
qty_raw       = min(qty_risk, qty_lev)
qty           = floor(qty_raw / qty_step) * qty_step
```

Reject the entry (log, do not error) unless **both**:

```
qty >= min_order_qty                (0.001 BTC)
qty * close[k] >= min_notional      ($5)
```

`qty_step`, `min_order_qty`, `min_notional` and `tick_size` must be read from
`get_instruments_info` at startup — never hardcoded. The values above are what
Bybit currently returns and are for sanity-checking only.

Realised leverage lands at a **median 1.71×** (plan §"eval_liquidation"); the
5× cap binds on roughly 3% of trades.

### 1.4 Stop loss — fixed, exchange-side

```
long:   stop_price = entry_price - 3.0 * ATR[at entry]
short:  stop_price = entry_price + 3.0 * ATR[at entry]
```

Quantise to `tick_size`. **The stop is set once at entry and never moved.** It
does not trail. It does not recompute as ATR changes.

It must live **on the exchange**, attached to the position, via
`set_trading_stop`. A stop held in bot memory disappears when the process dies,
and the bounded −5% average loss is the entire reason the strategy works.

### 1.5 Exit

Whichever comes first:

1. **Stop** — filled by the exchange. The bot discovers this via reconciliation.
2. **Channel exit**, on a closed bar:
   ```
   long  exits when close[k] < exit_low[k]
   short exits when close[k] > exit_high[k]
   ```
   Executed as a reduce-only **market** order.

No take-profit, no time stop, no partial exits. Winners run until the 20-day
channel turns — that is where the +26.6% average win comes from.

### 1.6 Capital policy — skim + refill

`base = 100.00 USDT`. After **every** closed trade:

```
if float > base:                       # bank the win
    bank  += float - base
    float  = base
elif float < base and bank > 0:         # refill from the bank
    top    = min(base - float, bank)
    float += top
    bank  -= top
```

`float` is what sizing uses. `bank` is withdrawn from the trading subaccount
(see §9). Never size off `float + bank`.

Do **not** implement the "bank but never refill" variant — it is a one-way
ratchet down and was the worst of four policies tested (plan §"eval_skimming").

---

## 2. Configuration — `.env.rapid`

Every tunable, with its committed value.

```ini
# ---- identity -------------------------------------------------------------
BOT_NAME=rapid_bot
SYMBOL=BTCUSDT
CATEGORY=linear
INTERVAL=60                     # minutes; 1h bars

# ---- credentials (separate SUBACCOUNT -- see §9) ---------------------------
BYBIT_API_KEY=
BYBIT_API_SECRET=
BYBIT_TESTNET=1                 # 1 until Phase 4
DRY_RUN=1                       # 1 = no orders sent

# ---- strategy (FROZEN -- plan §3.6 plateau, not an optimum) ---------------
ENTRY_CHANNEL_HOURS=960         # 40 days
EXIT_CHANNEL_HOURS=480          # 20 days
ATR_PERIOD=14
ATR_MULT=3.0
TREND_VETO_HOURS=720            # 30 days; 0 disables
ALLOW_LONG=1
ALLOW_SHORT=1

# ---- sizing ---------------------------------------------------------------
BASE_CAPITAL=100.0              # the trading float
RISK_FRAC=0.05                  # 5% of float risked per trade
MAX_LEVERAGE=5.0
CAPITAL_POLICY=skim_refill      # skim_refill | compound

# ---- risk / kill switches (§7) --------------------------------------------
KILL_TOTAL_BELOW=40.0           # float+bank floor, USDT
MAX_SINGLE_TRADE_LOSS_PCT=0.15  # stop failure detector
STALE_DATA_HALT_MIN=180         # no fresh bar for this long -> flatten
MAX_CONSEC_API_FAILURES=10

# ---- operations -----------------------------------------------------------
WARMUP_BARS=1200                # >= 960 + 14, padded
LOOP_INTERVAL_SEC=60
STATE_FILE=rapid_state.json
LOG_FILE=rapid_bot.log
TELEGRAM_BOT_TOKEN=             # MUST differ from bybitbot + shadow (§8)
TELEGRAM_CHAT_ID=
```

`.env.rapid` is read with the parent project's inline-comment stripping —
`set -a; source` strips a trailing ` #` comment but `systemd EnvironmentFile=`
does not, so the parser must handle it (`REMEDIATION_PLAN.md` A9.1).

---

## 3. Data

### 3.1 Historical (backtest / validation)

| file | purpose |
|---|---|
| `data/BTC_USDT_ext_1h.parquet` | 55,975 hourly bars, 2020-03-25 → 2026-08-13 |
| `data/BTC_USDT_ext_funding.parquet` | 6,997 rows, 8-hourly funding |

Columns: `open, high, low, close, volume, is_filled` indexed by UTC timestamp.
Funding: `fundingRate`, reindexed to hourly with `fill_value=0.0` so it is
charged only at real funding timestamps. **Verified correct** — do not "fix" it
by forward-filling, which would charge funding 24×/day instead of 3×.

### 3.2 Live

At startup, fetch `WARMUP_BARS=1200` closed 1h klines. Bybit returns newest
first, max 1000 per call, so this needs pagination — reuse the parent project's
backward-paging loop (`bybitbot.py:1340-1360`), which is already correct.

Requirement: **960 bars for the entry channel + 14 for ATR.** With fewer than
974 usable bars the bot must refuse to trade, not silently use a short channel.

Per loop, fetch only the last few bars and append. Detect a new closed bar by
timestamp change; act **once** per new bar.

### 3.3 Funding is the dominant cost

$500.46 over 6.4 years against $11.69 of trading fees (plan §"eval_liquidation").
Log the funding rate every 8h and accumulate realised funding into state — it is
a third of gross P&L and must be visible, not inferred.

---

## 4. Module layout

```
rapid_bot/
  IMPLEMENTATION.md      this file
  RAPID_BOT_PLAN.md      evidence and rationale
  screen_*.py            evidence screens (done)
  eval_*.py              evaluations (done)

  config.py     load + validate .env.rapid; fail loudly on missing/invalid
  exchange.py   Bybit V5 wrapper: retries, rate limits, DRY_RUN routing
  data.py       kline fetch/cache, bar-close detection, indicators
  strategy.py   pure functions: signals + sizing. NO I/O, NO exchange calls
  risk.py       kill switches, capital policy, reconciliation checks
  state.py      atomic JSON persistence
  notify.py     Telegram: queued, deduped, WARNING+
  main.py       the loop
  backtest.py   promoted from screen_breakout.py; shares strategy.py
  validate.py   Phase 1 walk-forward gate
  tests/        see §10
```

**`strategy.py` must be pure and shared by both `main.py` and `backtest.py`.**
If the live bot and the backtest compute signals from separate code they will
drift, and the backtest stops being evidence about the bot. This is the single
most important structural rule in the build.

---

## 5. Main loop

```
startup:
    cfg = load(.env.rapid); validate
    filters = get_instruments_info(symbol)      # tick, step, minQty, minNotional
    set_leverage(5)                             # tolerate "not modified" error
    #  DO NOT call switch_margin_mode -- ErrCode 100028 on UTA
    assert position_mode == one-way (positionIdx 0)
    bars = fetch_klines(1200)
    state = load_state() or fresh
    reconcile(state, exchange)                  # §6
    log viability: qty at current price, stop distance, liq distance

every LOOP_INTERVAL_SEC:
    if not new_closed_bar(): continue
    append bar; recompute ATR / channels / mom30

    if data_stale > STALE_DATA_HALT_MIN: flatten("stale data"); halt
    if kill_switch_tripped(): flatten(reason); halt

    pos = get_positions()
    reconcile(state, pos)                       # detects stop fills

    if pos is flat:
        if paused: continue
        sig = strategy.signal(bars, cfg)
        if sig: open_position(sig)              # §6
    else:
        if strategy.should_exit(bars, pos, cfg):
            close_position("channel exit")
```

Idempotency: every action is derived from the closed-bar index. If the loop
runs twice on the same bar it must do nothing the second time. Persist
`last_processed_bar_ts`.

---

## 6. Order lifecycle

### 6.1 Opening — entry then stop, with a guard

```
1. compute qty; reject if below minQty / minNotional
2. place_order(Market, side, qty, reduceOnly=False, positionIdx=0,
               orderLinkId=f"rapid-{bar_ts}-entry")
3. poll fill:  get_open_orders(orderId=...)  FIRST,
               fall back to get_order_history(orderId=...)
               -- history only returns FINAL-state orders (REMEDIATION_PLAN A0.1)
4. read the ACTUAL average fill price from the position, not the intended price
5. stop_price = avg_fill -/+ 3*ATR, quantised to tick_size
6. set_trading_stop(category, symbol, stopLoss=stop_price, positionIdx=0)
7. verify the stop is present via get_positions().stopLoss
8. if step 6 or 7 fails after retries -> CLOSE THE POSITION IMMEDIATELY
   and alert. An unprotected leveraged position is not an acceptable state.
```

Step 8 is not optional. Step 4 matters because the stop must be measured from
where you actually filled — a market order at 1.7× leverage can slip.

### 6.2 Closing

Reduce-only market order, `orderLinkId=f"rapid-{bar_ts}-exit"`. Confirm the
position is flat before applying the capital policy.

### 6.3 Every tick while in position

Verify the exchange-side stop still exists and matches state. If it has vanished
(manual cancel, exchange event), re-place it. If it cannot be re-placed, flatten.

---

## 7. Risk limits and kill switches

| trigger | action |
|---|---|
| `float + bank < 40 USDT` | flatten, halt permanently, alert |
| single trade loses > 15% of float | flatten, halt, alert — **the stop failed** |
| no fresh bar for 180 min | flatten, halt |
| 10 consecutive API failures | flatten if possible, halt, alert |
| stop missing and unfixable | flatten immediately |
| position exists that the bot did not open | halt, alert, do **not** auto-close |

**Not** kill criteria — these are normal and must not trigger intervention:

- 3+ consecutive stop-outs (win rate is 25%; this is expected)
- drawdown of 30–50% (plan §"The answer": expected, not a malfunction)
- months with no trade (a 40-day channel is quiet by design)

There is deliberately **no lifetime-drawdown halt** like `bybitbot.py`'s. Here
the stop bounds each trade and `KILL_TOTAL_BELOW` bounds the account; a
persistent cap would only lock in a drawdown the strategy is designed to
recover from.

---

## 8. State file

`rapid_state.json`, written atomically (temp file + `os.replace`).

```json
{
  "version": 1,
  "float_usdt": 100.0,
  "bank_usdt": 0.0,
  "last_processed_bar_ts": 1786512000000,
  "position": {
    "side": "Sell", "qty": 0.002,
    "entry_price": 63120.5, "stop_price": 64890.0,
    "entry_bar_ts": 1786500000000, "entry_order_id": "..."
  },
  "trades": [
    {"dir":"short","in":"2026-05-28T03:00:00Z","out":"2026-07-14T09:00:00Z",
     "entry":73232.0,"exit":64740.0,"pnl":25.41,"why":"channel"}
  ],
  "cum_fees": 0.0, "cum_funding": 0.0,
  "halted": false, "halt_reason": null, "paused": false
}
```

`position` is a **cache**, never the source of truth. The exchange is. On any
disagreement, trust `get_positions()` and log loudly.

---

## 9. Operational requirements

**9.1 Separate Bybit subaccount — blocking for Phase 4.** `bybitbot.py` trades
BTCUSDT; so does this. On one account in one-way mode the positions net and both
bots fight. A subaccount with its own API key also caps total exposure at its
balance. Fund it with $100; withdraw `bank` to the main account.

**9.2 Telegram token must be distinct** from `bybitbot`'s and the shadow
instance's. Two pollers on one token silently steal each other's `getUpdates`.

**9.3 UTA account.** Do not call `switch_margin_mode` — returns `ErrCode 100028`,
verified on live mainnet. `set_leverage` may return "leverage not modified";
treat that as success.

**9.4 Separate state, log and process** from the parent bot. No shared paths.

**9.5 Commands:** `/status` (float, bank, position, stop, distance to stop),
`/pause` (no new entries; existing position and its stop untouched), `/resume`,
`/flatten` (close now, then pause).

---

## 10. Build phases and gates

**Phase 1 — validate before writing the bot.** Promote `screen_breakout.py` to
`backtest.py` on top of `strategy.py`. Write `validate.py`: select on
2020→2022, evaluate untouched on 2023→2026, report train→test rank correlation
(`REMEDIATION_PLAN.md` Track D method).
*Gate:* if selection shows no out-of-sample edge — the expected result — freeze
40d/20d/3.0/5% permanently and never tune again. If it shows a large edge,
suspect a lookahead bug before believing it.

**Phase 2 — build, dry run.** `DRY_RUN=1`, live mainnet data, no orders. ≥2 weeks.
*Gate:* every signal the live bot generates matches `backtest.py` bar-for-bar
over the same window. Any mismatch is a bug in shared-code discipline (§4).

**Phase 3 — testnet.** `BYBIT_TESTNET=1`, `DRY_RUN=0`.
*Gates, all required:*
- one full entry → stop-out cycle executes correctly
- one full entry → channel-exit cycle executes correctly
- **kill the process mid-position; confirm the exchange stop survives, and that
  the bot re-attaches to the open position on restart without re-entering**
- capital policy moves the right amounts after a win and after a loss

**Phase 4 — live.** Subaccount, $100, `BYBIT_TESTNET=0`, `DRY_RUN=0`.
Do not add capital to a losing run. Do not restart a halted bot without a
written reason.

---

## 11. Tests

Unit (`tests/`, pure functions, no network):

- ATR matches a hand-computed 20-bar example
- channels exclude the current bar (lookahead regression test)
- sizing: risk path vs leverage path, and which binds
- quantisation: floor to step, min-qty and min-notional rejection
- capital policy: win → bank, loss → refill, refill capped by bank, and the
  ratchet-down variant is *absent*
- signal: no long when `mom30 < 0`; no entry while in position

Integration (testnet):

- entry places both order and stop; stop verified present
- forced stop-failure path closes the position (simulate by rejecting
  `set_trading_stop`)
- restart mid-position reconciles instead of double-entering
- duplicate bar processing is a no-op

Regression:

- `backtest.py` reproduces `eval_2026.py`: **+23.2% compound / +27.1%
  skim_refill, 8 trades, max DD −32.3%** on 2026-01-01 → 2026-08-13.
  This is the canary for accidental strategy drift.

---

## 12. Logging

Per closed bar at DEBUG: bar ts, close, channels, ATR, mom30, signal.
Per trade at INFO: direction, qty, entry, stop, distance-to-stop %, realised
leverage, exit reason, P&L, float and bank after the capital policy.
Per 8h at INFO: funding rate and cumulative funding paid.
Telegram at WARNING+ only: entries, exits, kill switches, stop anomalies.

Nothing in §7's "not kill criteria" list should page you.
