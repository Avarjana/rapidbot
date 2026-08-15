"""Historical data fetcher for Bybit BTCUSDT 1h klines and funding rates.
Saves to data/BTC_USDT_ext_1h.parquet and data/BTC_USDT_ext_funding.parquet.
"""

import os
import time
from pathlib import Path
import pandas as pd
import requests

DATA_DIR = Path("data")
KLINE_FILE = DATA_DIR / "BTC_USDT_ext_1h.parquet"
FUNDING_FILE = DATA_DIR / "BTC_USDT_ext_funding.parquet"

BYBIT_BASE_URL = "https://api.bybit.com"


def fetch_all_klines(symbol: str = "BTCUSDT", interval: str = "60", start_time_ms: int = 1585094400000) -> pd.DataFrame:
    """Fetch 1h klines from start_time_ms up to the most recent closed bar.
    1585094400000 is 2020-03-25 00:00:00 UTC.
    """
    print(f"Fetching {symbol} {interval}m klines from Bybit V5...")
    all_rows = []
    current_start = start_time_ms
    limit = 1000  # Max limit per request

    # Bybit returns klines newest first in descending order when querying by range or start/end
    # Using start & end to paginate forward
    now_ms = int(time.time() * 1000)

    while current_start < now_ms:
        # Window of 1000 bars = 1000 * 60 * 60 * 1000 ms = 3,600,000,000 ms (~41.6 days)
        current_end = min(current_start + (limit * 3600 * 1000), now_ms)
        url = f"{BYBIT_BASE_URL}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "start": current_start,
            "end": current_end,
            "limit": limit,
        }

        resp = None
        for attempt in range(5):
            try:
                r = requests.get(url, params=params, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("retCode") == 0:
                        resp = data.get("result", {}).get("list", [])
                        break
                print(f"Retry {attempt+1}: {r.text[:100]}")
            except Exception as e:
                print(f"Exception fetching klines: {e}, retry {attempt+1}")
            time.sleep(1)

        if resp is None or len(resp) == 0:
            # Advance start if no records in this window
            current_start = current_end + 3600 * 1000
            continue

        # Each record: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
        # Bybit returns newest first, so reverse to ascending
        for item in reversed(resp):
            ts = int(item[0])
            o = float(item[1])
            h = float(item[2])
            l = float(item[3])
            c = float(item[4])
            v = float(item[5])
            all_rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v, "is_filled": 0})

        # Advance current_start past the newest timestamp fetched
        newest_ts = max(int(item[0]) for item in resp)
        if newest_ts <= current_start:
            current_start = current_end + 3600 * 1000
        else:
            current_start = newest_ts + 3600 * 1000

        print(f"  Fetched up to {pd.to_datetime(newest_ts, unit='ms', utc=True)} (total bars: {len(all_rows)})")
        time.sleep(0.1)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No klines fetched")

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")
    return df


def fetch_all_funding_rates(symbol: str = "BTCUSDT", start_time_ms: int = 1585094400000) -> pd.DataFrame:
    """Fetch funding rate history."""
    print(f"Fetching {symbol} funding rate history from Bybit V5...")
    all_rows = []
    current_start = start_time_ms
    limit = 200
    now_ms = int(time.time() * 1000)

    while current_start < now_ms:
        # Window of 200 8h intervals = 200 * 8 * 3600 * 1000 ms
        current_end = min(current_start + (limit * 8 * 3600 * 1000), now_ms)
        url = f"{BYBIT_BASE_URL}/v5/market/funding/history"
        params = {
            "category": "linear",
            "symbol": symbol,
            "startTime": current_start,
            "endTime": current_end,
            "limit": limit,
        }

        resp = None
        for attempt in range(5):
            try:
                r = requests.get(url, params=params, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("retCode") == 0:
                        resp = data.get("result", {}).get("list", [])
                        break
                print(f"Retry {attempt+1}: {r.text[:100]}")
            except Exception as e:
                print(f"Exception fetching funding: {e}, retry {attempt+1}")
            time.sleep(1)

        if resp is None or len(resp) == 0:
            current_start = current_end + 8 * 3600 * 1000
            continue

        for item in reversed(resp):
            ts = int(item["fundingRateTimestamp"])
            rate = float(item["fundingRate"])
            all_rows.append({"timestamp": ts, "fundingRate": rate})

        newest_ts = max(int(item["fundingRateTimestamp"]) for item in resp)
        if newest_ts <= current_start:
            current_start = current_end + 8 * 3600 * 1000
        else:
            current_start = newest_ts + 8 * 3600 * 1000

        print(f"  Fetched funding up to {pd.to_datetime(newest_ts, unit='ms', utc=True)} (records: {len(all_rows)})")
        time.sleep(0.1)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No funding rates fetched")

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    kline_df = fetch_all_klines()
    kline_df.to_parquet(KLINE_FILE)
    print(f"Saved {len(kline_df)} klines to {KLINE_FILE}")

    funding_df = fetch_all_funding_rates()
    funding_df.to_parquet(FUNDING_FILE)
    print(f"Saved {len(funding_df)} funding records to {FUNDING_FILE}")


if __name__ == "__main__":
    main()
