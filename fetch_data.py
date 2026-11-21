"""
Standalone data-acquisition script for the Short-Horizon-Return-Model (SHRM)

Downloads historical OHLCV bars from Yahoo Finance 

It also writes the per-sector ticker-list CSVs 

This script is independent of the scripts in SHRM

Usage:
    python fetch_data.py
    python fetch_data.py --sectors tech financials --interval 1h --period 730d
    python fetch_data.py --skip-existing
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

SECTORS = {
    "tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "CSCO", "ORCL", "IBM", "ADBE", "CRM", "INTU", "SHOP"],
    "financials": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "PYPL"],
    "consumer": ["AMZN", "TSLA", "MCD", "NKE", "SBUX", "TGT", "HD", "LOW"],
    "staples": ["KO", "PEP", "WMT", "MRK", "COST"],
    "healthcare": ["JNJ", "MRK", "ABT", "UNH", "TMO", "PFE"],
    "energy": ["XOM", "CVX", "COP", "SLB"],
    "industry": ["BRK-B", "DIS", "NFLX", "LYFT", "UBER", "C"],
    "ffx": ["ES=F", "NQ=F", "CL=F", "GC=F", "SI=F", "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"],
}

REQUIRED_COLUMNS = ["date", "open", "high", "low", "adj_close", "volume"]

# yfinance max
MAX_INTRADAY_PERIOD = "730d"


def write_sector_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for sector, tickers in SECTORS.items():
        out_path = DATA_DIR / f"{sector}.csv"
        pd.DataFrame({"ticker": tickers}).to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(tickers)} tickers)")


def fetch_ticker(ticker: str, interval: str, period: str, retries: int = 3) -> pd.DataFrame | None:
    raw = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
            break
        except Exception as e:
            print(f"  [{ticker}] attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(3 * attempt)

    if raw is None or raw.empty:
        print(f"  [{ticker}] no data returned")
        return None

    raw = raw.reset_index()

    ts_col = "Datetime" if "Datetime" in raw.columns else "Date"
    raw = raw.rename(columns={ts_col: "date"})
    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]

    if "adj_close" not in raw.columns:
        raw["adj_close"] = raw["close"]

    if raw["date"].dt.tz is None:
        raw["date"] = raw["date"].dt.tz_localize("UTC")
    else:
        raw["date"] = raw["date"].dt.tz_convert("UTC")

    df = raw[REQUIRED_COLUMNS].copy()
    df = df.dropna(subset=REQUIRED_COLUMNS)

    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and clean OHLCV data for SHRM.")
    parser.add_argument(
        "--sectors", nargs="+", default=list(SECTORS.keys()),
        help=f"Sectors to download (default: all). Choices: {list(SECTORS.keys())}",
    )
    parser.add_argument("--interval", default="1h", help="yfinance bar interval (default: 1h)")
    parser.add_argument(
        "--period", default=MAX_INTRADAY_PERIOD,
        help=f"yfinance lookback period (default: {MAX_INTRADAY_PERIOD}, the max yfinance allows for 1h bars)",
    )
    parser.add_argument("--sleep", type=float, default=1.5, help="seconds to sleep between ticker requests (default: 1.5)")
    parser.add_argument("--skip-existing", action="store_true", help="skip tickers that already have a CSV in data/raw")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    write_sector_files()

    tickers = []
    for sector in args.sectors:
        if sector not in SECTORS:
            print(f"unknown sector '{sector}', skipping")
            continue
        tickers.extend(SECTORS[sector])
    tickers = list(dict.fromkeys(tickers))  # de-dupe

    print(f"\nFetching {len(tickers)} tickers @ interval={args.interval} period={args.period}\n")

    ok = 0
    failed = []
    for i, ticker in enumerate(tickers, 1):
        out_path = RAW_DIR / f"{ticker}.csv"
        if args.skip_existing and out_path.exists():
            print(f"[{i}/{len(tickers)}] {ticker}: already exists, skipping")
            ok += 1
            continue

        print(f"[{i}/{len(tickers)}] {ticker}: downloading...")
        df = fetch_ticker(ticker, args.interval, args.period)

        if df is None or df.empty:
            failed.append(ticker)
            continue

        df.to_csv(out_path, index=False)
        print(f"  saved {len(df)} rows -> {out_path}")
        ok += 1

        time.sleep(args.sleep)

    print(f"\nDone. {ok}/{len(tickers)} tickers saved.")
    if failed:
        print(f"Failed/empty: {failed}")


if __name__ == "__main__":
    main()
