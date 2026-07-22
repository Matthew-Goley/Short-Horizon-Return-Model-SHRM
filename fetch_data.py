"""
Standalone data-acquisition script for the Short-Horizon-Return-Model (SHRM).

Downloads historical OHLCV bars from Yahoo Finance (via yfinance) for the
tickers in each sector below, cleans them up, and writes them to
data/raw/{TICKER}.csv in the exact schema scripts/data_pipeline.py expects:
    date, open, high, low, adj_close, volume

It also writes the per-sector ticker-list CSVs (data/tech.csv, data/financials.csv,
data/consumer.csv, data/staples.csv, data/healthcare.csv, data/energy.csv,
data/industry.csv, data/ffx.csv) that scripts/data_pipeline.py::CCOMPUTEALL
reads to know which tickers belong to which sector.

This script is fully independent of the model code: it does not import from
or modify anything under scripts/. Run this first, then run scripts/train.py.

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

# Ticker lists per dev_notes.md. "ffx" (futures + forex) has no list in the
# project notes, so a reasonable default basket of liquid CME futures and
# major FX pairs is used instead — edit data/ffx.csv or the list below if you
# want a different set.
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

# yfinance limits how far back intraday intervals can go: 60m/1h bars max out
# at 730 days. Daily ("1d") bars have no such limit.
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

    # yfinance names the timestamp column "Date" for daily bars and
    # "Datetime" for intraday bars.
    ts_col = "Datetime" if "Datetime" in raw.columns else "Date"
    raw = raw.rename(columns={ts_col: "date"})
    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]

    if "adj_close" not in raw.columns:
        raw["adj_close"] = raw["close"]

    # Normalize to UTC so the CSV round-trips through a plain
    # pd.to_datetime() with a single, unambiguous offset (no DST-related
    # mixed-offset parsing issues), exactly what data_pipeline.py assumes.
    if raw["date"].dt.tz is None:
        raw["date"] = raw["date"].dt.tz_localize("UTC")
    else:
        raw["date"] = raw["date"].dt.tz_convert("UTC")

    df = raw[REQUIRED_COLUMNS].copy()
    df = df.dropna(subset=REQUIRED_COLUMNS)
    # Note: FX pairs always report volume == 0 in yfinance (no consolidated
    # tape), so a volume > 0 filter would silently wipe out the ffx sector.
    # Zero-volume bars are left in; log_volume/log(volume + 1) handles them.
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
    tickers = list(dict.fromkeys(tickers))  # de-dupe, keep order

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
