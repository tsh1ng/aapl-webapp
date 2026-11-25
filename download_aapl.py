#!/usr/bin/env python3
"""Download two years of AAPL daily OHLCV and save it as ``data/aapl.csv``.

Usage
-----
    python download_aapl.py

Output
------
``data/aapl.csv`` with a normal header row as the first line and ``Date`` as a
regular column (not an unnamed index)::

    Date,Open,High,Low,Close,Adj Close,Volume
    2024-11-25,231.19,233.24,230.71,232.78,231.42,41234500
    ...

The script creates ``data/`` if needed, retries transient download errors, and
prints the first five rows plus the column names once the file is written.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

TICKER = "AAPL"
PERIOD = "2y"  # trailing two years, anchored to today
RETRIES = 3
RETRY_DELAY_SECONDS = 4.0

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "aapl.csv"
CACHE_DIR = DATA_DIR / ".yf_cache"

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
COLUMN_ORDER = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def import_yfinance():
    """Import yfinance, or exit with an install hint if it is missing."""
    try:
        import yfinance as yf
    except ImportError:
        print(
            "ERROR: yfinance is not installed.\n"
            "Install it in the current environment with:\n"
            "    python -m pip install yfinance",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return yf


def configure_cache(yf) -> None:
    """Keep yfinance's SQLite timezone cache inside the project directory.

    By default yfinance writes to the user cache folder, which can be
    unavailable (e.g. read-only or sandboxed environments) and then fails with
    ``OperationalError: unable to open database file``.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    setter = getattr(yf, "set_tz_cache_location", None)
    if callable(setter):
        setter(str(CACHE_DIR))


def download_prices(yf) -> pd.DataFrame:
    """Download daily bars, retrying a few times on transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            frame = yf.download(
                TICKER,
                period=PERIOD,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                timeout=30,
            )
        except Exception as exc:  # network hiccups, rate limits, parsing issues
            last_error = exc
            print(f"Attempt {attempt}/{RETRIES} failed: {exc}", file=sys.stderr)
        else:
            if frame is not None and not frame.empty:
                return frame
            last_error = RuntimeError("yfinance returned an empty DataFrame")
            print(
                f"Attempt {attempt}/{RETRIES}: empty response "
                "(Yahoo may be rate limiting).",
                file=sys.stderr,
            )
        if attempt < RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError(f"Could not download {TICKER} data: {last_error}")


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance's MultiIndex and turn the date index into a column."""
    if isinstance(frame.columns, pd.MultiIndex):
        # Single-ticker downloads come back as (Price, Ticker) tuples.
        frame = frame.copy()
        frame.columns = [str(level[0]) for level in frame.columns]

    frame = frame.reset_index()

    date_col = next(
        (
            column
            for column in frame.columns
            if str(column).strip().lower() in {"date", "datetime", "index"}
        ),
        None,
    )
    if date_col is None:
        raise RuntimeError(f"No date column found; got columns {list(frame.columns)}")
    frame = frame.rename(columns={date_col: "Date"})

    dates = pd.to_datetime(frame["Date"])
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_localize(None)
    frame["Date"] = dates.dt.strftime("%Y-%m-%d")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns: {missing}")

    frame = frame[[c for c in COLUMN_ORDER if c in frame.columns]].copy()
    price_columns = [c for c in ("Open", "High", "Low", "Close", "Adj Close") if c in frame.columns]
    frame[price_columns] = frame[price_columns].astype(float).round(4)
    frame["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce").astype("Int64")

    frame = frame.dropna(subset=["Date", "Close"])
    frame = frame.sort_values("Date").reset_index(drop=True)

    if len(frame) < 200:
        raise RuntimeError(f"Only {len(frame)} rows downloaded; expected ~500.")

    return frame


def save_csv(frame: pd.DataFrame) -> None:
    """Write atomically so a failed run never leaves a half-written CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = CSV_PATH.with_suffix(".csv.tmp")
    frame.to_csv(tmp_path, index=False, lineterminator="\n")
    tmp_path.replace(CSV_PATH)


def main() -> int:
    yf = import_yfinance()
    configure_cache(yf)

    print(f"Downloading {TICKER} daily bars for the past {PERIOD} ...")
    frame = normalise(download_prices(yf))
    save_csv(frame)

    print(f"\nSaved {len(frame)} rows to {CSV_PATH}")
    print(f"Range: {frame['Date'].iloc[0]} -> {frame['Date'].iloc[-1]}")

    print("\nFirst 5 rows:")
    print(frame.head(5).to_string(index=False))

    print("\nColumn names:")
    print(list(frame.columns))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
