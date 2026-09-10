"""Readers for Binance public market data dumps.

Binance publishes two archives that matter here, and the difference between
them is the difference between testing this strategy and testing a candle
pattern.

**aggTrades** -- the tape, one row per aggregated trade, carrying
``is_buyer_maker``. When the buyer was the maker, the *seller* crossed the
spread, so the aggressor is a sell. This gives true footprints: volume at
price, split by who was impatient. Use it if you can.

**klines** -- OHLCV bars, but with ``taker_buy_base_asset_volume``: the portion
of the bar's volume where the buyer was the taker. That is aggressive buying,
measured rather than guessed, so bar delta is real. Only the *distribution* of
volume across price inside the bar is estimated. Far better than plain OHLCV,
and small enough to cover years.

Both readers accept ``.zip`` (as published), ``.csv``, or ``.csv.gz``, with or
without a header row, and cope with the millisecond and microsecond timestamp
conventions Binance has used at different times.
"""

import csv
import gzip
import io
import os
import zipfile
from typing import Iterable, Iterator, List, Optional, Sequence

from ..data import Bar, Trade
from ..footprint import EPS, Footprint
from ..funding import FundingSchedule

#: Column order of the published kline CSVs.
KLINE_COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume", "ignore",
)

#: Column order of the published aggTrades CSVs.
AGG_TRADE_COLUMNS = (
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker", "is_best_match",
)

#: Column order of the published fundingRate CSVs (USD-M futures).
FUNDING_COLUMNS = ("calc_time", "funding_interval_hours", "last_funding_rate")

TRUE_WORDS = {"true", "t", "1", "yes"}


def _to_seconds(raw: float) -> float:
    """Binance has published epochs in ms and, since 2025, in microseconds.

    Both are unambiguous by magnitude for any plausible trading date, so pick
    the scale from the value rather than from the filename.
    """
    if raw > 1e17:  # nanoseconds
        return raw / 1e9
    if raw > 1e14:  # microseconds
        return raw / 1e6
    if raw > 1e11:  # milliseconds
        return raw / 1e3
    return raw  # already seconds


def _open_rows(path: str) -> Iterator[List[str]]:
    """Yield CSV rows from a .zip, .csv.gz, or .csv path."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"{path}: archive contains no CSV")
            for name in sorted(names):
                with zf.open(name) as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                    for row in csv.reader(text):
                        yield row
    elif path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                yield row
    else:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                yield row


def _is_header(row: Sequence[str]) -> bool:
    """Newer dumps carry a header; older ones do not."""
    if not row:
        return True
    try:
        float(row[0])
        return False
    except ValueError:
        return True


def load_binance_klines(
    path: str,
    tick_size: float,
    levels_cap: int = 400,
) -> List[Bar]:
    """Read a Binance kline archive into bars with measured aggressor volume.

    ``tick_size`` here is the **footprint row size**, not the exchange's price
    increment. BTCUSDT ticks at $0.01, but a five minute bar can span hundreds
    of dollars, which at $0.01 rows would mean tens of thousands of price
    levels per bar -- unreadable as a footprint and ruinously slow. Real
    footprint charts on BTC use rows of $5 to $50. Pick a row size that gives
    roughly 20-50 rows per bar.
    """
    bars: List[Bar] = []
    for lineno, row in enumerate(_open_rows(path), start=1):
        if lineno == 1 and _is_header(row):
            continue
        if len(row) < 10:
            continue  # trailing blank line
        try:
            ts = _to_seconds(float(row[0]))
            o, h, l, c = (float(row[i]) for i in (1, 2, 3, 4))
            volume = float(row[5])
            taker_buy = float(row[9])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{path}:{lineno}: malformed kline row") from exc

        buy_frac = min(1.0, max(0.0, taker_buy / volume)) if volume > EPS else None
        fp = Footprint.from_ohlcv_proxy(
            o, h, l, c, volume, tick_size, levels_cap=levels_cap, buy_frac=buy_frac
        )
        bars.append(Bar(ts=ts, open=o, high=h, low=l, close=c, volume=volume, footprint=fp))

    bars.sort(key=lambda b: b.ts)
    return bars


def load_binance_agg_trades(path: str) -> List[Trade]:
    """Read a Binance aggTrades archive into a tape with real aggressor sides.

    ``is_buyer_maker == true`` means the buyer was resting and the seller
    crossed the spread, so the aggressor is a sell (-1).
    """
    trades: List[Trade] = []
    for lineno, row in enumerate(_open_rows(path), start=1):
        if lineno == 1 and _is_header(row):
            continue
        if len(row) < 7:
            continue
        try:
            price = float(row[1])
            qty = float(row[2])
            ts = _to_seconds(float(row[5]))
            buyer_is_maker = str(row[6]).strip().lower() in TRUE_WORDS
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{path}:{lineno}: malformed aggTrade row") from exc

        trades.append(
            Trade(ts=ts, price=price, size=qty, aggressor=-1 if buyer_is_maker else 1)
        )

    trades.sort(key=lambda t: t.ts)
    return trades


def load_binance_funding(path: str) -> FundingSchedule:
    """Read a Binance fundingRate archive.

    Rates are per settlement (typically 8-hourly), signed so that a positive
    rate means longs pay shorts.
    """
    times: List[float] = []
    rates: List[float] = []
    for lineno, row in enumerate(_open_rows(path), start=1):
        if lineno == 1 and _is_header(row):
            continue
        if len(row) < 3:
            continue
        try:
            times.append(_to_seconds(float(row[0])))
            rates.append(float(row[2]))
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{path}:{lineno}: malformed fundingRate row") from exc
    return FundingSchedule(times=times, rates=rates)


def load_binance_paths(
    paths: Iterable[str],
    tick_size: float,
    kind: str = "klines",
    levels_cap: int = 400,
):
    """Load and concatenate several monthly archives in chronological order.

    ``paths`` may name files or directories; directories are scanned for
    ``.zip``/``.csv``/``.csv.gz`` one level deep.
    """
    expanded: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            expanded.extend(
                os.path.join(p, n)
                for n in sorted(os.listdir(p))
                if n.endswith((".zip", ".csv", ".csv.gz"))
            )
        else:
            expanded.append(p)
    if not expanded:
        raise ValueError("no Binance data files found")

    if kind == "klines":
        out: List[Bar] = []
        for p in sorted(expanded):
            out.extend(load_binance_klines(p, tick_size, levels_cap=levels_cap))
        out.sort(key=lambda b: b.ts)
        return _dedupe(out, key=lambda b: b.ts)
    if kind == "aggTrades":
        tape: List[Trade] = []
        for p in sorted(expanded):
            tape.extend(load_binance_agg_trades(p))
        tape.sort(key=lambda t: t.ts)
        return tape
    if kind == "fundingRate":
        times: List[float] = []
        rates: List[float] = []
        for p in sorted(expanded):
            sched = load_binance_funding(p)
            times.extend(sched.times)
            rates.extend(sched.rates)
        return FundingSchedule(times=times, rates=rates)
    raise ValueError("kind must be 'klines', 'aggTrades' or 'fundingRate'")


def _dedupe(rows: Sequence, key) -> List:
    """Drop duplicate timestamps, which overlapping monthly archives produce."""
    out: List = []
    seen: Optional[float] = None
    for row in rows:
        k = key(row)
        if k != seen:
            out.append(row)
            seen = k
    return out
