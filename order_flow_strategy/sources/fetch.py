"""Download Binance public monthly archives.

Run this wherever outbound HTTPS to ``data.binance.vision`` is permitted -- your
laptop, or a session whose network policy allows it. It writes the raw ``.zip``
files unchanged, so the result is byte-identical to downloading them by hand and
can be checked against Binance's published ``.CHECKSUM`` files.

    # two years of 5 minute bars with measured taker-buy volume (~40 MB)
    python -m order_flow_strategy.sources.fetch --symbol BTCUSDT \
        --interval 5m --start 2023-08 --end 2025-07 --out data/btc

    # one month of the real tape, for true footprints (~2 GB, check disk first)
    python -m order_flow_strategy.sources.fetch --symbol BTCUSDT \
        --kind aggTrades --start 2025-06 --end 2025-06 --out data/btc-tape

Nothing else in this package needs the network; the loaders read local files.
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional, Sequence, Tuple

BASE = "https://data.binance.vision/data"
RETRIES = 4


def month_range(start: str, end: str) -> List[Tuple[int, int]]:
    """Inclusive list of ``(year, month)`` from ``YYYY-MM`` bounds."""
    try:
        sy, sm = (int(x) for x in start.split("-"))
        ey, em = (int(x) for x in end.split("-"))
    except ValueError as exc:
        raise ValueError("dates must look like YYYY-MM") from exc
    if (ey, em) < (sy, sm):
        raise ValueError("end month precedes start month")

    out: List[Tuple[int, int]] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def build_url(market: str, kind: str, symbol: str, interval: Optional[str], y: int, m: int) -> str:
    stem = f"{symbol}-{interval}-{y:04d}-{m:02d}" if interval else f"{symbol}-{kind}-{y:04d}-{m:02d}"
    parts = [BASE, market, "monthly", kind, symbol]
    if interval:
        parts.append(interval)
    return "/".join(parts) + f"/{stem}.zip"


def download(url: str, dest: str, timeout: float = 120.0) -> bool:
    """Fetch one archive. Returns False when the month is simply not published."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  have  {os.path.basename(dest)}")
        return True

    delay = 2.0
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                tmp = dest + ".part"
                with open(tmp, "wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                os.replace(tmp, dest)
            size = os.path.getsize(dest) / 1e6
            print(f"  got   {os.path.basename(dest)}  ({size:.1f} MB)")
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print(f"  none  {os.path.basename(dest)} (not published)")
                return False
            if exc.code in (403, 407):
                raise SystemExit(
                    f"blocked by network policy ({exc.code}) fetching {url}\n"
                    "This host is not permitted from here. Run this script somewhere "
                    "with open egress, or widen the environment's network policy."
                )
            print(f"  retry {exc.code} on {os.path.basename(dest)} ({attempt}/{RETRIES})")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  retry {exc} ({attempt}/{RETRIES})")
        if attempt < RETRIES:
            time.sleep(delay)
            delay *= 2
    print(f"  FAIL  {os.path.basename(dest)}")
    return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="order_flow_strategy.sources.fetch",
        description="Download Binance monthly klines or aggTrades archives.",
    )
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--market", default="spot", choices=("spot", "futures/um", "futures/cm"))
    p.add_argument("--kind", default="klines", choices=("klines", "aggTrades"))
    p.add_argument("--interval", default="5m", help="Kline interval; ignored for aggTrades.")
    p.add_argument("--start", required=True, metavar="YYYY-MM")
    p.add_argument("--end", required=True, metavar="YYYY-MM")
    p.add_argument("--out", required=True, metavar="DIR")
    args = p.parse_args(argv)

    try:
        months = month_range(args.start, args.end)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    interval = args.interval if args.kind == "klines" else None
    os.makedirs(args.out, exist_ok=True)

    if args.kind == "aggTrades":
        print(
            "Note: aggTrades archives are roughly 1-3 GB per month for BTCUSDT.\n"
            f"You asked for {len(months)} month(s). Check free disk before continuing.\n"
        )

    print(f"Fetching {len(months)} month(s) of {args.symbol} {args.kind} into {args.out}")
    ok = 0
    for y, m in months:
        url = build_url(args.market, args.kind, args.symbol, interval, y, m)
        dest = os.path.join(args.out, url.rsplit("/", 1)[-1])
        if download(url, dest):
            ok += 1

    print(f"\n{ok}/{len(months)} archives available in {args.out}")
    if ok:
        target = "--binance-klines" if args.kind == "klines" else "--binance-aggtrades"
        print(
            "\nNext:\n"
            f"  python -m order_flow_strategy backtest {target} {args.out} \\\n"
            "      --preset btc --timeframe 300"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
