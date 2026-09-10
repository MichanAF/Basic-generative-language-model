"""Bar construction helpers for the tests."""

from typing import Dict, Optional

from ..data import Bar
from ..footprint import Footprint, to_level

TICK = 0.25


def make_bar(
    ts: float,
    o: float,
    h: float,
    l: float,
    c: float,
    ask: Optional[Dict[float, float]] = None,
    bid: Optional[Dict[float, float]] = None,
    volume: Optional[float] = None,
    tick_size: float = TICK,
) -> Bar:
    """Build a bar with an explicit footprint.

    ``ask``/``bid`` map price -> volume. When omitted, volume is spread evenly
    across the range and split 50/50, which is deliberately neutral so tests
    that care about flow have to state it.
    """
    fp = Footprint(tick_size=tick_size)
    if ask is None and bid is None:
        total = volume if volume is not None else 100.0
        lo, hi = to_level(l, tick_size), to_level(h, tick_size)
        n = max(1, hi - lo + 1)
        for lvl in range(lo, hi + 1):
            fp.ask_volume[lvl] = total / (2 * n)
            fp.bid_volume[lvl] = total / (2 * n)
    else:
        for price, size in (ask or {}).items():
            fp.add(price, size, 1)
        for price, size in (bid or {}).items():
            fp.add(price, size, -1)

    return Bar(
        ts=ts,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=volume if volume is not None else fp.total_volume,
        footprint=fp,
    )


def flat_warmup(n: int, base: float = 95.0, start_ts: float = 0.0, step: float = 300.0):
    """``n`` quiet bars well below 100, to season ATR without touching levels."""
    bars = []
    for i in range(n):
        drift = 0.1 * ((i % 5) - 2)
        o = base + drift
        bars.append(
            make_bar(
                ts=start_ts + i * step,
                o=o,
                h=o + 0.5,
                l=o - 0.5,
                c=o + 0.1,
                volume=100.0,
            )
        )
    return bars
