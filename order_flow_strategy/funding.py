"""Perpetual futures funding.

A perp has no expiry, so the exchange keeps its price tethered to spot by making
one side pay the other every few hours. When the perp trades above spot -- the
usual state, because retail is structurally long and leveraged -- longs pay
shorts.

This matters more than most people model. At the 0.01% per 8h baseline a long
pays about 11% a year; in a strong bull it has run several times that. A
backtest that ignores funding overstates a held long by exactly that amount,
which is the difference between a viable carry and a slow bleed.

The sign convention here: ``rate > 0`` means longs pay. A long position's cash
flow is ``-rate * notional``; a short receives the same amount.
"""

import bisect
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

#: Binance and most venues settle every 8 hours.
DEFAULT_INTERVAL_HOURS = 8.0
#: The rate exchanges drift toward when the basis is flat.
BASELINE_RATE_PER_INTERVAL = 0.0001


@dataclass
class FundingSchedule:
    """Funding settlements over time, as ``(timestamp, rate)`` pairs.

    Rates are per settlement, not annualised -- 0.0001 means one basis point
    charged at that instant, not per year.
    """

    times: List[float] = field(default_factory=list)
    rates: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.times) != len(self.rates):
            raise ValueError("times and rates must be the same length")
        if any(b < a for a, b in zip(self.times, self.times[1:])):
            order = sorted(range(len(self.times)), key=lambda i: self.times[i])
            self.times = [self.times[i] for i in order]
            self.rates = [self.rates[i] for i in order]

    def __len__(self) -> int:
        return len(self.times)

    @property
    def is_empty(self) -> bool:
        return not self.times

    # ------------------------------------------------------------------
    def accrued_rate(self, start_ts: float, end_ts: float) -> float:
        """Total rate settled in ``(start_ts, end_ts]``.

        Half-open at the start so a settlement is charged exactly once as bars
        advance, never double-counted at a boundary.
        """
        if self.is_empty or end_ts <= start_ts:
            return 0.0
        lo = bisect.bisect_right(self.times, start_ts)
        hi = bisect.bisect_right(self.times, end_ts)
        return sum(self.rates[lo:hi])

    def cash_flow(
        self, direction: int, notional: float, start_ts: float, end_ts: float
    ) -> float:
        """Funding cash flow for a position held over the window.

        Negative means you paid. ``direction`` is +1 long, -1 short.
        """
        return -direction * self.accrued_rate(start_ts, end_ts) * abs(notional)

    # ------------------------------------------------------------------
    def annualised(self) -> float:
        """Mean rate expressed as a simple annual figure, for reporting."""
        if self.is_empty:
            return 0.0
        mean = sum(self.rates) / len(self.rates)
        per_year = 365.0 * 24.0 / DEFAULT_INTERVAL_HOURS
        return mean * per_year

    def summary(self) -> str:
        if self.is_empty:
            return "no funding data"
        pos = sum(1 for r in self.rates if r > 0)
        return (
            f"{len(self.rates)} settlements, mean {sum(self.rates)/len(self.rates):+.5f} "
            f"per {DEFAULT_INTERVAL_HOURS:.0f}h ({self.annualised():+.1%} annualised), "
            f"positive {pos / len(self.rates):.0%} of the time"
        )

    # ------------------------------------------------------------------
    @classmethod
    def constant(
        cls,
        start_ts: float,
        end_ts: float,
        rate: float = BASELINE_RATE_PER_INTERVAL,
        interval_hours: float = DEFAULT_INTERVAL_HOURS,
    ) -> "FundingSchedule":
        """A flat schedule, for sensitivity checks when you lack real history.

        Useful for answering "what would this cost me if funding sat at the
        baseline all year", but do not mistake it for a backtest -- real
        funding is autocorrelated and spikes exactly when you are most exposed.
        """
        step = interval_hours * 3600.0
        if step <= 0:
            raise ValueError("interval_hours must be positive")
        times, t = [], start_ts
        while t <= end_ts:
            times.append(t)
            t += step
        return cls(times=times, rates=[rate] * len(times))

    @classmethod
    def from_pairs(cls, pairs: Sequence[Tuple[float, float]]) -> "FundingSchedule":
        return cls(times=[p[0] for p in pairs], rates=[p[1] for p in pairs])


def zero_funding() -> Optional[FundingSchedule]:
    """Explicitly no funding -- spot, or futures with no perpetual leg."""
    return None
