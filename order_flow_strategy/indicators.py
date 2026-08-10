"""Small causal indicator set. Every value at index ``i`` uses only bars <= i."""

from typing import List, Optional, Sequence, Tuple

from .data import Bar
from .footprint import EPS


def true_range(bars: Sequence[Bar]) -> List[float]:
    out: List[float] = []
    prev_close: Optional[float] = None
    for b in bars:
        if prev_close is None:
            out.append(b.range)
        else:
            out.append(max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close)))
        prev_close = b.close
    return out


def atr(bars: Sequence[Bar], period: int) -> List[float]:
    """Wilder's ATR. Values before the seed window are a running mean."""
    tr = true_range(bars)
    out: List[float] = []
    running = 0.0
    for i, v in enumerate(tr):
        if i < period:
            running += v
            out.append(running / (i + 1))
        else:
            out.append((out[-1] * (period - 1) + v) / period)
    return out


def ema(values: Sequence[float], period: int) -> List[float]:
    out: List[float] = []
    k = 2.0 / (period + 1.0)
    prev: Optional[float] = None
    for v in values:
        prev = v if prev is None else v * k + prev * (1 - k)
        out.append(prev)
    return out


def rolling_quantile(values: Sequence[float], window: int, q: float) -> List[float]:
    """Quantile of the trailing ``window`` values, inclusive of the current one."""
    out: List[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = sorted(values[lo : i + 1])
        idx = min(len(chunk) - 1, int(q * (len(chunk) - 1)))
        out.append(chunk[idx])
    return out


def rolling_median(values: Sequence[float], window: int) -> List[float]:
    return rolling_quantile(values, window, 0.5)


def cumulative_delta(bars: Sequence[Bar]) -> List[float]:
    out: List[float] = []
    run = 0.0
    for b in bars:
        run += b.delta
        out.append(run)
    return out


class SwingTracker:
    """Confirms swing highs and lows with a right-hand lookback.

    A swing is only reported once ``lookback`` bars have printed to its right,
    which is what makes it usable in a backtest without peeking.
    """

    def __init__(self, lookback: int = 5):
        self.lookback = max(1, lookback)
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._n = 0
        #: Confirmed swings as ``(bar_index, price, cumulative_delta)``.
        self.swing_highs: List[Tuple[int, float, float]] = []
        self.swing_lows: List[Tuple[int, float, float]] = []

    def update(self, index: int, bar: Bar, cum_delta: float) -> None:
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._n += 1

        lb = self.lookback
        pivot = self._n - 1 - lb  # candidate index within our local arrays
        if pivot < lb:
            return

        window_lo, window_hi = pivot - lb, pivot + lb + 1
        hs = self._highs[window_lo:window_hi]
        ls = self._lows[window_lo:window_hi]
        pivot_index = index - lb

        if self._highs[pivot] >= max(hs):
            self.swing_highs.append((pivot_index, self._highs[pivot], cum_delta))
        if self._lows[pivot] <= min(ls):
            self.swing_lows.append((pivot_index, self._lows[pivot], cum_delta))

    def last_two_highs(self):
        return self.swing_highs[-2:] if len(self.swing_highs) >= 2 else None

    def last_two_lows(self):
        return self.swing_lows[-2:] if len(self.swing_lows) >= 2 else None


def slope(values: Sequence[float], span: int) -> float:
    """Simple normalized slope of the last ``span`` values."""
    if len(values) < span + 1 or span <= 0:
        return 0.0
    a, b = values[-span - 1], values[-1]
    return (b - a) / (abs(a) + EPS)
