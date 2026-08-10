"""Per-bar footprint: how much volume traded at each price, and on which side.

Terminology used throughout:

``ask_volume``
    Aggressive buying. A market buy lifting the offer. Printed at the ask.
``bid_volume``
    Aggressive selling. A market sell hitting the bid. Printed at the bid.
``delta``
    ``ask_volume - bid_volume``. Positive means aggressors were net buyers.

The single most useful thing a footprint tells you is when those two disagree
with price: heavy aggressive buying that produces no upward progress means a
passive seller is absorbing it, and that seller is the resistance.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

EPS = 1e-12


def to_level(price: float, tick_size: float) -> int:
    """Quantize a price to an integer tick index."""
    return int(round(price / tick_size))


def to_price(level: int, tick_size: float) -> float:
    return level * tick_size


@dataclass
class Footprint:
    """Volume at price, split by aggressor, for a single bar."""

    tick_size: float
    ask_volume: Dict[int, float] = field(default_factory=dict)
    bid_volume: Dict[int, float] = field(default_factory=dict)
    #: True when the split was estimated from OHLCV rather than measured from
    #: trades. Every consumer should treat proxy data with suspicion.
    is_proxy: bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add(self, price: float, size: float, aggressor: int) -> None:
        level = to_level(price, self.tick_size)
        book = self.ask_volume if aggressor > 0 else self.bid_volume
        book[level] = book.get(level, 0.0) + size

    @classmethod
    def from_ohlcv_proxy(
        cls,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        tick_size: float,
        levels_cap: int = 400,
        buy_frac: Optional[float] = None,
    ) -> "Footprint":
        """Estimate a footprint from an OHLCV bar.

        Volume is spread over the bar range with a triangular weighting that
        peaks at the bar's midpoint (a crude stand-in for the real profile).
        The buy/sell split comes from ``buy_frac`` when the feed reports real
        aggressor volumes; otherwise it falls back to where the bar closed
        inside its own range -- the standard "close location value" proxy.

        The CLV fallback is a stopgap, not a substitute. It cannot see
        absorption, because absorption is precisely the case where aggressive
        volume and price disagree, and the fallback *derives* the volume split
        from price. Use real tick data if you intend to trade this.
        """
        fp = cls(tick_size=tick_size, is_proxy=True)
        if volume <= 0:
            return fp

        lo_lvl = to_level(low, tick_size)
        hi_lvl = to_level(high, tick_size)
        if hi_lvl < lo_lvl:
            lo_lvl, hi_lvl = hi_lvl, lo_lvl
        n = hi_lvl - lo_lvl + 1
        if n > levels_cap:  # keep pathological ranges bounded
            step = n / levels_cap
            level_idx = [lo_lvl + int(i * step) for i in range(levels_cap)]
        else:
            level_idx = list(range(lo_lvl, hi_lvl + 1))

        if buy_frac is None:
            rng = max(high - low, EPS)
            buy_frac = (close - low) / rng
        buy_frac = min(1.0, max(0.0, buy_frac))

        mid = (len(level_idx) - 1) / 2.0
        weights = [1.0 + (mid - abs(i - mid)) for i in range(len(level_idx))]
        wsum = sum(weights) or 1.0
        for lvl, w in zip(level_idx, weights):
            v = volume * w / wsum
            fp.ask_volume[lvl] = fp.ask_volume.get(lvl, 0.0) + v * buy_frac
            fp.bid_volume[lvl] = fp.bid_volume.get(lvl, 0.0) + v * (1.0 - buy_frac)
        return fp

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------
    @property
    def levels(self) -> List[int]:
        return sorted(set(self.ask_volume) | set(self.bid_volume))

    def volume_at(self, level: int) -> float:
        return self.ask_volume.get(level, 0.0) + self.bid_volume.get(level, 0.0)

    @property
    def total_volume(self) -> float:
        return sum(self.ask_volume.values()) + sum(self.bid_volume.values())

    @property
    def delta(self) -> float:
        return sum(self.ask_volume.values()) - sum(self.bid_volume.values())

    @property
    def normalized_delta(self) -> float:
        """Delta as a fraction of volume, in ``[-1, 1]``.

        Comparable across bars and instruments, unlike raw delta.
        """
        total = self.total_volume
        return self.delta / total if total > EPS else 0.0

    @property
    def poc_level(self) -> Optional[int]:
        """Point of control: the most traded price level in the bar."""
        levels = self.levels
        if not levels:
            return None
        return max(levels, key=self.volume_at)

    # ------------------------------------------------------------------
    # Cluster reads -- the top / bottom slice of the bar's range
    # ------------------------------------------------------------------
    def cluster_levels(self, frac: float, top: bool) -> List[int]:
        """Levels in the top (or bottom) ``frac`` of the traded range."""
        levels = self.levels
        if not levels:
            return []
        lo, hi = levels[0], levels[-1]
        span = hi - lo
        if span == 0:
            return levels
        cut = span * frac
        if top:
            return [lvl for lvl in levels if lvl >= hi - cut]
        return [lvl for lvl in levels if lvl <= lo + cut]

    def cluster_flow(self, frac: float, top: bool) -> Tuple[float, float]:
        """``(ask_volume, bid_volume)`` traded inside the cluster."""
        lvls = self.cluster_levels(frac, top)
        ask = sum(self.ask_volume.get(l, 0.0) for l in lvls)
        bid = sum(self.bid_volume.get(l, 0.0) for l in lvls)
        return ask, bid

    def cluster_poc_price(self, frac: float, top: bool) -> Optional[float]:
        """Price of the heaviest level inside the cluster.

        This is where the passive order actually sat, which is a better level
        to mark than the bar's extreme wick tip.
        """
        lvls = self.cluster_levels(frac, top)
        if not lvls:
            return None
        best = max(lvls, key=self.volume_at)
        return to_price(best, self.tick_size)

    # ------------------------------------------------------------------
    # Imbalances
    # ------------------------------------------------------------------
    def diagonal_imbalances(
        self, ratio: float, min_volume: float
    ) -> Tuple[List[int], List[int]]:
        """Stacked-imbalance detection, standard footprint convention.

        Aggressive buying at price ``p`` is compared against aggressive selling
        one tick *below* it, because those are the two orders that met each
        other in the book.

        Returns ``(buy_imbalance_levels, sell_imbalance_levels)``.
        """
        buys: List[int] = []
        sells: List[int] = []
        for lvl in self.levels:
            ask_here = self.ask_volume.get(lvl, 0.0)
            bid_below = self.bid_volume.get(lvl - 1, 0.0)
            if ask_here >= min_volume and ask_here >= ratio * max(bid_below, EPS):
                buys.append(lvl)
            bid_here = self.bid_volume.get(lvl, 0.0)
            ask_above = self.ask_volume.get(lvl + 1, 0.0)
            if bid_here >= min_volume and bid_here >= ratio * max(ask_above, EPS):
                sells.append(lvl)
        return buys, sells

    @staticmethod
    def longest_stack(levels: Iterable[int]) -> Tuple[int, Optional[int]]:
        """Longest run of consecutive levels: ``(length, top_level)``."""
        lvls = sorted(set(levels))
        if not lvls:
            return 0, None
        best_len, best_top = 1, lvls[0]
        run_len, run_top = 1, lvls[0]
        for prev, cur in zip(lvls, lvls[1:]):
            if cur == prev + 1:
                run_len += 1
                run_top = cur
            else:
                run_len, run_top = 1, cur
            if run_len > best_len:
                best_len, best_top = run_len, run_top
        return best_len, best_top


def merge_footprints(fps: Iterable[Footprint], tick_size: float) -> Footprint:
    """Combine several footprints into one composite profile."""
    out = Footprint(tick_size=tick_size)
    for fp in fps:
        if fp.is_proxy:
            out.is_proxy = True
        for lvl, v in fp.ask_volume.items():
            out.ask_volume[lvl] = out.ask_volume.get(lvl, 0.0) + v
        for lvl, v in fp.bid_volume.items():
            out.bid_volume[lvl] = out.bid_volume.get(lvl, 0.0) + v
    return out


def high_volume_nodes(profile: Footprint, percentile: float) -> List[Tuple[float, float]]:
    """Local volume peaks at or above ``percentile`` of the profile.

    Returns ``(price, volume)`` pairs. These are prices the market has already
    agreed on; unfinished business tends to get revisited and defended there.
    """
    levels = profile.levels
    if len(levels) < 3:
        return []
    volumes = sorted(profile.volume_at(l) for l in levels)
    idx = min(len(volumes) - 1, int(percentile * (len(volumes) - 1)))
    threshold = volumes[idx]

    nodes: List[Tuple[float, float]] = []
    for i, lvl in enumerate(levels):
        v = profile.volume_at(lvl)
        if v < threshold or v <= 0:
            continue
        left = profile.volume_at(levels[i - 1]) if i > 0 else 0.0
        right = profile.volume_at(levels[i + 1]) if i < len(levels) - 1 else 0.0
        if v >= left and v >= right:  # local maximum
            nodes.append((to_price(lvl, profile.tick_size), v))
    return nodes
