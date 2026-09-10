"""Where resistance and support actually sit, derived from order flow.

A drawn line is a guess. What this module looks for is evidence that a passive
participant was working size at a price:

``absorption``
    Aggressive buyers hit the offer hard into the top of a bar and price still
    closed near the low. Somebody sold them everything they wanted. That
    somebody is the resistance, and their price is the level.
``stacked imbalance``
    Three or more consecutive price levels where one side dominated the
    diagonal comparison -- initiative that ran out at a specific price.
``high volume node``
    A price the market has already spent size agreeing on. Revisits get
    defended.
``delta divergence``
    A higher swing high on lower cumulative delta. The push was thinner than
    the one before it; the level that capped it matters.

Each piece of evidence adds weight. Levels decay if never revisited, gain
weight when they hold a retest, and are invalidated -- or flipped to the other
side -- when price closes decisively through them.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import StrategyConfig
from .data import Bar
from .footprint import EPS, Footprint, high_volume_nodes, to_level, to_price

RESISTANCE = 1
SUPPORT = -1

#: How many bars between full rebuilds of the rolling-profile HVN list.
HVN_REFRESH_BARS = 20

#: Evidence that actually comes from order flow. A level built only from wicks,
#: volume nodes and retests is a chart pattern -- worth marking, but trading it
#: is not trading order flow. ``StrategyConfig.require_flow_backed_levels``
#: enforces that a tradeable level carries at least one of these.
FLOW_EVIDENCE = frozenset({"absorption", "imbalance_stack", "delta_divergence"})


@dataclass(frozen=True)
class EvidenceWeights:
    absorption: float = 2.0
    imbalance_stack: float = 1.5
    delta_divergence: float = 1.5
    high_volume_node: float = 1.0
    wick_rejection: float = 0.5
    #: Added each time the level turns price away on a retest.
    retest_hold: float = 1.0
    #: Multiplier applied to strength when a broken level flips polarity.
    flip_retention: float = 0.6
    #: Hard ceiling on strength. Without it, a level sitting in the middle of a
    #: chop range accumulates retest credit every single bar and drowns out
    #: genuinely significant levels elsewhere.
    max_strength: float = 12.0
    #: Bars that must pass before the same level can be credited for another
    #: hold. Price oscillating across a level is one test, not thirty.
    touch_credit_gap: int = 3


@dataclass
class Level:
    price: float
    side: int  # RESISTANCE or SUPPORT
    strength: float
    created_index: int
    last_update_index: int
    volume: float = 0.0
    touches: int = 0
    holds: int = 0
    trades_taken: int = 0
    cooldown_until: int = -1
    evidence: Dict[str, float] = field(default_factory=dict)
    flipped_from: Optional[int] = None
    last_touch_index: int = -10_000
    #: Set when this level inherited flow evidence from a level it flipped
    #: from, whose own evidence keys are not carried across.
    inherited_flow: bool = False

    @property
    def is_flow_backed(self) -> bool:
        return self.inherited_flow or any(k in FLOW_EVIDENCE for k in self.evidence)

    def effective_strength(self, index: int, half_life: int) -> float:
        """Strength after time decay -- stale levels stop being tradeable."""
        if half_life <= 0:
            return self.strength
        age = max(0, index - self.last_update_index)
        return self.strength * (0.5 ** (age / half_life))

    def add_evidence(self, kind: str, weight: float, index: int, cap: float) -> None:
        room = max(0.0, cap - self.strength)
        applied = min(weight, room)
        if applied > 0:
            self.evidence[kind] = self.evidence.get(kind, 0.0) + applied
            self.strength += applied
        self.last_update_index = index

    def describe(self) -> str:
        bits = ", ".join(f"{k}x{v:.1f}" for k, v in sorted(self.evidence.items()))
        kind = "resistance" if self.side == RESISTANCE else "support"
        return f"{kind} @ {self.price:.2f} (strength {self.strength:.1f}; {bits})"


class RollingProfile:
    """Volume-at-price over a sliding window of bars, maintained incrementally."""

    def __init__(self, tick_size: float, window: int):
        self.tick_size = tick_size
        self.window = window
        self._ask: Dict[int, float] = {}
        self._bid: Dict[int, float] = {}
        self._queue: List[Footprint] = []

    def push(self, fp: Footprint) -> None:
        for lvl, v in fp.ask_volume.items():
            self._ask[lvl] = self._ask.get(lvl, 0.0) + v
        for lvl, v in fp.bid_volume.items():
            self._bid[lvl] = self._bid.get(lvl, 0.0) + v
        self._queue.append(fp)
        while len(self._queue) > self.window:
            old = self._queue.pop(0)
            for lvl, v in old.ask_volume.items():
                self._decrement(self._ask, lvl, v)
            for lvl, v in old.bid_volume.items():
                self._decrement(self._bid, lvl, v)

    @staticmethod
    def _decrement(book: Dict[int, float], lvl: int, v: float) -> None:
        remaining = book.get(lvl, 0.0) - v
        if remaining <= EPS:
            book.pop(lvl, None)
        else:
            book[lvl] = remaining

    def snapshot(self) -> Footprint:
        fp = Footprint(tick_size=self.tick_size)
        fp.ask_volume = dict(self._ask)
        fp.bid_volume = dict(self._bid)
        return fp

    def volume_at_price(self, price: float) -> float:
        lvl = to_level(price, self.tick_size)
        return self._ask.get(lvl, 0.0) + self._bid.get(lvl, 0.0)


# ----------------------------------------------------------------------
# Evidence detectors -- each returns (level_price, weight_key) or None
# ----------------------------------------------------------------------
def detect_absorption(
    bar: Bar, cfg: StrategyConfig, volume_threshold: float
) -> List[Tuple[float, int]]:
    """Heavy aggression into an extreme that produced no follow-through."""
    found: List[Tuple[float, int]] = []
    if bar.volume < volume_threshold or bar.range <= EPS:
        return found

    # Resistance: buyers were aggressive at the top, price closed at the bottom.
    ask_top, bid_top = bar.footprint.cluster_flow(cfg.cluster_frac, top=True)
    if (
        bar.close_location <= cfg.absorption_close_frac
        and ask_top >= cfg.absorption_flow_ratio * max(bid_top, EPS)
        and ask_top > 0
    ):
        price = bar.footprint.cluster_poc_price(cfg.cluster_frac, top=True)
        if price is not None:
            found.append((price, RESISTANCE))

    # Support: sellers were aggressive at the bottom, price closed at the top.
    ask_bot, bid_bot = bar.footprint.cluster_flow(cfg.cluster_frac, top=False)
    if (
        bar.close_location >= 1.0 - cfg.absorption_close_frac
        and bid_bot >= cfg.absorption_flow_ratio * max(ask_bot, EPS)
        and bid_bot > 0
    ):
        price = bar.footprint.cluster_poc_price(cfg.cluster_frac, top=False)
        if price is not None:
            found.append((price, SUPPORT))
    return found


def detect_imbalance_stacks(bar: Bar, cfg: StrategyConfig) -> List[Tuple[float, int]]:
    """Runs of consecutive one-sided price levels."""
    found: List[Tuple[float, int]] = []
    buys, sells = bar.footprint.diagonal_imbalances(cfg.imbalance_ratio, cfg.imbalance_min_volume)

    sell_len, sell_top = Footprint.longest_stack(sells)
    if sell_len >= cfg.imbalance_stack and sell_top is not None:
        # Sellers stacked -> supply. Mark the top of the stack.
        found.append((to_price(sell_top, cfg.tick_size), RESISTANCE))

    buy_len, buy_top = Footprint.longest_stack(buys)
    if buy_len >= cfg.imbalance_stack and buy_top is not None:
        bottom = buy_top - buy_len + 1
        found.append((to_price(bottom, cfg.tick_size), SUPPORT))
    return found


def detect_wick_rejection(bar: Bar, cfg: StrategyConfig) -> List[Tuple[float, int]]:
    found: List[Tuple[float, int]] = []
    if bar.range <= EPS:
        return found
    if bar.upper_wick_frac >= cfg.wick_frac and bar.normalized_delta <= cfg.c1_delta_max:
        price = bar.footprint.cluster_poc_price(cfg.cluster_frac, top=True) or bar.high
        found.append((price, RESISTANCE))
    if bar.lower_wick_frac >= cfg.wick_frac and bar.normalized_delta >= -cfg.c1_delta_max:
        price = bar.footprint.cluster_poc_price(cfg.cluster_frac, top=False) or bar.low
        found.append((price, SUPPORT))
    return found


def detect_delta_divergence(
    swing_highs: Sequence[Tuple[int, float, float]],
    swing_lows: Sequence[Tuple[int, float, float]],
    last_seen_high: int,
    last_seen_low: int,
) -> Tuple[List[Tuple[float, int]], int, int]:
    """Higher high on weaker delta (or lower low on stronger delta).

    ``last_seen_*`` are cursors so each confirmed swing is only scored once.
    """
    found: List[Tuple[float, int]] = []

    if len(swing_highs) >= 2 and len(swing_highs) > last_seen_high:
        (_, p_prev, d_prev), (_, p_cur, d_cur) = swing_highs[-2], swing_highs[-1]
        if p_cur > p_prev and d_cur < d_prev:
            found.append((p_cur, RESISTANCE))
        last_seen_high = len(swing_highs)

    if len(swing_lows) >= 2 and len(swing_lows) > last_seen_low:
        (_, p_prev, d_prev), (_, p_cur, d_cur) = swing_lows[-2], swing_lows[-1]
        if p_cur < p_prev and d_cur > d_prev:
            found.append((p_cur, SUPPORT))
        last_seen_low = len(swing_lows)

    return found, last_seen_high, last_seen_low


# ----------------------------------------------------------------------
# The book
# ----------------------------------------------------------------------
class LevelBook:
    """Maintains the live set of order-flow levels, bar by bar."""

    def __init__(self, cfg: StrategyConfig, weights: Optional[EvidenceWeights] = None):
        self.cfg = cfg
        self.w = weights or EvidenceWeights()
        self.levels: List[Level] = []
        self.profile = RollingProfile(cfg.tick_size, cfg.profile_lookback)
        self._hvn_cache: List[Tuple[float, float]] = []

    # -- lifecycle -----------------------------------------------------
    def update(
        self,
        index: int,
        bar: Bar,
        atr_value: float,
        volume_threshold: float,
        new_swing_evidence: Sequence[Tuple[float, int]] = (),
    ) -> None:
        """Ingest one closed bar."""
        self.profile.push(bar.footprint)

        for price, side in detect_absorption(bar, self.cfg, volume_threshold):
            self._register(price, side, self.w.absorption, "absorption", index, bar, atr_value)
        for price, side in detect_imbalance_stacks(bar, self.cfg):
            self._register(
                price, side, self.w.imbalance_stack, "imbalance_stack", index, bar, atr_value
            )
        for price, side in detect_wick_rejection(bar, self.cfg):
            self._register(
                price, side, self.w.wick_rejection, "wick_rejection", index, bar, atr_value
            )
        for price, side in new_swing_evidence:
            self._register(
                price, side, self.w.delta_divergence, "delta_divergence", index, bar, atr_value
            )

        if index % HVN_REFRESH_BARS == 0:
            self._refresh_hvns(index, bar, atr_value)

        self._score_touches(index, bar, atr_value)
        self._invalidate(index, bar, atr_value)
        self._prune(index)

    def _refresh_hvns(self, index: int, bar: Bar, atr_value: float) -> None:
        snapshot = self.profile.snapshot()
        self._hvn_cache = high_volume_nodes(snapshot, self.cfg.hvn_percentile)
        for price, volume in self._hvn_cache:
            if abs(price - bar.close) < 0.25 * atr_value:
                continue  # a node we are sitting inside is not a barrier
            side = RESISTANCE if price > bar.close else SUPPORT
            self._register(
                price, side, self.w.high_volume_node, "high_volume_node", index, bar, atr_value,
                volume=volume,
            )

    def _register(
        self,
        price: float,
        side: int,
        weight: float,
        kind: str,
        index: int,
        bar: Bar,
        atr_value: float,
        volume: float = 0.0,
    ) -> None:
        tol = max(self.cfg.level_merge_atr * atr_value, self.cfg.tick_size)
        existing = self._nearest(price, side, tol)
        if existing is not None:
            # Re-anchor slightly toward the fresh evidence, weighted by strength.
            total = existing.strength + weight
            existing.price = (existing.price * existing.strength + price * weight) / max(total, EPS)
            existing.add_evidence(kind, weight, index, self.w.max_strength)
            existing.volume = max(existing.volume, volume or self.profile.volume_at_price(price))
            return

        self.levels.append(
            Level(
                price=price,
                side=side,
                strength=min(weight, self.w.max_strength),
                created_index=index,
                last_update_index=index,
                volume=volume or self.profile.volume_at_price(price),
                evidence={kind: min(weight, self.w.max_strength)},
            )
        )

    def _nearest(self, price: float, side: int, tol: float) -> Optional[Level]:
        best: Optional[Level] = None
        best_d = tol
        for lv in self.levels:
            if lv.side != side:
                continue
            d = abs(lv.price - price)
            if d <= best_d:
                best, best_d = lv, d
        return best

    def _score_touches(self, index: int, bar: Bar, atr_value: float) -> None:
        """Credit levels that were tested and held on this bar."""
        tol = max(self.cfg.touch_tol_atr * atr_value, self.cfg.tick_size)
        for lv in self.levels:
            if lv.side == RESISTANCE:
                tested = bar.high >= lv.price - tol
                held = tested and bar.close < lv.price
            else:
                tested = bar.low <= lv.price + tol
                held = tested and bar.close > lv.price
            if not tested:
                continue
            # Chop across a level is one test, not one per bar.
            fresh_test = index - lv.last_touch_index >= self.w.touch_credit_gap
            lv.last_touch_index = index
            if not fresh_test:
                continue
            lv.touches += 1
            if held:
                lv.holds += 1
                lv.add_evidence("retest_hold", self.w.retest_hold, index, self.w.max_strength)
            else:
                lv.last_update_index = index

    def _invalidate(self, index: int, bar: Bar, atr_value: float) -> None:
        """Drop or flip levels that price closed decisively through."""
        buffer = self.cfg.level_break_atr * atr_value
        survivors: List[Level] = []
        broken: List[Level] = []
        for lv in self.levels:
            is_broken = (
                bar.close > lv.price + buffer
                if lv.side == RESISTANCE
                else bar.close < lv.price - buffer
            )
            (broken if is_broken else survivors).append(lv)

        self.levels = survivors

        # Broken resistance often becomes support, and vice versa. Merge the
        # flip into an existing level on the far side when one is already
        # there, so repeated breaks in chop cannot spawn stacks of near
        # duplicates.
        tol = max(self.cfg.level_merge_atr * atr_value, self.cfg.tick_size)
        for lv in broken:
            flipped_strength = lv.strength * self.w.flip_retention
            if flipped_strength < self.cfg.level_min_strength:
                continue
            existing = self._nearest(lv.price, -lv.side, tol)
            if existing is not None:
                existing.add_evidence(
                    "polarity_flip", flipped_strength, index, self.w.max_strength
                )
                existing.volume = max(existing.volume, lv.volume)
                existing.inherited_flow = existing.inherited_flow or lv.is_flow_backed
                if existing.flipped_from is None:
                    existing.flipped_from = lv.side
                continue
            self.levels.append(
                Level(
                    price=lv.price,
                    side=-lv.side,
                    strength=min(flipped_strength, self.w.max_strength),
                    created_index=index,
                    last_update_index=index,
                    volume=lv.volume,
                    evidence={"polarity_flip": min(flipped_strength, self.w.max_strength)},
                    flipped_from=lv.side,
                    inherited_flow=lv.is_flow_backed,
                )
            )

    def _prune(self, index: int) -> None:
        alive = [
            lv
            for lv in self.levels
            if lv.effective_strength(index, self.cfg.level_half_life) >= self.cfg.level_min_strength
        ]
        for side in (RESISTANCE, SUPPORT):
            side_levels = [lv for lv in alive if lv.side == side]
            if len(side_levels) > self.cfg.max_levels_per_side:
                side_levels.sort(
                    key=lambda l: l.effective_strength(index, self.cfg.level_half_life),
                    reverse=True,
                )
                drop = set(id(l) for l in side_levels[self.cfg.max_levels_per_side :])
                alive = [lv for lv in alive if id(lv) not in drop]
        self.levels = alive

    # -- queries -------------------------------------------------------
    def tradeable(self, index: int, side: int) -> List[Level]:
        cfg = self.cfg
        out = [
            lv
            for lv in self.levels
            if lv.side == side
            and lv.effective_strength(index, cfg.level_half_life) >= cfg.level_min_strength
            and lv.trades_taken < cfg.max_trades_per_level
            and index >= lv.cooldown_until
            and (lv.is_flow_backed or not cfg.require_flow_backed_levels)
        ]
        out.sort(key=lambda l: l.effective_strength(index, cfg.level_half_life), reverse=True)
        return out

    def nearest_opposing(self, price: float, side: int, index: int) -> Optional[Level]:
        """Closest level on the far side -- a natural profit target."""
        target_side = SUPPORT if side == RESISTANCE else RESISTANCE
        candidates = [
            lv
            for lv in self.levels
            if lv.side == target_side
            and lv.effective_strength(index, self.cfg.level_half_life) >= self.cfg.level_min_strength
            and ((lv.price < price) if side == RESISTANCE else (lv.price > price))
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda l: abs(l.price - price))

    def snapshot(self, index: int) -> List[Level]:
        return sorted(
            self.levels,
            key=lambda l: l.effective_strength(index, self.cfg.level_half_life),
            reverse=True,
        )
