"""The two-candle reversal: candle 1 gets rejected, candle 2 is where you get in.

The sequence, for a short at resistance (longs mirror it exactly):

**Candle 1 -- the test.** Price reaches a level from the level book, prints a
rejection wick, and closes back below the level. Order flow has to agree:
either aggressive buying was absorbed at the top of the bar, or sellers stacked
imbalances into the high, or the bar's delta was already negative despite the
push. A wick on its own is not evidence -- it is a picture of one.

**Candle 2 -- the entry.** This is the bar the position goes on, and the reason
the strategy waits for it. Candle 1 alone is a guess about who won; candle 2 is
the market confirming it. Candle 2 must close bearish, hold below candle 1's
high, close beyond a configurable fraction of candle 1's range, and carry
negative delta of its own. Any of those failing means the setup is dead, not
delayed -- there is no candle 3 in this strategy.

Cost of the extra bar: a worse entry price and some setups that resolve without
you. What you buy with it is skipping the tests that fail immediately, which is
where most of the losses in a naive "fade the level" rule come from.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .config import (
    ENTRY_CLOSE_OF_2,
    REGIME_COUNTER_TREND_ONLY,
    REGIME_WITH_TREND_ONLY,
    StrategyConfig,
)
from .data import Bar
from .footprint import EPS, Footprint
from .indicators import SwingTracker, atr, cumulative_delta, ema, rolling_median, rolling_quantile
from .levels import RESISTANCE, SUPPORT, Level, LevelBook, detect_delta_divergence

SHORT = -1
LONG = 1


@dataclass
class Indicators:
    """Causal indicator bundle, precomputed once per run."""

    atr: List[float]
    ema: List[float]
    volume_threshold: List[float]
    atr_median: List[float]
    cum_delta: List[float]


def prepare(bars: Sequence[Bar], cfg: StrategyConfig) -> Indicators:
    atr_vals = atr(bars, cfg.atr_period)
    return Indicators(
        atr=atr_vals,
        ema=ema([b.close for b in bars], cfg.ema_period),
        volume_threshold=rolling_quantile(
            [b.volume for b in bars], cfg.profile_lookback, cfg.absorption_volume_pct
        ),
        atr_median=rolling_median(atr_vals, cfg.profile_lookback),
        cum_delta=cumulative_delta(bars),
    )


@dataclass
class Setup:
    """An armed candle-1 rejection, waiting for its confirmation bar."""

    level: Level
    direction: int  # SHORT at resistance, LONG at support
    c1_index: int
    c1: Bar
    expires_at: int
    reasons: List[str] = field(default_factory=list)


@dataclass
class Signal:
    """A confirmed entry, generated on the close of candle 2."""

    index: int
    ts: float
    direction: int
    level: Level
    c1_index: int
    c2_index: int
    #: Immediate fill price (``close_of_2``) or resting stop price
    #: (``break_of_2``).
    entry_price: float
    stop: float
    target: float
    risk_per_unit: float
    atr: float
    entry_mode: str
    reasons: List[str] = field(default_factory=list)

    @property
    def r_multiple_target(self) -> float:
        return abs(self.target - self.entry_price) / max(self.risk_per_unit, EPS)


class SignalEngine:
    """Drives level discovery and the two-candle pattern, bar by bar."""

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.book = LevelBook(cfg)
        self.swings = SwingTracker(cfg.swing_lookback)
        self._seen_highs = 0
        self._seen_lows = 0
        self.setups: List[Setup] = []

    # ------------------------------------------------------------------
    def on_bar(self, index: int, bars: Sequence[Bar], ind: Indicators) -> List[Signal]:
        """Process one closed bar. Returns signals generated on *this* bar.

        Order matters: the current bar is first offered as candle 2 to setups
        armed earlier, then evaluated as a fresh candle 1. A single bar can do
        both jobs, for different levels.
        """
        bar = bars[index]
        atr_value = max(ind.atr[index], EPS)

        self.swings.update(index, bar, ind.cum_delta[index])
        divergences, self._seen_highs, self._seen_lows = detect_delta_divergence(
            self.swings.swing_highs, self.swings.swing_lows, self._seen_highs, self._seen_lows
        )

        signals = self._confirm(index, bar, atr_value)

        self.book.update(
            index=index,
            bar=bar,
            atr_value=atr_value,
            volume_threshold=ind.volume_threshold[index],
            new_swing_evidence=divergences,
        )

        if self._tradeable_conditions(index, bar, ind):
            self._arm(index, bar, atr_value)

        self.setups = [s for s in self.setups if s.expires_at >= index + 1]
        return signals

    # ------------------------------------------------------------------
    # Candle 1
    # ------------------------------------------------------------------
    def _arm(self, index: int, bar: Bar, atr_value: float) -> None:
        cfg = self.cfg
        tol = max(cfg.touch_tol_atr * atr_value, cfg.tick_size)

        if cfg.trade_shorts:
            for level in self.book.tradeable(index, RESISTANCE):
                reasons = self._c1_reasons_short(bar, level, tol)
                if reasons:
                    self.setups.append(
                        Setup(
                            level=level,
                            direction=SHORT,
                            c1_index=index,
                            c1=bar,
                            expires_at=index + cfg.setup_expiry_bars,
                            reasons=reasons,
                        )
                    )
                    break  # one setup per bar per side; strongest level wins

        if cfg.trade_longs:
            for level in self.book.tradeable(index, SUPPORT):
                reasons = self._c1_reasons_long(bar, level, tol)
                if reasons:
                    self.setups.append(
                        Setup(
                            level=level,
                            direction=LONG,
                            c1_index=index,
                            c1=bar,
                            expires_at=index + cfg.setup_expiry_bars,
                            reasons=reasons,
                        )
                    )
                    break

    def _c1_reasons_short(self, bar: Bar, level: Level, tol: float) -> List[str]:
        cfg = self.cfg
        if bar.high < level.price - tol:
            return []  # never reached the level
        if bar.close >= level.price:
            return []  # closed above it: that is acceptance, not rejection
        if bar.upper_wick_frac < cfg.wick_frac:
            return []

        reasons = [f"tested {level.describe()}", f"upper wick {bar.upper_wick_frac:.0%} of range"]
        flow = self._flow_evidence_short(bar)
        if cfg.require_flow_evidence and not flow:
            return []
        return reasons + flow

    def _c1_reasons_long(self, bar: Bar, level: Level, tol: float) -> List[str]:
        cfg = self.cfg
        if bar.low > level.price + tol:
            return []
        if bar.close <= level.price:
            return []
        if bar.lower_wick_frac < cfg.wick_frac:
            return []

        reasons = [f"tested {level.describe()}", f"lower wick {bar.lower_wick_frac:.0%} of range"]
        flow = self._flow_evidence_long(bar)
        if cfg.require_flow_evidence and not flow:
            return []
        return reasons + flow

    def _flow_evidence_short(self, bar: Bar) -> List[str]:
        cfg = self.cfg
        out: List[str] = []
        nd = bar.normalized_delta
        if nd <= cfg.c1_delta_max:
            out.append(f"delta {nd:+.0%} of volume")

        ask_top, bid_top = bar.footprint.cluster_flow(cfg.cluster_frac, top=True)
        if (
            ask_top >= cfg.absorption_flow_ratio * max(bid_top, EPS)
            and bar.close_location <= cfg.absorption_close_frac
            and ask_top > 0
        ):
            out.append(f"absorption: {ask_top:.0f} bought at the highs, closed at {bar.close_location:.0%}")

        _, sells = bar.footprint.diagonal_imbalances(cfg.imbalance_ratio, cfg.imbalance_min_volume)
        stack, _ = Footprint.longest_stack(sells)
        if stack >= cfg.imbalance_stack:
            out.append(f"{stack} stacked sell imbalances")
        return out

    def _flow_evidence_long(self, bar: Bar) -> List[str]:
        cfg = self.cfg
        out: List[str] = []
        nd = bar.normalized_delta
        if nd >= -cfg.c1_delta_max:
            out.append(f"delta {nd:+.0%} of volume")

        ask_bot, bid_bot = bar.footprint.cluster_flow(cfg.cluster_frac, top=False)
        if (
            bid_bot >= cfg.absorption_flow_ratio * max(ask_bot, EPS)
            and bar.close_location >= 1.0 - cfg.absorption_close_frac
            and bid_bot > 0
        ):
            out.append(f"absorption: {bid_bot:.0f} sold at the lows, closed at {bar.close_location:.0%}")

        buys, _ = bar.footprint.diagonal_imbalances(cfg.imbalance_ratio, cfg.imbalance_min_volume)
        stack, _ = Footprint.longest_stack(buys)
        if stack >= cfg.imbalance_stack:
            out.append(f"{stack} stacked buy imbalances")
        return out

    # ------------------------------------------------------------------
    # Candle 2
    # ------------------------------------------------------------------
    def _confirm(self, index: int, bar: Bar, atr_value: float) -> List[Signal]:
        cfg = self.cfg
        signals: List[Signal] = []
        surviving: List[Setup] = []

        for setup in self.setups:
            if setup.c1_index >= index:
                surviving.append(setup)
                continue
            if index > setup.expires_at:
                continue  # timed out, drop it

            ok, reasons, fatal = (
                self._c2_ok_short(bar, setup) if setup.direction == SHORT
                else self._c2_ok_long(bar, setup)
            )
            if not ok:
                # Fatal means the setup is wrong, not early: price took out
                # candle 1 or reclaimed the level. Merely unconfirmed setups
                # may wait, but only until setup_expiry_bars runs out (which at
                # the default of 1 means there is no candle 3).
                if not fatal:
                    surviving.append(setup)
                continue

            signal = self._build_signal(index, bar, setup, atr_value, reasons)
            if signal is not None:
                signals.append(signal)

        self.setups = surviving
        return signals

    def _c2_ok_short(self, bar: Bar, setup: Setup) -> Tuple[bool, List[str], bool]:
        """``(confirmed, reasons, fatal)``.

        ``fatal`` marks the setup as wrong rather than early -- price either
        took out candle 1's high or accepted back above the level, and no
        later bar can rescue it.
        """
        cfg = self.cfg
        c1 = setup.c1

        if not cfg.allow_c2_new_extreme and bar.high > c1.high:
            return False, ["candle 2 took out candle 1 high"], True
        if bar.close > setup.level.price:
            return False, ["candle 2 closed back above the level"], True

        if bar.close >= bar.open:
            return False, ["candle 2 was not bearish"], False
        if bar.body_frac < cfg.c2_body_frac_min:
            return False, ["candle 2 body too small"], False

        threshold = c1.high - cfg.c2_close_beyond_c1_frac * max(c1.range, EPS)
        if bar.close > threshold:
            return False, ["candle 2 did not close far enough through candle 1"], False
        if bar.normalized_delta > cfg.c2_delta_max:
            return False, ["candle 2 delta not negative enough"], False
        if cfg.c2_delta_must_worsen and bar.normalized_delta > c1.normalized_delta:
            return False, ["candle 2 delta weaker than candle 1"], False

        return (
            True,
            [
                f"candle 2 closed {bar.close:.2f}, below {threshold:.2f}",
                f"candle 2 delta {bar.normalized_delta:+.0%}",
            ],
            False,
        )

    def _c2_ok_long(self, bar: Bar, setup: Setup) -> Tuple[bool, List[str], bool]:
        cfg = self.cfg
        c1 = setup.c1

        if not cfg.allow_c2_new_extreme and bar.low < c1.low:
            return False, ["candle 2 took out candle 1 low"], True
        if bar.close < setup.level.price:
            return False, ["candle 2 closed back below the level"], True

        if bar.close <= bar.open:
            return False, ["candle 2 was not bullish"], False
        if bar.body_frac < cfg.c2_body_frac_min:
            return False, ["candle 2 body too small"], False

        threshold = c1.low + cfg.c2_close_beyond_c1_frac * max(c1.range, EPS)
        if bar.close < threshold:
            return False, ["candle 2 did not close far enough through candle 1"], False
        if bar.normalized_delta < -cfg.c2_delta_max:
            return False, ["candle 2 delta not positive enough"], False
        if cfg.c2_delta_must_worsen and bar.normalized_delta < c1.normalized_delta:
            return False, ["candle 2 delta weaker than candle 1"], False

        return (
            True,
            [
                f"candle 2 closed {bar.close:.2f}, above {threshold:.2f}",
                f"candle 2 delta {bar.normalized_delta:+.0%}",
            ],
            False,
        )

    # ------------------------------------------------------------------
    def _build_signal(
        self, index: int, bar: Bar, setup: Setup, atr_value: float, reasons: List[str]
    ) -> Optional[Signal]:
        cfg = self.cfg
        c1 = setup.c1
        buffer = cfg.stop_buffer_atr * atr_value

        if setup.direction == SHORT:
            stop = max(setup.level.price, c1.high) + buffer
            entry = bar.close if cfg.entry_mode == ENTRY_CLOSE_OF_2 else bar.low - cfg.tick_size
            risk = stop - entry
            if risk <= 0:
                return None
            target = entry - cfg.target_r * risk
        else:
            stop = min(setup.level.price, c1.low) - buffer
            entry = bar.close if cfg.entry_mode == ENTRY_CLOSE_OF_2 else bar.high + cfg.tick_size
            risk = entry - stop
            if risk <= 0:
                return None
            target = entry + cfg.target_r * risk

        return Signal(
            index=index,
            ts=bar.ts,
            direction=setup.direction,
            level=setup.level,
            c1_index=setup.c1_index,
            c2_index=index,
            entry_price=entry,
            stop=stop,
            target=target,
            risk_per_unit=risk,
            atr=atr_value,
            entry_mode=cfg.entry_mode,
            reasons=setup.reasons + reasons,
        )

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    def _tradeable_conditions(self, index: int, bar: Bar, ind: Indicators) -> bool:
        cfg = self.cfg
        if ind.atr[index] < cfg.min_atr_frac * max(ind.atr_median[index], EPS):
            return False  # too quiet; levels do not get tested properly
        if cfg.session_start_min is not None and cfg.session_end_min is not None:
            minute = int(bar.ts // 60) % 1440
            start, end = cfg.session_start_min, cfg.session_end_min
            inside = start <= minute < end if start <= end else (minute >= start or minute < end)
            if not inside:
                return False
        return True

    def regime_allows(self, index: int, direction: int, bars: Sequence[Bar], ind: Indicators) -> bool:
        """Trend gate, applied when a signal fires.

        ``with_trend_only`` keeps only reversals that point the same way as the
        prevailing trend -- selling a pullback into resistance inside a
        downtrend. ``counter_trend_only`` keeps only the pure fades.
        """
        cfg = self.cfg
        if cfg.regime_filter == "none":
            return True

        trend = self._trend(index, bars, ind)
        if trend == 0:
            return True  # no clear trend: balance, where fades work best
        if cfg.regime_filter == REGIME_WITH_TREND_ONLY:
            return trend == direction
        if cfg.regime_filter == REGIME_COUNTER_TREND_ONLY:
            return trend == -direction
        return True

    def _trend(self, index: int, bars: Sequence[Bar], ind: Indicators) -> int:
        cfg = self.cfg
        if index < cfg.ema_period:
            return 0
        e_now = ind.ema[index]
        e_prev = ind.ema[max(0, index - cfg.ema_period // 2)]
        close = bars[index].close
        rising = e_now > e_prev
        if close > e_now and rising:
            return LONG
        if close < e_now and not rising:
            return SHORT
        return 0
