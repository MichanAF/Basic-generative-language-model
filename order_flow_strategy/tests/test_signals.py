import unittest

from ..config import ENTRY_BREAK_OF_2, StrategyConfig
from ..levels import RESISTANCE, SUPPORT, Level
from ..signals import LONG, SHORT, SignalEngine, prepare
from .helpers import TICK, flat_warmup, make_bar

WARMUP = 30
LEVEL_PRICE = 100.0


def cfg(**overrides) -> StrategyConfig:
    base = dict(tick_size=TICK, tick_value=12.5, profile_lookback=60, level_half_life=120)
    base.update(overrides)
    return StrategyConfig(**base)


def resistance_level(index: int = WARMUP - 1) -> Level:
    return Level(
        price=LEVEL_PRICE,
        side=RESISTANCE,
        strength=10.0,
        created_index=0,
        last_update_index=index,
        evidence={"absorption": 10.0},
    )


def support_level(price: float, index: int = WARMUP - 1) -> Level:
    return Level(
        price=price,
        side=SUPPORT,
        strength=10.0,
        created_index=0,
        last_update_index=index,
        evidence={"absorption": 10.0},
    )


def rejection_candle(ts: float) -> "object":
    """Candle 1: pierces 100, heavy buying at the highs, closes at the low."""
    return make_bar(
        ts=ts,
        o=99.3,
        h=100.2,
        l=98.9,
        c=99.0,
        ask={100.0: 400, 100.25: 300},  # aggressive buyers hit the offer hard
        bid={99.0: 500, 98.9: 400},  # and got sold into
    )


def confirmation_candle(ts: float) -> "object":
    """Candle 2: bearish, stays under candle 1's high, sells off."""
    return make_bar(ts=ts, o=99.0, h=99.1, l=98.2, c=98.3, ask={99.0: 50}, bid={98.5: 600})


def run(bars, config, level: Level, inject_at: int = WARMUP):
    """Feed bars through the engine, planting ``level`` just before bar ``inject_at``."""
    engine = SignalEngine(config)
    ind = prepare(bars, config)
    signals = []
    for i in range(len(bars)):
        if i == inject_at:
            engine.book.levels = [level]
        signals.extend(engine.on_bar(i, bars, ind))
    return engine, signals


class TestTwoCandleReversal(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg()

    def test_confirmed_setup_fires_on_candle_two(self):
        bars = flat_warmup(WARMUP, base=98.5) + [
            rejection_candle(WARMUP * 300),
            confirmation_candle((WARMUP + 1) * 300),
        ]
        _, signals = run(bars, self.cfg, resistance_level())

        shorts = [s for s in signals if s.direction == SHORT]
        self.assertEqual(len(shorts), 1, "expected exactly one short signal")
        sig = shorts[0]
        self.assertEqual(sig.c1_index, WARMUP)
        self.assertEqual(sig.c2_index, WARMUP + 1, "entry must be on the second candle")
        self.assertAlmostEqual(sig.entry_price, 98.3, places=6)
        self.assertGreater(sig.stop, 100.2, "stop must sit beyond candle 1's high")
        self.assertLess(sig.target, sig.entry_price)
        self.assertAlmostEqual(sig.r_multiple_target, self.cfg.target_r, places=6)

    def test_target_is_measured_from_the_entry(self):
        bars = flat_warmup(WARMUP, base=98.5) + [
            rejection_candle(WARMUP * 300),
            confirmation_candle((WARMUP + 1) * 300),
        ]
        _, signals = run(bars, cfg(target_r=3.0), resistance_level())
        sig = signals[0]
        self.assertAlmostEqual(
            sig.entry_price - sig.target, 3.0 * sig.risk_per_unit, places=6
        )

    def test_new_high_on_candle_two_kills_the_setup(self):
        killer = make_bar(
            ts=(WARMUP + 1) * 300, o=99.0, h=100.5, l=98.2, c=98.3, bid={98.5: 600}
        )
        bars = flat_warmup(WARMUP, base=98.5) + [rejection_candle(WARMUP * 300), killer]
        engine, signals = run(bars, self.cfg, resistance_level())
        self.assertEqual([s for s in signals if s.direction == SHORT], [])
        # The killer bar is itself a fresh rejection, so it may arm a new
        # setup -- but the original one must be gone, not waiting.
        self.assertEqual(
            [s for s in engine.setups if s.c1_index == WARMUP],
            [],
            "an invalidated setup must not linger",
        )

    def test_bullish_candle_two_does_not_confirm(self):
        bullish = make_bar(
            ts=(WARMUP + 1) * 300, o=98.5, h=99.1, l=98.4, c=99.0, ask={99.0: 300}
        )
        bars = flat_warmup(WARMUP, base=98.5) + [rejection_candle(WARMUP * 300), bullish]
        _, signals = run(bars, self.cfg, resistance_level())
        self.assertEqual([s for s in signals if s.direction == SHORT], [])

    def test_shallow_close_does_not_confirm(self):
        """Candle 2 must close through candle 1, not merely tick down."""
        shallow = make_bar(
            ts=(WARMUP + 1) * 300, o=99.9, h=99.95, l=99.6, c=99.7, bid={99.7: 300}
        )
        bars = flat_warmup(WARMUP, base=98.5) + [rejection_candle(WARMUP * 300), shallow]
        _, signals = run(bars, self.cfg, resistance_level())
        self.assertEqual([s for s in signals if s.direction == SHORT], [])

    def test_there_is_no_candle_three(self):
        """An unconfirmed setup expires; a perfect third candle is too late."""
        indecisive = make_bar(
            ts=(WARMUP + 1) * 300, o=99.0, h=99.05, l=98.95, c=99.0, bid={99.0: 10}
        )
        bars = flat_warmup(WARMUP, base=98.5) + [
            rejection_candle(WARMUP * 300),
            indecisive,
            confirmation_candle((WARMUP + 2) * 300),
        ]
        _, signals = run(bars, self.cfg, resistance_level())
        late = [s for s in signals if s.direction == SHORT and s.c1_index == WARMUP]
        self.assertEqual(late, [])

    def test_expiry_window_can_be_widened(self):
        """setup_expiry_bars=2 explicitly allows the third candle to confirm."""
        indecisive = make_bar(
            ts=(WARMUP + 1) * 300, o=99.0, h=99.05, l=98.95, c=99.0, bid={99.0: 10}
        )
        bars = flat_warmup(WARMUP, base=98.5) + [
            rejection_candle(WARMUP * 300),
            indecisive,
            confirmation_candle((WARMUP + 2) * 300),
        ]
        _, signals = run(bars, cfg(setup_expiry_bars=2), resistance_level())
        matched = [s for s in signals if s.c1_index == WARMUP and s.direction == SHORT]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].c2_index, WARMUP + 2)

    def test_flow_evidence_is_required_by_default(self):
        """A rejection wick with buyers in control is not a short signal."""
        no_flow = make_bar(
            ts=WARMUP * 300,
            o=99.3,
            h=100.2,
            l=98.9,
            c=99.0,
            ask={100.0: 900, 99.5: 400},  # strongly positive delta
            bid={99.0: 20},
        )
        bars = flat_warmup(WARMUP, base=98.5) + [
            no_flow,
            confirmation_candle((WARMUP + 1) * 300),
        ]
        _, signals = run(bars, cfg(absorption_close_frac=0.05), resistance_level())
        self.assertEqual([s for s in signals if s.direction == SHORT], [])

    def test_break_of_two_places_a_resting_stop_order(self):
        bars = flat_warmup(WARMUP, base=98.5) + [
            rejection_candle(WARMUP * 300),
            confirmation_candle((WARMUP + 1) * 300),
        ]
        _, signals = run(bars, cfg(entry_mode=ENTRY_BREAK_OF_2), resistance_level())
        sig = [s for s in signals if s.direction == SHORT][0]
        self.assertAlmostEqual(sig.entry_price, 98.2 - TICK, places=6)


class TestLongMirror(unittest.TestCase):
    def test_support_produces_a_long_on_candle_two(self):
        level_price = 95.0
        c1 = make_bar(
            ts=WARMUP * 300,
            o=95.7,
            h=96.1,
            l=94.8,
            c=96.0,
            bid={95.0: 400, 94.8: 300},  # aggressive selling into the lows
            ask={96.0: 500, 96.1: 400},  # bought up
        )
        c2 = make_bar(
            ts=(WARMUP + 1) * 300, o=96.0, h=96.8, l=95.9, c=96.7, ask={96.5: 600}
        )
        bars = flat_warmup(WARMUP, base=96.5) + [c1, c2]
        _, signals = run(bars, cfg(), support_level(level_price))

        longs = [s for s in signals if s.direction == LONG]
        self.assertEqual(len(longs), 1)
        sig = longs[0]
        self.assertEqual(sig.c2_index, WARMUP + 1)
        self.assertLess(sig.stop, 94.8, "stop must sit below candle 1's low")
        self.assertGreater(sig.target, sig.entry_price)


class TestFilters(unittest.TestCase):
    def _engine_at(self, minute: int, config: StrategyConfig):
        bars = flat_warmup(WARMUP, base=98.5)
        bars[-1].ts = minute * 60.0
        engine = SignalEngine(config)
        return engine, bars, prepare(bars, config)

    def test_session_window_excludes_bars_outside_it(self):
        config = cfg(session_start_min=600, session_end_min=660)
        for minute, expected in ((630, True), (700, False), (599, False)):
            engine, bars, ind = self._engine_at(minute, config)
            self.assertEqual(
                engine._tradeable_conditions(WARMUP - 1, bars[-1], ind),
                expected,
                f"minute {minute}",
            )

    def test_session_window_wraps_past_midnight(self):
        config = cfg(session_start_min=1400, session_end_min=100)
        for minute, expected in ((1450, True), (50, True), (200, False)):
            engine, bars, ind = self._engine_at(minute, config)
            self.assertEqual(
                engine._tradeable_conditions(WARMUP - 1, bars[-1], ind),
                expected,
                f"minute {minute}",
            )

    def test_a_dead_market_is_skipped(self):
        config = cfg(min_atr_frac=5.0)  # demand ATR far above its own median
        engine, bars, ind = self._engine_at(630, config)
        self.assertFalse(engine._tradeable_conditions(WARMUP - 1, bars[-1], ind))


class TestNoLookahead(unittest.TestCase):
    """The single property that makes a backtest worth reading."""

    def _series(self):
        from ..data import SyntheticConfig, bars_from_trades, synthetic_trades

        sc = SyntheticConfig(n_ticks=40_000, tick_size=TICK, seed=11)
        return bars_from_trades(synthetic_trades(sc), 300.0, TICK)

    def test_indicators_at_i_ignore_bars_after_i(self):
        bars = self._series()
        k = len(bars) // 2
        full = prepare(bars, cfg())
        partial = prepare(bars[: k + 1], cfg())
        for i in range(k + 1):
            self.assertAlmostEqual(full.atr[i], partial.atr[i], places=9)
            self.assertAlmostEqual(full.ema[i], partial.ema[i], places=9)
            self.assertAlmostEqual(
                full.volume_threshold[i], partial.volume_threshold[i], places=9
            )
            self.assertAlmostEqual(full.cum_delta[i], partial.cum_delta[i], places=9)

    def test_signals_are_unchanged_by_future_bars(self):
        bars = self._series()
        k = len(bars) // 2
        config = cfg()

        engine_a = SignalEngine(config)
        ind_a = prepare(bars[: k + 1], config)
        early = []
        for i in range(k + 1):
            early.extend(engine_a.on_bar(i, bars[: k + 1], ind_a))

        engine_b = SignalEngine(config)
        ind_b = prepare(bars, config)
        full = []
        for i in range(len(bars)):
            full.extend(engine_b.on_bar(i, bars, ind_b))

        full_early = [s for s in full if s.c2_index <= k]
        self.assertEqual(
            [(s.c2_index, s.direction, round(s.entry_price, 6)) for s in early],
            [(s.c2_index, s.direction, round(s.entry_price, 6)) for s in full_early],
        )
        self.assertTrue(early, "the fixture should produce at least one signal")


if __name__ == "__main__":
    unittest.main()
