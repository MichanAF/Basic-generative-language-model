"""Tests for the momentum-candle condition.

The idea, from a trader describing his setup: the candle *leaving* the level
should expand beyond normal volatility. Candle 2 is that candle here, so the
test is its range against ATR.

What matters is that the condition is measured before it is enforced. Off by
default it must still tag, so ``confluence`` can say whether it earns its
place; on, it must actually reject.
"""

import unittest

from ..backtest import Backtester
from ..config import StrategyConfig
from ..data import SyntheticConfig, bars_from_trades, synthetic_trades
from ..signals import SignalEngine, prepare, tags_from_reasons
from .helpers import TICK, make_bar


def a_bar(rng: float, close_at: float = 0.5) -> "object":
    """A bar of the given range, closing ``close_at`` through it."""
    low = 100.0
    high = low + rng
    return make_bar(ts=0.0, o=high, h=high, l=low, c=low + rng * close_at)


class TestMomentumRule(unittest.TestCase):
    def _engine(self, **kw):
        return SignalEngine(StrategyConfig(tick_size=TICK, **kw))

    def test_expansion_beyond_atr_is_momentum(self):
        ok, reason, is_mom = self._engine()._momentum(a_bar(20.0), atr_value=10.0)
        self.assertTrue(ok)
        self.assertTrue(is_mom)
        self.assertIn("2.00x ATR", reason)

    def test_a_small_candle_is_not_momentum_but_is_allowed_by_default(self):
        ok, reason, is_mom = self._engine()._momentum(a_bar(5.0), atr_value=10.0)
        self.assertTrue(ok, "off by default, so it must not reject")
        self.assertFalse(is_mom)
        self.assertIsNone(reason, "no tag when the condition is absent")

    def test_requiring_it_rejects_a_small_candle(self):
        eng = self._engine(require_momentum=True)
        ok, reason, is_mom = eng._momentum(a_bar(5.0), atr_value=10.0)
        self.assertFalse(ok)
        self.assertFalse(is_mom)
        self.assertIn("0.50x ATR", reason)
        self.assertIn("momentum threshold", reason)

    def test_requiring_it_still_passes_an_expanding_candle(self):
        eng = self._engine(require_momentum=True)
        ok, _, is_mom = eng._momentum(a_bar(20.0), atr_value=10.0)
        self.assertTrue(ok)
        self.assertTrue(is_mom)

    def test_exactly_at_the_threshold_counts(self):
        ok, _, is_mom = self._engine()._momentum(a_bar(10.0), atr_value=10.0)
        self.assertTrue(is_mom)

    def test_the_multiple_is_configurable(self):
        strict = self._engine(momentum_atr_mult=3.0, require_momentum=True)
        ok, _, _ = strict._momentum(a_bar(20.0), atr_value=10.0)
        self.assertFalse(ok, "2x ATR must fail a 3x threshold")

    def test_a_zero_atr_refuses_to_judge_rather_than_calling_everything_momentum(self):
        """A zero denominator would make every candle infinitely expansive."""
        eng = self._engine(require_momentum=True)
        ok, reason, is_mom = eng._momentum(a_bar(5.0), atr_value=0.0)
        self.assertTrue(ok, "must not reject on a missing volatility estimate")
        self.assertFalse(is_mom, "and must not claim momentum either")
        self.assertIsNone(reason)

    def test_it_rejects_a_nonsense_multiple(self):
        with self.assertRaises(ValueError):
            StrategyConfig(tick_size=TICK, momentum_atr_mult=0.0)
        with self.assertRaises(ValueError):
            StrategyConfig(tick_size=TICK, momentum_atr_mult=-1.0)


class TestTagging(unittest.TestCase):
    def test_the_reason_string_becomes_a_confluence_tag(self):
        tags = tags_from_reasons(["momentum candle: range 2.10x ATR"])
        self.assertIn("momentum", tags)

    def test_an_ordinary_reason_does_not_produce_the_tag(self):
        tags = tags_from_reasons(["candle 2 closed 100.00, below 102.00"])
        self.assertNotIn("momentum", tags)


class TestEndToEnd(unittest.TestCase):
    def _bars(self):
        sc = SyntheticConfig(n_ticks=200_000, tick_size=TICK, seed=9)
        return bars_from_trades(synthetic_trades(sc), 300.0, TICK)

    def test_off_by_default_it_changes_nothing_about_which_trades_happen(self):
        """Measuring a condition must not alter the sample being measured."""
        bars = self._bars()
        base = StrategyConfig(tick_size=TICK)
        result = Backtester(base).run(bars)
        # A huge threshold makes the condition unreachable; with require off,
        # the same trades must still be taken.
        never = StrategyConfig(tick_size=TICK, momentum_atr_mult=99.0)
        other = Backtester(never).run(bars)
        self.assertEqual(
            [(t.entry_index, t.exit_index) for t in result.trades],
            [(t.entry_index, t.exit_index) for t in other.trades],
        )

    def test_requiring_it_means_every_resulting_trade_has_it(self):
        """The sound invariant.

        Note what is *not* asserted: that the strict trades are a subset of the
        loose ones. They are not. Blocking a setup leaves the engine free for a
        later one, and the level book evolves differently from there, so the
        two runs take genuinely different paths rather than one thinning the
        other.
        """
        bars = self._bars()
        strict = Backtester(
            StrategyConfig(tick_size=TICK, require_momentum=True)
        ).run(bars)
        self.assertTrue(strict.trades)
        for t in strict.trades:
            self.assertIn("momentum", t.tags)

    def test_a_strict_threshold_eventually_removes_everything(self):
        bars = self._bars()
        result = Backtester(
            StrategyConfig(tick_size=TICK, momentum_atr_mult=50.0,
                           require_momentum=True)
        ).run(bars)
        self.assertEqual(result.trades, [])

    def test_tagged_trades_carry_the_condition_into_confluence(self):
        from ..confluence import analyse

        bars = self._bars()
        result = Backtester(
            StrategyConfig(tick_size=TICK, momentum_atr_mult=0.5)
        ).run(bars)
        tagged = [t for t in result.trades if "momentum" in t.tags]
        self.assertTrue(tagged, "a low threshold should tag some trades")
        report = analyse(result.trades)
        self.assertIn("momentum", {f.tag for f in report.factors})


if __name__ == "__main__":
    unittest.main()
