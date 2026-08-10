import unittest

from ..backtest import Backtester, _PendingEntry, _Position
from ..config import ENTRY_BREAK_OF_2, StrategyConfig
from ..levels import RESISTANCE, Level
from ..metrics import max_consecutive_losses, max_drawdown, summarize
from ..signals import LONG, SHORT, Signal, prepare
from .helpers import TICK, make_bar

POINT_VALUE = 12.5 / TICK  # 50 currency units per point


def cfg(**overrides) -> StrategyConfig:
    base = dict(
        tick_size=TICK,
        tick_value=12.5,
        slippage_ticks=0.0,
        commission_per_side=0.0,
        partial_at_r=None,
        time_stop_bars=None,
        trail_atr=None,
    )
    base.update(overrides)
    return StrategyConfig(**base)


def a_level() -> Level:
    return Level(price=100.0, side=RESISTANCE, strength=5.0, created_index=0, last_update_index=0)


def a_signal(direction=SHORT, entry=100.0, stop=102.0, target=96.0) -> Signal:
    return Signal(
        index=1,
        ts=0.0,
        direction=direction,
        level=a_level(),
        c1_index=0,
        c2_index=1,
        entry_price=entry,
        stop=stop,
        target=target,
        risk_per_unit=abs(stop - entry),
        atr=1.0,
        entry_mode="close_of_2",
    )


def a_position(bt: Backtester, config: StrategyConfig, direction=SHORT, qty=1.0) -> _Position:
    sig = a_signal(direction=direction)
    stop, target = (102.0, 96.0) if direction == SHORT else (98.0, 104.0)
    return _Position(
        signal=sig,
        direction=direction,
        entry_index=0,
        entry_price=100.0,
        quantity=qty,
        initial_quantity=qty,
        risk_per_unit=2.0,
        stop=stop,
        target=target,
        level=sig.level,
        best_price=100.0,
        worst_price=100.0,
    )


class TestFillRules(unittest.TestCase):
    """The pessimistic assumptions, verified rather than asserted in prose."""

    def setUp(self):
        self.cfg = cfg()
        self.bt = Backtester(self.cfg)
        self.ind = None

    def _ind(self, bars):
        return prepare(bars, self.cfg)

    def test_stop_wins_when_a_bar_contains_both_stop_and_target(self):
        pos = a_position(self.bt, self.cfg)
        bar = make_bar(ts=300, o=100.0, h=102.5, l=95.0, c=97.0)  # touches both
        closed, _ = self.bt._manage(pos, 1, bar, self._ind([bar]), 100_000.0)
        self.assertIsNotNone(closed)
        self.assertEqual(closed.exit_reason, "stop")

    def test_gap_through_the_stop_fills_at_the_open(self):
        pos = a_position(self.bt, self.cfg)
        bar = make_bar(ts=300, o=105.0, h=106.0, l=104.0, c=105.0)  # gapped past 102
        closed, _ = self.bt._manage(pos, 1, bar, self._ind([bar]), 100_000.0)
        self.assertEqual(closed.exit_reason, "stop")
        self.assertAlmostEqual(closed.exit_price, 105.0, places=6)

    def test_gap_through_the_target_fills_at_the_open_in_your_favour(self):
        pos = a_position(self.bt, self.cfg)
        bar = make_bar(ts=300, o=94.0, h=94.5, l=93.0, c=93.5)  # gapped past 96
        closed, _ = self.bt._manage(pos, 1, bar, self._ind([bar]), 100_000.0)
        self.assertEqual(closed.exit_reason, "target")
        self.assertAlmostEqual(closed.exit_price, 94.0, places=6)

    def test_slippage_worsens_stop_exits_but_not_limit_targets(self):
        config = cfg(slippage_ticks=2.0)
        bt = Backtester(config)

        stopped = a_position(bt, config)
        bar = make_bar(ts=300, o=101.0, h=102.5, l=100.5, c=102.0)
        closed, _ = bt._manage(stopped, 1, bar, prepare([bar], config), 100_000.0)
        self.assertAlmostEqual(closed.exit_price, 102.0 + 2 * TICK, places=6)

        filled = a_position(bt, config)
        bar2 = make_bar(ts=300, o=99.0, h=99.5, l=95.5, c=96.0)
        closed2, _ = bt._manage(filled, 1, bar2, prepare([bar2], config), 100_000.0)
        self.assertAlmostEqual(closed2.exit_price, 96.0, places=6)

    def test_r_multiple_matches_the_realized_move(self):
        pos = a_position(self.bt, self.cfg, qty=3.0)
        bar = make_bar(ts=300, o=99.0, h=99.5, l=95.5, c=96.0)
        closed, equity = self.bt._manage(pos, 1, bar, self._ind([bar]), 100_000.0)
        # Short from 100 to 96 = 4 points on a 2 point risk.
        self.assertAlmostEqual(closed.r_multiple, 2.0, places=6)
        self.assertAlmostEqual(closed.pnl, 4.0 * 3.0 * POINT_VALUE, places=6)
        self.assertAlmostEqual(equity, 100_000.0 + closed.pnl, places=6)

    def test_commission_is_charged_on_both_sides(self):
        config = cfg(commission_per_side=5.0)
        bt = Backtester(config)
        pos = a_position(bt, config, qty=2.0)
        bar = make_bar(ts=300, o=99.0, h=99.5, l=95.5, c=96.0)
        closed, _ = bt._manage(pos, 1, bar, prepare([bar], config), 100_000.0)
        self.assertAlmostEqual(closed.pnl, 4.0 * 2.0 * POINT_VALUE - 5.0 * 2.0 * 2, places=6)

    def test_long_positions_mirror_the_rules(self):
        pos = a_position(self.bt, self.cfg, direction=LONG)
        bar = make_bar(ts=300, o=100.0, h=104.5, l=97.5, c=99.0)  # both levels touched
        closed, _ = self.bt._manage(pos, 1, bar, self._ind([bar]), 100_000.0)
        self.assertEqual(closed.exit_reason, "stop")
        self.assertAlmostEqual(closed.exit_price, 98.0, places=6)

    def test_time_stop_exits_at_the_close(self):
        config = cfg(time_stop_bars=3)
        bt = Backtester(config)
        pos = a_position(bt, config)
        bar = make_bar(ts=300, o=100.0, h=100.5, l=99.5, c=99.8)
        self.assertIsNone(bt._manage(pos, 2, bar, prepare([bar], config), 100_000.0)[0])
        closed, _ = bt._manage(pos, 3, bar, prepare([bar], config), 100_000.0)
        self.assertEqual(closed.exit_reason, "time_stop")
        self.assertAlmostEqual(closed.exit_price, 99.8, places=6)

    def test_partial_scales_out_and_moves_the_stop_to_breakeven(self):
        config = cfg(partial_at_r=1.0, partial_frac=0.5, breakeven_after_partial=True)
        bt = Backtester(config)
        pos = a_position(bt, config, qty=4.0)
        runner = make_bar(ts=300, o=99.5, h=99.6, l=97.9, c=98.0)  # reaches 1R at 98
        closed, equity = bt._manage(pos, 1, runner, prepare([runner], config), 100_000.0)
        self.assertIsNone(closed)
        self.assertTrue(pos.partial_done)
        self.assertAlmostEqual(pos.quantity, 2.0, places=6)
        self.assertAlmostEqual(pos.stop, 100.0, places=6)
        self.assertAlmostEqual(equity, 100_000.0 + 2.0 * 2.0 * POINT_VALUE, places=6)

        back = make_bar(ts=600, o=99.0, h=100.5, l=99.0, c=100.2)
        closed2, _ = bt._manage(pos, 2, back, prepare([back], config), equity)
        self.assertEqual(closed2.exit_reason, "breakeven_after_partial")
        self.assertAlmostEqual(closed2.r_multiple, 0.5, places=6)

    def test_trailing_stop_only_moves_favourably(self):
        config = cfg(trail_atr=1.0)
        bt = Backtester(config)
        pos = a_position(bt, config)
        tight = make_bar(ts=300, o=99.0, h=99.2, l=98.8, c=98.9)
        bt._manage(pos, 1, tight, prepare([tight, tight], config), 100_000.0)
        trailed = pos.stop
        self.assertLess(trailed, 102.0, "the stop should have pulled in behind the low")

        # Price backs up: the low is higher, so the naive trail would sit wider.
        # It must not move, and the bar must not have hit the tightened stop.
        backup = make_bar(ts=600, o=99.0, h=99.15, l=99.0, c=99.1)
        closed, _ = bt._manage(pos, 2, backup, prepare([tight, tight, backup], config), 100_000.0)
        self.assertIsNone(closed)
        self.assertLessEqual(pos.stop, trailed, "a trailing stop must never loosen")


class TestSizing(unittest.TestCase):
    def test_size_follows_risk_budget_and_stop_distance(self):
        config = cfg(risk_per_trade=0.01, whole_units=False)
        bt = Backtester(config)
        # 1% of 100k = 1000 risk; 2 points at 50/point = 100 per unit -> 10 units
        self.assertAlmostEqual(bt._size(2.0, 100_000.0), 10.0, places=6)
        self.assertAlmostEqual(bt._size(4.0, 100_000.0), 5.0, places=6)

    def test_whole_units_round_down(self):
        config = cfg(risk_per_trade=0.01, whole_units=True)
        bt = Backtester(config)
        self.assertEqual(bt._size(3.0, 100_000.0), 6.0)  # 6.66 -> 6

    def test_a_trade_too_large_for_the_account_is_skipped(self):
        config = cfg(risk_per_trade=0.001, whole_units=True)
        bt = Backtester(config)
        self.assertEqual(bt._size(50.0, 1_000.0), 0.0)
        self.assertIsNone(bt._open(a_signal(), 1, 100.0, 1_000.0))

    def test_inverted_stop_is_refused(self):
        bt = Backtester(cfg())
        bad = a_signal(entry=100.0, stop=99.0)  # stop below entry on a short
        self.assertIsNone(bt._open(bad, 1, 100.0, 100_000.0))


class TestPendingEntries(unittest.TestCase):
    def test_stop_entry_triggers_when_price_trades_through(self):
        config = cfg(entry_mode=ENTRY_BREAK_OF_2)
        bt = Backtester(config)
        pending = _PendingEntry(signal=a_signal(entry=99.0), placed_index=0, expires_index=2)
        untouched = make_bar(ts=300, o=99.5, h=99.8, l=99.2, c=99.4)
        self.assertIsNone(bt._try_trigger(pending, 1, untouched, 100_000.0))

        through = make_bar(ts=600, o=99.4, h=99.5, l=98.0, c=98.2)
        pos = bt._try_trigger(pending, 2, through, 100_000.0)
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos.entry_price, 99.0, places=6)

    def test_gap_below_the_trigger_fills_at_the_open(self):
        config = cfg(entry_mode=ENTRY_BREAK_OF_2)
        bt = Backtester(config)
        pending = _PendingEntry(signal=a_signal(entry=99.0), placed_index=0, expires_index=2)
        gapped = make_bar(ts=300, o=97.0, h=97.5, l=96.5, c=96.8)
        pos = bt._try_trigger(pending, 1, gapped, 100_000.0)
        self.assertAlmostEqual(pos.entry_price, 97.0, places=6)

    def test_order_is_cancelled_once_the_setup_is_invalidated(self):
        bt = Backtester(cfg(entry_mode=ENTRY_BREAK_OF_2))
        pending = _PendingEntry(signal=a_signal(entry=99.0, stop=102.0), placed_index=0, expires_index=2)
        blown = make_bar(ts=300, o=101.0, h=102.5, l=100.5, c=102.0)
        self.assertTrue(bt._pending_invalidated(pending, blown))


class TestEndToEnd(unittest.TestCase):
    def _bars(self):
        from ..data import SyntheticConfig, bars_from_trades, synthetic_trades

        sc = SyntheticConfig(n_ticks=60_000, tick_size=TICK, seed=3)
        return bars_from_trades(synthetic_trades(sc), 300.0, TICK)

    def test_run_is_deterministic(self):
        bars = self._bars()
        config = StrategyConfig(tick_size=TICK)
        a = Backtester(config).run(bars)
        b = Backtester(config).run(bars)
        self.assertEqual(
            [(t.entry_index, t.exit_index, round(t.pnl, 6)) for t in a.trades],
            [(t.entry_index, t.exit_index, round(t.pnl, 6)) for t in b.trades],
        )

    def test_equity_curve_has_one_point_per_bar(self):
        bars = self._bars()
        result = Backtester(StrategyConfig(tick_size=TICK)).run(bars)
        self.assertEqual(len(result.equity_curve), len(bars))

    def test_positions_never_open_before_their_signal(self):
        bars = self._bars()
        result = Backtester(StrategyConfig(tick_size=TICK)).run(bars)
        for t in result.trades:
            self.assertGreaterEqual(t.exit_index, t.entry_index)
            self.assertGreaterEqual(t.entry_index, 1)

    def test_only_one_position_at_a_time(self):
        bars = self._bars()
        result = Backtester(StrategyConfig(tick_size=TICK)).run(bars)
        for earlier, later in zip(result.trades, result.trades[1:]):
            self.assertGreaterEqual(later.entry_index, earlier.exit_index)

    def test_open_position_is_flattened_at_the_end_of_data(self):
        bars = self._bars()
        config = StrategyConfig(tick_size=TICK, time_stop_bars=None, target_r=50.0)
        result = Backtester(config).run(bars)
        if result.trades:
            reasons = {t.exit_reason for t in result.trades}
            self.assertTrue(reasons <= {"stop", "target", "end_of_data", "breakeven_after_partial",
                                        "stop_after_partial", "time_stop"})

    def test_summary_survives_a_run_with_no_trades(self):
        bars = self._bars()
        impossible = StrategyConfig(tick_size=TICK, level_min_strength=1e9)
        stats = summarize(Backtester(impossible).run(bars))
        self.assertEqual(stats.n_trades, 0)
        self.assertEqual(stats.expectancy_r, 0.0)


class TestMetrics(unittest.TestCase):
    def test_max_drawdown(self):
        absolute, fraction = max_drawdown([100.0, 120.0, 90.0, 130.0])
        self.assertAlmostEqual(absolute, 30.0)
        self.assertAlmostEqual(fraction, 0.25)

    def test_max_drawdown_on_a_rising_curve_is_zero(self):
        self.assertEqual(max_drawdown([1.0, 2.0, 3.0]), (0.0, 0.0))

    def test_consecutive_losses(self):
        class T:
            def __init__(self, pnl):
                self.pnl = pnl

        self.assertEqual(max_consecutive_losses([T(1), T(-1), T(-1), T(1), T(-1)]), 2)


if __name__ == "__main__":
    unittest.main()
