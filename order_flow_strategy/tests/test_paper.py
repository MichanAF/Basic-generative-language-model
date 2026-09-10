"""Tests for the paper-trading harness.

The properties worth testing here are not "does it make money" -- that is the
backtest's job. They are the operational ones: does it survive a restart, does
it act once per bar, and does the position it thinks it holds match what the
engine says it should hold.
"""

import json
import os
import tempfile
import unittest

from ..backtest import BacktestResult, ClosedTrade, OpenPosition
from ..config import StrategyConfig
from ..data import SyntheticConfig, bars_from_trades, synthetic_trades
from ..paper import PaperState, PaperTrader, load_state, save_state
from .helpers import TICK, make_bar


def a_result(
    open_position=None, trades=None, equity=100_000.0, cfg=None
) -> BacktestResult:
    """A minimal result, so tests can drive the reconciler directly."""
    cfg = cfg or StrategyConfig(tick_size=TICK)
    return BacktestResult(
        trades=list(trades or []),
        equity_curve=[equity],
        bar_timestamps=[0.0],
        signals=[],
        config=cfg,
        bars_tested=1,
        used_proxy_footprints=True,
        open_position=open_position,
    )


def a_position(direction=1, entry_ts=100.0, qty=2.0, stop=95.0, target=115.0):
    return OpenPosition(
        direction=direction,
        entry_index=1,
        entry_ts=entry_ts,
        entry_price=100.0,
        quantity=qty,
        stop=stop,
        target=target,
    )


def a_trade(pnl=250.0, r=1.5, direction=1, reason="target"):
    return ClosedTrade(
        direction=direction, entry_index=1, entry_ts=100.0, entry_price=100.0,
        exit_index=4, exit_ts=400.0, exit_price=110.0, quantity=2.0,
        pnl=pnl, r_multiple=r, mae_r=-0.2, mfe_r=1.6, bars_held=3,
        exit_reason=reason, level_price=100.0, level_strength=4.0,
    )


class _Scripted:
    """Returns a queued result per call, so a test can script the engine."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def __call__(self, bars):
        self.calls += 1
        return self.results.pop(0)


class TestState(unittest.TestCase):
    def test_round_trips_through_disk(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "state.json")
            st = PaperState(
                equity=101_234.5, direction=-1, quantity=3.0, entry_price=99.5,
                stop=101.0, target=94.0, entry_ts=500.0, last_bar_ts=900.0,
                n_trades_seen=4, bars_seen=250, n_decisions=17,
            )
            save_state(path, st)
            self.assertEqual(load_state(path, 0.0), st)

    def test_missing_file_starts_at_the_configured_equity(self):
        with tempfile.TemporaryDirectory() as d:
            st = load_state(os.path.join(d, "nope.json"), 50_000.0)
            self.assertEqual(st.equity, 50_000.0)
            self.assertTrue(st.is_flat)

    def test_corrupt_file_raises_rather_than_forgetting_a_position(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            with self.assertRaises(RuntimeError):
                load_state(path, 100_000.0)

    def test_non_object_json_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("[1, 2, 3]")
            with self.assertRaises(RuntimeError):
                load_state(path, 100_000.0)

    def test_unknown_fields_are_ignored_so_old_files_still_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"equity": 7.0, "retired_field": 1}, fh)
            self.assertEqual(load_state(path, 0.0).equity, 7.0)

    def test_unrealised_is_signed_by_direction(self):
        long = PaperState(direction=1, quantity=2.0, entry_price=100.0)
        short = PaperState(direction=-1, quantity=2.0, entry_price=100.0)
        self.assertAlmostEqual(long.unrealised(110.0, 1.0), 20.0)
        self.assertAlmostEqual(short.unrealised(110.0, 1.0), -20.0)
        self.assertEqual(PaperState().unrealised(110.0, 1.0), 0.0)


class TestReconciliation(unittest.TestCase):
    def setUp(self):
        self.cfg = StrategyConfig(tick_size=TICK)
        self.dir = tempfile.TemporaryDirectory()
        self.state = os.path.join(self.dir.name, "state.json")
        self.journal = os.path.join(self.dir.name, "journal.jsonl")

    def tearDown(self):
        self.dir.cleanup()

    def _trader(self, results):
        return PaperTrader(
            self.cfg, state_path=self.state, journal_path=self.journal,
            runner=_Scripted(results),
        )

    def _bars(self, n):
        return [make_bar(ts=300.0 * i, o=100, h=101, l=99, c=100) for i in range(1, n + 1)]

    def test_flat_and_wanted_flat_is_a_recorded_non_decision(self):
        bars = self._bars(2)
        pt = self._trader([a_result(), a_result()])
        pt.update(bars[:1])
        out = pt.update(bars)
        self.assertEqual([d.action for d in out], ["no_signal"])

    def test_opening_adopts_the_engine_position(self):
        bars = self._bars(2)
        pos = a_position()
        pt = self._trader([a_result(), a_result(open_position=pos)])
        pt.update(bars[:1])
        out = pt.update(bars)
        self.assertEqual([d.action for d in out], ["open_long"])
        self.assertEqual(pt.state.direction, 1)
        self.assertEqual(pt.state.entry_ts, pos.entry_ts)
        self.assertEqual(pt.state.stop, pos.stop)

    def test_short_is_labelled_as_a_short(self):
        bars = self._bars(2)
        pt = self._trader(
            [a_result(), a_result(open_position=a_position(direction=-1))]
        )
        pt.update(bars[:1])
        self.assertEqual([d.action for d in pt.update(bars)], ["open_short"])

    def test_an_unchanged_position_holds(self):
        bars = self._bars(3)
        pos = a_position()
        pt = self._trader(
            [a_result(), a_result(open_position=pos), a_result(open_position=pos)]
        )
        pt.update(bars[:1])
        pt.update(bars[:2])
        self.assertEqual([d.action for d in pt.update(bars)], ["hold"])

    def test_a_moved_stop_produces_an_adjust(self):
        bars = self._bars(3)
        pt = self._trader([
            a_result(),
            a_result(open_position=a_position(stop=95.0)),
            a_result(open_position=a_position(stop=99.0)),
        ])
        pt.update(bars[:1])
        pt.update(bars[:2])
        out = pt.update(bars)
        self.assertEqual([d.action for d in out], ["adjust"])
        self.assertIn("stop", out[0].reason)
        self.assertEqual(pt.state.stop, 99.0)

    def test_a_scale_out_produces_an_adjust(self):
        bars = self._bars(3)
        pt = self._trader([
            a_result(),
            a_result(open_position=a_position(qty=2.0)),
            a_result(open_position=a_position(qty=1.0)),
        ])
        pt.update(bars[:1])
        pt.update(bars[:2])
        out = pt.update(bars)
        self.assertEqual([d.action for d in out], ["adjust"])
        self.assertIn("size", out[0].reason)
        self.assertEqual(pt.state.quantity, 1.0)

    def test_a_closed_trade_is_journalled_from_the_engine_not_the_close_price(self):
        bars = self._bars(3)
        trade = a_trade(pnl=250.0, r=1.5)
        pt = self._trader([
            a_result(),
            a_result(open_position=a_position()),
            a_result(trades=[trade], equity=100_250.0),
        ])
        pt.update(bars[:1])
        pt.update(bars[:2])
        out = pt.update(bars)
        self.assertEqual([d.action for d in out], ["close", "no_signal"])
        self.assertEqual(out[0].price, trade.exit_price)
        self.assertEqual(out[0].r_multiple, 1.5)
        self.assertIn("target", out[0].reason)
        self.assertTrue(pt.state.is_flat)
        self.assertAlmostEqual(pt.state.equity, 100_250.0)

    def test_a_round_trip_between_updates_is_not_mistaken_for_a_hold(self):
        """Same direction, different entry -- direction alone would miss it."""
        bars = self._bars(3)
        first = a_position(entry_ts=100.0)
        second = a_position(entry_ts=900.0)
        pt = self._trader([
            a_result(),
            a_result(open_position=first),
            a_result(open_position=second, trades=[]),
        ])
        pt.update(bars[:1])
        pt.update(bars[:2])
        out = pt.update(bars)
        self.assertEqual([d.action for d in out], ["close", "open_long"])
        self.assertEqual(pt.state.entry_ts, 900.0)

    def test_a_position_vanishing_without_a_trade_forces_flat_and_says_so(self):
        bars = self._bars(3)
        pt = self._trader([
            a_result(),
            a_result(open_position=a_position()),
            a_result(open_position=None, trades=[]),
        ])
        pt.update(bars[:1])
        pt.update(bars[:2])
        out = pt.update(bars)
        self.assertEqual([d.action for d in out], ["close"])
        self.assertIn("forced flat", out[0].reason)
        self.assertTrue(pt.state.is_flat)

    def test_close_and_reopen_on_one_bar_produces_both(self):
        bars = self._bars(3)
        pt = self._trader([
            a_result(),
            a_result(open_position=a_position(entry_ts=100.0)),
            a_result(
                open_position=a_position(direction=-1, entry_ts=900.0),
                trades=[a_trade()],
            ),
        ])
        pt.update(bars[:1])
        pt.update(bars[:2])
        out = pt.update(bars)
        self.assertEqual([d.action for d in out], ["close", "open_short"])

    def test_the_first_run_adopts_history_without_journalling_it(self):
        bars = self._bars(1)
        pt = self._trader([a_result(trades=[a_trade(), a_trade()])])
        out = pt.update(bars)
        self.assertEqual([d.action for d in out], ["no_signal"])
        self.assertEqual(pt.state.n_trades_seen, 2)
        self.assertTrue(any("historical trades" in w for w in pt.warnings))


class TestOperational(unittest.TestCase):
    def setUp(self):
        self.cfg = StrategyConfig(tick_size=TICK)
        self.dir = tempfile.TemporaryDirectory()
        self.state = os.path.join(self.dir.name, "state.json")
        self.journal = os.path.join(self.dir.name, "journal.jsonl")
        sc = SyntheticConfig(n_ticks=120_000, tick_size=TICK, seed=11)
        self.bars = bars_from_trades(synthetic_trades(sc), 300.0, TICK)

    def tearDown(self):
        self.dir.cleanup()

    def _trader(self):
        return PaperTrader(self.cfg, state_path=self.state, journal_path=self.journal)

    def test_empty_bars_do_nothing(self):
        self.assertEqual(self._trader().update([]), [])

    def test_the_same_bar_is_never_acted_on_twice(self):
        pt = self._trader()
        pt.update(self.bars[:300])
        first = pt.update(self.bars[:301])
        self.assertTrue(first)
        self.assertEqual(pt.update(self.bars[:301]), [])
        self.assertEqual(pt.update(self.bars[:300]), [])  # older, also ignored

    def test_the_journal_grows_by_exactly_the_decisions_taken(self):
        pt = self._trader()
        pt.update(self.bars[:300])
        before = len(pt.read_journal())
        taken = pt.update(self.bars[:302])
        self.assertEqual(len(pt.read_journal()), before + len(taken))

    def test_journal_lines_are_valid_json_with_a_stable_shape(self):
        pt = self._trader()
        pt.update(self.bars[:300])
        pt.update(self.bars[:305])
        entries = pt.read_journal()
        self.assertTrue(entries)
        for e in entries:
            self.assertIn(e["action"], {
                "open_long", "open_short", "close", "adjust", "hold", "no_signal",
            })
            self.assertIsInstance(e["bar_ts"], float)
            self.assertIn("equity", e)

    def test_a_restart_resumes_the_same_book(self):
        pt = self._trader()
        for i in range(300, 340):
            pt.update(self.bars[: i + 1])
        resumed = self._trader()
        self.assertEqual(resumed.state, pt.state)

    def test_a_restart_does_not_replay_bars_it_already_processed(self):
        pt = self._trader()
        pt.update(self.bars[:320])
        resumed = self._trader()
        self.assertEqual(resumed.update(self.bars[:320]), [])

    def test_every_open_is_eventually_closed_or_still_held(self):
        """The journal must not describe an impossible book."""
        pt = self._trader()
        pt.update(self.bars[:250])
        for i in range(250, len(self.bars)):
            pt.update(self.bars[: i + 1])

        held = 0
        for e in pt.read_journal():
            if e["action"] in ("open_long", "open_short"):
                self.assertEqual(held, 0, "opened while already positioned")
                held = 1
            elif e["action"] == "close":
                self.assertEqual(held, 1, "closed while flat")
                held = 0
            elif e["action"] == "adjust":
                self.assertEqual(held, 1, "adjusted while flat")
            elif e["action"] == "hold":
                self.assertEqual(held, 1, "held while flat")
            elif e["action"] == "no_signal":
                self.assertEqual(held, 0, "no_signal while positioned")
        self.assertEqual(held, 0 if pt.state.is_flat else 1)

    def test_equity_comes_from_the_engine_not_a_second_implementation(self):
        from ..backtest import Backtester

        pt = self._trader()
        pt.update(self.bars[:250])
        pt.update(self.bars)
        engine = Backtester(self.cfg).run(self.bars, flatten_at_end=False)
        self.assertAlmostEqual(pt.state.equity, engine.equity_curve[-1], places=6)

    def test_a_shrinking_history_warns_because_equity_gets_rebased(self):
        pt = self._trader()
        pt.update(self.bars[:400])
        pt.update(self.bars[:200] + self.bars[400:402])
        self.assertTrue(any("history shrank" in w for w in pt.warnings))

    def test_warnings_reset_between_updates(self):
        pt = self._trader()
        pt.update(self.bars[:400])
        pt.update(self.bars[:401])
        self.assertEqual(pt.warnings, [])

    def test_status_reports_flat_and_positioned_books(self):
        pt = self._trader()
        pt.update(self.bars[:300])
        self.assertIn("Paper trading status", pt.status(self.bars[299].close))
        pt.state.direction, pt.state.quantity = 1, 2.0
        pt.state.entry_price, pt.state.stop = 100.0, 95.0
        text = pt.status(110.0)
        self.assertIn("long 2 @ 100.00", text)
        self.assertIn("Unrealised", text)

    def test_status_never_claims_to_have_placed_an_order(self):
        self.assertIn("No orders are placed", self._trader().status())


if __name__ == "__main__":
    unittest.main()
