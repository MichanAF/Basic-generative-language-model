"""Tests for the set-and-forget allocator and the shared persistence layer.

The properties that matter are the ones you cannot check by watching it run
for a week: that the rails never block an exit, that a restart resumes the
same book mid-trend, that hysteresis actually suppresses churn, and that the
halt is sticky.
"""

import json
import math
import os
import tempfile
import unittest

from ..autopilot import (
    RAIL_BAD_DATA,
    RAIL_HALTED,
    RAIL_MIN_HOLD,
    RAIL_STALE,
    RAIL_WARMUP,
    RISK_OFF,
    RISK_ON,
    Autopilot,
    AutopilotConfig,
    AutopilotRunner,
    AutopilotState,
    format_autopilot,
)
from ..data import Bar
from ..footprint import Footprint
from ..persistence import (
    append_jsonl,
    atomic_write_json,
    load_dataclass,
    read_json_strict,
    read_jsonl,
)

DAY = 86400.0


def a_bar(i: int, price: float) -> Bar:
    fp = Footprint.from_ohlcv_proxy(
        open_=price, high=price * 1.005, low=price * 0.995, close=price,
        volume=100.0, tick_size=10.0,
    )
    return Bar(
        ts=i * DAY, open=price, high=price * 1.005, low=price * 0.995,
        close=price, volume=100.0, footprint=fp,
    )


def bars_from(prices):
    return [a_bar(i, p) for i, p in enumerate(prices)]


def flat_then(n_flat: int, tail, level: float = 100.0):
    """``n_flat`` bars at ``level`` to fill the average, then ``tail``."""
    return bars_from([level] * n_flat + list(tail))


SHORT = AutopilotConfig(sma_period=10, band_pct=0.03, min_hold_bars=0)


class TestConfig(unittest.TestCase):
    def test_rejects_nonsense(self):
        for kwargs in (
            {"sma_period": 1},
            {"band_pct": 1.0},
            {"band_pct": -0.1},
            {"min_hold_bars": -1},
            {"target_exposure": 0.0},
            {"target_exposure": 1.5},
            {"fee_rate": -0.1},
            {"halt_drawdown_pct": 0.0},
            {"halt_drawdown_pct": 1.0},
            {"starting_equity": 0.0},
        ):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                AutopilotConfig(**kwargs)

    def test_cost_drag_is_two_sided(self):
        cfg = AutopilotConfig(fee_rate=0.001, slippage_rate=0.0005)
        self.assertAlmostEqual(cfg.cost_rate, 0.0015)
        # Six round trips a year at 15bp a side is 1.8% of the account.
        self.assertAlmostEqual(cfg.annual_cost_drag(6), 0.018)

    def test_partial_exposure_scales_the_drag(self):
        cfg = AutopilotConfig(target_exposure=0.5)
        full = AutopilotConfig(target_exposure=1.0)
        self.assertAlmostEqual(cfg.annual_cost_drag(6), full.annual_cost_drag(6) / 2)


class TestRegimeRule(unittest.TestCase):
    def _decide(self, bars, regime=RISK_OFF, since=10_000, cfg=SHORT, now=None):
        return Autopilot(cfg).decide(bars, len(bars) - 1, regime, since, now=now)

    def test_warmup_keeps_it_out(self):
        v = self._decide(bars_from([100.0] * 5))
        self.assertEqual(v.regime, RISK_OFF)
        self.assertEqual(v.blocked_by, RAIL_WARMUP)
        self.assertEqual(v.target_exposure, 0.0)

    def test_clearing_the_upper_band_turns_it_on(self):
        v = self._decide(flat_then(10, [130.0]))
        self.assertEqual(v.regime, RISK_ON)
        self.assertIsNone(v.blocked_by)
        self.assertEqual(v.target_exposure, 1.0)

    def test_inside_the_band_nothing_changes(self):
        """The whole point of hysteresis: no flip in the dead zone."""
        bars = flat_then(10, [101.0])
        self.assertEqual(self._decide(bars, regime=RISK_OFF).regime, RISK_OFF)
        self.assertEqual(self._decide(bars, regime=RISK_ON).regime, RISK_ON)

    def test_breaking_the_lower_band_turns_it_off(self):
        v = self._decide(flat_then(10, [70.0]), regime=RISK_ON)
        self.assertEqual(v.regime, RISK_OFF)
        self.assertEqual(v.target_exposure, 0.0)

    def test_the_sma_is_causal(self):
        """A verdict must not change when later bars are appended."""
        prices = [100.0] * 10 + [130.0, 60.0, 200.0]
        full = bars_from(prices)
        early = Autopilot(SHORT).decide(full[:11], 10, RISK_OFF, 10_000)
        later = Autopilot(SHORT).decide(full, 10, RISK_OFF, 10_000)
        self.assertEqual((early.regime, early.sma), (later.regime, later.sma))


class TestRailsNeverBlockAnExit(unittest.TestCase):
    """The design principle, tested three ways."""

    def test_min_hold_blocks_entry(self):
        cfg = AutopilotConfig(sma_period=10, band_pct=0.03, min_hold_bars=5)
        v = Autopilot(cfg).decide(flat_then(10, [130.0]), 10, RISK_OFF, 2)
        self.assertEqual(v.regime, RISK_OFF)
        self.assertEqual(v.blocked_by, RAIL_MIN_HOLD)

    def test_min_hold_does_not_block_exit(self):
        cfg = AutopilotConfig(sma_period=10, band_pct=0.03, min_hold_bars=5)
        v = Autopilot(cfg).decide(flat_then(10, [70.0]), 10, RISK_ON, 0)
        self.assertEqual(v.regime, RISK_OFF)
        self.assertIsNone(v.blocked_by)

    def test_a_violent_bar_blocks_entry(self):
        cfg = AutopilotConfig(sma_period=10, band_pct=0.03, min_hold_bars=0,
                              max_gap_pct=0.20)
        v = Autopilot(cfg).decide(flat_then(10, [200.0]), 10, RISK_OFF, 99)
        self.assertEqual(v.blocked_by, RAIL_BAD_DATA)
        self.assertEqual(v.regime, RISK_OFF)

    def test_a_violent_bar_does_not_block_exit(self):
        """A 50% crash might be bad data or might be real. Selling costs a
        round trip if wrong; not selling costs the crash if right."""
        cfg = AutopilotConfig(sma_period=10, band_pct=0.03, min_hold_bars=0,
                              max_gap_pct=0.20)
        v = Autopilot(cfg).decide(flat_then(10, [40.0]), 10, RISK_ON, 99)
        self.assertEqual(v.regime, RISK_OFF)
        self.assertIsNone(v.blocked_by)

    def test_stale_data_blocks_entry(self):
        cfg = AutopilotConfig(sma_period=10, band_pct=0.03, min_hold_bars=0,
                              max_bar_age_seconds=3600)
        bars = flat_then(10, [130.0])
        v = Autopilot(cfg).decide(bars, 10, RISK_OFF, 99, now=bars[-1].ts + 99_999)
        self.assertEqual(v.blocked_by, RAIL_STALE)

    def test_stale_data_does_not_block_exit(self):
        cfg = AutopilotConfig(sma_period=10, band_pct=0.03, min_hold_bars=0,
                              max_bar_age_seconds=3600)
        bars = flat_then(10, [70.0])
        v = Autopilot(cfg).decide(bars, 10, RISK_ON, 99, now=bars[-1].ts + 99_999)
        self.assertEqual(v.regime, RISK_OFF)
        self.assertIsNone(v.blocked_by)


class TestBookKeeping(unittest.TestCase):
    def test_going_long_spends_cash_and_pays_a_fee(self):
        cfg = AutopilotConfig(sma_period=10, band_pct=0.03, min_hold_bars=0,
                              fee_rate=0.001, slippage_rate=0.0,
                              starting_equity=10_000.0)
        r = Autopilot(cfg).backtest(flat_then(10, [130.0]))
        self.assertEqual(len(r.fills), 1)
        fill = r.fills[0]
        self.assertEqual(fill.side, "buy")
        self.assertAlmostEqual(fill.cost, fill.units * fill.price * 0.001)
        self.assertGreaterEqual(r.state.cash, -1e-9)

    def test_a_round_trip_returns_to_cash_minus_costs(self):
        cfg = AutopilotConfig(sma_period=10, band_pct=0.03, min_hold_bars=0,
                              fee_rate=0.001, slippage_rate=0.0,
                              starting_equity=10_000.0)
        # Up through the band, then back to the starting price and below it.
        r = Autopilot(cfg).backtest(flat_then(10, [130.0, 130.0, 60.0]))
        self.assertEqual([f.side for f in r.fills], ["buy", "sell"])
        self.assertTrue(r.state.is_flat)
        self.assertGreater(r.state.cash, 0.0)

    def test_equity_never_goes_negative_and_cash_never_does_either(self):
        cfg = AutopilotConfig(sma_period=10, band_pct=0.02, min_hold_bars=0,
                              fee_rate=0.005, slippage_rate=0.005)
        prices = [100.0] * 10
        for i in range(60):
            prices.append(100.0 * (1.5 if i % 2 else 0.6))
        r = Autopilot(cfg).backtest(bars_from(prices))
        self.assertGreaterEqual(min(r.equity_curve), 0.0)
        self.assertGreaterEqual(r.state.cash, -1e-9)

    def test_a_small_drift_is_not_worth_a_fee(self):
        cfg = AutopilotConfig(sma_period=10, band_pct=0.03, min_hold_bars=0,
                              rebalance_threshold=0.5)
        r = Autopilot(cfg).backtest(flat_then(10, [130.0, 131.0, 132.0, 133.0]))
        self.assertEqual(len(r.fills), 1)  # the entry, and nothing after it

    def test_partial_exposure_leaves_cash_behind(self):
        cfg = AutopilotConfig(sma_period=10, band_pct=0.03, min_hold_bars=0,
                              target_exposure=0.5, starting_equity=10_000.0)
        r = Autopilot(cfg).backtest(flat_then(10, [130.0]))
        held = r.state.units * 130.0
        self.assertAlmostEqual(held / r.state.equity(130.0), 0.5, places=2)


class TestHalt(unittest.TestCase):
    def _crashing(self):
        cfg = AutopilotConfig(sma_period=10, band_pct=0.01, min_hold_bars=0,
                              halt_drawdown_pct=0.10, starting_equity=10_000.0)
        # Rise enough to get in, then collapse inside a single bar so the
        # exit cannot outrun the loss.
        return cfg, Autopilot(cfg).backtest(
            flat_then(10, [130.0, 130.0, 20.0, 20.0, 200.0, 300.0])
        )

    def test_a_deep_drawdown_halts(self):
        _, r = self._crashing()
        self.assertTrue(r.state.halted)
        self.assertIn("below the peak", r.state.halt_reason)

    def test_a_halt_is_sticky_and_forces_flat(self):
        _, r = self._crashing()
        self.assertTrue(r.state.is_flat)
        self.assertEqual(r.verdicts[-1].blocked_by, RAIL_HALTED)
        self.assertEqual(r.verdicts[-1].target_exposure, 0.0)

    def test_a_halted_run_says_so_instead_of_reporting_a_truncated_number(self):
        """A halted run's headline return describes a bot that stopped
        trading, which is not the strategy's result."""
        _, r = self._crashing()
        self.assertGreater(r.bars_halted, 0)
        text = format_autopilot(r)
        self.assertIn("THIS RUN HALTED", text)
        self.assertIn("stopped trading", text)

    def test_an_unhalted_run_says_nothing_about_halting(self):
        r = Autopilot(
            AutopilotConfig(sma_period=10, band_pct=0.03, min_hold_bars=0)
        ).backtest(flat_then(10, [130.0] * 5))
        self.assertEqual(r.bars_halted, 0)
        self.assertNotIn("HALTED", format_autopilot(r))

    def test_the_default_threshold_tolerates_an_ordinary_trend_drawdown(self):
        """A 40% drawdown is normal for a trend filter riding one decline down
        to its exit. A halt that fires on that is worse than no halt."""
        self.assertGreater(AutopilotConfig().halt_drawdown_pct, 0.5)

    def test_clearing_a_halt_rebases_the_peak(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = AutopilotConfig(sma_period=10, halt_drawdown_pct=0.10)
            runner = AutopilotRunner(
                cfg, os.path.join(d, "s.json"), os.path.join(d, "j.jsonl")
            )
            runner.state.halted = True
            runner.state.halt_reason = "test"
            runner.state.cash = 500.0
            runner.state.peak_equity = 10_000.0
            runner.state.last_close = 1.0
            runner.clear_halt()
            self.assertFalse(runner.state.halted)
            self.assertAlmostEqual(runner.state.peak_equity, 500.0)


class TestHysteresisSuppressesChurn(unittest.TestCase):
    def _oscillating(self):
        prices = [100.0] * 30
        for i in range(200):
            prices.append(100.0 * (1.02 if i % 2 == 0 else 0.98))
        return bars_from(prices)

    def test_a_band_produces_fewer_flips_than_none(self):
        bars = self._oscillating()
        base = AutopilotConfig(sma_period=20, band_pct=0.0, min_hold_bars=0)
        wide = AutopilotConfig(sma_period=20, band_pct=0.05, min_hold_bars=0)
        self.assertLess(
            Autopilot(wide).backtest(bars).n_flips,
            Autopilot(base).backtest(bars).n_flips,
        )

    def test_a_wide_band_stops_the_churn_entirely(self):
        r = Autopilot(
            AutopilotConfig(sma_period=20, band_pct=0.10, min_hold_bars=0)
        ).backtest(self._oscillating())
        self.assertEqual(r.n_flips, 0)
        self.assertEqual(len(r.fills), 0)

    def test_min_hold_also_reduces_flips(self):
        bars = self._oscillating()
        base = AutopilotConfig(sma_period=20, band_pct=0.0, min_hold_bars=0)
        held = AutopilotConfig(sma_period=20, band_pct=0.0, min_hold_bars=20)
        self.assertLess(
            Autopilot(held).backtest(bars).n_flips,
            Autopilot(base).backtest(bars).n_flips,
        )


class TestResultMetrics(unittest.TestCase):
    def setUp(self):
        prices = [100.0] * 30 + [100.0 * math.exp(0.01 * i) for i in range(300)]
        self.r = Autopilot(
            AutopilotConfig(sma_period=20, band_pct=0.03, min_hold_bars=0)
        ).backtest(bars_from(prices))

    def test_it_participates_in_a_clean_uptrend(self):
        self.assertGreater(self.r.total_return, 0.0)
        self.assertGreater(self.r.time_in_market, 0.5)
        self.assertEqual(self.r.n_flips, 1)

    def test_drawdown_is_a_fraction_between_zero_and_one(self):
        self.assertGreaterEqual(self.r.max_drawdown, 0.0)
        self.assertLess(self.r.max_drawdown, 1.0)

    def test_years_and_cagr_agree_with_the_total_return(self):
        implied = (1 + self.r.total_return) ** (1 / self.r.years) - 1
        self.assertAlmostEqual(self.r.cagr, implied, places=6)

    def test_the_report_shows_both_sides_of_the_trade(self):
        text = format_autopilot(self.r)
        self.assertIn("buy and hold", text)
        self.assertIn("Max drawdown", text)
        self.assertIn("regime flips a year", text)

    def test_it_refuses_to_run_on_nothing(self):
        with self.assertRaises(ValueError):
            Autopilot().backtest([])


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.state = os.path.join(self.dir.name, "state.json")
        self.journal = os.path.join(self.dir.name, "journal.jsonl")
        self.cfg = AutopilotConfig(sma_period=10, band_pct=0.03, min_hold_bars=0)
        self.bars = flat_then(10, [130.0 + i for i in range(20)])

    def tearDown(self):
        self.dir.cleanup()

    def _runner(self):
        return AutopilotRunner(self.cfg, self.state, self.journal)

    def _now(self):
        return self.bars[-1].ts + 60.0

    def test_first_run_establishes_the_regime_without_trading_history(self):
        r = self._runner()
        out = r.update(self.bars, now=self._now())
        self.assertEqual(len(out), 1)  # one decision, not one per bar
        self.assertEqual(r.state.bars_seen, 1)
        self.assertTrue(any("first run" in w for w in r.warnings))
        self.assertEqual(r.state.regime, RISK_ON)
        self.assertFalse(r.state.is_flat)

    def test_the_same_bar_is_never_processed_twice(self):
        r = self._runner()
        r.update(self.bars, now=self._now())
        before = len(r.read_journal())
        self.assertEqual(r.update(self.bars, now=self._now()), [])
        self.assertEqual(len(r.read_journal()), before)

    def test_a_restart_resumes_mid_trend(self):
        first = self._runner()
        first.update(self.bars, now=self._now())
        resumed = self._runner()
        self.assertEqual(resumed.state, first.state)
        self.assertEqual(resumed.state.regime, RISK_ON)

    def test_a_restart_does_not_re_bootstrap(self):
        self._runner().update(self.bars, now=self._now())
        resumed = self._runner()
        resumed.update(self.bars, now=self._now())
        self.assertFalse(any("first run" in w for w in resumed.warnings))

    def test_new_bars_are_folded_in_one_at_a_time(self):
        r = self._runner()
        r.update(self.bars, now=self._now())
        more = self.bars + [a_bar(len(self.bars) + i, 20.0) for i in range(3)]
        out = r.update(more, now=more[-1].ts + 60.0)
        self.assertEqual(len(out), 3)
        self.assertEqual(r.state.regime, RISK_OFF)
        self.assertTrue(r.state.is_flat)

    def test_a_backlog_of_bars_warns(self):
        r = self._runner()
        r.update(self.bars, now=self._now())
        more = self.bars + [a_bar(len(self.bars) + i, 150.0) for i in range(5)]
        r.update(more, now=more[-1].ts + 60.0)
        self.assertTrue(any("arrived at once" in w for w in r.warnings))

    def test_too_little_history_is_refused(self):
        with self.assertRaises(ValueError):
            self._runner().update(self.bars[:5], now=self._now())

    def test_no_bars_is_a_no_op(self):
        self.assertEqual(self._runner().update([], now=0.0), [])

    def test_the_journal_records_every_decision_with_a_reason(self):
        r = self._runner()
        r.update(self.bars, now=self._now())
        entries = r.read_journal()
        self.assertTrue(entries)
        for e in entries:
            self.assertIn(e["regime"], (RISK_ON, RISK_OFF))
            self.assertTrue(e["reason"])
            self.assertIn("equity", e)

    def test_status_warns_when_the_feed_is_behind(self):
        r = self._runner()
        r.update(self.bars, now=self._now())
        fresh = r.status(now=self.bars[-1].ts + 60.0)
        self.assertNotIn("STALE", fresh)
        stale = r.status(now=self.bars[-1].ts + self.cfg.max_bar_age_seconds * 4)
        self.assertIn("STALE", stale)

    def test_status_never_claims_to_have_placed_an_order(self):
        self.assertIn("No orders are placed", self._runner().status(now=0.0))

    def test_live_and_backtest_paths_agree_on_the_regime(self):
        """One rule, two callers. If these diverge the design has failed."""
        r = self._runner()
        r.update(self.bars, now=self._now())
        for i in range(1, 6):
            more = self.bars + [a_bar(len(self.bars) + j, 200.0) for j in range(i)]
            r.update(more, now=more[-1].ts + 60.0)
        replay = Autopilot(self.cfg).backtest(
            self.bars + [a_bar(len(self.bars) + j, 200.0) for j in range(5)]
        )
        self.assertEqual(r.state.regime, replay.state.regime)


class TestPersistence(unittest.TestCase):
    def test_atomic_write_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "deep", "state.json")
            atomic_write_json(path, AutopilotState(cash=5.0))
            self.assertFalse(os.path.exists(path + ".tmp"))
            self.assertEqual(read_json_strict(path)["cash"], 5.0)

    def test_missing_file_reads_as_none_not_as_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(read_json_strict(os.path.join(d, "nope.json")))

    def test_corrupt_state_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{truncated")
            with self.assertRaises(RuntimeError):
                read_json_strict(path)

    def test_a_json_array_is_not_a_state_object(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("[]")
            with self.assertRaises(RuntimeError):
                read_json_strict(path)

    def test_load_dataclass_drops_fields_the_class_no_longer_has(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"cash": 3.0, "removed_in_v2": True}, fh)
            self.assertEqual(load_dataclass(path, AutopilotState).cash, 3.0)

    def test_journal_appends_and_reads_back_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "j.jsonl")
            self.assertEqual(read_jsonl(path), [])
            for i in range(3):
                append_jsonl(path, {"i": i})
            self.assertEqual([e["i"] for e in read_jsonl(path)], [0, 1, 2])

    def test_a_malformed_journal_line_raises_rather_than_being_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "j.jsonl")
            append_jsonl(path, {"i": 0})
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("{bad\n")
            with self.assertRaises(RuntimeError) as ctx:
                read_jsonl(path)
            self.assertIn("line 2", str(ctx.exception))


class TestSyntheticDailyBars(unittest.TestCase):
    def test_it_is_deterministic(self):
        from ..data import synthetic_daily_bars

        a = synthetic_daily_bars(n_bars=200, seed=5)
        b = synthetic_daily_bars(n_bars=200, seed=5)
        self.assertEqual([x.close for x in a], [x.close for x in b])

    def test_a_different_seed_gives_a_different_path(self):
        from ..data import synthetic_daily_bars

        a = synthetic_daily_bars(n_bars=200, seed=5)
        b = synthetic_daily_bars(n_bars=200, seed=6)
        self.assertNotEqual([x.close for x in a], [x.close for x in b])

    def test_bars_are_well_formed_and_ordered(self):
        from ..data import synthetic_daily_bars

        bars = synthetic_daily_bars(n_bars=500, seed=2)
        self.assertEqual(len(bars), 500)
        for i, b in enumerate(bars):
            self.assertGreater(b.close, 0.0)
            self.assertGreaterEqual(b.high, max(b.open, b.close))
            self.assertLessEqual(b.low, min(b.open, b.close))
            self.assertGreater(b.volume, 0.0)
            if i:
                self.assertGreater(b.ts, bars[i - 1].ts)

    def test_it_refuses_nonsense(self):
        from ..data import synthetic_daily_bars

        with self.assertRaises(ValueError):
            synthetic_daily_bars(n_bars=0)
        with self.assertRaises(ValueError):
            synthetic_daily_bars(start_price=0.0)


if __name__ == "__main__":
    unittest.main()
