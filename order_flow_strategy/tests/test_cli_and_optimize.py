import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from ..cli import config_from_args, main
from ..config import ENTRY_BREAK_OF_2, REGIME_WITH_TREND_ONLY, StrategyConfig
from ..data import SyntheticConfig, bars_from_trades, load_bars_csv, load_trades_csv, synthetic_trades, write_bars_csv
from ..optimize import DEFAULT_GRID, FAST_GRID, expand, format_recommendation, walk_forward
from .helpers import TICK


def small_bars(n_ticks=60_000, seed=5):
    sc = SyntheticConfig(n_ticks=n_ticks, tick_size=TICK, seed=seed)
    return bars_from_trades(synthetic_trades(sc), 300.0, TICK)


class TestConfigValidation(unittest.TestCase):
    def test_rejects_unknown_entry_mode(self):
        with self.assertRaises(ValueError):
            StrategyConfig(entry_mode="on_a_hunch")

    def test_rejects_partial_beyond_target(self):
        with self.assertRaises(ValueError):
            StrategyConfig(partial_at_r=3.0, target_r=2.0)

    def test_rejects_trading_neither_direction(self):
        with self.assertRaises(ValueError):
            StrategyConfig(trade_shorts=False, trade_longs=False)

    def test_rejects_absurd_risk(self):
        with self.assertRaises(ValueError):
            StrategyConfig(risk_per_trade=1.5)

    def test_with_copies_and_overrides(self):
        base = StrategyConfig()
        changed = base.with_(target_r=5.0)
        self.assertEqual(base.target_r, 2.0)
        self.assertEqual(changed.target_r, 5.0)


class TestGrid(unittest.TestCase):
    def test_expand_produces_the_cartesian_product(self):
        combos = expand({"a": (1, 2), "b": ("x", "y", "z")})
        self.assertEqual(len(combos), 6)
        self.assertIn({"a": 1, "b": "z"}, combos)

    def test_default_grid_is_a_manageable_size(self):
        self.assertLessEqual(len(expand(DEFAULT_GRID)), 200)
        self.assertLess(len(expand(FAST_GRID)), len(expand(DEFAULT_GRID)))


class TestWalkForward(unittest.TestCase):
    def test_refuses_when_there_is_not_enough_data(self):
        with self.assertRaises(ValueError):
            walk_forward(small_bars()[:80], StrategyConfig(tick_size=TICK), folds=4)

    def test_produces_recommendations_and_never_trains_on_test_data(self):
        bars = small_bars(n_ticks=200_000, seed=9)
        rec = walk_forward(
            bars, StrategyConfig(tick_size=TICK), grid=FAST_GRID, folds=2, min_trades=1
        )
        self.assertTrue(rec.robust_params)
        for key in FAST_GRID:
            self.assertIn(key, rec.robust_params)
            self.assertIn(rec.robust_params[key], FAST_GRID[key])
        for fold in rec.folds:
            self.assertLessEqual(
                fold.train_span[1], fold.test_span[0], "training must precede testing"
            )

    def test_report_renders(self):
        bars = small_bars(n_ticks=200_000, seed=9)
        rec = walk_forward(
            bars, StrategyConfig(tick_size=TICK), grid=FAST_GRID, folds=2, min_trades=1
        )
        text = format_recommendation(rec, StrategyConfig(tick_size=TICK))
        self.assertIn("Parameter plateaus", text)
        self.assertIn("Per-fold selection", text)
        for key in FAST_GRID:
            self.assertIn(key, text)

    def _recommendation(self, test_score: float):
        from ..metrics import PerformanceStats
        from ..optimize import FoldOutcome, Recommendation

        fold = FoldOutcome(
            fold=0,
            train_span=(0, 100),
            test_span=(100, 200),
            params={"target_r": 2.0},
            train_score=1.0,
            test_score=test_score,
            test_stats=PerformanceStats(n_trades=40, t_stat=3.0),
        )
        return Recommendation(
            best_params={"target_r": 2.0},
            robust_params={"target_r": 2.0},
            folds=[fold],
            param_sensitivity={"target_r": [(2.0, test_score, 1)]},
            oos_stats=None,
        )

    def test_a_positive_result_is_presented_as_a_recommendation(self):
        text = format_recommendation(self._recommendation(1.5), StrategyConfig(tick_size=TICK))
        self.assertIn("Recommended configuration", text)
        self.assertNotIn("NOT a recommendation", text)

    def test_a_negative_result_refuses_to_endorse(self):
        text = format_recommendation(self._recommendation(-1.5), StrategyConfig(tick_size=TICK))
        self.assertIn("NOT a recommendation", text)
        self.assertNotIn("Recommended configuration", text)


class TestCsvRoundTrip(unittest.TestCase):
    def test_bars_survive_a_write_and_read(self):
        bars = small_bars()[:40]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bars.csv")
            write_bars_csv(path, bars)
            reloaded = load_bars_csv(path, TICK)
        self.assertEqual(len(reloaded), len(bars))
        for original, copy in zip(bars, reloaded):
            self.assertAlmostEqual(original.close, copy.close, places=6)
            self.assertAlmostEqual(original.volume, copy.volume, places=3)
            # The aggressor split is preserved even though price distribution is not.
            self.assertAlmostEqual(
                original.normalized_delta, copy.normalized_delta, places=3
            )
            self.assertTrue(copy.footprint.is_proxy)

    def test_trade_csv_parses_side_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ticks.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("timestamp,price,qty,aggressor\n")
                fh.write("1,100.0,5,buy\n2,100.25,3,SELL\n3,100.5,1,-1\n")
            trades = load_trades_csv(path)
        self.assertEqual([t.aggressor for t in trades], [1, -1, -1])
        self.assertEqual(trades[0].size, 5)

    def test_missing_columns_are_reported_with_a_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("ts,price\n1,100.0\n")
            with self.assertRaises(ValueError) as ctx:
                load_trades_csv(path)
        self.assertIn(":2", str(ctx.exception))


class TestCliArgs(unittest.TestCase):
    class Args:
        def __init__(self, **kw):
            defaults = dict(
                tick_size=0.25, tick_value=12.5, entry_mode=None, regime_filter=None,
                target_r=None, stop_buffer_atr=None, risk_per_trade=None,
                starting_equity=None, slippage_ticks=None, commission_per_side=None,
                wick_frac=None, c2_close_beyond_c1_frac=None, time_stop_bars=None,
                partial_at_r=None, trail_atr=None, no_longs=False, no_shorts=False,
                allow_loose_flow=False, allow_chart_levels=False,
                session_start_min=None, session_end_min=None,
            )
            defaults.update(kw)
            self.__dict__.update(defaults)

    def test_defaults_survive_when_nothing_is_passed(self):
        built = config_from_args(self.Args())
        self.assertEqual(built.target_r, StrategyConfig().target_r)

    def test_zero_disables_partials_and_trailing(self):
        built = config_from_args(self.Args(partial_at_r=0.0, trail_atr=0.0))
        self.assertIsNone(built.partial_at_r)
        self.assertIsNone(built.trail_atr)

    def test_direction_and_flow_switches(self):
        built = config_from_args(self.Args(no_longs=True, allow_loose_flow=True))
        self.assertFalse(built.trade_longs)
        self.assertTrue(built.trade_shorts)
        self.assertFalse(built.require_flow_evidence)
        self.assertTrue(built.require_flow_backed_levels, "unrelated switch must not move")

    def test_chart_levels_switch(self):
        self.assertTrue(config_from_args(self.Args()).require_flow_backed_levels)
        self.assertFalse(
            config_from_args(self.Args(allow_chart_levels=True)).require_flow_backed_levels
        )

    def test_session_window_is_passed_through(self):
        built = config_from_args(self.Args(session_start_min=810, session_end_min=1230))
        self.assertEqual(built.session_start_min, 810)
        self.assertEqual(built.session_end_min, 1230)

    def test_overrides_are_applied(self):
        built = config_from_args(
            self.Args(entry_mode=ENTRY_BREAK_OF_2, regime_filter=REGIME_WITH_TREND_ONLY, target_r=4.0)
        )
        self.assertEqual(built.entry_mode, ENTRY_BREAK_OF_2)
        self.assertEqual(built.regime_filter, REGIME_WITH_TREND_ONLY)
        self.assertEqual(built.target_r, 4.0)


class TestCliCommands(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_demo(self):
        code, out = self._run(["demo", "--ticks", "50000"])
        self.assertEqual(code, 0)
        self.assertIn("Demo backtest", out)
        self.assertIn("Active order-flow levels", out)
        self.assertIn("not financial advice", out)

    def test_backtest_on_synthetic_data(self):
        code, out = self._run(["backtest", "--synthetic", "--ticks", "50000"])
        self.assertEqual(code, 0)
        self.assertIn("Backtest over", out)

    def test_scan_reports_levels(self):
        code, out = self._run(["scan", "--synthetic", "--ticks", "50000"])
        self.assertEqual(code, 0)
        self.assertIn("Active order-flow levels", out)

    def test_optimize_runs_a_small_grid(self):
        code, out = self._run(
            ["optimize", "--synthetic", "--ticks", "200000", "--fast", "--folds", "2",
             "--min-trades", "1"]
        )
        self.assertEqual(code, 0)
        self.assertIn("Walk-forward recommendation", out)

    def test_backtest_exports_trades(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trades.csv")
            code, out = self._run(
                ["backtest", "--synthetic", "--ticks", "80000", "--export-trades", path]
            )
            self.assertEqual(code, 0)
            with open(path, encoding="utf-8") as fh:
                header = fh.readline()
        self.assertIn("r_multiple", header)

    def test_paper_creates_a_book_and_is_safe_to_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = os.path.join(tmp, "paper", "state.json")
            journal = os.path.join(tmp, "paper", "journal.jsonl")
            argv = [
                "paper", "--synthetic", "--ticks", "60000",
                "--state", state, "--journal", journal,
            ]
            code, out = self._run(argv)
            self.assertEqual(code, 0)
            self.assertIn("Paper trading status", out)
            self.assertIn("No orders are placed", out)
            self.assertTrue(os.path.exists(state))

            # A scheduler that fires twice on the same data must not act twice.
            with open(journal, encoding="utf-8") as fh:
                first = fh.read()
            code, out = self._run(argv)
            self.assertEqual(code, 0)
            self.assertIn("Nothing new", out)
            with open(journal, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), first)

    def test_paper_status_only_does_not_process_bars(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = os.path.join(tmp, "state.json")
            code, out = self._run(
                ["paper", "--synthetic", "--ticks", "60000", "--state", state,
                 "--journal", os.path.join(tmp, "j.jsonl"), "--status-only"]
            )
            self.assertEqual(code, 0)
            self.assertIn("none yet", out)
            self.assertFalse(os.path.exists(state))

    def test_requires_a_data_source(self):
        with self.assertRaises(SystemExit):
            self._run(["backtest"])

    def test_rejects_a_series_that_is_too_short(self):
        with self.assertRaises(SystemExit):
            self._run(["backtest", "--synthetic", "--ticks", "300"])


class TestProxyWarning(unittest.TestCase):
    def test_ohlcv_input_is_flagged_as_proxy(self):
        from ..backtest import Backtester
        from ..metrics import summarize
        from ..report import warnings_for

        bars = small_bars()[:120]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bars.csv")
            write_bars_csv(path, bars)
            reloaded = load_bars_csv(path, TICK)

        stats = summarize(Backtester(StrategyConfig(tick_size=TICK)).run(reloaded))
        self.assertTrue(stats.used_proxy_footprints)
        self.assertTrue(any("estimated from OHLCV" in w for w in warnings_for(stats)))


if __name__ == "__main__":
    unittest.main()
