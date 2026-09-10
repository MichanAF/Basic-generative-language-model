import io
import os
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout

from ..backtest import Backtester
from ..baselines import SIZE_FULL, _sma, buy_and_hold, trend_baseline
from ..cli import main
from ..config import StrategyConfig
from ..data import SyntheticConfig, bars_from_trades, synthetic_trades
from ..funding import DEFAULT_INTERVAL_HOURS, FundingSchedule
from ..metrics import summarize
from ..presets import btc_5m
from ..sources.binance import load_binance_funding
from .helpers import TICK, make_bar

HOUR = 3600.0
EIGHT_H = 8 * HOUR


def bars(n_ticks=200_000, seed=5, tf=300.0):
    sc = SyntheticConfig(n_ticks=n_ticks, tick_size=TICK, seed=seed)
    return bars_from_trades(synthetic_trades(sc), tf, TICK)


class TestFundingSchedule(unittest.TestCase):
    def setUp(self):
        self.s = FundingSchedule(times=[100.0, 200.0, 300.0], rates=[0.001, 0.002, -0.003])

    def test_window_is_half_open_so_settlements_are_charged_once(self):
        # (100, 300] picks up 200 and 300, not the 100 already settled.
        self.assertAlmostEqual(self.s.accrued_rate(100.0, 300.0), 0.002 - 0.003)
        self.assertAlmostEqual(self.s.accrued_rate(0.0, 100.0), 0.001)
        # Consecutive windows must sum to the whole, with nothing double counted.
        a = self.s.accrued_rate(0.0, 150.0)
        b = self.s.accrued_rate(150.0, 400.0)
        self.assertAlmostEqual(a + b, sum(self.s.rates))

    def test_empty_and_reversed_windows_are_zero(self):
        self.assertEqual(self.s.accrued_rate(300.0, 100.0), 0.0)
        self.assertEqual(FundingSchedule().accrued_rate(0.0, 1e9), 0.0)

    def test_longs_pay_and_shorts_receive_when_the_rate_is_positive(self):
        long_flow = self.s.cash_flow(1, 10_000.0, 0.0, 100.0)
        short_flow = self.s.cash_flow(-1, 10_000.0, 0.0, 100.0)
        self.assertAlmostEqual(long_flow, -10.0)   # 0.001 * 10k paid
        self.assertAlmostEqual(short_flow, +10.0)
        self.assertAlmostEqual(long_flow, -short_flow)

    def test_negative_rates_flip_who_pays(self):
        s = FundingSchedule(times=[50.0], rates=[-0.001])
        self.assertGreater(s.cash_flow(1, 10_000.0, 0.0, 60.0), 0, "long receives")
        self.assertLess(s.cash_flow(-1, 10_000.0, 0.0, 60.0), 0, "short pays")

    def test_unsorted_input_is_ordered(self):
        s = FundingSchedule(times=[300.0, 100.0, 200.0], rates=[3.0, 1.0, 2.0])
        self.assertEqual(s.times, [100.0, 200.0, 300.0])
        self.assertEqual(s.rates, [1.0, 2.0, 3.0])

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            FundingSchedule(times=[1.0, 2.0], rates=[0.1])

    def test_constant_schedule_spans_the_window(self):
        s = FundingSchedule.constant(0.0, 3 * EIGHT_H, rate=0.0001)
        self.assertEqual(len(s), 4)
        self.assertAlmostEqual(s.accrued_rate(0.0, 3 * EIGHT_H), 0.0003)

    def test_annualisation_matches_the_hand_calculation(self):
        s = FundingSchedule.constant(0.0, 10 * EIGHT_H, rate=0.0001)
        expected = 0.0001 * 365 * 24 / DEFAULT_INTERVAL_HOURS  # ~10.95%
        self.assertAlmostEqual(s.annualised(), expected, places=6)
        self.assertIn("annualised", s.summary())


class TestFundingLoader(unittest.TestCase):
    ROWS = [
        "1690848000000,8,0.00010000",
        "1690876800000,8,0.00025000",
        "1690905600000,8,-0.00005000",
    ]

    def test_parses_rates_and_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(self.ROWS) + "\n")
            s = load_binance_funding(path)
        self.assertEqual(len(s), 3)
        self.assertAlmostEqual(s.times[0], 1690848000.0)
        self.assertAlmostEqual(s.rates[1], 0.00025)
        self.assertLess(s.rates[2], 0)

    def test_reads_the_published_zip_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "BTCUSDT-fundingRate-2023-08.zip")
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("BTCUSDT-fundingRate-2023-08.csv", "\n".join(self.ROWS) + "\n")
            self.assertEqual(len(load_binance_funding(path)), 3)

    def test_header_row_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("calc_time,funding_interval_hours,last_funding_rate\n")
                fh.write("\n".join(self.ROWS) + "\n")
            self.assertEqual(len(load_binance_funding(path)), 3)


class TestFundingInTheBacktest(unittest.TestCase):
    def _cfg(self, **kw):
        base = dict(tick_size=10.0, tick_value=10.0, whole_units=False,
                    slippage_ticks=0.0, commission_per_side=0.0,
                    partial_at_r=None, time_stop_bars=None, trail_atr=None)
        base.update(kw)
        return StrategyConfig(**base)

    def _position(self, bt, cfg, direction=1):
        from ..backtest import _Position
        from ..levels import RESISTANCE, Level
        from ..signals import Signal

        lvl = Level(price=100.0, side=RESISTANCE, strength=5.0,
                    created_index=0, last_update_index=0)
        sig = Signal(index=0, ts=0.0, direction=direction, level=lvl, c1_index=0,
                     c2_index=0, entry_price=100.0, stop=98.0 if direction > 0 else 102.0,
                     target=110.0, risk_per_unit=2.0, atr=1.0, entry_mode="close_of_2")
        return _Position(
            signal=sig, direction=direction, entry_index=0, entry_price=100.0,
            quantity=10.0, initial_quantity=10.0, risk_per_unit=2.0,
            stop=98.0 if direction > 0 else 102.0, target=110.0, level=lvl,
            best_price=100.0, worst_price=100.0, last_funding_ts=0.0,
        )

    def test_a_long_pays_funding_and_a_short_receives_it(self):
        cfg = self._cfg()
        sched = FundingSchedule(times=[EIGHT_H], rates=[0.001])
        bar = make_bar(ts=2 * EIGHT_H, o=100.0, h=100.5, l=99.5, c=100.0, tick_size=10.0)

        bt = Backtester(cfg)
        bt.funding = sched
        long_pos = self._position(bt, cfg, direction=1)
        flow_long = bt._accrue_funding(long_pos, bar)

        bt2 = Backtester(cfg)
        bt2.funding = sched
        flow_short = bt2._accrue_funding(self._position(bt2, cfg, direction=-1), bar)

        # notional = 10 units * 100 * point_value 1 = 1000; 0.1% of that = 1.0
        self.assertAlmostEqual(flow_long, -1.0, places=6)
        self.assertAlmostEqual(flow_short, +1.0, places=6)
        self.assertAlmostEqual(long_pos.funding_cash, -1.0, places=6)
        self.assertAlmostEqual(long_pos.cost_cash, 1.0, places=6, msg="payment is a cost")

    def test_no_schedule_means_no_funding(self):
        bt = Backtester(self._cfg())
        pos = self._position(bt, self._cfg())
        bar = make_bar(ts=EIGHT_H, o=100.0, h=100.5, l=99.5, c=100.0, tick_size=10.0)
        self.assertEqual(bt._accrue_funding(pos, bar), 0.0)
        self.assertEqual(pos.funding_cash, 0.0)

    def test_settlements_are_not_charged_twice_across_bars(self):
        cfg = self._cfg()
        bt = Backtester(cfg)
        bt.funding = FundingSchedule(times=[EIGHT_H], rates=[0.001])
        pos = self._position(bt, cfg)
        b1 = make_bar(ts=2 * EIGHT_H, o=100.0, h=100.5, l=99.5, c=100.0, tick_size=10.0)
        b2 = make_bar(ts=3 * EIGHT_H, o=100.0, h=100.5, l=99.5, c=100.0, tick_size=10.0)
        first = bt._accrue_funding(pos, b1)
        second = bt._accrue_funding(pos, b2)
        self.assertAlmostEqual(first, -1.0, places=6)
        self.assertEqual(second, 0.0, "the same settlement must not be charged again")

    def test_funding_shows_up_in_the_run_and_the_report(self):
        series = bars()
        cfg = btc_5m(starting_equity=100_000.0)
        span = FundingSchedule.constant(series[0].ts, series[-1].ts, rate=0.0005)
        with_f = summarize(Backtester(cfg).run(series, funding=span))
        without = summarize(Backtester(cfg).run(series))
        if with_f.n_trades:
            self.assertNotEqual(with_f.total_funding, 0.0)
            self.assertEqual(without.total_funding, 0.0)


class TestTrendBaseline(unittest.TestCase):
    def test_sma_is_causal_and_none_until_the_window_fills(self):
        vals = _sma([1.0, 2.0, 3.0, 4.0], 3)
        self.assertIsNone(vals[0])
        self.assertIsNone(vals[1])
        self.assertAlmostEqual(vals[2], 2.0)
        self.assertAlmostEqual(vals[3], 3.0)

    def test_goes_long_above_the_average_and_flat_below(self):
        cfg = StrategyConfig(tick_size=TICK, slippage_ticks=0.0, commission_per_side=0.0)
        rising = [make_bar(ts=i * 300, o=100 + i, h=101 + i, l=99 + i, c=100 + i)
                  for i in range(30)]
        falling = [make_bar(ts=(30 + i) * 300, o=130 - 3 * i, h=131 - 3 * i,
                            l=129 - 3 * i, c=130 - 3 * i) for i in range(20)]
        result = trend_baseline(rising + falling, cfg, sma_period=10)
        self.assertTrue(result.trades)
        for t in result.trades:
            self.assertEqual(t.direction, 1, "the baseline is long-only")
        self.assertIn(result.trades[0].exit_reason, ("trend_exit", "end_of_data"))

    def test_risk_is_floored_so_r_multiples_stay_sane(self):
        """A cross sits *at* the average; unfloored that gives near-zero risk."""
        cfg = StrategyConfig(tick_size=TICK)
        result = trend_baseline(bars(), cfg, sma_period=50)
        for t in result.trades:
            self.assertLess(abs(t.r_multiple), 50.0,
                            f"implausible R multiple {t.r_multiple}")
            self.assertLess(t.cost_r, 5.0)

    def test_equity_curve_has_one_point_per_bar(self):
        series = bars()
        result = trend_baseline(series, StrategyConfig(tick_size=TICK), sma_period=50)
        self.assertEqual(len(result.equity_curve), len(series))

    def test_leverage_cap_binds_risk_sizing(self):
        """A 0.5% risk budget behind a tight stop implies leverage silently."""
        series = bars()
        capped = btc_5m()                      # max_leverage 1.0
        loose = btc_5m(max_leverage=None)      # uncapped
        self.assertEqual(capped.max_leverage, 1.0)

        a = trend_baseline(series, capped, sma_period=50)
        b = trend_baseline(series, loose, sma_period=50)
        if a.trades and b.trades:
            self.assertLess(a.trades[0].quantity, b.trades[0].quantity,
                            "the cap must actually bind")
            notional = a.trades[0].quantity * a.trades[0].entry_price
            self.assertLessEqual(notional, capped.starting_equity * 1.01,
                                 "capped notional must not exceed equity")

    def test_full_size_mode_uses_the_whole_stake(self):
        series = bars()
        cfg = btc_5m()
        full = trend_baseline(series, cfg, sma_period=50, size_mode=SIZE_FULL)
        if full.trades:
            notional = full.trades[0].quantity * full.trades[0].entry_price
            self.assertAlmostEqual(notional / cfg.starting_equity, 1.0, places=1)

    def test_negative_leverage_cap_rejected(self):
        with self.assertRaises(ValueError):
            StrategyConfig(max_leverage=0.0)

    def test_funding_reduces_a_long_only_baseline(self):
        series = bars()
        cfg = btc_5m()
        span = FundingSchedule.constant(series[0].ts, series[-1].ts, rate=0.001)
        plain = summarize(trend_baseline(series, cfg, sma_period=50))
        taxed = summarize(trend_baseline(series, cfg, sma_period=50, funding=span))
        if plain.n_trades:
            self.assertLess(taxed.total_pnl, plain.total_pnl,
                            "paying funding on a long must reduce P&L")

    def test_rejects_bad_arguments(self):
        cfg = StrategyConfig(tick_size=TICK)
        with self.assertRaises(ValueError):
            trend_baseline(bars(), cfg, sma_period=1)
        with self.assertRaises(ValueError):
            trend_baseline(bars(), cfg, size_mode="yolo")


class TestBuyAndHold(unittest.TestCase):
    def test_one_trade_spanning_the_data(self):
        series = bars()
        result = buy_and_hold(series, btc_5m())
        self.assertEqual(len(result.trades), 1)
        t = result.trades[0]
        self.assertEqual(t.entry_index, 0)
        self.assertEqual(t.exit_index, len(series) - 1)
        self.assertEqual(len(result.equity_curve), len(series))

    def test_tracks_the_underlying_direction(self):
        up = [make_bar(ts=i * 300, o=100 + i, h=101 + i, l=99 + i, c=100 + i)
              for i in range(50)]
        cfg = StrategyConfig(tick_size=TICK, slippage_ticks=0.0,
                             commission_per_side=0.0, whole_units=False)
        self.assertGreater(buy_and_hold(up, cfg).trades[0].pnl, 0)

    def test_needs_at_least_two_bars(self):
        with self.assertRaises(ValueError):
            buy_and_hold(bars()[:1], btc_5m())


class TestCompareCommand(unittest.TestCase):
    def test_compare_runs_and_lists_all_three(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["compare", "--synthetic", "--ticks", "200000", "--sma", "50"])
        out = buf.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Comparison", out)
        self.assertIn("order flow", out)
        self.assertIn("trend SMA50", out)
        self.assertIn("buy and hold", out)

    def test_warns_when_the_history_is_too_short_for_the_average(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["compare", "--synthetic", "--ticks", "60000", "--sma", "400"])
        self.assertIn("WARNING", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
