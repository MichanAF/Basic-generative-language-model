import gzip
import os
import tempfile
import unittest
import zipfile

from ..backtest import Backtester
from ..config import StrategyConfig
from ..data import bars_from_trades
from ..presets import btc_5m, build, es_5m
from ..sources.binance import (
    _to_seconds,
    load_binance_agg_trades,
    load_binance_klines,
    load_binance_paths,
)
from ..sources.fetch import build_url, month_range
from .helpers import make_bar

# One published kline row: open_time, o, h, l, c, volume, close_time,
# quote_vol, trades, taker_buy_base, taker_buy_quote, ignore
KLINE_ROWS_MS = [
    "1690848000000,29000.00,29150.00,28950.00,29100.00,120.5,1690848299999,3500000,900,90.0,2600000,0",
    "1690848300000,29100.00,29200.00,29050.00,29060.00,100.0,1690848599999,2900000,800,25.0,700000,0",
]

# Same first row with the microsecond epochs Binance moved to in 2025.
KLINE_ROW_US = (
    "1690848000000000,29000.00,29150.00,28950.00,29100.00,120.5,"
    "1690848299999999,3500000,900,90.0,2600000,0"
)

KLINE_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)

# agg_trade_id, price, quantity, first_id, last_id, transact_time, is_buyer_maker, is_best_match
AGG_ROWS = [
    "1,29000.00,1.5,10,11,1690848000000,true,true",
    "2,29010.00,2.0,12,13,1690848001000,false,true",
    "3,29005.00,0.5,14,15,1690848002000,True,true",
]


def write_zip(path: str, name: str, lines):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name, "\n".join(lines) + "\n")


class TestTimestampScales(unittest.TestCase):
    def test_scales_are_detected_by_magnitude(self):
        self.assertAlmostEqual(_to_seconds(1690848000.0), 1690848000.0)
        self.assertAlmostEqual(_to_seconds(1690848000000.0), 1690848000.0)
        self.assertAlmostEqual(_to_seconds(1690848000000000.0), 1690848000.0)
        self.assertAlmostEqual(_to_seconds(1690848000000000000.0), 1690848000.0)


class TestKlines(unittest.TestCase):
    def test_taker_buy_volume_becomes_real_delta(self):
        """90 of 120.5 bought by takers -> delta = 90 - 30.5, not a guess."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "k.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(KLINE_ROWS_MS) + "\n")
            bars = load_binance_klines(path, tick_size=10.0)

        self.assertEqual(len(bars), 2)
        first = bars[0]
        self.assertAlmostEqual(first.open, 29000.0)
        self.assertAlmostEqual(first.high, 29150.0)
        self.assertAlmostEqual(first.volume, 120.5)
        self.assertAlmostEqual(first.delta, 90.0 - 30.5, places=4)
        self.assertGreater(first.normalized_delta, 0)
        # The second bar was sold into despite closing mid-range.
        self.assertLess(bars[1].delta, 0)

    def test_delta_is_independent_of_close_location(self):
        """The whole point: measured flow, not inferred from the close."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "k.csv")
            # Closes at the very high, but takers were overwhelmingly sellers.
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    "1690848000000,29000,29150,28950,29150,100.0,1690848299999,0,0,10.0,0,0\n"
                )
            bars = load_binance_klines(path, tick_size=10.0)
        self.assertLess(bars[0].delta, 0, "close at the high must not force positive delta")
        self.assertAlmostEqual(bars[0].delta, 10.0 - 90.0, places=4)

    def test_header_row_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "k.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(KLINE_HEADER + "\n" + "\n".join(KLINE_ROWS_MS) + "\n")
            self.assertEqual(len(load_binance_klines(path, 10.0)), 2)

    def test_microsecond_epochs_land_in_the_same_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            ms = os.path.join(tmp, "ms.csv")
            us = os.path.join(tmp, "us.csv")
            with open(ms, "w", encoding="utf-8") as fh:
                fh.write(KLINE_ROWS_MS[0] + "\n")
            with open(us, "w", encoding="utf-8") as fh:
                fh.write(KLINE_ROW_US + "\n")
            self.assertAlmostEqual(
                load_binance_klines(ms, 10.0)[0].ts, load_binance_klines(us, 10.0)[0].ts
            )

    def test_reads_zip_and_gzip(self):
        with tempfile.TemporaryDirectory() as tmp:
            z = os.path.join(tmp, "BTCUSDT-5m-2023-08.zip")
            write_zip(z, "BTCUSDT-5m-2023-08.csv", KLINE_ROWS_MS)
            g = os.path.join(tmp, "k.csv.gz")
            with gzip.open(g, "wt", encoding="utf-8") as fh:
                fh.write("\n".join(KLINE_ROWS_MS) + "\n")
            self.assertEqual(len(load_binance_klines(z, 10.0)), 2)
            self.assertEqual(len(load_binance_klines(g, 10.0)), 2)

    def test_row_size_controls_footprint_resolution(self):
        """The reason tick_size must be a row height, not the exchange tick."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "k.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(KLINE_ROWS_MS[0] + "\n")  # 200 dollar range
            coarse = load_binance_klines(path, tick_size=10.0)[0]
            fine = load_binance_klines(path, tick_size=0.01)[0]
        self.assertLessEqual(len(coarse.footprint.levels), 25)
        self.assertGreater(len(fine.footprint.levels), 100)
        # Volume is conserved regardless of resolution.
        self.assertAlmostEqual(coarse.footprint.total_volume, 120.5, places=4)
        self.assertAlmostEqual(fine.footprint.total_volume, 120.5, places=4)

    def test_malformed_row_names_the_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "k.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(KLINE_ROWS_MS[0] + "\n")
                fh.write("1690848300000,oops,29200,29050,29060,100,0,0,0,25,0,0\n")
            with self.assertRaises(ValueError) as ctx:
                load_binance_klines(path, 10.0)
        self.assertIn(":2", str(ctx.exception))


class TestAggTrades(unittest.TestCase):
    def test_buyer_maker_flag_maps_to_the_aggressor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(AGG_ROWS) + "\n")
            trades = load_binance_agg_trades(path)

        # is_buyer_maker=true -> the seller crossed the spread -> aggressor -1
        self.assertEqual([t.aggressor for t in trades], [-1, 1, -1])
        self.assertAlmostEqual(trades[0].price, 29000.0)
        self.assertAlmostEqual(trades[0].size, 1.5)
        self.assertAlmostEqual(trades[0].ts, 1690848000.0)

    def test_tape_builds_true_footprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(AGG_ROWS) + "\n")
            bars = bars_from_trades(load_binance_agg_trades(path), 300.0, 10.0)

        self.assertEqual(len(bars), 1)
        self.assertFalse(bars[0].footprint.is_proxy, "real tape must not be flagged proxy")
        self.assertAlmostEqual(bars[0].delta, 2.0 - 1.5 - 0.5, places=6)

    def test_trades_are_sorted_by_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("2,29010.00,2.0,12,13,1690848005000,false,true\n")
                fh.write("1,29000.00,1.5,10,11,1690848000000,true,true\n")
            trades = load_binance_agg_trades(path)
        self.assertEqual([t.ts for t in trades], sorted(t.ts for t in trades))


class TestMultiFileLoading(unittest.TestCase):
    def test_directory_of_archives_concatenates_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_zip(
                os.path.join(tmp, "BTCUSDT-5m-2023-09.zip"), "b.csv",
                ["1693526400000,29500,29600,29400,29550,50,0,0,0,25,0,0"],
            )
            write_zip(
                os.path.join(tmp, "BTCUSDT-5m-2023-08.zip"), "a.csv", KLINE_ROWS_MS
            )
            bars = load_binance_paths([tmp], tick_size=10.0, kind="klines")
        self.assertEqual(len(bars), 3)
        self.assertEqual([b.ts for b in bars], sorted(b.ts for b in bars))

    def test_overlapping_archives_are_deduped(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_zip(os.path.join(tmp, "a.zip"), "a.csv", KLINE_ROWS_MS)
            write_zip(os.path.join(tmp, "b.zip"), "b.csv", KLINE_ROWS_MS)
            bars = load_binance_paths([tmp], tick_size=10.0, kind="klines")
        self.assertEqual(len(bars), 2, "duplicate timestamps must collapse")

    def test_unknown_kind_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_zip(os.path.join(tmp, "a.zip"), "a.csv", KLINE_ROWS_MS)
            with self.assertRaises(ValueError):
                load_binance_paths([tmp], 10.0, kind="orderbook")

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            load_binance_paths([], 10.0)


class TestFetchPlumbing(unittest.TestCase):
    """The download itself needs network; its URL and date maths do not."""

    def test_month_range_is_inclusive_and_crosses_years(self):
        self.assertEqual(
            month_range("2023-11", "2024-02"),
            [(2023, 11), (2023, 12), (2024, 1), (2024, 2)],
        )

    def test_two_years_is_twenty_four_months(self):
        self.assertEqual(len(month_range("2023-08", "2025-07")), 24)

    def test_backwards_range_is_rejected(self):
        with self.assertRaises(ValueError):
            month_range("2024-05", "2024-01")

    def test_bad_format_is_rejected(self):
        with self.assertRaises(ValueError):
            month_range("August 2024", "2024-09")

    def test_kline_url_matches_the_published_layout(self):
        self.assertEqual(
            build_url("spot", "klines", "BTCUSDT", "5m", 2024, 3),
            "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/"
            "BTCUSDT-5m-2024-03.zip",
        )

    def test_a_policy_denial_explains_itself_instead_of_retrying(self):
        """403 means an egress policy said no; hammering it will not help."""
        import urllib.error

        from ..sources import fetch as fetch_mod

        calls = []

        def deny(url, timeout=0):
            calls.append(url)
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

        original = fetch_mod.urllib.request.urlopen
        fetch_mod.urllib.request.urlopen = deny
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(SystemExit) as ctx:
                    fetch_mod.download("https://data.binance.vision/x.zip",
                                       os.path.join(tmp, "x.zip"))
        finally:
            fetch_mod.urllib.request.urlopen = original

        self.assertEqual(len(calls), 1, "a policy denial must not be retried")
        self.assertIn("network policy", str(ctx.exception))

    def test_a_missing_month_is_not_an_error(self):
        import urllib.error

        from ..sources import fetch as fetch_mod

        def missing(url, timeout=0):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        original = fetch_mod.urllib.request.urlopen
        fetch_mod.urllib.request.urlopen = missing
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ok = fetch_mod.download("https://x/y.zip", os.path.join(tmp, "y.zip"))
        finally:
            fetch_mod.urllib.request.urlopen = original
        self.assertFalse(ok)

    def test_bad_dates_exit_cleanly(self):
        from ..sources.fetch import main

        with self.assertRaises(SystemExit):
            main(["--start", "2024-05", "--end", "2024-01", "--out", "/tmp/nope"])

    def test_aggtrades_url_omits_the_interval(self):
        self.assertEqual(
            build_url("spot", "aggTrades", "BTCUSDT", None, 2024, 12),
            "https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/"
            "BTCUSDT-aggTrades-2024-12.zip",
        )


class TestPresets(unittest.TestCase):
    def test_btc_preset_prices_one_unit_as_one_coin(self):
        cfg = btc_5m()
        self.assertAlmostEqual(cfg.tick_value / cfg.tick_size, 1.0)
        self.assertFalse(cfg.whole_units, "BTC positions are fractional")
        self.assertGreater(cfg.commission_pct, 0)
        self.assertIsNone(cfg.session_start_min, "crypto trades continuously")

    def test_es_preset_keeps_contract_conventions(self):
        cfg = es_5m()
        self.assertTrue(cfg.whole_units)
        self.assertAlmostEqual(cfg.tick_value / cfg.tick_size, 50.0)
        self.assertEqual(cfg.commission_pct, 0.0)
        self.assertIsNotNone(cfg.session_start_min)

    def test_overrides_apply_on_top_of_a_preset(self):
        self.assertEqual(build("btc", target_r=4.0).target_r, 4.0)

    def test_unknown_preset_is_rejected(self):
        with self.assertRaises(ValueError):
            build("dogecoin")


class TestProportionalCosts(unittest.TestCase):
    """BTC ranged several-fold over two years; fixed per-unit fees cannot model that."""

    def _short(self, cfg, entry=100.0, stop=102.0):
        from ..backtest import _Position
        from ..levels import RESISTANCE, Level
        from ..signals import SHORT, Signal

        level = Level(price=100.0, side=RESISTANCE, strength=5.0,
                      created_index=0, last_update_index=0)
        sig = Signal(index=1, ts=0.0, direction=SHORT, level=level, c1_index=0, c2_index=1,
                     entry_price=entry, stop=stop, target=96.0, risk_per_unit=2.0,
                     atr=1.0, entry_mode="close_of_2")
        return _Position(signal=sig, direction=SHORT, entry_index=0, entry_price=entry,
                         quantity=3.0, initial_quantity=3.0, risk_per_unit=2.0, stop=stop,
                         target=96.0, level=level, best_price=entry, worst_price=entry)

    def _cfg(self, **kw):
        base = dict(tick_size=10.0, tick_value=10.0, whole_units=False, slippage_ticks=0.0,
                    commission_per_side=0.0, partial_at_r=None, time_stop_bars=None,
                    trail_atr=None)
        base.update(kw)
        return StrategyConfig(**base)

    def test_commission_scales_with_notional(self):
        cfg = self._cfg(commission_pct=0.001)
        bt = Backtester(cfg)
        pos = self._short(cfg)
        bar = make_bar(ts=300, o=99.0, h=99.5, l=95.5, c=96.0, tick_size=10.0)
        from ..signals import prepare

        closed, _ = bt._manage(pos, 1, bar, prepare([bar], cfg), 100_000.0)
        # gross 4 * 3 = 12; fees 0.001 * (100*3) in + 0.001 * (96*3) out
        self.assertAlmostEqual(closed.pnl, 12.0 - 0.300 - 0.288, places=6)

    def test_the_same_fee_rate_costs_more_at_a_higher_price(self):
        cfg = self._cfg(commission_pct=0.001)
        bt = Backtester(cfg)
        cheap = bt._commission(20_000.0, 1.0)
        dear = bt._commission(100_000.0, 1.0)
        self.assertAlmostEqual(dear / cheap, 5.0, places=6)

    def test_proportional_slippage_worsens_the_entry(self):
        cfg = self._cfg(slippage_pct=0.002)
        bt = Backtester(cfg)
        pos = bt._open(self._short(cfg).signal, 1, 100.0, 100_000.0)
        self.assertAlmostEqual(pos.entry_price, 100.0 - 0.2, places=6)

    def test_fixed_and_proportional_components_add(self):
        cfg = self._cfg(slippage_ticks=0.5, slippage_pct=0.001)  # 0.5*10 + 0.001*100
        self.assertAlmostEqual(Backtester(cfg)._slippage(100.0), 5.0 + 0.1, places=6)

    def test_negative_percentage_costs_are_rejected(self):
        with self.assertRaises(ValueError):
            StrategyConfig(commission_pct=-0.001)
        with self.assertRaises(ValueError):
            StrategyConfig(slippage_pct=-0.001)


if __name__ == "__main__":
    unittest.main()
