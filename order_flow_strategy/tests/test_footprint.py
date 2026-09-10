import unittest

from ..data import Trade, bars_from_trades
from ..footprint import Footprint, high_volume_nodes, merge_footprints, to_level

TICK = 0.25


class TestFootprint(unittest.TestCase):
    def test_delta_and_normalization(self):
        fp = Footprint(tick_size=TICK)
        fp.add(100.0, 30, 1)
        fp.add(100.0, 10, -1)
        self.assertEqual(fp.total_volume, 40)
        self.assertEqual(fp.delta, 20)
        self.assertAlmostEqual(fp.normalized_delta, 0.5)

    def test_normalized_delta_is_zero_on_empty(self):
        self.assertEqual(Footprint(tick_size=TICK).normalized_delta, 0.0)

    def test_poc_is_the_heaviest_level(self):
        fp = Footprint(tick_size=TICK)
        fp.add(100.0, 5, 1)
        fp.add(100.5, 50, -1)
        fp.add(101.0, 5, 1)
        self.assertEqual(fp.poc_level, to_level(100.5, TICK))

    def test_diagonal_imbalance_compares_across_the_spread(self):
        """Ask at p is compared with bid at p-1 tick, not bid at p."""
        fp = Footprint(tick_size=TICK)
        fp.add(100.25, 100, 1)  # aggressive buying
        fp.add(100.00, 5, -1)  # thin selling one tick below
        buys, sells = fp.diagonal_imbalances(ratio=3.0, min_volume=10)
        self.assertIn(to_level(100.25, TICK), buys)
        self.assertNotIn(to_level(100.0, TICK), sells)

    def test_longest_stack_finds_the_run_and_its_top(self):
        length, top = Footprint.longest_stack([1, 2, 3, 7, 9, 10])
        self.assertEqual(length, 3)
        self.assertEqual(top, 3)

    def test_longest_stack_on_empty(self):
        self.assertEqual(Footprint.longest_stack([]), (0, None))

    def test_cluster_flow_splits_top_and_bottom(self):
        fp = Footprint(tick_size=TICK)
        for i in range(9):  # 100.00 .. 102.00
            price = 100.0 + i * 0.25
            fp.add(price, 10, 1 if i >= 6 else -1)
        ask_top, bid_top = fp.cluster_flow(frac=0.30, top=True)
        self.assertGreater(ask_top, 0)
        self.assertEqual(bid_top, 0)
        ask_bot, bid_bot = fp.cluster_flow(frac=0.30, top=False)
        self.assertEqual(ask_bot, 0)
        self.assertGreater(bid_bot, 0)

    def test_cluster_poc_price_points_at_the_heavy_level(self):
        fp = Footprint(tick_size=TICK)
        for i in range(9):
            fp.add(100.0 + i * 0.25, 5, 1)
        fp.add(101.75, 500, 1)  # the wall, inside the top cluster
        self.assertAlmostEqual(fp.cluster_poc_price(0.30, top=True), 101.75)

    def test_proxy_split_follows_close_location(self):
        strong = Footprint.from_ohlcv_proxy(100, 102, 100, 102, 1000, TICK)
        weak = Footprint.from_ohlcv_proxy(100, 102, 100, 100, 1000, TICK)
        self.assertGreater(strong.normalized_delta, 0.9)
        self.assertLess(weak.normalized_delta, -0.9)
        self.assertTrue(strong.is_proxy)

    def test_proxy_honours_a_measured_buy_fraction(self):
        """A feed that reports real aggressor volume must override the guess."""
        fp = Footprint.from_ohlcv_proxy(100, 102, 100, 102, 1000, TICK, buy_frac=0.25)
        self.assertAlmostEqual(fp.normalized_delta, -0.5, places=6)

    def test_proxy_caps_level_count_on_huge_ranges(self):
        fp = Footprint.from_ohlcv_proxy(0, 1000, 0, 500, 10, 0.01, levels_cap=50)
        self.assertLessEqual(len(fp.levels), 50)
        self.assertAlmostEqual(fp.total_volume, 10, places=6)

    def test_merge_marks_proxy_contamination(self):
        real = Footprint(tick_size=TICK)
        real.add(100.0, 10, 1)
        proxy = Footprint.from_ohlcv_proxy(100, 101, 99, 100, 50, TICK)
        merged = merge_footprints([real, proxy], TICK)
        self.assertTrue(merged.is_proxy)
        self.assertAlmostEqual(merged.total_volume, 60, places=6)

    def test_high_volume_nodes_are_local_maxima(self):
        fp = Footprint(tick_size=TICK)
        for i in range(21):
            fp.add(100.0 + i * 0.25, 10, 1)
        fp.add(102.5, 900, 1)  # a clear node
        nodes = high_volume_nodes(fp, percentile=0.85)
        self.assertTrue(any(abs(p - 102.5) < 1e-9 for p, _ in nodes))


class TestBarAggregation(unittest.TestCase):
    def test_bars_from_trades_builds_ohlc_and_flow(self):
        trades = [
            Trade(ts=0, price=100.0, size=1, aggressor=1),
            Trade(ts=10, price=101.0, size=2, aggressor=1),
            Trade(ts=20, price=99.0, size=3, aggressor=-1),
            Trade(ts=30, price=100.5, size=4, aggressor=1),
            Trade(ts=70, price=105.0, size=5, aggressor=1),  # next bucket
        ]
        bars = bars_from_trades(trades, timeframe_seconds=60, tick_size=TICK)
        self.assertEqual(len(bars), 2)
        first = bars[0]
        self.assertEqual((first.open, first.high, first.low, first.close), (100.0, 101.0, 99.0, 100.5))
        self.assertEqual(first.volume, 10)
        self.assertEqual(first.delta, 1 + 2 - 3 + 4)
        self.assertEqual(bars[1].open, 105.0)

    def test_empty_tape_yields_no_bars(self):
        self.assertEqual(bars_from_trades([], 60, TICK), [])

    def test_rejects_nonpositive_timeframe(self):
        with self.assertRaises(ValueError):
            bars_from_trades([Trade(0, 1, 1, 1)], 0, TICK)


if __name__ == "__main__":
    unittest.main()
