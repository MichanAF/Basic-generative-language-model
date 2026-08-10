import unittest

from ..config import StrategyConfig
from ..footprint import Footprint
from ..levels import (
    RESISTANCE,
    SUPPORT,
    EvidenceWeights,
    Level,
    LevelBook,
    RollingProfile,
    detect_absorption,
    detect_delta_divergence,
    detect_imbalance_stacks,
)
from .helpers import TICK, make_bar


def cfg(**overrides) -> StrategyConfig:
    base = dict(tick_size=TICK, profile_lookback=50)
    base.update(overrides)
    return StrategyConfig(**base)


class TestRollingProfile(unittest.TestCase):
    def test_old_bars_are_evicted(self):
        profile = RollingProfile(TICK, window=2)
        for price in (100.0, 101.0, 102.0):
            fp = Footprint(tick_size=TICK)
            fp.add(price, 10, 1)
            profile.push(fp)
        self.assertEqual(profile.volume_at_price(100.0), 0.0, "should have rolled off")
        self.assertEqual(profile.volume_at_price(101.0), 10.0)
        self.assertEqual(profile.volume_at_price(102.0), 10.0)

    def test_snapshot_totals_the_window(self):
        profile = RollingProfile(TICK, window=10)
        for _ in range(3):
            fp = Footprint(tick_size=TICK)
            fp.add(100.0, 10, 1)
            fp.add(100.0, 4, -1)
            profile.push(fp)
        snap = profile.snapshot()
        self.assertAlmostEqual(snap.total_volume, 42.0)
        self.assertAlmostEqual(snap.delta, 18.0)


class TestEvidenceDetectors(unittest.TestCase):
    def test_absorption_needs_aggression_that_failed(self):
        """Buyers hit the offer at the highs and the bar closed at the low."""
        bar = make_bar(
            ts=0,
            o=99.5,
            h=100.5,
            l=99.0,
            c=99.2,
            ask={100.25: 400, 100.5: 300},
            bid={99.2: 100, 99.0: 80},
        )
        found = detect_absorption(bar, cfg(), volume_threshold=100.0)
        self.assertEqual([side for _, side in found], [RESISTANCE])
        self.assertGreaterEqual(found[0][0], 100.0, "level should sit at the offer, not the low")

    def test_absorption_is_ignored_below_the_volume_threshold(self):
        bar = make_bar(
            ts=0, o=99.5, h=100.5, l=99.0, c=99.2, ask={100.5: 40}, bid={99.0: 10}
        )
        self.assertEqual(detect_absorption(bar, cfg(), volume_threshold=10_000.0), [])

    def test_a_strong_close_is_not_resistance_absorption(self):
        bar = make_bar(
            ts=0, o=99.2, h=100.5, l=99.0, c=100.4, ask={100.25: 400}, bid={99.0: 50}
        )
        sides = [side for _, side in detect_absorption(bar, cfg(), volume_threshold=100.0)]
        self.assertNotIn(RESISTANCE, sides)

    def test_support_absorption_mirrors(self):
        bar = make_bar(
            ts=0,
            o=99.8,
            h=100.5,
            l=99.0,
            c=100.4,
            bid={99.0: 400, 99.25: 300},
            ask={100.4: 100},
        )
        found = detect_absorption(bar, cfg(), volume_threshold=100.0)
        self.assertEqual([side for _, side in found], [SUPPORT])

    def test_stacked_sell_imbalances_mark_resistance(self):
        bar = make_bar(
            ts=0,
            o=100.0,
            h=100.75,
            l=99.5,
            c=99.6,
            bid={100.0: 200, 100.25: 200, 100.5: 200},
            ask={100.75: 5, 99.6: 20},
        )
        found = detect_imbalance_stacks(bar, cfg())
        self.assertIn(RESISTANCE, [side for _, side in found])

    def test_delta_divergence_needs_a_higher_high_on_weaker_flow(self):
        highs = [(1, 100.0, 500.0), (10, 102.0, 300.0)]  # higher price, lower delta
        found, seen_h, _ = detect_delta_divergence(highs, [], 0, 0)
        self.assertEqual(found, [(102.0, RESISTANCE)])
        self.assertEqual(seen_h, 2)

        again, _, _ = detect_delta_divergence(highs, [], seen_h, 0)
        self.assertEqual(again, [], "each swing is scored once")

    def test_higher_high_on_stronger_flow_is_not_divergence(self):
        highs = [(1, 100.0, 300.0), (10, 102.0, 500.0)]
        found, _, _ = detect_delta_divergence(highs, [], 0, 0)
        self.assertEqual(found, [])


class TestLevelBook(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg()
        self.book = LevelBook(self.cfg)

    def _absorption_bar(self, ts=0.0):
        return make_bar(
            ts=ts,
            o=99.5,
            h=100.5,
            l=99.0,
            c=99.2,
            ask={100.25: 400, 100.5: 300},
            bid={99.2: 100, 99.0: 80},
        )

    def test_evidence_creates_a_level(self):
        self.book.update(0, self._absorption_bar(), atr_value=1.0, volume_threshold=100.0)
        resistances = [l for l in self.book.levels if l.side == RESISTANCE]
        self.assertTrue(resistances)
        self.assertIn("absorption", resistances[0].evidence)

    def test_nearby_evidence_merges_instead_of_duplicating(self):
        for i in range(3):
            self.book.update(i, self._absorption_bar(ts=i * 300), 1.0, 100.0)
        near = [l for l in self.book.levels if l.side == RESISTANCE and abs(l.price - 100.4) < 0.5]
        self.assertEqual(len(near), 1, "repeated evidence at one price is one level")

    def test_strength_is_capped(self):
        for i in range(60):
            self.book.update(i, self._absorption_bar(ts=i * 300), 1.0, 100.0)
        for level in self.book.levels:
            self.assertLessEqual(level.strength, EvidenceWeights().max_strength + 1e-9)

    def test_chop_across_a_level_is_not_thirty_tests(self):
        book = LevelBook(cfg())
        level = Level(
            price=100.0, side=RESISTANCE, strength=8.0, created_index=0, last_update_index=0
        )
        book.levels = [level]
        for i in range(1, 10):  # a bar straddling the level every single bar
            bar = make_bar(ts=i * 300, o=99.8, h=100.1, l=99.5, c=99.7)
            book._score_touches(i, bar, atr_value=1.0)
        self.assertLessEqual(level.touches, 4, "touch credit must be throttled")

    def test_decay_makes_stale_levels_untradeable(self):
        level = Level(
            price=100.0, side=RESISTANCE, strength=4.0, created_index=0, last_update_index=0
        )
        self.assertAlmostEqual(level.effective_strength(120, half_life=120), 2.0)
        self.assertAlmostEqual(level.effective_strength(240, half_life=120), 1.0)
        self.assertEqual(level.effective_strength(0, half_life=0), 4.0)

    def test_a_decisive_close_through_flips_polarity(self):
        book = LevelBook(cfg())
        book.levels = [
            Level(price=100.0, side=RESISTANCE, strength=8.0, created_index=0, last_update_index=0)
        ]
        broke = make_bar(ts=300, o=100.0, h=102.0, l=99.9, c=101.5)
        book._invalidate(1, broke, atr_value=1.0)
        self.assertEqual([l.side for l in book.levels], [SUPPORT])
        self.assertEqual(book.levels[0].flipped_from, RESISTANCE)

    def test_flips_merge_into_an_existing_level_on_the_far_side(self):
        book = LevelBook(cfg())
        book.levels = [
            Level(price=100.0, side=RESISTANCE, strength=8.0, created_index=0, last_update_index=0),
            Level(price=100.0, side=SUPPORT, strength=5.0, created_index=0, last_update_index=0),
        ]
        broke = make_bar(ts=300, o=100.0, h=102.0, l=99.9, c=101.5)
        book._invalidate(1, broke, atr_value=1.0)
        self.assertEqual(len(book.levels), 1, "the flip must not spawn a duplicate")
        self.assertIn("polarity_flip", book.levels[0].evidence)

    def test_weak_broken_levels_disappear_entirely(self):
        book = LevelBook(cfg(level_min_strength=5.0))
        book.levels = [
            Level(price=100.0, side=RESISTANCE, strength=2.0, created_index=0, last_update_index=0)
        ]
        book._invalidate(1, make_bar(ts=300, o=100.0, h=102.0, l=99.9, c=101.5), atr_value=1.0)
        self.assertEqual(book.levels, [])

    def test_a_level_with_no_flow_evidence_is_not_tradeable(self):
        """A wick plus a volume node is a chart pattern, not order flow."""
        book = LevelBook(cfg())
        chart_only = Level(
            price=100.0,
            side=RESISTANCE,
            strength=8.0,
            created_index=0,
            last_update_index=10,
            evidence={"wick_rejection": 4.0, "high_volume_node": 4.0},
        )
        book.levels = [chart_only]
        self.assertFalse(chart_only.is_flow_backed)
        self.assertEqual(book.tradeable(10, RESISTANCE), [])

        chart_only.evidence["absorption"] = 2.0
        self.assertTrue(chart_only.is_flow_backed)
        self.assertEqual(book.tradeable(10, RESISTANCE), [chart_only])

    def test_the_flow_requirement_can_be_disabled(self):
        book = LevelBook(cfg(require_flow_backed_levels=False))
        chart_only = Level(
            price=100.0, side=RESISTANCE, strength=8.0, created_index=0,
            last_update_index=10, evidence={"wick_rejection": 8.0},
        )
        book.levels = [chart_only]
        self.assertEqual(book.tradeable(10, RESISTANCE), [chart_only])

    def test_a_flip_carries_flow_backing_across(self):
        """Flipped levels lose their evidence keys, so the flag must survive."""
        book = LevelBook(cfg())
        book.levels = [
            Level(price=100.0, side=RESISTANCE, strength=8.0, created_index=0,
                  last_update_index=0, evidence={"absorption": 8.0})
        ]
        book._invalidate(1, make_bar(ts=300, o=100.0, h=102.0, l=99.9, c=101.5), atr_value=1.0)
        self.assertEqual(len(book.levels), 1)
        self.assertTrue(book.levels[0].is_flow_backed)

    def test_a_flip_does_not_invent_flow_backing(self):
        book = LevelBook(cfg())
        book.levels = [
            Level(price=100.0, side=RESISTANCE, strength=8.0, created_index=0,
                  last_update_index=0, evidence={"wick_rejection": 8.0})
        ]
        book._invalidate(1, make_bar(ts=300, o=100.0, h=102.0, l=99.9, c=101.5), atr_value=1.0)
        self.assertFalse(book.levels[0].is_flow_backed)

    def test_tradeable_respects_cooldown_and_trade_count(self):
        book = LevelBook(cfg(max_trades_per_level=2))
        level = Level(
            price=100.0, side=RESISTANCE, strength=8.0, created_index=0,
            last_update_index=10, evidence={"absorption": 8.0},
        )
        book.levels = [level]
        self.assertEqual(book.tradeable(10, RESISTANCE), [level])

        level.cooldown_until = 15
        self.assertEqual(book.tradeable(12, RESISTANCE), [])
        self.assertEqual(book.tradeable(15, RESISTANCE), [level])

        level.trades_taken = 2
        self.assertEqual(book.tradeable(15, RESISTANCE), [])

    def test_pruning_keeps_only_the_strongest_per_side(self):
        book = LevelBook(cfg(max_levels_per_side=2))
        book.levels = [
            Level(price=100.0 + i, side=RESISTANCE, strength=2.0 + i,
                  created_index=0, last_update_index=0)
            for i in range(5)
        ]
        book._prune(0)
        self.assertEqual(len(book.levels), 2)
        self.assertEqual(sorted(l.strength for l in book.levels), [5.0, 6.0])

    def test_nearest_opposing_finds_a_target(self):
        book = LevelBook(cfg())
        book.levels = [
            Level(price=95.0, side=SUPPORT, strength=8.0, created_index=0, last_update_index=0),
            Level(price=90.0, side=SUPPORT, strength=8.0, created_index=0, last_update_index=0),
            Level(price=105.0, side=SUPPORT, strength=8.0, created_index=0, last_update_index=0),
        ]
        found = book.nearest_opposing(price=100.0, side=RESISTANCE, index=0)
        self.assertAlmostEqual(found.price, 95.0, msg="must be below, and the closest one")

    def test_nearest_opposing_returns_none_when_there_is_nothing_below(self):
        book = LevelBook(cfg())
        book.levels = [
            Level(price=105.0, side=SUPPORT, strength=8.0, created_index=0, last_update_index=0)
        ]
        self.assertIsNone(book.nearest_opposing(price=100.0, side=RESISTANCE, index=0))


if __name__ == "__main__":
    unittest.main()
