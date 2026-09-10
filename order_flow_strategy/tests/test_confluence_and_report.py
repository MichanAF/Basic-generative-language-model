import io
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout

from ..backtest import Backtester, ClosedTrade
from ..baselines import buy_and_hold, trend_baseline
from ..cli import main
from ..confluence import analyse, format_confluence
from ..config import StrategyConfig
from ..data import SyntheticConfig, bars_from_trades, synthetic_trades
from ..metrics import summarize
from ..presets import btc_5m
from ..reporting_html import build_report
from ..signals import tags_from_reasons
from .helpers import TICK

SHORT = -1


def a_trade(r: float, tags, pnl=None) -> ClosedTrade:
    return ClosedTrade(
        direction=SHORT, entry_index=0, entry_ts=0.0, entry_price=100.0,
        exit_index=1, exit_ts=300.0, exit_price=98.0, quantity=1.0,
        pnl=r * 100.0 if pnl is None else pnl, r_multiple=r, mae_r=0.2, mfe_r=0.5,
        bars_held=1, exit_reason="target", level_price=100.0, level_strength=5.0,
        tags=list(tags),
    )


def series(n_ticks=300_000, seed=5):
    sc = SyntheticConfig(n_ticks=n_ticks, tick_size=TICK, seed=seed)
    return bars_from_trades(synthetic_trades(sc), 300.0, TICK)


class TestTagExtraction(unittest.TestCase):
    def test_reads_the_phrases_the_engine_writes(self):
        tags = tags_from_reasons([
            "tested resistance @ 100.00 (strength 8.0; absorptionx2.0, retest_holdx1.0)",
            "upper wick 65% of range",
            "3 stacked sell imbalances",
        ])
        self.assertIn("absorption", tags)
        self.assertIn("retest_hold", tags)
        self.assertIn("imbalance_stack", tags)
        self.assertIn("wick_rejection", tags)

    def test_buy_and_sell_stacks_share_one_tag(self):
        self.assertEqual(tags_from_reasons(["4 stacked buy imbalances"]), ["imbalance_stack"])
        self.assertEqual(tags_from_reasons(["4 stacked sell imbalances"]), ["imbalance_stack"])

    def test_no_duplicates_and_empty_input_is_empty(self):
        tags = tags_from_reasons(["absorption here", "absorption there"])
        self.assertEqual(tags, ["absorption"])
        self.assertEqual(tags_from_reasons([]), [])


class TestConfluenceAnalysis(unittest.TestCase):
    def test_splits_by_condition_and_measures_the_lift(self):
        trades = [a_trade(2.0, ["absorption"]) for _ in range(15)]
        trades += [a_trade(-1.0, []) for _ in range(15)]
        rep = analyse(trades)
        f = next(f for f in rep.factors if f.tag == "absorption")
        self.assertEqual(f.n_with, 15)
        self.assertEqual(f.n_without, 15)
        self.assertAlmostEqual(f.exp_with, 2.0)
        self.assertAlmostEqual(f.exp_without, -1.0)
        self.assertAlmostEqual(f.lift, 3.0)
        self.assertEqual(f.verdict, "earns its place")

    def test_a_harmful_condition_is_called_out(self):
        trades = [a_trade(-1.5, ["high_volume_node"]) for _ in range(12)]
        trades += [a_trade(1.0, []) for _ in range(12)]
        f = next(f for f in analyse(trades).factors if f.tag == "high_volume_node")
        self.assertLess(f.lift, 0)
        self.assertEqual(f.verdict, "hurts -- consider dropping")

    def test_small_subgroups_refuse_to_judge(self):
        trades = [a_trade(3.0, ["absorption"]) for _ in range(3)]
        trades += [a_trade(-1.0, []) for _ in range(30)]
        f = next(f for f in analyse(trades).factors if f.tag == "absorption")
        self.assertEqual(f.verdict, "too few to judge",
                         "three trades must not produce a recommendation")

    def test_no_effect_is_reported_as_such(self):
        trades = [a_trade(0.5, ["wick_rejection"]) for _ in range(15)]
        trades += [a_trade(0.5, []) for _ in range(15)]
        f = next(f for f in analyse(trades).factors if f.tag == "wick_rejection")
        self.assertEqual(f.verdict, "no measurable effect")

    def test_factors_are_ranked_by_lift(self):
        trades = [a_trade(2.0, ["good"]) for _ in range(12)]
        trades += [a_trade(-2.0, ["bad"]) for _ in range(12)]
        trades += [a_trade(0.0, []) for _ in range(12)]
        tags = [f.tag for f in analyse(trades).factors]
        self.assertLess(tags.index("good"), tags.index("bad"))

    def test_grouping_by_condition_count(self):
        trades = [a_trade(1.0, ["a", "b"]) for _ in range(6)]
        trades += [a_trade(-1.0, ["a"]) for _ in range(6)]
        rep = analyse(trades)
        self.assertEqual(rep.by_count[2].n, 6)
        self.assertEqual(rep.by_count[1].n, 6)
        self.assertAlmostEqual(rep.by_count[2].expectancy, 1.0)
        self.assertTrue(rep.more_is_better)

    def test_detects_when_stacking_does_not_help(self):
        trades = [a_trade(-1.0, ["a", "b", "c"]) for _ in range(8)]
        trades += [a_trade(1.0, ["a"]) for _ in range(8)]
        self.assertFalse(analyse(trades).more_is_better)

    def test_empty_input_is_safe(self):
        rep = analyse([])
        self.assertEqual(rep.n_trades, 0)
        self.assertEqual(rep.factors, [])
        self.assertIn("No tagged trades", format_confluence(rep))

    def test_text_output_carries_the_caveat(self):
        trades = [a_trade(2.0, ["absorption"]) for _ in range(12)]
        trades += [a_trade(-1.0, []) for _ in range(12)]
        text = format_confluence(analyse(trades))
        self.assertIn("absorption", text)
        self.assertIn("hypothesis", text, "the multiple-testing caveat must survive")


class TestHtmlReport(unittest.TestCase):
    def _sections(self, bars):
        cfg = btc_5m()
        of = Backtester(cfg).run(bars)
        tr = trend_baseline(bars, cfg, 50)
        bh = buy_and_hold(bars, cfg)
        return cfg, [
            ("Order flow", of, summarize(of)),
            ("Trend SMA50", tr, summarize(tr)),
            ("Buy and hold", bh, summarize(bh)),
        ], analyse(of.trades)

    def test_page_is_self_contained_and_well_formed(self):
        bars = series()
        cfg, sections, conf = self._sections(bars)
        page = build_report(sections, cfg, confluence=conf, title="Test Report")

        self.assertIn("<title>Test Report</title>", page)
        # The artifact host supplies the document skeleton.
        for tag in ("<!doctype", "<html", "<body>"):
            self.assertNotIn(tag, page.lower()[:200])
        self.assertEqual(page.count("<svg"), page.count("</svg>"), "unbalanced svg")
        self.assertIn("</style>", page)
        # Only the webfont may be remote; data and styles ship inline.
        remote = re.findall(r'https?://[^"\')\s]+', page)
        for url in remote:
            self.assertTrue(
                "fonts.googleapis.com" in url or "fonts.gstatic.com" in url,
                f"unexpected remote resource: {url}",
            )

    def test_every_theme_token_is_defined_in_the_bare_root(self):
        """A token defined only under a media query renders unreadably."""
        cfg, sections, conf = self._sections(series())
        page = build_report(sections, cfg, confluence=conf)
        bare = page.split(":root {", 1)[1].split("}", 1)[0]
        declared = set(re.findall(r"(--[a-z0-9-]+):", bare))
        used = set(re.findall(r"var\((--[a-z0-9-]+)\)", page))
        self.assertFalse(used - declared, f"undeclared tokens: {used - declared}")

    def test_both_theme_overrides_are_present(self):
        cfg, sections, conf = self._sections(series())
        page = build_report(sections, cfg, confluence=conf)
        self.assertIn('prefers-color-scheme: dark', page)
        self.assertIn(':root:not([data-theme="light"])', page)
        self.assertIn(':root[data-theme="dark"]', page)
        self.assertIn("background:var(--ground)", page.replace(" ", "").replace("\n", ""))

    def test_verdict_names_the_benchmark(self):
        cfg, sections, conf = self._sections(series())
        page = build_report(sections, cfg, confluence=conf)
        self.assertIn("hold", page.lower())

    def test_survives_a_run_with_no_trades(self):
        bars = series()
        cfg = StrategyConfig(tick_size=TICK, level_min_strength=1e9)
        empty = Backtester(cfg).run(bars)
        bh = buy_and_hold(bars, cfg)
        page = build_report(
            [("Order flow", empty, summarize(empty)), ("Buy and hold", bh, summarize(bh))],
            cfg, confluence=analyse(empty.trades),
        )
        self.assertIn("no trades", page.lower())

    def test_escapes_titles(self):
        cfg, sections, conf = self._sections(series())
        page = build_report(sections, cfg, confluence=conf, title="<script>x</script>")
        self.assertNotIn("<script>x</script>", page)
        self.assertIn("&lt;script&gt;", page)


class TestReportCommand(unittest.TestCase):
    def test_writes_a_file_and_prints_the_confluence_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "r.html")
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(["report", "--synthetic", "--ticks", "300000",
                             "--sma", "50", "--out", out])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.getsize(out) > 5000)
            with open(out, encoding="utf-8") as fh:
                self.assertIn("<title>", fh.read(200))
        self.assertIn("Confluence", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
