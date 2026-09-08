"""Which conditions actually earn their keep.

The strategy stacks several pieces of evidence before it fires: absorption,
stacked imbalances, delta divergence, a volume node, a wick, a level that has
held before. Traders call the stack "confluence" and generally assume more of
it is better.

That is an assumption, and it is testable. For each condition this module
splits the trades into those that had it and those that did not, and reports
what each subset actually did. A condition that earns its place shows a
materially better expectancy when present. A condition that shows no
difference is costing you trades for nothing, and one that shows a *worse*
expectancy is actively harmful -- which happens more often than people expect,
because a condition can be a proxy for "the move already happened".

The obvious caveat: these are subgroups of one sample, not independent
experiments. Split six ways and one will look good by luck. Treat a lift as a
hypothesis to test on fresh data, never as a reason to add a filter.
"""

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Sequence

from .backtest import ClosedTrade
from .footprint import EPS


@dataclass
class FactorStats:
    """One condition, measured on the trades that had it and those that did not."""

    tag: str
    n_with: int = 0
    n_without: int = 0
    exp_with: float = 0.0
    exp_without: float = 0.0
    win_with: float = 0.0
    win_without: float = 0.0

    @property
    def lift(self) -> float:
        """Expectancy difference, in R, attributable to the condition."""
        return self.exp_with - self.exp_without

    @property
    def coverage(self) -> float:
        total = self.n_with + self.n_without
        return self.n_with / total if total else 0.0

    @property
    def verdict(self) -> str:
        """A plain reading, deliberately conservative about small subgroups."""
        if min(self.n_with, self.n_without) < 10:
            return "too few to judge"
        if self.lift > 0.15:
            return "earns its place"
        if self.lift < -0.15:
            return "hurts -- consider dropping"
        return "no measurable effect"


@dataclass
class ConfluenceReport:
    factors: List[FactorStats] = field(default_factory=list)
    #: Expectancy grouped by how many conditions were present at once.
    by_count: Dict[int, "CountStats"] = field(default_factory=dict)
    n_trades: int = 0

    @property
    def more_is_better(self) -> bool:
        """Does expectancy actually rise with the number of conditions?"""
        counts = sorted(self.by_count)
        usable = [c for c in counts if self.by_count[c].n >= 5]
        if len(usable) < 2:
            return False
        return self.by_count[usable[-1]].expectancy > self.by_count[usable[0]].expectancy


@dataclass
class CountStats:
    count: int
    n: int
    expectancy: float
    win_rate: float


def _expectancy(trades: Sequence[ClosedTrade]) -> float:
    return mean([t.r_multiple for t in trades]) if trades else 0.0


def _win_rate(trades: Sequence[ClosedTrade]) -> float:
    return sum(1 for t in trades if t.pnl > 0) / len(trades) if trades else 0.0


def analyse(trades: Sequence[ClosedTrade]) -> ConfluenceReport:
    """Split trades by each condition and by how many conditions co-occurred."""
    report = ConfluenceReport(n_trades=len(trades))
    if not trades:
        return report

    all_tags: List[str] = []
    for t in trades:
        for tag in t.tags:
            if tag not in all_tags:
                all_tags.append(tag)

    for tag in sorted(all_tags):
        has = [t for t in trades if tag in t.tags]
        hasnt = [t for t in trades if tag not in t.tags]
        report.factors.append(
            FactorStats(
                tag=tag,
                n_with=len(has),
                n_without=len(hasnt),
                exp_with=_expectancy(has),
                exp_without=_expectancy(hasnt),
                win_with=_win_rate(has),
                win_without=_win_rate(hasnt),
            )
        )
    report.factors.sort(key=lambda f: f.lift, reverse=True)

    buckets: Dict[int, List[ClosedTrade]] = {}
    for t in trades:
        buckets.setdefault(len(t.tags), []).append(t)
    for count, group in sorted(buckets.items()):
        report.by_count[count] = CountStats(
            count=count,
            n=len(group),
            expectancy=_expectancy(group),
            win_rate=_win_rate(group),
        )
    return report


def format_confluence(report: ConfluenceReport) -> str:
    """Plain-text version, for the terminal."""
    if not report.factors:
        return "No tagged trades to analyse.\n"

    rule = "=" * 72
    thin = "-" * 72
    lines = [
        rule,
        "Confluence: does each condition earn its place?",
        rule,
        f"{'condition':<20}{'seen in':>9}{'with':>8}{'without':>9}{'lift':>8}  verdict",
        thin,
    ]
    for f in report.factors:
        lines.append(
            f"{f.tag:<20}{f.coverage*100:>8.0f}%{f.exp_with:>+8.2f}"
            f"{f.exp_without:>+9.2f}{f.lift:>+8.2f}  {f.verdict}"
        )

    lines += [thin, "Expectancy by number of conditions present:", ""]
    for count, cs in sorted(report.by_count.items()):
        bar = "#" * min(40, max(0, int((cs.expectancy + 1) * 12)))
        lines.append(f"  {count} condition(s)  n={cs.n:<5} {cs.expectancy:>+6.2f} R  {bar}")

    lines += [
        thin,
        "More confluence did " + ("" if report.more_is_better else "NOT ")
        + "improve expectancy on this sample.",
        "These are subgroups of one dataset, not independent tests. Six splits",
        "means one looks good by chance. Treat any lift as a hypothesis for",
        "fresh data, not as a filter to add today.",
        rule,
        "",
    ]
    return "\n".join(lines)
