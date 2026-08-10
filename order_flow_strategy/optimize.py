"""Walk-forward parameter search, built to resist fooling itself.

Three habits do most of the work here:

1. **Walk forward.** Parameters are chosen on data that precedes the data they
   are scored on. An in-sample grid search alone tells you nothing.
2. **Prefer plateaus to peaks.** The per-parameter recommendation is the value
   with the best *median* score across every other combination, not the value
   in the single best run. A peak surrounded by cliffs is a fitting artifact;
   a broad plateau survives contact with new data.
3. **Report the degradation.** The gap between in-sample and out-of-sample
   score is printed rather than hidden. A large gap means the search found
   noise, and the honest response is to use fewer parameters.
"""

import itertools
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .backtest import Backtester
from .config import ENTRY_BREAK_OF_2, ENTRY_CLOSE_OF_2, REGIME_NONE, REGIME_WITH_TREND_ONLY, StrategyConfig
from .data import Bar
from .metrics import PerformanceStats, robustness_score, summarize
from .report import RULE, THIN
from .signals import Indicators, prepare

#: Parameters worth searching, and sensible ranges for each.
DEFAULT_GRID: Dict[str, Sequence[Any]] = {
    "target_r": (1.5, 2.0, 3.0),
    "c2_close_beyond_c1_frac": (0.33, 0.5, 0.75),
    "stop_buffer_atr": (0.15, 0.25, 0.40),
    "entry_mode": (ENTRY_CLOSE_OF_2, ENTRY_BREAK_OF_2),
    "regime_filter": (REGIME_NONE, REGIME_WITH_TREND_ONLY),
}

#: A deliberately small grid for quick runs.
FAST_GRID: Dict[str, Sequence[Any]] = {
    "target_r": (1.5, 2.0, 3.0),
    "c2_close_beyond_c1_frac": (0.33, 0.5),
    "entry_mode": (ENTRY_CLOSE_OF_2, ENTRY_BREAK_OF_2),
}


class _IndicatorCache:
    """Indicators only depend on a few parameters; recomputing is the slow part."""

    def __init__(self):
        self._cache: Dict[Tuple, Indicators] = {}

    def get(self, bars: Sequence[Bar], cfg: StrategyConfig) -> Indicators:
        key = (
            id(bars),
            len(bars),
            cfg.atr_period,
            cfg.ema_period,
            cfg.profile_lookback,
            cfg.absorption_volume_pct,
        )
        if key not in self._cache:
            self._cache[key] = prepare(bars, cfg)
        return self._cache[key]


def expand(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def evaluate(
    bars: Sequence[Bar],
    cfg: StrategyConfig,
    cache: Optional[_IndicatorCache] = None,
) -> PerformanceStats:
    cache = cache or _IndicatorCache()
    result = Backtester(cfg).run(bars, cache.get(bars, cfg))
    return summarize(result)


@dataclass
class FoldOutcome:
    fold: int
    train_span: Tuple[int, int]
    test_span: Tuple[int, int]
    params: Dict[str, Any]
    train_score: float
    test_score: float
    test_stats: PerformanceStats


@dataclass
class Recommendation:
    best_params: Dict[str, Any]
    robust_params: Dict[str, Any]
    folds: List[FoldOutcome]
    param_sensitivity: Dict[str, List[Tuple[Any, float, int]]]
    oos_stats: Optional[PerformanceStats]
    notes: List[str] = field(default_factory=list)

    @property
    def mean_train_score(self) -> float:
        scores = [f.train_score for f in self.folds if f.train_score > float("-inf")]
        return statistics.mean(scores) if scores else 0.0

    @property
    def mean_test_score(self) -> float:
        scores = [f.test_score for f in self.folds if f.test_score > float("-inf")]
        return statistics.mean(scores) if scores else 0.0


def walk_forward(
    bars: Sequence[Bar],
    base: StrategyConfig,
    grid: Optional[Dict[str, Sequence[Any]]] = None,
    folds: int = 4,
    min_trades: int = 15,
) -> Recommendation:
    """Anchored walk-forward: train on everything before the test window."""
    grid = grid or DEFAULT_GRID
    combos = expand(grid)
    cache = _IndicatorCache()

    n = len(bars)
    if folds < 1:
        raise ValueError("folds must be >= 1")
    segment = n // (folds + 1)
    if segment < 100:
        raise ValueError(
            f"need at least {100 * (folds + 1)} bars for {folds} folds; got {n}"
        )

    outcomes: List[FoldOutcome] = []
    # Score surface for the plateau analysis: params -> scores across all folds.
    surface: List[Tuple[Dict[str, Any], float]] = []

    for k in range(folds):
        train_end = segment * (k + 1)
        test_end = segment * (k + 2) if k < folds - 1 else n
        train = bars[:train_end]
        test = bars[train_end:test_end]
        if len(test) < 50:
            continue

        best_params, best_score = None, float("-inf")
        for params in combos:
            cfg = base.with_(**params)
            score = robustness_score(evaluate(train, cfg, cache), min_trades)
            surface.append((params, score))
            if score > best_score:
                best_params, best_score = params, score

        if best_params is None:  # nothing cleared the trade minimum
            continue

        test_cfg = base.with_(**best_params)
        test_stats = evaluate(test, test_cfg, cache)
        outcomes.append(
            FoldOutcome(
                fold=k,
                train_span=(0, train_end),
                test_span=(train_end, test_end),
                params=best_params,
                train_score=best_score,
                test_score=robustness_score(test_stats, min_trades=1),
                test_stats=test_stats,
            )
        )

    sensitivity = _sensitivity(surface, grid)
    robust = {k: values[0][0] for k, values in sensitivity.items() if values}
    best = _most_common_params(outcomes) or robust

    oos_stats = None
    if outcomes:
        # A single continuous out-of-sample pass with the robust parameters,
        # which is what actually trading them would have produced.
        first_test_start = outcomes[0].test_span[0]
        oos_stats = evaluate(bars[first_test_start:], base.with_(**robust), cache)

    return Recommendation(
        best_params=best,
        robust_params=robust,
        folds=outcomes,
        param_sensitivity=sensitivity,
        oos_stats=oos_stats,
        notes=_notes(outcomes, oos_stats, robust, best),
    )


def _sensitivity(
    surface: Sequence[Tuple[Dict[str, Any], float]], grid: Dict[str, Sequence[Any]]
) -> Dict[str, List[Tuple[Any, float, int]]]:
    """Median score per parameter value, best first -- the plateau map."""
    out: Dict[str, List[Tuple[Any, float, int]]] = {}
    for key, values in grid.items():
        rows: List[Tuple[Any, float, int]] = []
        for v in values:
            scores = [s for p, s in surface if p.get(key) == v and s > float("-inf")]
            rows.append((v, statistics.median(scores) if scores else float("-inf"), len(scores)))
        rows.sort(key=lambda r: r[1], reverse=True)
        out[key] = rows
    return out


def _most_common_params(outcomes: Sequence[FoldOutcome]) -> Optional[Dict[str, Any]]:
    """The value each parameter took most often across winning folds."""
    if not outcomes:
        return None
    keys = outcomes[0].params.keys()
    picked: Dict[str, Any] = {}
    for k in keys:
        counts: Dict[Any, int] = {}
        for o in outcomes:
            counts[o.params[k]] = counts.get(o.params[k], 0) + 1
        picked[k] = max(counts.items(), key=lambda kv: kv[1])[0]
    return picked


def _notes(
    outcomes: Sequence[FoldOutcome],
    oos: Optional[PerformanceStats],
    robust: Dict[str, Any],
    best: Dict[str, Any],
) -> List[str]:
    notes: List[str] = []
    if not outcomes:
        return [
            "No fold produced enough trades to rank parameters. Either the data is "
            "too short, or the entry filters are too strict to ever fire."
        ]

    train = statistics.mean([o.train_score for o in outcomes])
    test = statistics.mean([o.test_score for o in outcomes])
    if train > 0 and test < 0.4 * train:
        notes.append(
            f"Out-of-sample score ({test:.2f}) is far below in-sample ({train:.2f}). "
            "The search is fitting noise -- shrink the grid before trusting it."
        )
    if test <= 0:
        notes.append(
            "Out-of-sample score is not positive. On this data the rules do not have "
            "an edge; do not trade them because the in-sample chart looked good."
        )
    if robust != best:
        notes.append(
            "The most frequently selected parameters differ from the most robust "
            "ones. Prefer the robust set -- it sits on a plateau rather than a spike."
        )
    if oos is not None and oos.is_statistically_thin:
        notes.append(
            f"Continuous out-of-sample run has {oos.n_trades} trades at t={oos.t_stat:+.2f}. "
            "Treat the result as a sanity check, not as validation."
        )
    return notes


def format_recommendation(rec: Recommendation, base: StrategyConfig) -> str:
    lines = [RULE, "Walk-forward recommendation", RULE]

    if not rec.folds:
        lines.append("No usable folds.")
        lines += [f"  ! {n}" for n in rec.notes]
        return "\n".join(lines) + "\n"

    lines.append("Per-fold selection (trained on everything before the test window):")
    lines.append(
        f"  {'fold':<5}{'train bars':<13}{'test bars':<13}{'train':>8}{'test':>8}"
        f"{'trades':>8}  params"
    )
    for o in rec.folds:
        params = ", ".join(f"{k}={v}" for k, v in sorted(o.params.items()))
        lines.append(
            f"  {o.fold:<5}"
            f"{f'{o.train_span[0]}-{o.train_span[1]}':<13}"
            f"{f'{o.test_span[0]}-{o.test_span[1]}':<13}"
            f"{o.train_score:>8.2f}{o.test_score:>8.2f}{o.test_stats.n_trades:>8}  {params}"
        )

    lines += [
        THIN,
        f"Mean in-sample score      {rec.mean_train_score:>8.2f}",
        f"Mean out-of-sample score  {rec.mean_test_score:>8.2f}",
        THIN,
        "Parameter plateaus (median score across all other settings):",
    ]
    for key, rows in sorted(rec.param_sensitivity.items()):
        rendered = "   ".join(
            f"{v}: {s:.2f}" if s > float("-inf") else f"{v}: n/a" for v, s, _ in rows
        )
        lines.append(f"  {key:<26} {rendered}")

    if rec.mean_test_score > 0:
        lines += [THIN, "Recommended configuration:"]
    else:
        lines += [
            THIN,
            "Least-bad configuration -- NOT a recommendation. Every option scored",
            "negative out of sample, so this is the pick you would make if forced,",
            "not a set of parameters worth trading:",
        ]
    for k, v in sorted(rec.robust_params.items()):
        lines.append(f"  {k:<26} {v}")

    if rec.oos_stats is not None:
        s = rec.oos_stats
        lines += [
            THIN,
            "Continuous out-of-sample run with those parameters:",
            f"  {s.n_trades} trades, {s.win_rate * 100:.1f}% win rate, "
            f"{s.expectancy_r:+.3f} R expectancy, t={s.t_stat:+.2f}, "
            f"profit factor {s.profit_factor:.2f}",
        ]

    if rec.notes:
        lines += [THIN, "Caveats:"]
        lines += [f"  ! {n}" for n in rec.notes]

    lines.append(RULE)
    return "\n".join(lines) + "\n"
